"""
Produit le fichier de centres SHC consommé par main/losses/center_hash_loss.py.

SHC (Chen et al., ACM TOIS 2025) est un pipeline en trois étages. Son étage 3
est *exactement* la loss de CSQ — leur `train.py` définit littéralement
`class CSQLoss`. Ce qui distingue SHC de CSQ, ce sont donc uniquement les
centres, produits par les étages 1 et 2. Ce script produit ces centres :

  Étage 1 — une matrice de similarité inter-classes S (C x C), dérivée des
            confusions d'un classifieur entraîné sur le jeu de données.
  Étage 2 — une optimisation ALM sous contrainte de distance minimale, qui
            transforme S en centres binaires. Code des auteurs, vendorisé tel
            quel dans main/losses/shc_center_generation.py.

GÉNÉRALISATION MULTI-LABEL DE L'ÉTAGE 1
Le code publié par les auteurs est mono-label de bout en bout :
`torch.max(targets, 1)` pour désigner LA classe vraie, `CrossEntropyLoss`, et
une accumulation dans une seule ligne de S par image. Appliqué tel quel à un
jeu multi-label comme MIRFLICKR, il assignerait chaque image à une seule de ses
classes, arbitrairement et silencieusement.

La généralisation appliquée ici est mécanique — partout où le code dit "la
classe vraie", on lit "l'ensemble des classes actives" :

  1. CrossEntropyLoss -> BCEWithLogitsLoss (le classifieur multi-label standard).
  2. Le masque porte sur TOUT l'ensemble actif avant le softmax, pas sur une
     seule classe. C'est ce qui préserve la sémantique d'origine : chez les
     auteurs, masquer l'unique classe vraie laisse un softmax sur des classes
     toutes absentes, ce qui mesure la CONFUSION. Ne masquer que la classe c
     laisserait les autres labels co-présents dominer, et l'on mesurerait la
     CO-OCCURRENCE — une tout autre quantité.
  3. La distribution obtenue est accumulée dans la ligne de chaque classe active.

Le cas mono-label est exactement retrouvé quand une seule classe est active,
donc ce script reproduit le comportement d'origine sur un jeu mono-label.

Usage (depuis la racine du dépôt) :

    python -m studies.generate_shc_centers \
        --dataset MIRFlickrHashing \
        --data_dir /content/mirflickr \
        --num_classes 38 \
        --bits 64 \
        --backbone resnet34 \
        --classifier_epochs 100 \
        --out data/shc_centers_mflickr_64.pt

La matrice S est mise en cache : elle ne dépend pas du nombre de bits, donc un
seul entraînement de classifieur suffit pour toutes les longueurs de code.
L'étage 2 coûte ~25 s sur CPU pour 38 classes et 64 bits.
"""
import argparse
import inspect
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from main import datasets
from main.losses.shc_center_generation import GenerateSemanticHashCenters


def build_dataset(name, data_dir, mode, transform):
    """Instancie un dataset du dépôt. `download` est propre à VOC2012Hashing,
    on ne le passe donc que si la classe le déclare."""
    cls = getattr(datasets, name)
    extra = {"download": False} if "download" in inspect.signature(cls.__init__).parameters else {}
    return cls(data_dir=data_dir, mode=mode, transform=transform, **extra)


def build_classifier(backbone, num_classes, device, freeze_backbone=False):
    if backbone == "resnet34":
        import torchvision.models as models
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif backbone == "dinov2":
        trunk = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if freeze_backbone:
            for p in trunk.parameters():
                p.requires_grad = False

        class DinoClassifier(nn.Module):
            def __init__(self, trunk, num_classes):
                super().__init__()
                self.trunk = trunk
                self.head = nn.Linear(384, num_classes)

            def forward(self, x):
                feats = self.trunk(x)
                return self.head(feats)

        net = DinoClassifier(trunk, num_classes)
    else:
        raise ValueError(f"backbone inconnu : {backbone}")
    return net.to(device)


def train_classifier(net, loader, device, epochs, lr):
    """Classifieur multi-label. Mêmes hyperparamètres que l'étage 1 des auteurs
    (RMSprop, weight_decay 1e-5, cosine annealing), seule la loss change :
    BCEWithLogits au lieu de CrossEntropy."""
    criterion = nn.BCEWithLogitsLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.RMSprop(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        net.train()
        total, running = 0, 0.0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            total += images.size(0)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [classifieur] epoch {epoch + 1}/{epochs}  BCE={running / total:.4f}")
    return net


@torch.no_grad()
def compute_similarity_matrix(net, loader, num_classes, device):
    """Étage 1 généralisé : pour chaque image, on masque TOUTES ses classes
    actives, on renormalise sur les classes absentes, et on accumule la
    distribution obtenue dans la ligne de chacune de ses classes actives."""
    net.eval()
    S = torch.zeros(num_classes, num_classes, device=device)
    n_skipped = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()
        logits = net(images)

        for i in range(images.size(0)):
            active = labels[i] > 0
            n_active = int(active.sum().item())
            if n_active == 0 or n_active == num_classes:
                # Rien à mesurer : soit aucune classe, soit toutes masquées.
                n_skipped += 1
                continue
            masked = logits[i].clone()
            masked[active] = float("-inf")
            q = F.softmax(masked, dim=0)          # distribution sur les classes ABSENTES
            S[active] += q                         # une contribution par classe active

    if n_skipped:
        print(f"  [S] {n_skipped} images ignorées (aucune classe active, ou toutes actives)")

    # Symétrisation et normalisation par ligne, exactement comme les auteurs.
    S = (S + S.T) / 2
    for i in range(num_classes):
        s_max, s_min, s_mean = S[i].max(), S[i].min(), S[i].mean()
        denom = max(abs((s_max - s_mean).item()), abs((s_min - s_mean).item()))
        if denom > 0:
            S[i] = (S[i] - s_mean) / denom
    S[torch.eye(num_classes, dtype=torch.bool, device=device)] = 1.0
    return S


def report_centers(H, bit):
    """Contrôle de sanité : si les centres sont tous exactement à bit/2 les uns
    des autres, l'étage 2 n'a rien fait et on a en réalité du CSQ/Hadamard."""
    C = H.t().contiguous() if H.shape[0] == bit else H
    d = (bit - C.float() @ C.float().t()) / 2
    off = d[~torch.eye(C.shape[0], dtype=torch.bool)]
    print(f"\n=== Géométrie des centres ({C.shape[0]} classes, {bit} bits) ===")
    print(f"  distance de Hamming  min={off.min():.1f}  moyenne={off.mean():.2f}  max={off.max():.1f}")
    if off.min() == off.max():
        print("  ATTENTION : toutes les paires sont à la même distance. L'étage 2 n'a "
              "probablement rien changé — ces centres sont équivalents à du Hadamard. "
              "Vérifiez que loss1 décroît, et ajustez eta.")
    else:
        print("  OK : les distances varient, l'étage 2 a bien structuré les centres.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Nom de classe dans main/datasets.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--mode", default="train",
                    help="Split utilisé pour l'étage 1 (défaut: train — celui sur lequel "
                         "la loss s'entraîne).")
    ap.add_argument("--backbone", default="resnet34", choices=["resnet34", "dinov2"],
                    help="resnet34 est fidèle au dépôt des auteurs ; dinov2 est cohérent "
                         "avec le reste du pipeline et plus rapide si gelé.")
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Ne concerne que dinov2 : n'entraîne que la tête linéaire.")
    ap.add_argument("--classifier_epochs", type=int, default=100,
                    help="100 = protocole des auteurs (défaut de leur run.py et de "
                         "leur README). A ne changer que pour une raison documentée.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--similarity_cache", default=None,
                    help="Chemin du cache de S. S ne dépend pas du nombre de bits : "
                         "un seul entraînement sert à toutes les longueurs de code.")
    ap.add_argument("--out", required=True, help="Fichier .pt de centres à écrire.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")

    cache = args.similarity_cache or os.path.join(
        os.path.dirname(os.path.abspath(args.out)) or ".",
        f"shc_similarity_{args.dataset}_{args.mode}.pt")

    # ---------- Étage 1 ----------
    if os.path.exists(cache):
        print(f"[étage 1] matrice S déjà calculée, rechargée depuis {cache}")
        S = torch.load(cache, map_location=device)
    else:
        print("[étage 1] entraînement du classifieur multi-label…")
        transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        dts = build_dataset(args.dataset, args.data_dir, args.mode, transform)
        loader = DataLoader(dts, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=False)
        print(f"  {len(dts)} images, {args.num_classes} classes")

        net = build_classifier(args.backbone, args.num_classes, device, args.freeze_backbone)
        net = train_classifier(net, loader, device, args.classifier_epochs, args.lr)

        print("[étage 1] calcul de la matrice de similarité…")
        eval_loader = DataLoader(dts, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        S = compute_similarity_matrix(net, eval_loader, args.num_classes, device)
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        torch.save(S.cpu(), cache)
        print(f"  S écrite dans {cache}")

    S = S.to(device)

    # ---------- Étage 2 ----------
    print("\n[étage 2] optimisation des centres (code des auteurs, vendorisé)…")
    print("  surveillez loss1 : si elle ne décroît pas, ajustez eta dans "
          "main/losses/shc_center_generation.py")

    class _Args:
        code_length = args.bits
        num_classes = args.num_classes
        dataset = f"{args.dataset}_{args.mode}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = GenerateSemanticHashCenters(_Args(), S)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(H.cpu(), args.out)
    print(f"\ncentres écrits dans {args.out}  (forme {tuple(H.shape)})")
    report_centers(H.cpu(), args.bits)
    print("\nÀ utiliser via :  loss.0.kwargs.centers: " + args.out)


if __name__ == "__main__":
    main()
