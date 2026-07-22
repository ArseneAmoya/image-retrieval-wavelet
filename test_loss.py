import torch
import torch.optim as optim
import sys
import os
from omegaconf import OmegaConf

try:
    from roadmap.getter import Getter
except ImportError:
    print("Erreur : Lancez ce script depuis la racine du projet 'image-retrieval-wavelet'.")
    sys.exit(1)

def test_config_loading():
    yaml_rel_path = "config/loss/multi_roadmap_distillated.yaml"

    print(f"\n{'='*20} TEST CONFIG: {yaml_rel_path} {'='*20}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(yaml_rel_path):
        print(f"ERREUR FATALE : Le fichier '{yaml_rel_path}' est introuvable.")
        print("   Vérifiez que vous êtes bien à la racine du projet.")
        return

    print(f"-> Chargement via OmegaConf...")
    loss_cfg = OmegaConf.load(yaml_rel_path)

    params_to_inject = {
        "num_classes": 10,
        "embedding_size": 128,
        "writer": None
    }

    print("-> Instanciation via Getter().get_loss()...")
    getter = Getter()

    try:
        criterion_list = getter.get_loss(loss_cfg)

        print(f"Structure retournée par Getter: {type(criterion_list)}")

        if isinstance(criterion_list, list) and len(criterion_list) > 0:
            criterion, weight = criterion_list[0]
            print(f"Loss récupérée (Poids: {weight})")
        else:
            criterion = criterion_list

        criterion = criterion.to(device)
        print(f"Loss prête : {type(criterion).__name__}")

        if hasattr(criterion, 'losses'):
            print(f"   Nombre de branches détectées : {len(criterion.losses)}")
            if hasattr(criterion, 'branch_weights'):
                 print(f"   Poids globaux des branches : {criterion.branch_weights}")

    except Exception as e:
        print(f"ÉCHEC de l'initialisation : {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n-> Test Forward/Backward (Validation du graphe)...")

    batch_size = 8
    num_classes = params_to_inject["num_classes"]
    embed_dim = params_to_inject["embedding_size"]

    if hasattr(criterion, 'losses'):
        num_branches = len(criterion.losses)
    else:
        num_branches = 1

    print(f"   Simulation pour {num_branches} branches.")

    inputs_list = []
    layers = []

    x = torch.randn(batch_size, embed_dim).to(device)

    for i in range(num_branches):
        layer = torch.nn.Linear(embed_dim, embed_dim).to(device)
        layers.append(layer)

        out = layer(x)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        inputs_list.append(out)

    targets = torch.randint(0, num_classes, (batch_size,)).to(device)

    optimizer = optim.SGD([p for l in layers for p in l.parameters()], lr=0.01)
    optimizer.zero_grad()

    try:
        loss_output = criterion(inputs_list, targets)

        if isinstance(loss_output, tuple):
            loss_val = loss_output[0]
        else:
            loss_val = loss_output

        print(f"   Loss value : {loss_val.item():.4f}")

        loss_val.backward()
        print("Backward réussi.")

        if len(layers) > 0 and layers[0].weight.grad is not None and layers[0].weight.grad.norm() > 0:
            print("Les gradients remontent bien dans les branches.")
        else:
            print("Attention : Gradient nul ou non détecté sur la branche 0.")

    except Exception as e:
        print(f"Erreur lors du calcul : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_loading()
