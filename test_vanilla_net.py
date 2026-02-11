import torch
import os
from omegaconf import OmegaConf
from roadmap.getter import Getter
from roadmap.engine.evaluate import evaluate
from roadmap.utils import get_set_random_state

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Chemins relatifs basés sur votre structure de fichiers
MODEL_CONFIG_PATH = 'config/model/dino_v3.yaml' 
DATASET_CONFIG_PATH = 'config/dataset/cifar100.yaml'
TRANSFORM_CONFIG_PATH = 'config/transform/cifar_resize.yaml'

def test_dino_zeroshot():
    print(f"\n=== Lancement de l'évaluation Zero-Shot DINOv3 sur CIFAR100 ===")
    print(f"Device: {DEVICE}")
    
    # 1. Initialisation du Getter
    getter = Getter()
    
    # 2. Chargement des configurations
    print("-> Chargement des fichiers de configuration...")
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError(f"Config introuvable: {MODEL_CONFIG_PATH}")
        
    model_cfg = OmegaConf.load(MODEL_CONFIG_PATH)
    data_cfg = OmegaConf.load(DATASET_CONFIG_PATH)
    transform_cfg = OmegaConf.load(TRANSFORM_CONFIG_PATH)

    # 3. Création des Transformations
    # ASTUCE CRITIQUE : Pour une évaluation Zero-Shot "propre" sur le Train, 
    # on utilise la transformation de TEST (Resize + CenterCrop) pour le train aussi.
    # Cela évite que le RandomCrop ne fausse les métriques de clustering.
    print("-> Création des transformations (Mode Test pour tout le monde)...")
    
    # On récupère uniquement la transfo de test
    eval_transform = getter.get_transform(transform_cfg.test)
    
    # 4. Création des Datasets
    print("-> Chargement des datasets CUB...")
    # On force l'utilisation de la transfo 'eval_transform' pour le train set aussi
    train_dataset = getter.get_dataset(eval_transform, "train", data_cfg) 
    test_dataset = getter.get_dataset(eval_transform, "test", data_cfg)
    
    print(f"   Train set size: {len(train_dataset)} images")
    print(f"   Test set size: {len(test_dataset)} images")

    # 5. Chargement du Modèle (Foundation Model)
    print("-> Chargement du modèle DINOv2 (Poids pré-entraînés)...")
    model = getter.get_model(model_cfg).to(DEVICE)
    
    # CRITIQUE : Passer en mode eval()
    # Dans votre classe DinoModel_ce, la méthode forward() contient :
    # if self.training: return logits
    # else: return F.normalize(features['x_norm_clstoken'])
    # Le mode eval() est donc OBLIGATOIRE pour obtenir les features et non les classes.
    model.eval() 
    print("   Modèle passé en mode eval() -> Output: Normalized Embeddings")

    # 6. Lancement de l'évaluation
    print("-> Calcul des métriques (Recall@K, MAP)...")
    
    # La fonction evaluate() de votre moteur gère tout :
    # Elle extrait les embeddings par batch et calcule les distances.
    results = evaluate(
        net=model,
        train_dataset=train_dataset, # Évalue sur le train (Zero-shot clustering)
        test_dataset=test_dataset,   # Évalue sur le test (Zero-shot generalization)
        batch_size=32,       # Ajustez selon votre VRAM (32 passe sur 8Go, 64+ sur 24Go)
        num_workers=4,       
        k=2048,                # On regarde le Recall jusqu'à 2048 voisins
        exclude_ranks=None   
    )

    # 7. Affichage propre des résultats
    print("\n" + "="*50)
    print(" RÉSULTATS ZERO-SHOT (DINOv2 sans fine-tuning)")
    print("="*50)
    
    # evaluate() retourne un dictionnaire imbriqué
    # Structure typique : results['test']['precision_at_1']
    
    for split_name, metrics in results.items():
        print(f"\n--- Split: {split_name.upper()} ---")
        # On filtre pour afficher les métriques les plus parlantes
        keys_to_show = ["precision_at_1", "map_at_r", "r_precision", "mean_average_precision", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "recall_at_100", "recall_at_2", "recall_at_4", "recall_at_8"]
        
        found = False
        for k, v in metrics.items():
            if any(x in k for x in keys_to_show):
                print(f"  {k:<30} : {v:.4f}")
                found = True
        
        if not found: # Si les noms sont différents, on affiche tout
             for k, v in metrics.items():
                 print(f"  {k:<30} : {v:.4f}")

if __name__ == "__main__":
    # Fixer l'aléatoire via votre utilitaire pour reproductibilité
    # Note: get_set_random_state est un décorateur dans votre code, 
    # mais on peut appeler torch.manual_seed directement si besoin.
    torch.manual_seed(42)
    test_dino_zeroshot()