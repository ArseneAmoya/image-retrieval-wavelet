import torch
import os
from omegaconf import OmegaConf
from main.getter import Getter
from main.engine.evaluate import evaluate
from main.utils import get_set_random_state

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CONFIG_PATH = 'config/model/dino.yaml'
DATASET_CONFIG_PATH = 'config/dataset/cifar100.yaml'
TRANSFORM_CONFIG_PATH = 'config/transform/cifar_resize.yaml'

def test_dino_zeroshot():
    print(f"\n=== Lancement de l'évaluation Zero-Shot DINOv3 sur CIFAR100 ===")
    print(f"Device: {DEVICE}")

    getter = Getter()

    print("-> Chargement des fichiers de configuration...")
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError(f"Config introuvable: {MODEL_CONFIG_PATH}")

    model_cfg = OmegaConf.load(MODEL_CONFIG_PATH)
    data_cfg = OmegaConf.load(DATASET_CONFIG_PATH)
    transform_cfg = OmegaConf.load(TRANSFORM_CONFIG_PATH)

    print("-> Création des transformations (Mode Test pour tout le monde)...")

    # Using the test transform (Resize + CenterCrop) for the train split too, so RandomCrop
    # doesn't distort the zero-shot clustering metrics.
    eval_transform = getter.get_transform(transform_cfg.test)

    print("-> Chargement des datasets CUB...")
    train_dataset = getter.get_dataset(eval_transform, "train", data_cfg)
    test_dataset = getter.get_dataset(eval_transform, "test", data_cfg)

    print(f"   Train set size: {len(train_dataset)} images")
    print(f"   Test set size: {len(test_dataset)} images")

    print("-> Chargement du modèle DINOv2 (Poids pré-entraînés)...")
    model = getter.get_model(model_cfg).to(DEVICE)

    # DinoModel_ce.forward() returns class logits in train() mode and normalized
    # embeddings in eval() mode, so eval() is required here.
    model.eval()
    print("   Modèle passé en mode eval() -> Output: Normalized Embeddings")

    print("-> Calcul des métriques (Recall@K, MAP)...")

    results = evaluate(
        net=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=4,
        k=54000,
        exclude_ranks=None
    )

    print("\n" + "="*50)
    print(" RÉSULTATS ZERO-SHOT (DINOv2 sans fine-tuning)")
    print("="*50)

    for split_name, metrics in results.items():
        print(f"\n--- Split: {split_name.upper()} ---")
        keys_to_show = ["precision_at_1", "map_at_r", "r_precision", "mean_average_precision", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "recall_at_100", "recall_at_2", "recall_at_4", "recall_at_8"]

        found = False
        for k, v in metrics.items():
            if any(x in k for x in keys_to_show):
                print(f"  {k:<30} : {v:.4f}")
                found = True

        if not found:
             for k, v in metrics.items():
                 print(f"  {k:<30} : {v:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    test_dino_zeroshot()
