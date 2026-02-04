import torch
import torch.optim as optim
import sys
import os
from omegaconf import OmegaConf

# Assurez-vous d'être à la racine pour les imports
try:
    from roadmap.getter import Getter
except ImportError:
    print("❌ Erreur : Lancez ce script depuis la racine du projet 'image-retrieval-wavelet'.")
    sys.exit(1)

def test_config_loading():
    # --- 1. CONFIGURATION ---
    # Le fichier que vous voulez tester
    yaml_rel_path = "config/loss/multi_roadmap_distillated.yaml" 
    
    print(f"\n{'='*20} TEST CONFIG: {yaml_rel_path} {'='*20}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Vérification de l'existence du fichier
    if not os.path.exists(yaml_rel_path):
        print(f"❌ ERREUR FATALE : Le fichier '{yaml_rel_path}' est introuvable.")
        print("   Vérifiez que vous êtes bien à la racine du projet.")
        return

    # --- 2. CHARGEMENT ET PRÉPARATION OMEGACONF ---
    print(f"-> Chargement via OmegaConf...")
    loss_cfg = OmegaConf.load(yaml_rel_path)
    
    # On force des valeurs par défaut pour le test (simulation de ce que fait run.py)
    params_to_inject = {
        "num_classes": 10,
        "embedding_size": 128,  # Taille fictive pour le test
        "writer": None          # Si votre loss essaie d'écrire des logs
    }
    
    # Injection des dépendances (Commentée comme demandé, mais souvent nécessaire si le YAML a des ${})
    # print("-> Injection des dépendances (num_classes, embedding_size)...")
    # loss_cfg = OmegaConf.merge(loss_cfg, OmegaConf.create(params_to_inject))

    # --- 3. INSTANCIATION VIA GETTER ---
    print("-> Instanciation via Getter().get_loss()...")
    getter = Getter()
    
    try:
        # C'est le moment de vérité : Getter va lire le "name" dans le YAML et appeler MultiLoss
        # Le getter retourne généralement une liste de tuples [(loss_fn, weight), ...]
        criterion_list = getter.get_loss(loss_cfg)
        
        print(f"Structure retournée par Getter: {type(criterion_list)}")
        
        # On suppose que le Getter retourne une liste contenant notre MultiLoss comme premier élément
        # Ou alors, si MultiLoss est la loss principale, elle est peut-être le premier élément du tuple
        if isinstance(criterion_list, list) and len(criterion_list) > 0:
            # Cas standard : [(MultiLoss(...), 1.0)]
            criterion, weight = criterion_list[0]
            print(f"✅ Loss récupérée (Poids: {weight})")
        else:
            # Cas direct (peu probable avec votre Getter mais possible)
            criterion = criterion_list

        criterion = criterion.to(device)
        print(f"✅ Loss prête : {type(criterion).__name__}")
        
        # Inspection rapide spécifique à MultiLoss
        if hasattr(criterion, 'losses'):
            print(f"   Nombre de branches détectées : {len(criterion.losses)}")
            if hasattr(criterion, 'branch_weights'):
                 print(f"   Poids globaux des branches : {criterion.branch_weights}")
            
    except Exception as e:
        print(f"❌ ÉCHEC de l'initialisation : {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. SIMULATION D'ENTRAÎNEMENT ---
    print("\n-> Test Forward/Backward (Validation du graphe)...")
    
    # On simule un batch : 8 images, dimension 128
    batch_size = 8
    # Si les params n'ont pas été injectés dans la config, on utilise des valeurs par défaut pour les tenseurs
    num_classes = params_to_inject["num_classes"]
    embed_dim = params_to_inject["embedding_size"]
    
    # IMPORTANT : MultiLoss attend une LISTE d'embeddings (un par branche)
    # On détecte le nombre de branches pour adapter l'input
    if hasattr(criterion, 'losses'):
        num_branches = len(criterion.losses)
    else:
        num_branches = 1 # Fallback
        
    print(f"   Simulation pour {num_branches} branches.")
    
    inputs_list = []
    layers = []
    
    # Input source
    x = torch.randn(batch_size, embed_dim).to(device)
    
    # On crée une petite couche par branche pour vérifier les gradients
    for i in range(num_branches):
        layer = torch.nn.Linear(embed_dim, embed_dim).to(device)
        layers.append(layer)
        
        # Forward pass simulé + Normalisation (souvent requise en retrieval)
        out = layer(x)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        inputs_list.append(out)
        
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Optimiseur
    optimizer = optim.SGD([p for l in layers for p in l.parameters()], lr=0.01)
    optimizer.zero_grad()

    try:
        # Appel de la loss
        # MultiLoss attend (inputs_list, targets)
        loss_output = criterion(inputs_list, targets)
        
        # Gestion tuple/scalar (certaines losses retournent (loss, logs))
        if isinstance(loss_output, tuple):
            loss_val = loss_output[0]
        else:
            loss_val = loss_output
            
        print(f"   Loss value : {loss_val.item():.4f}")
        
        # Backward
        loss_val.backward()
        print("✅ Backward réussi.")
        
        # Vérif gradient branche 0
        if len(layers) > 0 and layers[0].weight.grad is not None and layers[0].weight.grad.norm() > 0:
            print("✅ Les gradients remontent bien dans les branches.")
        else:
            print("⚠️ Attention : Gradient nul ou non détecté sur la branche 0.")

    except Exception as e:
        print(f"❌ Erreur lors du calcul : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_loading()