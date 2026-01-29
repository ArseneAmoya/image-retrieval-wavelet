import torch.nn as nn
from roadmap.getter import Getter
from omegaconf import OmegaConf
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_transforms(transform_cfg_path='config/transform/cub_dwt_dec_resize.yaml'):
    # Charge la config
    transform_cfg = OmegaConf.load(transform_cfg_path)
    print(transform_cfg)
    # Crée les transformations via le getter
    getter = Getter()
    train_transform = getter.get_transform(transform_cfg.train)
    test_transform = getter.get_transform(transform_cfg.test)
    return train_transform, test_transform


def freeze_batch_norm(model):
    for module in filter(lambda m: type(m) == nn.BatchNorm2d, model.modules()):
        module.eval()
        module.train = lambda _: None
        print(f"Freezing {module} in {model.__class__.__name__}")
    return model

def test_retrievalnet_with_wresnet(freeze_bn=False):
    import torch
    from roadmap.models.net import RetrievalNet

    # Paramètres du batch
    batch_size = 15
    channels = 3
    height = 224
    width = 224

    # Batch d'images aléatoires
    x = Image.open("../../data/car.jpg").convert('RGB')
    transforms, _ = test_transforms()
    x_transformed = transforms(x).to(device)

    print("Input shape:", x_transformed.shape)  # Doit être [3, H, W]

    model_configs = OmegaConf.load('config/model/multi_dino.yaml')
    getter = Getter()
    # Instanciation du modèle RetrievalNet avec wresnet
    model = getter.get_model(model_configs).to(device)
    if freeze_bn:
        freeze_batch_norm(model)
        print("BatchNorm layers have been frozen.")
    
    # for modules in model.modules():
    #         if isinstance(modules, nn.BatchNorm2d) or isinstance(modules, nn.SyncBatchNorm):
    #             print("BatchNorm layer found:", modules)

    print(model)

    # FourBranchResNet expects input of shape [Batch, 3, 4, H, W]
    # Replicate the image 4 times for the 4 branches
    batch_size = 1
    print("--- Vérification des couches sélectionnées ---")
    i = 0
    for name, param in model.named_parameters():
        # Votre filtre actuel (qui marche grâce au substring)
        if "att_block" in name: 
            print(f"[{i}] Trouvé : {name}")
            print(f"    Shape: {param.shape}")
            print(f"type: {type(param)}")
            i += 1
    
    model.eval()
    with torch.no_grad():
        output = model(x_transformed.unsqueeze(0))  # Shape [batch_size, 3, H, W]
    
    if isinstance(output, list):
        print("Output is a list of embeddings from each branch.")
        for i, out in enumerate(output):
            print(f"Branch {i} output shape:", out.shape)  # Doit être [batch_size, embed_dim]
    else:
        print("Output shape:", output.shape)  # Doit être [batch_size, embed_dim]
    

if __name__ == "__main__":
    # Simule config.freeze_batch_norm=True
    test_retrievalnet_with_wresnet(freeze_bn=False)
