import torch.nn as nn


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
    x = torch.randn(batch_size, channels, height, width)

    # Instanciation du modèle RetrievalNet avec wresnet
    model = RetrievalNet(
        backbone_name='wresnet',
        embed_dim=512,
        norm_features=False,
        without_fc=False,
        with_autocast=False,
        pooling='default',
        projection_normalization_layer='none',
        pretrained=False  # ou True si tu veux charger des poids pré-entraînés
    )

    if freeze_bn:
        freeze_batch_norm(model)
        print("BatchNorm layers have been frozen.")

    model.eval()
    with torch.no_grad():
        output = model(x)
    print("Output shape:", output.shape)  # Doit être [batch_size, embed_dim]

if __name__ == "__main__":
    # Simule config.freeze_batch_norm=True
    test_retrievalnet_with_wresnet(freeze_bn=True)
