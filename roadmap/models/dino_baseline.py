import torch
import torch.nn as nn

class DINOHashBaseline(nn.Module):
    def __init__(self, dino_backbone='dinov2_vits14', embed_dim=384, binary_config={'nbits': 64}, frozen=False, **kwargs):
        super().__init__()

        self.backbone = torch.hub.load('facebookresearch/dinov2', dino_backbone)

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

            self.backbone.train = lambda mode=False: None

        self.hash_head = nn.Sequential(
            nn.Linear(embed_dim, binary_config['nbits'], bias=False),
            nn.BatchNorm1d(binary_config['nbits'])
        )

    def forward(self, x):
        with torch.set_grad_enabled(not getattr(self.backbone, 'frozen', True)):
            features = self.backbone(x)

        if isinstance(features, dict):
            features = features['x_norm_clstoken']

        logits = self.hash_head(features)

        if self.training:
            return logits
        else:
            return torch.sign(logits)
