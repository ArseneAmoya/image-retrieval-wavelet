import torch
import torch.nn as nn

class DINOHashBaseline(nn.Module):
    def __init__(self, dino_backbone='dinov2_vits14', embed_dim=384, binary_config={'nbits': 64}, frozen=False, **kwargs):
        super().__init__()

        self.backbone = torch.hub.load('facebookresearch/dinov2', dino_backbone)

        # Was: forward() called getattr(self.backbone, 'frozen', True), but nothing
        # ever set that attribute on the backbone -- the constructor arg below only
        # touches requires_grad. The getattr therefore always fell through to its
        # default True, so set_grad_enabled(not True) disabled gradients on EVERY
        # forward and the backbone was frozen at its pretrained weights regardless of
        # this flag. That silently turned any DINOHashBaseline run into a
        # frozen-feature + linear-head baseline (e.g. mflickr_vitb_capacity_control,
        # which is meant to be a fine-tuned parameter-matched control against a fully
        # fine-tuned MBW-DINO). Store the flag on self instead.
        self.frozen = frozen

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
        # `and torch.is_grad_enabled()` so this never *re-enables* grad inside an
        # outer torch.no_grad() block (evaluate.py / compute_all_embeddings run under
        # no_grad; re-enabling there would silently build a graph and waste memory).
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            features = self.backbone(x)

        if isinstance(features, dict):
            features = features['x_norm_clstoken']

        logits = self.hash_head(features)

        if self.training:
            return logits
        else:
            return torch.sign(logits)
