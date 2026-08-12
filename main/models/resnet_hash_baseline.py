import timm
import torch
import torch.nn as nn


class ResNetHashBaseline(nn.Module):
    """ResNet50 counterpart of `DINOHashBaseline`, with a deliberately identical
    output contract so the two can be compared without an architectural confound.

    Why not reuse `config/model/resnet_hashing.yaml` (`backbone_name: resnet50_tanh`):
    that path runs through `RetrievalNet`, whose `ResNetHashing` backbone already
    squashes its output, and `RetrievalNet.forward` then L2-normalizes it (the
    `'hash' in self.backbone_name` early-return does not fire for the literal name
    `resnet50_tanh`). Paired with `HashLoss`, which applies `torch.tanh` itself, that
    stacks a second squashing on top of a normalization -- the same family of problem
    found in `SingleBandNet`. Comparing that against a clean `DINOHashBaseline` would
    measure the head plumbing, not the backbone.

    So this class mirrors DINOHashBaseline exactly:
      backbone -> pooled feature vector -> Linear(embed_dim, nbits, bias=not use_bn)
      -> BatchNorm1d(nbits) -> raw logits in training, sign() at eval.
    `HashLoss` applies the tanh; `SCHLoss` needs `apply_tanh: true` (see
    main/losses/dsch.py).
    """

    def __init__(self, backbone_name='resnet50', embed_dim=2048,
                 binary_config={'nbits': 64}, pretrained=True, frozen=False,
                 use_bn=True, **kwargs):
        super().__init__()
        self.frozen = frozen
        self.use_bn = use_bn

        # num_classes=0 -> timm returns the pooled feature vector (2048 for resnet50),
        # matching DINOv2's CLS token role in DINOHashBaseline.
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        feat_dim = getattr(self.backbone, 'num_features', embed_dim)
        if feat_dim != embed_dim:
            raise ValueError(
                f"embed_dim={embed_dim} does not match {backbone_name}'s feature "
                f"dimension ({feat_dim}). Fix the model config rather than silently "
                f"projecting from the wrong size."
            )

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            self.backbone.train = lambda mode=False: None

        nbits = binary_config['nbits']
        self.hash_head = nn.Sequential(
            nn.Linear(embed_dim, nbits, bias=not use_bn),
            nn.BatchNorm1d(nbits) if use_bn else nn.Identity(),
        )

    def forward(self, x):
        # `and torch.is_grad_enabled()` so this never re-enables grad inside an outer
        # no_grad block (evaluate.py), same guard as DINOHashBaseline.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            features = self.backbone(x)

        if isinstance(features, dict):
            features = features['x_norm_clstoken']
        if features.dim() > 2:
            features = features.flatten(1)

        logits = self.hash_head(features)

        return logits if self.training else torch.sign(logits)
