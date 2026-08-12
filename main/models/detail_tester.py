import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DetailTesterNet(nn.Module):
    def __init__(self, backbone_name='resnet50', detail_index=1, output_dim=64, **kwargs):
        """
        Args:
            backbone_name: 'resnet50', 'convnextv2_tiny.fcmae', or 'dinov2_vits14'
            detail_index: 1 for LH, 2 for HL, 3 for HH (0 is LL / the global image)
            output_dim: hashing dimension (e.g. 64 bits)
        """
        super().__init__()
        self.detail_index = detail_index

        if 'dinov2' in backbone_name:
            self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
            dim = self.backbone.embed_dim
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
            self.backbone.forward = self.backbone.forward_features if hasattr(self.backbone, 'forward_features') else self.backbone.forward
            dim = self.backbone.num_features

        self.pool = nn.AdaptiveAvgPool2d(1) if not 'dinov2' in backbone_name else nn.Identity()

        self.bn = nn.BatchNorm1d(dim)
        self.hash_fc = nn.Linear(dim, output_dim)

        nn.init.normal_(self.hash_fc.weight, std=0.01)
        nn.init.constant_(self.hash_fc.bias, 0)

    def forward(self, x):
        # x is the SWT output: (B, Channels=3, Sub-bands=4, H, W); we extract the target sub-band.
        detail_band = x[:, :, self.detail_index, :, :]

        if isinstance(self.backbone, timm.models.resnet.ResNet) or 'convnext' in self.backbone.__class__.__name__.lower():
            feat_map = self.backbone(detail_band)
            feat = self.pool(feat_map).flatten(1)
        else:
            out = self.backbone(detail_band)
            feat = out['x_norm_clstoken'] if isinstance(out, dict) else out

        feat = self.bn(feat)
        logits = self.hash_fc(feat)

        if self.training:
            return torch.tanh(logits)
        else:
            return torch.sign(logits)



class SingleBandNet(nn.Module):
    """Single DINOv2 backbone on one SWT sub-band.

    When `is_hashing=True` the head is deliberately built to mirror
    `MultiDinoHashing` (main/models/multi_dino_attention.py) exactly, because the
    whole point of mflickr_single_band_ablation is that its mAP is directly
    comparable to the full fusion model's. Two things have to match:

    1. BatchNorm before the hash projection (`use_bn`, default True, same as
       MultiDinoHashing's `use_bn: true`). BN centers each bit via batch statistics;
       without it the bits drift off-center and bit_balance -- one of the metrics
       this study reports -- is not measuring the same thing as in the reference.
    2. **No tanh in the model.** `HashLoss.forward` (main/losses/hash_loss.py) already
       applies `torch.tanh(embeddings)` itself. MultiDinoHashing therefore returns raw
       logits during training. This class used to return `torch.tanh(out)`, which made
       the effective activation `tanh(tanh(x))`: bounded by tanh(1) = 0.7616, so the
       quantization term `mean(||emb| - 1|)` could never fall below ~0.238 and pushed
       the weights up forever while the gradient was squashed twice.

    The `is_hashing=False` path (plain L2-normalized embeddings) is unchanged.
    """

    def __init__(self, detail_index=0, output_dim=384, is_hashing=False, use_bn=True, **kwargs):
        super().__init__()
        self.detail_index = detail_index
        self.is_hashing = is_hashing
        self.use_bn = use_bn if is_hashing else False

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        embed_dim = self.backbone.embed_dim

        if output_dim == embed_dim and not is_hashing:
            self.head = nn.Identity()
        else:
            # bias=not use_bn matches MultiDinoHashing: BN's own shift makes the
            # linear bias redundant (and BN would cancel it anyway).
            self.head = nn.Linear(embed_dim, output_dim, bias=not self.use_bn)
            nn.init.normal_(self.head.weight, std=0.01)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

        self.bn = nn.BatchNorm1d(output_dim) if self.use_bn else nn.Identity()

    def forward(self, x):
        if x.dim() == 5:
            x = x[:, :, self.detail_index, :, :]

        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['x_norm_clstoken']

        out = self.head(features)

        if self.is_hashing:
            out = self.bn(out)
            # Raw logits in training -- HashLoss applies the tanh. Mirrors
            # MultiDinoHashing.forward's `return logits if self.training else sign`.
            return out if self.training else torch.sign(out)
        else:
            return F.normalize(out, p=2, dim=1)
