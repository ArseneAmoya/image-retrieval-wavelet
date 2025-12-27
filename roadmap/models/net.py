import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

import roadmap.utils as lib

from .create_projection_head import create_projection_head
from .wresnet import WaveResNet, WaveResNetCE, WCNN, WCNN_Attention, WCNN_Attention_CE
from .resnet_ce import ResNetCE
from .mtwavenet import FourBranchResNet, FourBranchResNet50, FourBranchResNet50Fusion, HybridMultiBranch


def get_backbone(name, pretrained=True, **kwargs):
    if name == 'resnet18':
        lib.LOGGER.info("using ResNet-18")
        out_dim = 512
        backbone = models.resnet18(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'resnet50':
        lib.LOGGER.info("using ResNet-50")
        out_dim = 2048
        backbone = models.resnet50(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'resnet101':
        lib.LOGGER.info("using ResNet-101")
        out_dim = 2048
        backbone = models.resnet101(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'vit':
        lib.LOGGER.info("using ViT-S")
        out_dim = 768
        backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit':
        lib.LOGGER.info("using DeiT-S")
        out_dim = 384
        try:
            backbone = timm.create_model('vit_deit_small_patch16_224', pretrained=pretrained)
        except RuntimeError:
            backbone = timm.create_model('deit_small_patch16_224', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit_distilled':
        lib.LOGGER.info("using DeiT-S distilled")
        try:
            deit = timm.create_model('vit_deit_small_patch16_224')
            deit_distilled = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=pretrained)
        except RuntimeError:
            deit = timm.create_model('deit_small_patch16_224')
            deit_distilled = timm.create_model('deit_small_distilled_patch16_224', pretrained=pretrained)
        deit_distilled.pos_embed = torch.nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 384
        pooling = nn.Identity()
    elif name == 'vit_deit_base':
        lib.LOGGER.info("using DeiT-B")
        out_dim = 768
        try:
            backbone = timm.create_model('vit_deit_base_patch16_224', pretrained=pretrained)
        except RuntimeError:
            backbone = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit_base_distilled':
        lib.LOGGER.info("using DeiT-B distilled")
        try:
            deit = timm.create_model('deit_base_patch16_224')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
        except RuntimeError:
            deit = timm.create_model('deit_base_patch16_224')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
        deit_distilled.pos_embed = torch.nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 768
        pooling = nn.Identity()
    elif name == 'vit_deit_base_384':
        lib.LOGGER.info("using DeiT-B 384")
        out_dim = 768
        try:
            backbone = timm.create_model('vit_deit_base_patch16_384', pretrained=pretrained)
        except RuntimeError:
            backbone = timm.create_model('deit_base_patch16_384', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit_base_384_distilled':
        lib.LOGGER.info("using DeiT-B 384 distilled")
        try:
            deit = timm.create_model('deit_base_patch16_384')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_384', pretrained=pretrained)
        except RuntimeError:
            deit = timm.create_model('deit_base_patch16_384')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_384', pretrained=pretrained)
        deit_distilled.pos_embed = torch.nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 768
        pooling = nn.Identity()
    elif name == 'wresnet':
        lib.LOGGER.info(f"using WResNet, attention : {kwargs.get('attention', True)}, decom_level :, {kwargs.get('decom_level', 3)}, wave :,{ kwargs.get('wave', 'haar')} feature size : {kwargs.get('feature_size', 512)} {kwargs.get('feature_size', 512)}")
        out_dim = 2048
        backbone = WaveResNet(**kwargs)#(decom_level=2, wave='haar',ll_only=False, attention=True)
        pooling = nn.Identity()
    elif name == 'wresnet_ce':
        lib.LOGGER.info(f"using WResNet, attention : {kwargs.get('attention', True)}, decom_level :, {kwargs.get('decom_level', 3)}, wave :,{ kwargs.get('wave', 'haar')} feature size : {kwargs.get('feature_size', 512)} {kwargs.get('feature_size', 512)}")
        out_dim = 512
        backbone = WaveResNetCE(**kwargs)#(decom_level=2, wave='haar',ll_only=False, attention=True)
        pooling = nn.Identity()
    elif name == 'wcnn_attention_ce':
        lib.LOGGER.info(f"using WCNN with Attention for cross entropy, decom_level :, {kwargs.get('decom_level', 'not specified')}, wave :,{ kwargs.get('wave', 'haar')} feature size : {kwargs.get('feature_size', 'unspecidied')} {kwargs.get('feature_size', 'unspecified')}, coarse_only : {kwargs.get('coarse_only', False)}   ")
        out_dim = 512
        backbone = WCNN_Attention_CE(pretrained=pretrained,**kwargs)#(decom_level=2, wave='haar',ll_only=False, attention=True)
        pooling = nn.Identity()
    elif name == 'wcnn':
        lib.LOGGER.info(f"using WCNN, decom_level :, {kwargs.get('decom_level', 3)}, wave :,{ kwargs.get('wave', 'haar')} feature size : {kwargs.get('feature_size', 512)} {kwargs.get('feature_size', 512)}")
        out_dim = 2048
        backbone = WCNN(pretrained=pretrained,**kwargs)#(decom_level=2, wave='haar',ll_only=False, attention=True)
        pooling = nn.Identity()
    # Dans get_backbone(...)
    elif name == 'resnet_ce':
        lib.LOGGER.info("using ResNet-CE (Boudiaf et al. reproduction)")
        # On récupère num_classes et dropout depuis kwargs
        num_classes = kwargs.pop('num_classes', None)
        dropout = kwargs.pop('dropout', 0.5)
        
        if num_classes is None:
            raise ValueError("ResNet-CE requires 'num_classes' to be defined in kwargs")

        # Instanciation propre
        backbone = ResNetCE(
            num_classes=num_classes, 
            dropout=dropout, 
            pretrained=pretrained,
            freeze_bn=True, # Forcé à True selon le papier
            **kwargs
        )
        out_dim = 2048
        pooling = nn.Identity() # Le pooling est déjà fait dans ResNetCE.features
    elif name == 'resnet18_ce':
        lib.LOGGER.info(f"using ResNet 18 for cross entropy, num classes : {kwargs.get('num_classes', "not specified")}")
        backbone = ResNetCE(pretrained=pretrained, backbone_name="resnet18",**kwargs)
        out_dim = 512
        pooling = nn.Identity()
    elif name == 'wcnn_attention':
        lib.LOGGER.info(f"using WCNN with Attention, decom_level :, {kwargs.get('decom_level', 3)}, wave :,{ kwargs.get('wave', 'haar')} feature size : {kwargs.get('feature_size', 512)} {kwargs.get('feature_size', 512)}, coarse_only : {kwargs.get('coarse_only', False)}   ")
        out_dim = 512
        backbone = WCNN_Attention(pretrained=pretrained,**kwargs)#(decom_level=2, wave='haar',ll_only=False, attention=True)
        pooling = nn.Identity()
    elif name == 'mtwavenet':
        lib.LOGGER.info(f"using Multi-Branch TWaveNet, num classes : {kwargs.get('num_classes', 'not specified')}")
        # Avoid passing `num_classes` twice if it is already present in kwargs
    
        backbone = FourBranchResNet(pretrained=pretrained, **kwargs)
        out_dim = 512 * 4
        pooling = nn.Identity()
    elif name == 'mtwavenet50_fusion':
        lib.LOGGER.info(f"using Multi-Branch TWaveNet50 with Fusion Module, num classes : {kwargs.get('num_classes', 'not specified')}")
        # Avoid passing `num_classes` twice if it is already present in kwargs
    
        backbone = FourBranchResNet50Fusion(pretrained=pretrained, **kwargs)
        out_dim = 2048  # Because fusion does a weighted sum, not a concatenation
        pooling = nn.Identity()
    elif name == 'mtwavenet50':
        lib.LOGGER.info(f"using Multi-Branch TWaveNet50, num classes : {kwargs.get('num_classes', 'not specified')}")
        # Avoid passing `num_classes` twice if it is already present in kwargs
    
        backbone = FourBranchResNet50(pretrained=pretrained, **kwargs)
        out_dim = 2048 * 4
        pooling = nn.Identity()
    elif name == "wcnn_ce":
        lib.LOGGER.info(f"using WCNN with Cross Entropy, decom_level :, {kwargs.get('decom_level', 'not specified')}, wave :,{ kwargs.get('wave', 'haar')} feature size : {kwargs.get('feature_size', 'unspecified')}, coarse_only : {kwargs.get('coarse_only', False)}  dropout : {kwargs.get('dropout', 0.0)}   ")
        out_dim = 2048 * 4
        backbone = WCNN(pretrained=pretrained,**kwargs)#(decom_level=2, wave='haar',ll_only=False, attention=True)
        pooling = nn.Identity()
    elif name == 'hybrid_mtwavenet_ce':
        lib.LOGGER.info(f"using Hybrid Multi-Branch TWaveNet with Cross Entropy, num classes : {kwargs.get('num_classes', 'not specified')}")
        # Avoid passing `num_classes` twice if it is already present in kwargs
    
        backbone = HybridMultiBranch(pretrained=pretrained, **kwargs)
        out_dim = 2048 + 1024* 3
        pooling = nn.Identity()
       
    else:
        raise ValueError(f"{name} is not recognized")

    return (backbone, pooling, out_dim)


class RetrievalNet(nn.Module):

    def __init__(
        self,
        backbone_name,
        embed_dim=512,
        norm_features=False,
        without_fc=False,
        with_autocast=False,
        pooling='default',
        projection_normalization_layer='none',
        pretrained=False,
        *args, **kwargs
    ):
        super().__init__()

        norm_features = lib.str_to_bool(norm_features)
        without_fc = lib.str_to_bool(without_fc)
        with_autocast = lib.str_to_bool(with_autocast)

        assert isinstance(without_fc, bool)
        assert isinstance(norm_features, bool)
        assert isinstance(with_autocast, bool)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.with_autocast = with_autocast
        self.with_classifier = bool(kwargs.get('num_classes', None))
        if with_autocast:
            lib.LOGGER.info("Using mixed precision")
        
        self.backbone_name = backbone_name

        self.backbone, default_pooling, out_features = get_backbone(backbone_name, pretrained=pretrained, embed_dim=embed_dim, **kwargs)
        if pooling == 'default':
            self.pooling = default_pooling
        elif pooling == 'none':
            self.pooling = nn.Identity()
        elif pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        lib.LOGGER.info(f"Pooling is {self.pooling}")

        if self.norm_features:
            lib.LOGGER.info("Using a LayerNorm layer")
            self.standardize = nn.LayerNorm(out_features, elementwise_affine=False)
        else:
            self.standardize = nn.Identity()

        if not self.without_fc:
            self.fc = create_projection_head(out_features, embed_dim, projection_normalization_layer)
            lib.LOGGER.info(f"Projection head : \n{self.fc}")
        else:
            self.fc = nn.Identity()
            lib.LOGGER.info("Not using a linear projection layer")

    def forward(self, X):
        with torch.amp.autocast('cuda',enabled=self.with_autocast):
            X = self.backbone(X)
            if self.with_classifier or self.backbone_name in ['mtwavenet', 'mtwavenet50']:
                return X
            if self.backbone_name in ['mtwavenet50_fusion'] and self.training:
                X[-1] = self.fc(X[-1])
                X = [F.normalize(x, p=2, dim=1) for x in X]
                return X
            X = self.pooling(X)

            X = X.view(X.size(0), -1)
            X = self.standardize(X)
            X = self.fc(X)
            X = F.normalize(X, p=2, dim=1)
            return X