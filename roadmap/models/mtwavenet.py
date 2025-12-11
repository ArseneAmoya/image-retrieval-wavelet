import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import roadmap.utils as lib

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=1):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        channel_att_sum = self.mlp(max_out) + self.mlp(avg_out)
        scale = F.sigmoid( channel_att_sum )
        return scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale
    
class CrossBandAttention(nn.Module):
    def __init__(self, channels_per_branch, num_branches=4, reduction_ratio=1, no_spatial=True):
        super(CrossBandAttention, self).__init__()
        self.num_channels = channels_per_branch * num_branches
        self.ChannelGate = ChannelGate(self.num_channels, reduction_ratio)
        self.num_branches = num_branches
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x = torch.cat(x, dim=1)  # Concatène le long du canal
        b, c, h, w = x.shape
        att = self.ChannelGate(x)
        att =att.unsqueeze(-1).unsqueeze(-1).expand(b, c, h, w)
        x_out = att * x
        if not self.no_spatial:
            att = self.SpatialGate(x_out).expand(b, c, h, w)
            x_out = att * x_out
        x_out = list(torch.split(x_out, self.num_channels // 4, dim=1))

        return x_out
    def alphas(self, x):
        x_out = self.ChannelGate(x)
        return x_out


class ResNetStage(nn.Module):
    """ Enveloppe pour extraire une partie spécifique d'un ResNet """
    def __init__(self, block_layer):
        super().__init__()
        self.block = block_layer

    def forward(self, x):
        return self.block(x)

class FourBranchResNet(nn.Module):
    def __init__(self, num_classes=None, *args, **kwargs):
        super(FourBranchResNet, self).__init__()
        
        
        self.branches = nn.ModuleList()
        
        for _ in range(4):
            base_resnet = models.resnet18(pretrained=models.ResNet18_Weights.IMAGENET1K_V1)
            lib.LOGGER.info(f"Using ResNet18 backbone, pretrained={kwargs.get('pretrained', False)}")   
            
            stem = nn.Sequential(
                base_resnet.conv1,
                base_resnet.bn1,
                base_resnet.relu,
                base_resnet.maxpool
            )
            
            layer1 = base_resnet.layer1 # 64 channels
            layer2 = base_resnet.layer2 # 128 channels
            layer3 = base_resnet.layer3 # 256 channels
            layer4 = base_resnet.layer4 # 512 channels
            module_list = [stem, layer1, layer2, layer3, layer4]
            if num_classes is not None:
                module_list.append(nn.Linear(512, num_classes))

                classifier = nn.Linear(512, num_classes)
    
                nn.init.constant_(classifier.weight, 0)
                nn.init.constant_(classifier.bias, 0)
                
                module_list.append(nn.Sequential(
                    nn.Dropout(p=0.5),
                    classifier
                ))
            else:
                module_list.append(nn.Identity())
            self.branches.append(nn.ModuleList(module_list))



        self.att_block1 = CrossBandAttention(channels_per_branch=64)
        self.att_block2 = CrossBandAttention(channels_per_branch=128)
        self.att_block3 = CrossBandAttention(channels_per_branch=256)
        self.att_block4 = CrossBandAttention(channels_per_branch=512)
        
        # --- Tête de classification finale ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 512 channels * 4 branches (si on concatène à la fin)
        self.freeze_bn_flag = kwargs.get('freeze_batch_norm', False)
        if self.freeze_bn_flag:
            lib.LOGGER.info("Freezing Batch Normalization layers (Boudiaf et al. protocol)")
            self._freeze_bn()
        
    def _freeze_bn(self):
        # Freezes all BatchNorm layers in the backbone
        for m in self.modules():
            for sub_m in m.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()              # Fix running stats (mean/var)
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    lib.LOGGER.info(f"Freezed BatchNorm layer: {m}")
    def forward(self, x):
        """
        x_list: tensor [Batch, 3, 4, H, W]
        """
        assert x.size(-3) == 4, "Il faut exactement 4 entrées."
        
        # --- Stage 0 (Stem) ---
        feats = []
        for i in range(4):
            feats.append(self.branches[i][0](x[..., i, :, :])) # Stem
            
        # --- Stage 1 + Attention ---
        for i in range(4):
            feats[i] = self.branches[i][1](feats[i]) # ResNet Layer 1
        feats = self.att_block1(feats)               # Interaction
        
        # --- Stage 2 + Attention ---
        for i in range(4):
            feats[i] = self.branches[i][2](feats[i]) # ResNet Layer 2
        feats = self.att_block2(feats)               # Interaction

        # --- Stage 3 + Attention ---
        for i in range(4):
            feats[i] = self.branches[i][3](feats[i]) # ResNet Layer 3
        feats = self.att_block3(feats)               # Interaction

        # --- Stage 4 + Attention ---
        for i in range(4):
            feats[i] = self.branches[i][4](feats[i]) # ResNet Layer 4
        feats = self.att_block4(feats)               # Interaction
        
        # --- Aggregation Finale ---
        embeddings = []
        for i in range(4):
            x = self.avgpool(feats[i])
            x = torch.flatten(x, 1)
            embeddings.append(x)
        
        if self.training:
            # Pendant l'entraînement, on retourne les classes de chaque branche
            return [self.branches[i][-1](embeddings[i]) for i in range(4)]
        else:
            final_vec = torch.cat(embeddings, dim=1)
            final_vec = F.normalize(final_vec, dim=1, p=2)
                
        return final_vec
    def train(self, mode=True):
        # 1. Standard behavior: set everything to train mode (Dropout, etc.)
        super(FourBranchResNet, self).train(mode)
        
        # 2. Override: If freezing is on, hunt down every BN layer and force it to eval
        if self.freeze_bn_flag and mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
        
        return self