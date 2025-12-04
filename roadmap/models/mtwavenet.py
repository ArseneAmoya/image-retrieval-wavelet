import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
        scale = F.sigmoid( channel_att_sum ).unsqueeze(-1)
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
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x) * x
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out) * x_out
        return x_out#(x.permute(0,2,1)@y).squeeze(-1)/self.chan
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
    def __init__(self, num_classes=1000):
        super(FourBranchResNet, self).__init__()
        
        
        self.branches = nn.ModuleList()
        
        for _ in range(4):
            base_resnet = models.resnet18(pretrained=True)
            
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
            
            self.branches.append(nn.ModuleList([stem, layer1, layer2, layer3, layer4]))


        self.att_block1 = CrossBandAttention(channels_per_branch=64)
        self.att_block2 = CrossBandAttention(channels_per_branch=128)
        self.att_block3 = CrossBandAttention(channels_per_branch=256)
        self.att_block4 = CrossBandAttention(channels_per_branch=512)
        
        # --- Tête de classification finale ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 512 channels * 4 branches (si on concatène à la fin)
        self.fc = nn.Linear(512 * 4, num_classes) 

    def forward(self, x_list):
        """
        x_list: liste de 4 images [Batch, 3, H, W]
        """
        assert len(x_list) == 4, "Il faut exactement 4 entrées."
        
        # --- Stage 0 (Stem) ---
        feats = []
        for i in range(4):
            feats.append(self.branches[i][0](x_list[i]))
            
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
            
        # Concaténation des 4 vecteurs finaux
        final_vec = torch.cat(embeddings, dim=1)
        
        # Classification
        out = self.fc(final_vec)
        
        return out

# --- Test du modèle ---
if __name__ == "__main__":
    model = FourBranchResNet(num_classes=10)
    
    # Création de 4 "images" aléatoires
    inputs = [torch.randn(2, 3, 224, 224) for _ in range(4)]
    
    output = model(inputs)
    print(f"Output shape: {output.shape}") # Doit être [2, 10]