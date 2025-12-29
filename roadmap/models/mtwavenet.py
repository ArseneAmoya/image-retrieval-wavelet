import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import roadmap.utils as lib

# --- Classes Utilitaires d'Amoya et al. (Renommées pour éviter les conflits) ---


class ChannelGate1D(nn.Module):
    """
    Implémentation exacte d'Amoya et al. pour la fusion 1D.
    Prend en entrée (Batch, Branches, Features).
    """
    def __init__(self, gate_channels, reduction_ratio=1, pool_types=['avg', 'max']):
        super(ChannelGate1D, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        # x shape: [Batch, 4 (Branches), 2048 (Features)]
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # Moyenne sur la dimension des features (dim 2) -> [Batch, 4, 1]
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                # Max sur la dimension des features -> [Batch, 4, 1]
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(max_pool)
            # (Note: lp et lse retirés car nécessitent input 2D, ici on est en 1D)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # Sigmoid pour obtenir les poids par branche [Batch, 4, 1]
        scale = torch.sigmoid(channel_att_sum).unsqueeze(-1)
        
        # Opération de fusion d'Amoya : 
        # (Batch, Features, Branches) @ (Batch, Branches, 1) -> (Batch, Features, 1)
        # Cela revient à une somme pondérée des branches
        out = (x.permute(0, 2, 1) @ scale).squeeze(-1)
        
        # Division optionnelle par gate_channels comme dans le snippet original
        # out = out / self.gate_channels 
        
        return out

class FusionModule(nn.Module):
    def __init__(self, num_branches=4, reduction_ratio=1, input_dim=2048, hidden_dim=2048): 
        super(FusionModule, self).__init__()
        
        # 1. Le Gating Amoya (Inchangé)
        self.channel_gate = ChannelGate1D(
            gate_channels=num_branches, 
            reduction_ratio=reduction_ratio,
            pool_types=['avg', 'max']
        )

        # 2. Le FCN (Nouveau)
        # On ajoute une couche de projection non-linéaire après la somme pondérée
        self.fcn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # Projection
            nn.BatchNorm1d(hidden_dim),            # Normalisation (Important !)
            nn.ReLU(inplace=True),                 # Non-linéarité
            nn.Dropout(p=0.5)                      # Régularisation
        )

    def forward(self, embeddings_list):
        """
        Entrée : Liste de 4 tenseurs [Batch, 2048]
        Sortie : Tenseur fusionné et projeté [Batch, 2048]
        """
        # A. Empilement [Batch, 4, 2048]
        stacked_features = torch.stack(embeddings_list, dim=1)
        
        # B. Fusion Amoya (Somme pondérée) -> [Batch, 2048]
        fused_features = self.channel_gate(stacked_features)
        
        
        return fused_features

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
    

# ... (Keep your helper classes: CrossBandAttention, etc.) ...

class FourBranchResNet50(nn.Module):
    def __init__(self, num_classes=None, *args, **kwargs):
        super(FourBranchResNet50, self).__init__()
        
        self.branches = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.classifiers = nn.ModuleList() # Separate list for heads
        
        for i in range(4):
            # --- 1. BACKBONE ---
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                base_resnet = models.resnet50(weights=weights)
            except:
                base_resnet = models.resnet50(pretrained=True)
            
            if i == 0: lib.LOGGER.info(f"Using ResNet50 backbone")
            
            # Feature Extractor parts
            stem = nn.Sequential(base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool)
            
            # We store the backbone layers in a ModuleList
            # Structure: branches[i] = [stem, layer1, layer2, layer3, layer4]
            self.branches.append(nn.ModuleList([
                stem, 
                base_resnet.layer1, 
                base_resnet.layer2, 
                base_resnet.layer3, 
                base_resnet.layer4
            ]))
            
            # --- 2. NORMALIZATION (One per branch) ---
            self.layer_norms.append(nn.LayerNorm(2048))
            
            # --- 3. CLASSIFIER HEAD ---
            if num_classes:
                clf = nn.Linear(2048, num_classes)
                nn.init.constant_(clf.weight, 0)
                nn.init.constant_(clf.bias, 0)
                
                self.classifiers.append(nn.Sequential(
                    nn.Dropout(p=0.5),
                    clf
                ))
            else:
                self.classifiers.append(nn.Identity())

        # --- 4. ATTENTION & POOLING ---
        self.att_block1 = CrossBandAttention(256)
        self.att_block2 = CrossBandAttention(512)
        self.att_block3 = CrossBandAttention(1024)
        self.att_block4 = CrossBandAttention(2048)
        if kwargs.get('pooling_mode', 'avg') == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            lib.LOGGER.info("Using Average Pooling")
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            lib.LOGGER.info("Using Max Pooling")
        
        # BN Freezing
        self.freeze_bn_flag = kwargs.get('freeze_batch_norm', False)
        if self.freeze_bn_flag:
            lib.LOGGER.info("Freezing Batch Normalization layers")
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.branches.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        assert x.size(-3) == 4
        
        # --- Feature Extraction with Attention ---
        feats = []
        
        # 1. Stem
        for i in range(4):
            feats.append(self.branches[i][0](x[..., i, :, :]))
            
        # 2. Stages 1-4 with Attention
        attentions = [self.att_block1, self.att_block2, self.att_block3, self.att_block4]
        
        for stage_idx, att_block in enumerate(attentions):
            # Pass through ResNet Layer (indices 1 to 4 in self.branches[i])
            for i in range(4):
                feats[i] = self.branches[i][stage_idx + 1](feats[i])
            
            # Apply Cross-Band Attention
            feats = att_block(feats)

        # --- Aggregation & Normalization ---
        embeddings = []
        for i in range(4):
            # A. Max Pool
            feat = self.pool(feats[i])
            feat = torch.flatten(feat, 1)
            
            # B. Layer Norm (Specific to branch i)
            feat = self.layer_norms[i](feat)
            
            embeddings.append(feat)
        
        if self.training:
            # Return list of logits for MultiCrossEntropyLoss
            return [self.classifiers[i](embeddings[i]) for i in range(4)]
        else:
            # Test: Concatenate normalized embeddings
            final_vec = torch.cat(embeddings, dim=1)
            final_vec = F.normalize(final_vec, dim=1, p=2)
            return final_vec

    def train(self, mode=True):
        super(FourBranchResNet50, self).train(mode)
        if self.freeze_bn_flag and mode:
            for m in self.branches.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        return self
    
class FourBranchResNet50Fusion(FourBranchResNet50):
    def __init__(self, num_classes=None, *args, **kwargs):
        # 1. Initialiser la classe parente (crée les 4 branches, BN frozen, etc.)
        super(FourBranchResNet50Fusion, self).__init__(num_classes, *args, **kwargs)
        
        lib.LOGGER.info("Adding Amoya Fusion Module to the architecture")

        # 2. Ajouter le Module de Fusion Amoya
        self.fusion_module = FusionModule(num_branches=4, reduction_ratio=1)
        
        # 3. Ajouter le Classifieur Principal (Main Head)
        # Il prend 2048 en entrée (car Amoya fait une somme pondérée, pas une concat)
        if num_classes:
            self.main_classifier = nn.Linear(2048, num_classes)
            nn.init.constant_(self.main_classifier.weight, 0)
            nn.init.constant_(self.main_classifier.bias, 0)
            
            self.main_head = nn.Sequential(
                nn.Dropout(0.5),
                self.main_classifier
            )
        else:
            self.main_head = nn.Identity()
    def forward(self, x):
        # On doit réécrire le forward pour intercepter les embeddings avant la sortie
        # Mais on réutilise tous les attributs de self (qui viennent du parent)
        
        assert x.size(-3) == 4
        
        # --- Feature Extraction (Copie de la logique parent utilisant les attributs hérités) ---
        feats = []
        for i in range(4):
            feats.append(self.branches[i][0](x[..., i, :, :])) # Stem
            
        attentions = [self.att_block1, self.att_block2, self.att_block3, self.att_block4]
        for stage_idx, att_block in enumerate(attentions):
            for i in range(4):
                feats[i] = self.branches[i][stage_idx + 1](feats[i])
            feats = att_block(feats)

        # --- Aggregation & Normalisation ---
        embeddings = []
        for i in range(4):
            feat = self.pool(feats[i])
            feat = torch.flatten(feat, 1)
            feat = self.layer_norms[i](feat)
            embeddings.append(feat)
        
        # --- C'EST ICI QUE ÇA CHANGE : FUSION AMOYA ---
        fused_embedding = self.fusion_module(embeddings) # [Batch, 2048]
        
        if self.training:
            outputs = []
            # 1. Sorties Auxiliaires (des 4 branches)
            # self.classifiers vient du parent (si vous l'avez nommé self.classifiers dans le parent)
            # Sinon utilisez self.branches[i][-1] selon votre implémentation parent exacte
            for i in range(4):
                outputs.append(self.classifiers[i](embeddings[i]))
            
            # 2. Sortie Principale (Fusionnée)
            outputs.append(self.main_head(fused_embedding))
            
            return outputs # Liste de 5 logits
        else:
            # En Test : On renvoie le vecteur fusionné normalisé
            return F.normalize(fused_embedding, dim=1, p=2)


class HybridMultiBranch(nn.Module):
    def __init__(self, num_classes=200, pretrained=True, dropout=0.5, freeze_resnet_bn=True, *args, **kwargs):
        super(HybridMultiBranch, self).__init__()
        
        lib.LOGGER.info("Initializing Hybrid Multi-Branch Network")

        # --- BRANCHE 1 : ResNet50 pour l'Approximation (LL) ---
        resnet = models.resnet50(pretrained=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.resnet_branch = nn.Sequential(*list(resnet.children())[:-2]) # Output: 2048 channels
        self.resnet_dim = 2048
        
        # Gestion du gel des Batch Norms pour ResNet uniquement
        if freeze_resnet_bn:
            for m in self.resnet_branch.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval() # Passe en mode évaluation (utilise running stats)
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    m.train = lambda mode: None  # Désactive complètement le mode train
            lib.LOGGER.info("BatchNorm layers in ResNet branch have been frozen.")

        # --- BRANCHES 2, 3, 4 : DenseNet121 pour les Détails (LH, HL, HH) ---
        # On crée 3 instances indépendantes
        self.dense_branches = nn.ModuleList()
        self.dense_dim = 1024
        
        for _ in range(3): # Pour LH, HL, HH
            dense = models.densenet121(pretrained=pretrained)
            features = dense.features
            self.dense_branches.append(features)

        # --- CLASSIFIERS (Têtes) ---
        # On a 4 sorties indépendantes pour la Multi-CE Loss
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

        # Classifier pour ResNet (LL)
        self.fc_resnet = nn.Linear(self.resnet_dim, num_classes)
        
        # Classifiers pour DenseNets (LH, HL, HH)
        self.fc_dense_lh = nn.Linear(self.dense_dim, num_classes)
        self.fc_dense_hl = nn.Linear(self.dense_dim, num_classes)
        self.fc_dense_hh = nn.Linear(self.dense_dim, num_classes)

        # Initialisation des classifiers
        for fc in [self.fc_resnet, self.fc_dense_lh, self.fc_dense_hl, self.fc_dense_hh]:
            nn.init.xavier_normal_(fc.weight)
            nn.init.constant_(fc.bias, 0)

    def forward(self, x):
        # 1. Décomposition DWT
        # x: [B, 3, 448, 448] -> LL, LH, HL, HH: [B, 3, 224, 224]
        assert x.size(-3) == 4, "Input must have 4 subbands."

        # --- Branche 1: Approximation (ResNet) ---
        # Note: Si BN est figé, resnet_branch est en mode eval() pour les BN
        f_ll = self.resnet_branch(x[:, :, 0])
        f_ll = self.gap(f_ll).flatten(1)

        # --- Branches Détails (DenseNet) ---
        # DenseNet nécessite une activation finale (ReLU) + GAP
        
        # LH
        f_lh = self.dense_branches[0](x[:, :, 1])
        f_lh = F.relu(f_lh, inplace=True)
        f_lh = self.gap(f_lh).flatten(1)

        # HL
        f_hl = self.dense_branches[1](x[:, :, 2])
        f_hl = F.relu(f_hl, inplace=True)
        f_hl = self.gap(f_hl).flatten(1)

        # HH
        f_hh = self.dense_branches[2](x[:, :, 3])
        f_hh = F.relu(f_hh, inplace=True)
        f_hh = self.gap(f_hh).flatten(1)

        if self.training:
            out_ll = self.fc_resnet(self.dropout(f_ll))
            out_lh = self.fc_dense_lh(self.dropout(f_lh))
            out_hl = self.fc_dense_hl(self.dropout(f_hl))
            out_hh = self.fc_dense_hh(self.dropout(f_hh))
            return [out_ll, out_lh, out_hl, out_hh]
        
        else:
            return F.normalize(torch.cat([f_ll, f_lh, f_hl, f_hh], dim=1), p=2, dim=1)
       
class HybridMultiBranchV2(nn.Module):
    def __init__(self, num_classes=200, pretrained=True, dropout=0.5, freeze_resnet_bn=True, *args, **kwargs):
        super(HybridMultiBranchV2, self).__init__()
        
        lib.LOGGER.info("Initializing Hybrid Multi-Branch Network")

        # --- BRANCHE 1 : ResNet50 pour l'Approximation (LL) ---
        resnet = models.resnet50(pretrained=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.resnet_branch = nn.Sequential(*list(resnet.children())[:-2]) # Output: 2048 channels
        self.resnet_dim = 2048
        
        # Gestion du gel des Batch Norms pour ResNet uniquement
        if freeze_resnet_bn:
            for m in self.resnet_branch.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval() # Passe en mode évaluation (utilise running stats)
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    m.train = lambda mode: None  # Désactive complètement le mode train
            lib.LOGGER.info("BatchNorm layers in ResNet branch have been frozen.")

        # --- BRANCHES 2, 3, 4 : DenseNet121 pour les Détails (LH, HL, HH) ---
        # On crée 3 instances indépendantes
        self.dense_branches = nn.ModuleList()
        self.dense_dim = 1024
        
        for _ in range(2): # Pour LH, HL, HH
            dense = models.densenet121(pretrained=pretrained)
            features = dense.features
            self.dense_branches.append(features)

        # --- CLASSIFIERS (Têtes) ---
        # On a 4 sorties indépendantes pour la Multi-CE Loss
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier pour ResNet (LL)
        self.fc_resnet = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.resnet_dim, num_classes))
        
        # Classifiers pour DenseNets (LH, HL, HH)
        self.fc_dense_lh = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.dense_dim, num_classes))
        self.fc_dense_hl = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.dense_dim, num_classes))

        # Initialisation des classifiers
        for fc in [self.fc_resnet, self.fc_dense_lh, self.fc_dense_hl]:

            nn.init.xavier_normal_(fc[1].weight)
            nn.init.constant_(fc[1].bias, 0)

    def forward(self, x):
        # 1. Décomposition DWT
        # x: [B, 3, 448, 448] -> LL, LH, HL, HH: [B, 3, 224, 224]
        assert x.size(-3) == 4, "Input must have 4 subbands."

        # --- Branche 1: Approximation (ResNet) ---
        # Note: Si BN est figé, resnet_branch est en mode eval() pour les BN
        f_ll = self.resnet_branch(x[:, :, 0])
        f_ll = self.gap(f_ll).flatten(1)

        # --- Branches Détails (DenseNet) ---
        # DenseNet nécessite une activation finale (ReLU) + GAP
        
        # LH
        f_lh = self.dense_branches[0](x[:, :, 1])
        f_lh = F.relu(f_lh, inplace=True)
        f_lh = self.gap(f_lh).flatten(1)

        # HL
        f_hl = self.dense_branches[1](x[:, :, 2])
        f_hl = F.relu(f_hl, inplace=True)
        f_hl = self.gap(f_hl).flatten(1)

        # HH - Ignoré dans cette version

        if self.training:
            out_ll = self.fc_resnet(f_ll)
            out_lh = self.fc_dense_lh(f_lh)
            out_hl = self.fc_dense_hl(f_hl)
            return [out_ll, out_lh, out_hl]
        
        else:
            return F.normalize(torch.cat([f_ll, f_lh, f_hl], dim=1), p=2, dim=1)
       


