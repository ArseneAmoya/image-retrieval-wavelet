import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. MODULES DE FUSION (CBAM, ECA, SELF ATTENTION) + SELECTEUR AUTOMATIQUE
# ==========================================

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
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

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=1, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = nn.AdaptiveAvgPool1d(1)(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = nn.AdaptiveMaxPool1d(1)(x)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(-1)
        return (x.permute(0, 2, 1) @ scale).squeeze(-1) / self.gate_channels

    def alphas(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = nn.AdaptiveAvgPool1d(1)(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = nn.AdaptiveMaxPool1d(1)(x)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(-1)
        return scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
        
    def alphas(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels=4, reduction_ratio=1, pool_types=['avg', 'max'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
            
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
        
    def alphas(self, x):
        x_out = self.ChannelGate.alphas(x)
        if not self.no_spatial:
            x_out = self.SpatialGate.alphas(x_out)
        return x_out

class Eca1D_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(Eca1D_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.chan = channel

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return (x.permute(0, 2, 1) @ y).squeeze(-1) / self.chan
        
    def alphas(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return y

# --- MODULE DE LIAISON (WRAPPER) ---
class AdvancedFusionModule(nn.Module):
    def __init__(self, fusion_type='cbam', num_branches=4, reduction_ratio=1, input_dim=384, hidden_dim=384):
        super(AdvancedFusionModule, self).__init__()
        
        if fusion_type == 'cbam':
            self.gate = CBAM(gate_channels=num_branches, reduction_ratio=reduction_ratio, pool_types=['avg', 'max'], no_spatial=True)
        elif fusion_type == 'eca':
            self.gate = Eca1D_layer(channel=num_branches, k_size=3)
            
        # Projection et régularisation optionnelles
        self.fcn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, embeddings_list):
        # Transforme la liste de tenseurs en [Batch, 4_Branches, 384_Features]
        x = torch.stack(embeddings_list, dim=1) 
        
        # Applique CBAM ou ECA (Sortie: [Batch, 384_Features])
        fused = self.gate(x) 
        
        # Applique la normalisation
        out = self.fcn(fused)
        return out


class StandardFusionHead(nn.Module):
    def __init__(self, input_dims, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity() for dim in input_dims])
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(self, features_list):
        batch_size = features_list[0].shape[0]
        projected_feats = [proj(f) for proj, f in zip(self.projections, features_list)]
        kv = torch.stack(projected_feats, dim=1)
        q = self.query_token.expand(batch_size, -1, -1)
        attn_output, _ = self.attn(query=q, key=kv, value=kv)
        x = self.norm1(attn_output)
        x = x + self.mlp(x)
        return self.norm2(x).squeeze(1)

class TemperatureFusionHead(nn.Module):
    def __init__(self, input_dims, embed_dim=512, num_heads=4, dropout=0.1, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projections = nn.ModuleList([nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity() for dim in input_dims])
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(self, features_list):
        batch_size = features_list[0].shape[0]
        projected_feats = [proj(f) for proj, f in zip(self.projections, features_list)]
        kv = torch.stack(projected_feats, dim=1)
        q = self.query_token.expand(batch_size, -1, -1)
        q_scaled = q / self.temperature
        attn_output, _ = self.attn(query=q_scaled, key=kv, value=kv)
        x = self.norm1(attn_output)
        x = x + self.mlp(x)
        return self.norm2(x).squeeze(1)

class SemanticFusionHead(nn.Module):
    def __init__(self, input_dims, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity() for dim in input_dims])
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))

    def forward(self, features_list):
        projected_feats = [proj(f) for proj, f in zip(self.projections, features_list)]
        q = projected_feats[0].unsqueeze(1)
        kv = torch.stack(projected_feats, dim=1)
        attn_output, _ = self.attn(query=q, key=kv, value=kv)
        x = self.norm1(attn_output)
        x = x + self.mlp(x)
        return self.norm2(x).squeeze(1)

class GatedFusionHead(nn.Module):
    def __init__(self, input_dims, embed_dim=512, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity() for dim in input_dims])
        self.gate_network = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))

    def forward(self, features_list):
        projected_feats = [proj(f) for proj, f in zip(self.projections, features_list)]
        fused_out = 0
        for feat in projected_feats:
            gate = self.gate_network(feat)
            fused_out = fused_out + (feat * gate)
        x = self.norm1(fused_out)
        x = x + self.mlp(x)
        return self.norm2(x)

# --- LA NOUVELLE METHODE (COMBINAISON 1 + 3) ---
class TemperatureGatedFusionHead(nn.Module):
    def __init__(self, input_dims, embed_dim=512, dropout=0.1, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projections = nn.ModuleList([nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity() for dim in input_dims])
        # Note: Pas de Sigmoid ici, on garde le logit brut pour lui appliquer la température après
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, features_list):
        projected_feats = [proj(f) for proj, f in zip(self.projections, features_list)]
        fused_out = 0
        for feat in projected_feats:
            raw_gate = self.gate_network(feat) # Logit (ex: -0.5, 1.2)
            # Application de la température + Sigmoid (Hard Gating)
            hard_gate = torch.sigmoid(raw_gate / self.temperature)
            fused_out = fused_out + (feat * hard_gate)

        x = self.norm1(fused_out)
        x = x + self.mlp(x)
        return self.norm2(x)

# =========================================================================
# 2. SELECTEUR AUTOMATIQUE
# =========================================================================

def get_fusion_head(fusion_config, output_dims):
    fusion_type = fusion_config.get('type', 'standard')
    embed_dim = fusion_config['output_dim']
    num_heads = fusion_config.get('num_heads', 8)
    dropout = fusion_config.get('dropout', 0.1)

    if fusion_type == 'temperature':
        temp = fusion_config.get('temperature', 0.1)
        return TemperatureFusionHead(output_dims, embed_dim, num_heads, dropout, temperature=temp)
    elif fusion_type == 'semantic':
        return SemanticFusionHead(output_dims, embed_dim, num_heads, dropout)
    elif fusion_type == 'gated':
        return GatedFusionHead(output_dims, embed_dim, dropout)
    elif fusion_type == 'temperature_gated': # L'ajout au routeur
        temp = fusion_config.get('temperature', 0.1)
        return TemperatureGatedFusionHead(output_dims, embed_dim, dropout, temperature=temp)
    else:
        return StandardFusionHead(output_dims, embed_dim, num_heads, dropout)


# =========================================================================
# 3. VOS MODELES
# =========================================================================

class MultiDinoAttention(nn.Module):
    def __init__(self, backbones_config, fusion_config, **kwargs):
        super().__init__()
        self.backbones = nn.ModuleList()
        output_dims = []
        for bb_cfg in backbones_config:
            model = torch.hub.load('facebookresearch/dinov2', bb_cfg['name'])
            dim = model.embed_dim
            if bb_cfg.get('frozen', True):
                for p in model.parameters(): p.requires_grad = False
                model.eval()
            self.backbones.append(model)
            output_dims.append(dim)
        self.fusion_head = get_fusion_head(fusion_config, output_dims)

    def forward(self, x):
        features = []
        for i, backbone in enumerate(self.backbones):
            out = backbone(x[..., i, :, :])
            features.append(out['x_norm_clstoken'] if isinstance(out, dict) else out)
        final_embedding = self.fusion_head(features)
        return F.normalize(final_embedding, p=2, dim=1)

class MultiDinoHashing(nn.Module):
    def __init__(self, backbones_config, fusion_config, binary_config, **kwargs):
        super().__init__()
        self.backbones = nn.ModuleList()
        output_dims = []
        for bb_cfg in backbones_config:
            model = torch.hub.load('facebookresearch/dinov2', bb_cfg['name'])
            dim = model.embed_dim
            if bb_cfg.get('frozen', True):
                for p in model.parameters(): p.requires_grad = False
                model.eval()
            self.backbones.append(model)
            output_dims.append(dim)
        self.fusion_head = get_fusion_head(fusion_config, output_dims)
        self.bn = nn.BatchNorm1d(fusion_config['output_dim'])
        self.nbits = binary_config['nbits']
        self.hash_fc = nn.Linear(fusion_config['output_dim'], self.nbits)
        nn.init.normal_(self.hash_fc.weight, std=0.01)
        nn.init.constant_(self.hash_fc.bias, 0)

    def forward(self, x):
        features = []
        for i, backbone in enumerate(self.backbones):
            out = backbone(x[..., i, :, :])
            features.append(out['x_norm_clstoken'] if isinstance(out, dict) else out)
        fused_embedding = self.fusion_head(features)
        fused_embedding = self.bn(fused_embedding)
        logits = self.hash_fc(fused_embedding)
        return torch.tanh(logits) if self.training else torch.sign(logits)

class MultiDinoHashingTF(nn.Module):
    def __init__(self, backbones_config, fusion_config, binary_config, pretrained_paths=None, **kwargs):
        super().__init__()
        self.backbones = nn.ModuleList()
        output_dims = []
        for bb_cfg in backbones_config:
            model = torch.hub.load('facebookresearch/dinov2', bb_cfg['name'])
            self.backbones.append(model)
            output_dims.append(model.embed_dim)
        if pretrained_paths is not None:
            keys_for_branches = ['ll', 'lh', 'lh', 'hh']
            for i in range(len(self.backbones)):
                target_key = keys_for_branches[i]
                if target_key in pretrained_paths and pretrained_paths[target_key] is not None:
                    self._load_expert_weights(self.backbones[i], pretrained_paths[target_key])

        self.num_branches = len(backbones_config)
        self.branch_embeddings = nn.Parameter(torch.randn(self.num_branches, output_dims[0]) * 0.02)
        self.fusion_head = get_fusion_head(fusion_config, output_dims)
        self.bn = nn.BatchNorm1d(fusion_config['output_dim'])
        self.hash_fc = nn.Linear(fusion_config['output_dim'], binary_config['nbits'])

    def _load_expert_weights(self, target_backbone, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        backbone_state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['net_state'].items() if k.startswith('backbone.')}
        target_backbone.load_state_dict(backbone_state_dict, strict=True)

    def forward(self, x):
        features = []
        for i, backbone in enumerate(self.backbones):
            out = backbone(x[..., i, :, :])
            features.append(out['x_norm_clstoken'] if isinstance(out, dict) else out)
        fused_embedding = self.fusion_head(features)
        fused_embedding = self.bn(fused_embedding)
        logits = self.hash_fc(fused_embedding)
        return torch.tanh(logits) if self.training else torch.sign(logits)
class PretrainedMultiDinoHashing(nn.Module):
    """
    Architecture SOTA pour l'évaluation du Hachage.
    Charge un modèle continu pré-entraîné, fige DINO + Attention,
    et n'entraîne qu'une tête de hachage (Linear + Tanh).
    """
    def __init__(self, backbones_config, fusion_config, binary_config, pretrained_ckpt_path=None, **kwargs):
        super().__init__()
        
        # 1. Reconstruction du réseau de base
        self.backbones = nn.ModuleList()
        output_dims = []
        for bb_cfg in backbones_config:
            model = torch.hub.load('facebookresearch/dinov2', bb_cfg['name'])
            for p in model.parameters(): p.requires_grad = False
            model.eval()
            self.backbones.append(model)
            output_dims.append(model.embed_dim)
            
        self.fusion_head = get_fusion_head(fusion_config, output_dims)

        # 2. Chargement des poids de votre meilleur run continu (ex: 98.5%)
        if pretrained_ckpt_path is not None:
            print(f"[INFO] Chargement du modèle continu depuis : {pretrained_ckpt_path}")
            ckpt = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=False)
            state_dict = {}
            for k, v in ckpt['net_state'].items():
                if k.startswith('fusion_head.'):
                    state_dict[k.replace('fusion_head.', '')] = v
            self.fusion_head.load_state_dict(state_dict, strict=True)
        else:
            print("[ATTENTION] Aucun chemin de checkpoint fourni. L'attention ne sera pas initialisée avec vos meilleurs poids !")

        # 3. Figer l'attention (Tout le réseau extracteur est maintenant gelé)
        for p in self.fusion_head.parameters():
            p.requires_grad = False
        self.fusion_head.eval()

        # 4. La couche de Hachage (La seule partie qui s'entraîne)
        self.output_dim = fusion_config['output_dim']
        self.nbits = binary_config['nbits']
        
        self.bn = nn.BatchNorm1d(self.output_dim)
        self.hash_fc = nn.Linear(self.output_dim, self.nbits)
        
        nn.init.normal_(self.hash_fc.weight, std=0.01)
        nn.init.constant_(self.hash_fc.bias, 0)

    def forward(self, x):
        # Partie figée : On désactive les gradients pour économiser de la VRAM et accélérer à 100%
        with torch.no_grad():
            features = []
            for i, backbone in enumerate(self.backbones):
                out = backbone(x[..., i, :, :])
                features.append(out['x_norm_clstoken'] if isinstance(out, dict) else out)
            
            fused_embedding = self.fusion_head(features)
            fused_embedding = F.normalize(fused_embedding, p=2, dim=1) # Même format que votre loss continue

        # Partie entraînable : Hashing Head
        x_hash = self.bn(fused_embedding)
        logits = self.hash_fc(x_hash)
        
        return torch.tanh(logits) if self.training else torch.sign(logits)