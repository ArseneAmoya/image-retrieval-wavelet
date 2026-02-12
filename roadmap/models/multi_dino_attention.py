import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionHead(nn.Module):
    """
    Module de fusion SOTA basé sur l'attention (Cross-Attention).
    Il utilise un 'Query Token' apprenable qui va agréger l'information 
    pertinente des N branches (Keys/Values).
    """
    def __init__(self, input_dims, embed_dim=512, num_heads=4, dropout=0.1):
        """
        Args:
            input_dims (list[int]): Liste des dimensions de sortie de chaque branche (ex: [384, 384, 768, 768])
            embed_dim (int): Dimension commune de projection et de sortie.
            num_heads (int): Nombre de têtes d'attention (doit diviser embed_dim).
        """
        super().__init__()
        
        # 1. Projecteurs : Pour ramener toutes les branches à la même dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim) if dim != embed_dim else nn.Identity()
            for dim in input_dims
        ])
        
        # 2. Le Token d'Agrégation (Query) - C'est lui qui "pose la question" aux branches
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 3. Multi-Head Cross Attention
        # batch_first=True attend (Batch, Seq_len, Dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 4. Normalisation et Feed-Forward (Architecture type Transformer Block)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Init des poids
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(self, features_list):
        """
        Args:
            features_list (list[Tensor]): Liste de tenseurs (B, D_i) issus des backbones
        """
        batch_size = features_list[0].shape[0]
        
        # A. Projection & Stack -> Création de la séquence (Keys/Values)
        # On projette chaque branche et on les empile : (B, N_branches, embed_dim)
        projected_feats = [proj(f) for proj, f in zip(self.projections, features_list)]
        kv = torch.stack(projected_feats, dim=1) 
        
        # B. Préparation du Query (B, 1, embed_dim)
        q = self.query_token.expand(batch_size, -1, -1)
        
        # C. Attention Cross : Le Query regarde les Keys/Values
        # attn_output shape: (B, 1, embed_dim)
        attn_output, _ = self.attn(query=q, key=kv, value=kv)
        
        # D. Connexion résiduelle + Norm (Standard Transformer)
        # Note: On peut ajouter q à l'output, mais ici q est constant, donc on utilise l'output direct
        x = self.norm1(attn_output)
        
        # E. Feed Forward Network
        x = x + self.mlp(x)
        x = self.norm2(x)
        
        # Retourne le vecteur final (B, embed_dim)
        return x.squeeze(1)


class MultiDinoAttention(nn.Module):
    """
    Wrapper qui gère plusieurs Backbones et fusionne avec Attention.
    """
    def __init__(self, backbones_config, fusion_config, **kwargs):
        super().__init__()
        
        self.backbones = nn.ModuleList()
        output_dims = []
        
        # 1. Chargement des Backbones (ex: DINOv2 Base, Small, Wavelet...)
        for bb_cfg in backbones_config:
            # On suppose que 'DinoModel' ou équivalent est accessible
            # Ici on instancie via getter ou directement si importé
            # Pour l'exemple, j'instancie DinoModel (adaptez selon votre getter)
            
            # Exemple simple si vous utilisez 'roadmap.models.dino_models.DinoModel'
            # model = DinoModel(name=bb_cfg['name'], ...)
            
            # Si vous utilisez timm ou torch.hub direct :
            if 'dinov2' in bb_cfg['name']:
                 model = torch.hub.load('facebookresearch/dinov2', bb_cfg['name'])
                 # Récupération dimension
                 dim = model.embed_dim 
            else:
                 raise NotImplementedError("Seul DINOv2 géré dans cet exemple")
            
            # Freeze ou non
            if bb_cfg.get('frozen', True):
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()
            
            self.backbones.append(model)
            output_dims.append(dim)
            
        # 2. Module de Fusion
        self.fusion_head = AttentionFusionHead(
            input_dims=output_dims,
            embed_dim=fusion_config['output_dim'],
            num_heads=fusion_config.get('num_heads', 8),
            dropout=fusion_config.get('dropout', 0.1)
        )
        
    def forward(self, x):
        features = []
        
        # Passage dans chaque backbone
        for i, backbone in enumerate(self.backbones):
        
            # Extraction feature DINO (souvent output directe ou via forward_features)
            out = backbone(x[..., i, :, :])  # Adaptez selon votre format d'entrée (ex: [B, 3, H, W] -> [B, 3, H, W] pour chaque branche)
            
            # Gestion des outputs DINO (parfois dict, parfois tensor)
            if isinstance(out, dict):
                feat = out['x_norm_clstoken'] # Si DINO retourne un dict
            else:
                feat = out # Si retourne direct (B, Dim)
            
            features.append(feat)
            
        # Fusion par Attention
        final_embedding = self.fusion_head(features)
        
        # Normalisation finale pour le Retrieval (Hypersphère)
        return F.normalize(final_embedding, p=2, dim=1)