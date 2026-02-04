import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoModel_ce(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes, feature_mode="default", dropout=0.5):
        super(DinoModel_ce, self).__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        # Define the classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        
    def forward(self, x):
        # Extract features using the base model
        features = self.base_model.forward_features(x)

        if self.training:
            # Pass the features through the classification head
            logits = self.classification_head(features['x_norm_clstoken'])
            return logits

        return F.normalize(features['x_norm_clstoken'], dim=-1)
    
class Multi_DinoModel(nn.Module):
    def __init__(self, base_model, branches= [0,1,2,3]):
        super(Multi_DinoModel, self).__init__()
        self.base_model = base_model
        self.branche_selected = branches # one can choose to discard some branches by specifying indices to keep
        self.n_branches = len(branches)
        self.branches = nn.ModuleList([copy.deepcopy(self.base_model) for _ in range(self.n_branches)])        
        
    def forward(self, x):
        # Extract features using the base model
        b, c, s, h, w = x.shape  # x shape: [B, 3, 4, H, W]
        assert s >= self.n_branches, f"Expected at least {self.n_branches} branches, but got {s}"
        outputs = []
        for i, b in enumerate(self.branche_selected):
            xi = x[:, :, b, :, :]  # Shape: [B, 3, H, W]
            features = self.branches[i].forward_features(xi)['x_norm_clstoken']
            if self.training:
                features = F.normalize(features, dim=-1)
            outputs.append(features)
        if not self.training:
            out = F.normalize(torch.cat(outputs, dim=-1), dim=-1)
            return out
        return outputs