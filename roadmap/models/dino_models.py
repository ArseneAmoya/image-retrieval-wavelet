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
        
        return F.normalize(logits, dim=-1)