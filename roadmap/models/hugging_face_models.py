
from torch import nn
from transformers import AutoModel


class HuggingFaceVisionWrapper(nn.Module):
    def __init__(self, model_name, pretrained=True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.kwargs = kwargs

        # Load the Hugging Face model
        self.model = AutoModel.from_pretrained(model_name) if pretrained else AutoModel.from_config(model_name)
        self.model = self.model.vision_model
        self.feature_dim = self.model.config.hidden_size

    def forward(self, x):

        _, cls_emb = self.model(x, return_dict=False)
        return cls_emb