import math
import torch
from torch import nn

from _network import ResNet50
from _network import build_model as build_net


def build_model(args, pretrained=True):
    net = build_net(args, pretrained, ResNet50Mod)
    return net.to(args.device)


class ResNet50Mod(nn.Module):
    def __init__(self, n_bits, pretrained, **kwargs):
        super().__init__()
        self.tanh = kwargs.pop("tanh", False)
        self.net = ResNet50(n_bits, pretrained, **kwargs)
        self.alpha = 1.0

    def set_alpha(self, epoch):
        """
        continuation methods from HashNet
        """
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

    def forward(self, x):
        x = self.net(x)
        if self.tanh:
            x = torch.tanh(self.alpha * x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    n = ResNet50Mod(16, True, tanh=True)
    z = n(x)
    print(z)
