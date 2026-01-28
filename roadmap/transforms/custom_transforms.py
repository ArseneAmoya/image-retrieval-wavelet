from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from collections.abc import Sequence

from .wavelets import fast_haar_2d_op, fast_cdf97_2d_op

class Cdf97Lifting(nn.Module):
    def __init__(self, n_levels=1, *args, **kwargs):
        super(Cdf97Lifting, self).__init__(*args, **kwargs)
        self.n_levels = n_levels
    def forward_one(self, x):
        c, h, w = x.shape[-3:]
        pad_right = (4 - (w % 4)) % 4
        pad_bottom = (4 - (h % 4)) % 4
        x = F.pad(x, (0, pad_right, 0, pad_bottom))
        ll, lh, hl, hh = fast_cdf97_2d_op(x)
        return ll, torch.stack([lh, hl, hh], dim=-3)
    def forward(self, x):
        details = []
        approx = []
        for _ in range(self.n_levels):
            x, high = self.forward_one(x)
            details.append(high)
            approx.append(x)
        return approx, details



class HaarLifting(nn.Module):
    def __init__(self, n_levels=1, *args, **kwargs):
        super(HaarLifting, self).__init__(*args, **kwargs)
        self.n_levels = n_levels
    def forward_one(self, x):
        c, h, w = x.shape[-3:]
        pad_right = w % 2
        pad_bottom = h % 2
        x = F.pad(x, (0, pad_right, 0, pad_bottom))
        ll, lh, hl, hh = fast_haar_2d_op(x)
        return ll, torch.stack([lh, hl, hh], dim=-3)
    def forward(self, x):
        details = []
        approx = []
        for _ in range(self.n_levels):
            x, high = self.forward_one(x)
            details.append(high)
            approx.append(x)
        return approx, details

class ResizeSubBands(nn.Module):
    def __init__(self, size,  interpolation=F2.InterpolationMode.BILINEAR, max_size=None, antialias=True):
        super(ResizeSubBands, self).__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size
        if isinstance(interpolation, int):
            interpolation = F2._interpolation_modes_from_int(interpolation)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (Tuple of tensors): Subbands to be scaled.

        Returns:
            Tensor: Rescaled subbands.
        """
        l, h = img

        l = (F2.resize(li, self.size, self.interpolation, self.max_size, self.antialias).unsqueeze(-3) for li in l)
        subbands = [*l, *(F2.resize(im, self.size, self.interpolation, self.max_size, self.antialias) for im in h)]
        resized_and_stacked = torch.cat(subbands, dim=-3)
        return resized_and_stacked

WAVELET_DICT = {
    "haar": HaarLifting,
    "cdf97": Cdf97Lifting
}
class CustomTransform:
    def __init__(self, decompose_levels=3, basis="haar", coarse_only=True, ll_only=False, device='cuda'):
        self.dwt = WAVELET_DICT[basis](n_levels=decompose_levels)
        self.coarse_only = coarse_only
        self.ll_only = ll_only
        self.decompose_levels = decompose_levels
        # Don't store device in transform - let DataLoader handle device placement
    
    def __call__(self, img):
        # Keep data on CPU during transforms
        l, h = self.dwt(img)  # Let model move data to GPU
        if not self.ll_only:
            if self.coarse_only:

                return torch.cat([l[self.decompose_levels-1].unsqueeze(-3), h[self.decompose_levels-1]], dim=-3)
            else:
                subbands = [li.unsqueeze(-3) for li in l] + [hi for hi in h]
                return torch.cat(subbands, dim=-3)
        else:
            if self.coarse_only:
                return l[self.decompose_levels-1]
            else:
                return torch.cat(l, dim=-3)
        
# img = torch.randn(2,3, 64, 64)
# transform = CustomTransform(decompose_levels=3, basis="haar", coarse_only=True)
# transformed_img = transform(img)
# print(transformed_img.shape)  # Should print the shape of the transformed image
