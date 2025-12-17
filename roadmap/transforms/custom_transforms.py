from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from collections.abc import Sequence

from .wavelets import fast_haar_2d_op


class Cdf97Lifting(nn.Module):
    def __init__(self, n_levels = 2, *args, **kwargs) -> None:
        super(Cdf97Lifting, self).__init__(*args, **kwargs)
        self.n_levels = n_levels

    def forward_one(self, x):
        size = x.shape #(batch, height, width) or (batch, channel, height, width)

        if(len(size) == 3):
            b, h, w = size
        elif(len(size) == 4):
            b, c, h, w = size
            assert size[1] == 3, f"Invalid number of channels, expected 3, got {c}"
            x = x.reshape(-1, h, w )
        else:
            raise("Exepected in put to be of fromat B,C,H,W for color images or B,H,W for gray level image")
        pad_right = 0
        pad_bottom =  0
        # Padding in the case odd length size
        if w % 2 != 0:
            pad_right = 1
            w += 1
        if h % 2 != 0:
            pad_bottom = 1
            h += 1
        x = F.pad(x, (0, pad_right, 0, pad_bottom))
        y = self.fwt97_batch(self.fwt97_batch(x).transpose(2,1)).transpose(2,1)
        ll = y[:, :h//2, :w//2].view(b,-1,h//2, w//2).squeeze(1)
        details = torch.stack([y[:, h//2:, :w//2].view(b,-1,h//2, w//2), y[:, :h//2, w//2:].view(b,-1,h//2, w//2), y[:, h//2:, w//2:].view(b,-1,h//2, w//2)], dim = -3).squeeze(1)

        return ll, details

    def fwt97_batch(self, X):
        '''fwt97 forwards a 9/7 wavelets transform of on a batch of 2D image X
        This function only does one pass i.e, The low pass filter and the high pass filter.
        X should be of size B, H, W for Batch, Height, Width respectively.
        '''
        #9/7 cofficients
        a1 = -1.586134342
        a2 = -0.05298011854
        a3 = 0.8829110762
        a4 = 0.4435068522
        #scaling factors
        k1 = 0.81289306611596146 # 1/1.230174104914
        k2 = 0.61508705245700002 # 1.230174104914/2
        # Another k used by P. Getreuer is 1.1496043988602418
        X[:, 1:-1:2]  += a1 * (X[:, 0:-2:2] + X[:, 2::2]) #predict 1
        X[:, -1] += 2 * a1 * X[:, -2] #Symetric extension

        X[:, 2::2] += a2 * (X[:, 1:-1:2] + X[:, 3::2]) #update 1
        X[:,0] += 2 * a2 * X[:, 1] #Symetric extension

        X[:, 1:-1:2]  += a3 * (X[:, 0:-2:2] + X[:, 2::2]) #predict 2
        X[:, -1] += 2 * a3 * X[:, -2] #Symetric extension

        X[:, 2::2] += a4 * (X[:, 1:-1:2] + X[:, 3::2]) #update 2
        X[:,0] += 2 * a4 * X[:, 1] #Symetric extension
        #de-interleave
        b, h, w = X.shape
        temp_bank = torch.zeros(b, h, w, device=X.device)
        temp_bank[:, :int(h/2)] = k1 * X[:, ::2] #even
        temp_bank[:, int(h/2):] = k2 * X[:, 1::2] #odd
        X = temp_bank
        return X
    def forward(self, x):
        det = []
        for _ in range(self.n_levels):
            x, high = self.forward_one(x)
            det.append(high)
        return x, det

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
    def __init__(self, decompose_levels=3, basis="haar", coarse_only=True, device='cuda'):
        self.dwt = WAVELET_DICT[basis](n_levels=decompose_levels)
        self.coarse_only = coarse_only
        self.decompose_levels = decompose_levels
        # Don't store device in transform - let DataLoader handle device placement
    
    def __call__(self, img):
        # Keep data on CPU during transforms
        l, h = self.dwt(img)  # Let model move data to GPU
        if self.coarse_only:
            return torch.cat([l[self.decompose_levels-1].unsqueeze(-3), h[self.decompose_levels-1]], dim=-3)
        else:
            subbands = [li.unsqueeze(-3) for li in l] + [hi for hi in h]
            return torch.cat(subbands, dim=-3)
    
# img = torch.randn(2,3, 64, 64)
# transform = CustomTransform(decompose_levels=3, basis="haar", coarse_only=True)
# transformed_img = transform(img)
# print(transformed_img.shape)  # Should print the shape of the transformed image
