from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from collections.abc import Sequence

import numpy as np
import pywt
import torch
from PIL import Image

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
            imgs tensor: Subbands to be scaled.

        Returns:
            Tensor: Rescaled subbands.
        """
        size = img.shape

        # l = (F2.resize(li, self.size, self.interpolation, self.max_size, self.antialias).unsqueeze(-3) for li in l)
        # subbands = [*l, *(F2.resize(im, self.size, self.interpolation, self.max_size, self.antialias) for im in h)]
        # resized_and_stacked = torch.cat(subbands, dim=-3)
        return F2.resize(img, self.size, self.interpolation, self.max_size, self.antialias)

WAVELET_DICT = {
    "haar": HaarLifting,
    "cdf97": Cdf97Lifting
}
class CustomTransform:
    def __init__(self, decompose_levels=3, basis="haar", coarse_only=True, ll_only=False, device='cuda', dwt_mode='dwt'):
        self.dwt = WAVELET_DICT[basis](n_levels=decompose_levels)
        self.coarse_only = coarse_only
        self.ll_only = ll_only
        self.decompose_levels = decompose_levels
        # Don't store device in transform - let DataLoader handle device placement
        print(f'll_only: {ll_only}, coarse_only: {coarse_only}, decompose_levels: {decompose_levels}, basis: {basis}')
    def __call__(self, img):
        # Keep data on CPU during transforms
        l, h = self.dwt(img)  # Let model move data to GPU
        if not self.ll_only:
            if self.coarse_only:

                return torch.cat([l[self.decompose_levels-1].unsqueeze(-3), h[self.decompose_levels-1]], dim=-3)
            else:
                if self.decompose_levels >1:
                    raise NotImplementedError("Full subbands not implemented yet for decompose_levels > 1 ")
                subbands = [li.unsqueeze(-3) for li in l] + [hi for hi in h] # relèvera une erreur car les li n'ont pas la même taille pareil avec les hi
                return torch.cat(subbands, dim=-3)
        else:

            if self.coarse_only:
                return l[self.decompose_levels-1]
            else:
                if self.decompose_levels >1:
                    raise NotImplementedError("Full approx not implemented yet for decompose_levels > 1 ")
                return torch.cat(l, dim=-3) #retournera une erreur car les l n'ont pas la même taille
        
# img = torch.randn(2,3, 64, 64)
# transform = CustomTransform(decompose_levels=3, basis="haar", coarse_only=True)
# transformed_img = transform(img)
# print(transformed_img.shape)  # Should print the shape of the transformed image



class BaseWaveletTransform(object):
    """
    Classe Mère abstraite gérant le pipeline commun aux ondelettes :
    - Ajustement de la taille de l'image
    - Boucle sur les canaux RGB
    - Conversion finale en Tenseur PyTorch
    """
    def __init__(self, level=1, wavelet='haar'):
        self.level = level
        self.wavelet = wavelet

    def fix_size(self, image):
        w, h = image.size
        # Le facteur dépend du niveau pour s'assurer que la division euclidienne passe
        factor = 2 ** self.level 
        new_w = int(np.ceil(w / factor) * factor)
        new_h = int(np.ceil(h / factor) * factor)
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        return image

    def _apply_wavelet(self, channel_pixels):
        """
        Méthode polymorphe. Doit être écrasée par les classes filles.
        Prend un tableau 2D (H, W) et retourne un tableau 3D (4, H', W').
        """
        raise NotImplementedError("Cette méthode doit être définie dans la sous-classe.")

    def __call__(self, img):
        img = self.fix_size(img)
        img_np = np.array(img).astype(np.float32) / 255.0
        
        channels_output = []
        for c in range(3):
            channel_pixels = img_np[:, :, c]
            subbands = self._apply_wavelet(channel_pixels)
            
            channels_output.append(subbands)

        output = np.stack(channels_output)
        return torch.from_numpy(output).float()


# ==========================================
# LES CLASSES FILLES
# ==========================================

class SWTTransform(BaseWaveletTransform):
    """ Transformée Stationnaire (Taille préservée: H, W) """
    
    def _apply_wavelet(self, channel_pixels):
        coeffs = pywt.swt2(channel_pixels, self.wavelet, level=self.level)
        cA, (cH, cV, cD) = coeffs[0]
        return np.stack([cA, cH, cV, cD])

    def __repr__(self):
        return f"SWTTransform(shape='C,S,H,W', wavelet={self.wavelet}, level={self.level})"


class DWTTransform(BaseWaveletTransform):
    """ Transformée Discrète multi-niveaux (Taille divisée par 2^level) """

    def __init__(self, level=1, wavelet='haar'):
        super().__init__(level=level, wavelet=wavelet)

    def _apply_wavelet(self, channel_pixels):
        coeffs = pywt.wavedec2(channel_pixels, self.wavelet, level=self.level)
        cA = coeffs[0]
        cH, cV, cD = coeffs[1]
        return np.stack([cA, cH, cV, cD])

    def __repr__(self):
        factor = 2 ** self.level
        return f"DWTTransform(shape='C,S,H/{factor},W/{factor}', wavelet={self.wavelet}, level={self.level})"