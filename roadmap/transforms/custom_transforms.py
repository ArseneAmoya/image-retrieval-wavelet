from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        details = torch.stack([y[:, h//2:, :w//2].view(b,-1,h//2, w//2), y[:, :h//2, w//2:].view(b,-1,h//2, w//2), y[:, h//2:, w//2:].view(b,-1,h//2, w//2)], dim = 2).squeeze(1)

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

class CustomTransform:
    def __init__(self, decompose_levels=3, basis="haar", coarse_only = True, device='cuda'):
        self.dwt = Cdf97Lifting(n_levels = decompose_levels) if basis == "cdf97" else DWTForward(J=decompose_levels, wave= basis, mode='zero')
        self.coarse_only = coarse_only
        self.decompose_levels = decompose_levels
        self.device = device
    def __call__(self, img):
        # This is a placeholder for a real transformation.
        # For now, it just returns the image as is.
        img = img.to(self.device)
        l, h = self.dwt(img)
        if self.coarse_only:
            return torch.cat([l.unsqueeze(-3),h[self.decompose_levels -1]], dim=-3)
        else:
            raise NotImplementedError("Full decomposition not implemented yet.")
            # return torch.stack([l.unsqueeze(1)] + h, dim=1)
    
# img = torch.randn(2,3, 64, 64)
# transform = CustomTransform(decompose_levels=2, basis="cdf97", corse_only=True)
# transformed_img = transform(img)
# print(transformed_img.shape)  # Should print the shape of the transformed image
