import torch
from torch import nn
import copy
import numpy as np
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models
import roadmap.utils as lib

WEIGHTS_DICT = {
    'resnet18': models.ResNet18_Weights,
}


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

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
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
            if pool_type=='avg':
                avg_pool = nn.AdaptiveAvgPool1d(1)(x)# F.avg_pool1d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = nn.AdaptiveMaxPool1d(1)(x)# F.max_pool1d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(-1)
        return (x.permute(0,2,1)@scale).squeeze(-1)/self.gate_channels#, scale
    def alphas(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = nn.AdaptiveAvgPool1d(1)(x)# F.avg_pool1d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = nn.AdaptiveMaxPool1d(1)(x)# F.max_pool1d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(-1)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
    def alphas(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels=4, reduction_ratio=1, pool_types=['avg', 'max'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out#(x.permute(0,2,1)@y).squeeze(-1)/self.chan
    def alphas(self, x):
        x_out = self.ChannelGate.alphas(x)
        if not self.no_spatial:
            x_out = self.SpatialGate.alphas(x_out)
        return x_out
class Eca1D_layer(nn.Module):
    """Constructs a 1D ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(Eca1D_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.chan = channel

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        #print(y.shape)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return (x.permute(0,2,1)@y).squeeze(-1)/self.chan#, y
    def alphas(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        #print(y.shape)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y
class WaveResNet(nn.Module):
    def __init__(self, decom_level = 3, wave="haar", *args, **kwargs) -> None:
        self.OUT_SIZE = kwargs.get("feature_size", 2048)
        super(WaveResNet, self).__init__()
        self.dwt = Cdf97Lifting(n_levels = decom_level) if wave == "cdf97" else DWTForward(J=decom_level, wave= wave, mode='zero')
        self.backbone = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()#nn.Linear(2048, 1024) #=
        #self.backbone.pre_logit = nn.Linear(2048, 1024)
        self.backbone.conv1 = nn.Conv2d(3, 64, (1,1))
        self.backbone.maxpool = nn.Identity()


#         if(maps):
#             self.backbone.avgpool = nn.Identity()
#             self.backbone.flatten =  nn.Identity()

        # else:

        self.backbone.avgpool = nn.AdaptiveAvgPool3d((self.OUT_SIZE, 1, 1))
        #ct = 0
        self.att = kwargs.get("attention", False)
#         for child in self.backbone.children():
#             #ct += 1
#             #if ct < 7:
#             for param in child.parameters():
#                 param.requires_grad = False
        self.lh_backbone = copy.deepcopy(self.backbone)
        self.hl_backbone = copy.deepcopy(self.backbone)
        self.hh_backbone = copy.deepcopy(self.backbone)
        if self.att:
            if kwargs.get('attention_type', None) == "eca":
                self.attention = Eca1D_layer(4)
                lib.LOGGER.info("Using ECA attention")
            else:
                self.attention = CBAM() #ChannelAttention(self.OUT_SIZE)
                lib.LOGGER.info("Using CBAM attention")
        else :
            self.attention = nn.Identity()

        self.level = decom_level -1
        self.ll_only = ll_only
        # ct = 0
        # for child in self.left.children():
        #     ct += 1
        #     if ct < 7:
        #         for param in child.parameters():
        #             param.requires_grad = False
    def forward(self, x):
        x, high = self.dwt(x)
        high = high[self.level]
        x = self.backbone(x)
        #print(y.shape, "y")
        if not self.ll_only:
            x = torch.cat([x, self.lh_backbone(high[:,:, 0]), self.hl_backbone(high[:,:, 1]), self.hh_backbone(high[:,:, 2])], dim=1)
            #print(y.shape, "all")
        if(self.att):
            x = x.view(x.size(0), 4, self.OUT_SIZE)
            x = self.attention(x)

        return x#torch.flatten(x, 1)
    def alphas(self, x):
        x, high = self.dwt(x)
        high = high[self.level]
        x = self.backbone(x)
        if not self.ll_only:
            x = torch.cat([x, self.lh_backbone(high[:,:, 0]), self.hl_backbone(high[:,:, 1]), self.hh_backbone(high[:,:, 2])], dim=1)
        if(self.att):
            x = x.view(x.size(0), 4, self.OUT_SIZE)
            x = self.attention.alphas(x)
            return x
        else:
            return None

class WaveResNetCE(nn.Module):
    def __init__(self, decom_level = 3, wave="haar", ll_only =False, *args, **kwargs) -> None:
        self.OUT_SIZE = kwargs.get("feature_size", 2048)
        super(WaveResNetCE, self).__init__()
        self.dwt = Cdf97Lifting(n_levels = decom_level) if wave == "cdf97" else DWTForward(J=decom_level, wave= wave, mode='zero')
        self.backbone = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()#nn.Linear(2048, 1024) #=
        #self.backbone.pre_logit = nn.Linear(2048, 1024)
        self.backbone.conv1 = nn.Conv2d(3, 64, (1,1))
        self.backbone.maxpool = nn.Identity()


#         if(maps):
#             self.backbone.avgpool = nn.Identity()
#             self.backbone.flatten =  nn.Identity()

        # else:

        self.backbone.avgpool = nn.AdaptiveAvgPool3d((self.OUT_SIZE, 1, 1))
        #ct = 0
        self.att = kwargs.get("attention", False)
#         for child in self.backbone.children():
#             #ct += 1
#             #if ct < 7:
#             for param in child.parameters():
#                 param.requires_grad = False
        if not ll_only :
            self.lh_backbone = copy.deepcopy(self.backbone)
            self.hl_backbone = copy.deepcopy(self.backbone)
            self.hh_backbone = copy.deepcopy(self.backbone)
        if self.att:
            if kwargs.get('attention_type', None) == "eca":
                self.attention = Eca1D_layer(4)
                lib.LOGGER.info("Using ECA attention")
            else:
                self.attention = CBAM() #ChannelAttention(self.OUT_SIZE)
                lib.LOGGER.info("Using CBAM attention")
        else :
            self.attention = nn.Identity()

        self.level = decom_level -1
        self.ll_only = ll_only

        in_features = self.OUT_SIZE * (4 if (not ll_only) and not self.att else 1)
        self.classifier = nn.Linear(in_features, kwargs.get("num_classes", 100))
        # ct = 0
        # for child in self.left.children():
        #     ct += 1
        #     if ct < 7:
        #         for param in child.parameters():
        #             param.requires_grad = False
    def forward(self, x):
        x, high = self.dwt(x)
        high = high[self.level]
        x = self.backbone(x)
        #print(y.shape, "y")
        if not self.ll_only:
            x = torch.cat([x, self.lh_backbone(high[:,:, 0]), self.hl_backbone(high[:,:, 1]), self.hh_backbone(high[:,:, 2])], dim=1)
            #print(y.shape, "all")
        if(self.att):
            x = x.view(x.size(0), 4, self.OUT_SIZE)
            x = self.attention(x)
        if self.training:          # ⇢ mode entrainement : logits
            return self.classifier(x)
        else:                      # ⇢ mode évaluation / test : embeddings
            return x
    def alphas(self, x):
        x, high = self.dwt(x)
        high = high[self.level]
        x = self.backbone(x)
        if not self.ll_only:
            x = torch.cat([x, self.lh_backbone(high[:,:, 0]), self.hl_backbone(high[:,:, 1]), self.hh_backbone(high[:,:, 2])], dim=1)
        if(self.att):
            x = x.view(x.size(0), 4, self.OUT_SIZE)
            x = self.attention.alphas(x)
            return x
        else:
            return None
        

class WCNN(nn.Module):
    '''
    Implementing a Multibranch CNN where each branch process one sub-band of the DWT, the subbands are extracted outside the network.
    '''
    def __init__(self, backbone = "resnet50", num_classes=None, pretrained=True, *args, **kwargs) -> None:
        self.OUT_SIZE = kwargs.get("feature_size", 2048)
        super(WCNN, self).__init__()

        self.backbone = getattr(models, backbone)(weights = WEIGHTS_DICT[backbone].DEFAULT if pretrained else None)
        lib.LOGGER.info(f"Instantiating WCNN with backbone {backbone} and pretrained={pretrained}")
        self.backbone.fc = nn.Identity()
        self.backbone.conv1 = nn.Conv2d(3, 64, (1,1))
        self.backbone.maxpool = nn.Identity()

        self.lh_backbone = copy.deepcopy(self.backbone)
        self.hl_backbone = copy.deepcopy(self.backbone)
        self.hh_backbone = copy.deepcopy(self.backbone)

        if num_classes is not None:
            self.ll_classifier = nn.Linear(self.OUT_SIZE, num_classes)
            self.lh_classifier = nn.Linear(self.OUT_SIZE, num_classes)
            self.hl_classifier = nn.Linear(self.OUT_SIZE, num_classes)
            self.hh_classifier = nn.Linear(self.OUT_SIZE, num_classes)
        else:
            self.ll_classifier = nn.Identity()
            self.lh_classifier = nn.Identity()
            self.hl_classifier = nn.Identity()
            self.hh_classifier = nn.Identity()
    

        in_features = self.OUT_SIZE * 4
        self.classifier = nn.Linear(in_features, kwargs.get("num_classes", 100))
    def forward(self, x):
        assert x.shape[2] ==4, f"Expected input with 4 subbands, got {x.shape[2]}"
        assert len(x.shape) ==5, f"Expected input of shape B,S,C,H,W got {x.shape}"

        x = [self.backbone(x[:, :, 0]), self.lh_backbone(x[:,:, 1]), self.hl_backbone(x[:,:, 2]), self.hh_backbone(x[:,:, 3])]
            #print(y.shape, "all")
        if self.training:
            return [self.ll_classifier(x[0]), self.lh_classifier(x[1]), self.hl_classifier(x[2]), self.hh_classifier(x[3])]
        return F.normalize(torch.cat(x, dim=1), dim=1)

class WCNN_ALL(nn.Module):
    '''
    Implementing a Multibranch CNN where each branch process one sub-band of the DWT, the subbands are extracted outside the network.
    '''
    def __init__(self, backbone = "resnet18", pretrained=True, *args, **kwargs) -> None:
        self.OUT_SIZE = kwargs.get("feature_size", 512)
        super(WCNN_ALL, self).__init__()
        self.backbone = getattr(models, backbone)(weights = WEIGHTS_DICT[backbone].DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()#nn.Linear(2048, 1024) #=
        #self.backbone.pre_logit = nn.Linear(2048, 1024)

        #self.backbone.avgpool = nn.AdaptiveAvgPool3d((self.OUT_SIZE, 1, 1))
        #ct = 0
       
#         for child in self.backbone.children():
#             #ct += 1
#             #if ct < 7:
#             for param in child.parameters():
#                 param.requires_grad = False
        self.backbone2 = copy.deepcopy(self.backbone)
        self.lh_backbone = copy.deepcopy(self.backbone)
        self.hl_backbone = copy.deepcopy(self.backbone)
        self.hh_backbone = copy.deepcopy(self.backbone)

        self.lh2_backbone = copy.deepcopy(self.backbone)
        self.hl2_backbone = copy.deepcopy(self.backbone)
        self.hh2_backbone = copy.deepcopy(self.backbone)

    

        # in_features = self.OUT_SIZE * 7
        # self.classifier = nn.Linear(in_features, kwargs.get("num_classes", 100))
    def forward(self, x):
        x = torch.cat([self.backbone(x[:, :, 0]), self.backbone2(x[:, :, 1]), self.lh_backbone(x[:,:, 2]), self.hl_backbone(x[:,:, 3]), self.hh_backbone(x[:,:, 4]), self.lh2_backbone(x[:,:, 5]), self.hl2_backbone(x[:,:, 6]), self.hh2_backbone(x[:,:, 7])], dim=1)
            #print(y.shape, "all")
        return x    

    
class WCNN_Attention(nn.Module):
    '''
        Combining WCNN with attention mechanism (CBAM or ECA)
    '''
    def __init__(self, backbone = "wcnn", multibranch_backbone = "resnet18", pretrained=True, attention_type="cbam", *args, **kwargs) -> None:
        super(WCNN_Attention, self).__init__()
        coarse_only = kwargs.get("coarse_only", False)

        if not coarse_only:
            backbone = backbone + "_all"
        lib.LOGGER.info(f"Instantiating WCNN_Attention with backbone {backbone} and multibranch_backbone {multibranch_backbone}")
        if backbone == "wcnn":
            self.backbone = WCNN(backbone= multibranch_backbone, pretrained=pretrained, *args, **kwargs)
            self.num_subands = 4
        elif backbone == "wcnn_all":
            self.backbone = WCNN_ALL(backbone= multibranch_backbone, pretrained=pretrained, *args, **kwargs)
            self.num_subands = 4* (kwargs.get("decom_level", 1))
        else:
            raise ValueError(f"Backbone {backbone} not recognized for WCNN_Attention")

        if attention_type == "eca":
            self.attention = Eca1D_layer(self.num_subands)
            lib.LOGGER.info("Using ECA attention")
        else:
            self.attention = CBAM(gate_channels=self.num_subands) #ChannelAttention(self.OUT_SIZE)
            lib.LOGGER.info("Using CBAM attention")
        

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), self.num_subands, -1)
        x = self.attention(x)
        return x
    
    def alphas(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), self.num_subands, -1)
        x = self.attention.alphas(x)
        return x
    

class WCNN_Attention_CE(WCNN_Attention):
    '''
        WCNN with attention for classification task
    '''
    def __init__(self, backbone = "wcnn", multibranch_backbone = "resnet18", pretrained=True, attention_type="cbam", *args, **kwargs) -> None:
        super(WCNN_Attention_CE, self).__init__(backbone, multibranch_backbone, pretrained, attention_type, *args, **kwargs)
        features_size = kwargs.get("feature_size", None)
        num_classes = kwargs.get("num_classes", None)
        assert features_size is not None, "feature_size must be specified for WCNN_Attention_CE"
        assert num_classes is not None, "num_classes must be specified for WCNN_Attention_CE"
        lib.LOGGER.info(f"Training with cross entropy with feature size {features_size} with {num_classes} classes")
        self.classifier = nn.Linear(features_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), self.num_subands, -1)
        x = self.attention(x)
        if self.training:
            return self.classifier(x)
        return x
    
    