from .blackbox_ap import BlackBoxAP
from .calibration_loss import CalibrationLoss
from .fast_ap import FastAP
from .softbin_ap import SoftBinAP
from .pair_loss import PairLoss
from .cross_entropy_loss import CrossEntropy
from torch.nn import CrossEntropyLoss
from .multi_ce_loss import MultiCrossEntropyLoss
from .smooth_rank_ap import (
    HeavisideAP,
    SmoothAP,
    SupAP,
)


__all__ = [
    'BlackBoxAP',
    'CalibrationLoss',
    'FastAP',
    'SoftBinAP',
    'PairLoss',
    'HeavisideAP',
    'SmoothAP',
    'SupAP',
    'SoftmaxCrossEntropy'
    'CrossEntropy',
    'MultiCrossEntropyLoss',
]
