from pytorch_metric_learning import losses, distances
from .. import utils as lib
import torch
import torch.optim as optim


def get_optimizer_for_arcface(loss, optim_cfg):
    # Accept either a dict config like {'name': 'AdamW', 'kwargs': {...}}
    # or an object with attributes `name` and `kwargs`.
    if isinstance(optim_cfg, dict):
        opt_name = optim_cfg.get('name', 'SGD')
        opt_kwargs = optim_cfg.get('kwargs', {})
    else:
        opt_name = getattr(optim_cfg, 'name', 'SGD')
        opt_kwargs = getattr(optim_cfg, 'kwargs', {}) or {}

    optimizer = getattr(optim, opt_name)(loss.parameters(), **opt_kwargs)
    return optimizer
class ArcFaceLoss(losses.ArcFaceLoss):
    takes_embeddings = True
    def __init__(self, num_classes, embedding_size, margin=28.6, scale=64, **kwargs):
        super(ArcFaceLoss, self).__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=margin,
            scale=scale,
            distance=distances.CosineSimilarity(),
        )   
        self.loss_optimizer = get_optimizer_for_arcface(self, kwargs.get('optimizer', {'name': 'AdamW', 'kwargs': {'lr': 0.000001, 'weight_decay': 0.0005}}))

    def step(self):
        self.loss_optimizer.step()
        self.loss_optimizer.zero_grad()
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd['optimizer_state'] = self.loss_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        optimizer_state = state_dict.pop('optimizer_state', None)
        super().load_state_dict(state_dict, strict)
        if optimizer_state is not None:
            self.loss_optimizer.load_state_dict(optimizer_state)