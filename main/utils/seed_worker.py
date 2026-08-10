import random

import numpy as np
import torch


def seed_worker(worker_id):
    """DataLoader `worker_init_fn`: seeds Python's `random` and NumPy's global RNG
    inside each worker process.

    PyTorch automatically seeds each worker's own `torch` RNG (as `base_seed +
    worker_id`, where `base_seed` is drawn from the DataLoader's `generator`) --
    but it does NOT seed `random`/`numpy`, which torchvision transforms and other
    augmentation code may use internally. Without this, two runs with the same
    `experience.seed` can still see different per-image augmentations (crop
    location, flip decisions) because the `random`/`numpy` streams inside worker
    processes aren't tied to the run's seed at all.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader -- this
    is the pattern recommended there, `worker_seed = torch.initial_seed() % 2**32`.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_worker_generator(seed):
    """A torch.Generator seeded deterministically from `seed`, to pass as the
    DataLoader's `generator=` argument. This is what PyTorch derives each
    worker's `torch.initial_seed()` from -- without passing this explicitly,
    the DataLoader falls back to the *global* default RNG, whose state at
    `DataLoader.__iter__()` time depends on everything that happened earlier in
    the process (model init order, prior forward passes, etc.), which is fragile
    and not reliably reproducible across two separately-launched runs even with
    the same `experience.seed`.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
