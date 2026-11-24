"""Global seeding for reproducibility across python / numpy / torch."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic_torch: bool = True) -> int:
    """Seed python, numpy, and (if available) torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
