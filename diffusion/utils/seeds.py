import random
import numpy as np
import torch


def str_to_bool(v):
    """Argparse type for booleans; tolerates non-breaking spaces from copy-paste."""
    v = str(v).strip()
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise ValueError(f"Cannot convert {v!r} to bool")


def clean_args(args):
    """Strip unicode whitespace (including U+00A0) from all string args."""
    for key, val in vars(args).items():
        if isinstance(val, str):
            setattr(args, key, val.strip())
    return args


def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
