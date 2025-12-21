import random
import numpy as np
from scipy.signal import resample
import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq

class Reshape(object):
    def __call__(self, seq):
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class NormalizeFixed:
    def __init__(self, stats, eps: float = 1e-8):
        self.stats = stats
        self.eps = eps

    def __call__(self, x):
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        # x shape normalize: (C,H,W) standard
        if x_np.ndim == 1:
            x_np = x_np[None, :]           # (L,) -> (1,L)
        elif x_np.ndim == 2:
            x_np = x_np[None, :, :]        # (H,W) -> (1,H,W)
        elif x_np.ndim != 3:
            raise ValueError(f"NormalizeFixed expects ndim 1/2/3, got {x_np.ndim}, shape={x_np.shape}")

        if self.stats["type"] == "zscore":
            mean = self.stats["mean"].astype(np.float32)
            std  = self.stats["std"].astype(np.float32)

            if x_np.ndim == 3:             # (C,H,W)
                mean = mean[:, None, None]
                std  = std[:, None, None]
            else:                            # (C,L)
                mean = mean[:, None]
                std  = std[:, None]

            x_np = (x_np - mean) / (std + self.eps)

        elif self.stats["type"] == "minmax":
            mn = self.stats["min"].astype(np.float32)
            mx = self.stats["max"].astype(np.float32)

            if x_np.ndim == 3:             # (C,H,W)
                mn = mn[:, None, None]
                mx = mx[:, None, None]
            else:                            # (C,L)
                mn = mn[:, None]
                mx = mx[:, None]

            denom = (mx - mn)
            denom[np.abs(denom) < self.eps] = 1.0
            x_np = (x_np - mn) / denom

        elif self.stats["type"] == "none":
            pass
        else:
            raise ValueError(f"Unknown stats type: {self.stats['type']}")

        if is_tensor:
            return torch.from_numpy(x_np).to(x.device).type_as(x)
        return x_np
