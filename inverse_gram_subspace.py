"""Project-defined inverse Gram subspace loss; this is not Kim et al.'s IGSM.

The DARE-GRAM repository motivates the comparison, but the formula below is
specified by this project's experiment and is deliberately a separate loss.
"""

from __future__ import annotations

import torch


ENERGY = 0.90
EPS_RELATIVE = 1e-8


def _basis(features: torch.Tensor, energy: float, eps_relative: float):
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError(f"Expected batch x 512 ResNet features, got {tuple(features.shape)}")
    # Double precision keeps the thin SVD and its backward pass better conditioned.
    z = torch.cat((torch.ones_like(features[:, :1]), features), dim=1).double()
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite input features")
    _, sigma, vh = torch.linalg.svd(z, full_matrices=False)
    squared = sigma.square()
    cumulative = torch.cumsum(squared.detach(), dim=0) / squared.detach().sum()
    rank = int(torch.searchsorted(cumulative, energy, right=False).item()) + 1
    rank = min(rank, sigma.numel())
    eps = squared[0].detach().clamp_min(1.0) * eps_relative
    if not torch.isfinite(sigma).all() or not torch.isfinite(eps):
        raise FloatingPointError("Nonfinite singular values or epsilon")
    return sigma, vh, rank, eps


def inverse_gram_subspace_loss(source: torch.Tensor, target: torch.Tensor,
                               energy: float = ENERGY, eps_relative: float = EPS_RELATIVE):
    """Squared Frobenius distance of unit-norm truncated inverse Gramians.

    Z=[1,H]=U diag(sigma) V^T; k is the smaller 90%-energy rank.  The inverse
    Gram subspace matrix is V_k diag(1/(sigma_k^2+eps)) V_k^T.  eps is detached
    max(sigma_1^2,1)*1e-8 for each domain, so rank/regularization do not use
    target labels or inject a gradient through a discrete selection rule.
    """
    if not 0 < energy <= 1 or eps_relative <= 0:
        raise ValueError("Invalid energy or relative epsilon")
    sig_s, vh_s, rank_s, eps_s = _basis(source, energy, eps_relative)
    sig_t, vh_t, rank_t, eps_t = _basis(target, energy, eps_relative)
    k = min(rank_s, rank_t)

    def matrix(sigma, vh, eps):
        v = vh[:k].T
        inv = (sigma[:k].square() + eps).reciprocal()
        m = (v * inv.unsqueeze(0)) @ v.T
        return m / torch.linalg.matrix_norm(m, ord="fro")

    ms, mt = matrix(sig_s, vh_s, eps_s), matrix(sig_t, vh_t, eps_t)
    loss = torch.linalg.matrix_norm(ms - mt, ord="fro").square()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite alignment loss; source sigma={sig_s.detach().cpu().tolist()}; "
                                 f"target sigma={sig_t.detach().cpu().tolist()}")
    detail = {"k": k, "rank_source": rank_s, "rank_target": rank_t,
              "sigma_source_max": float(sig_s[0].detach()),
              "sigma_source_min": float(sig_s[-1].detach()),
              "sigma_source_kept_min": float(sig_s[k-1].detach()),
              "sigma_target_max": float(sig_t[0].detach()),
              "sigma_target_min": float(sig_t[-1].detach()),
              "sigma_target_kept_min": float(sig_t[k-1].detach()),
              "epsilon_source": float(eps_s), "epsilon_target": float(eps_t)}
    return loss, detail
