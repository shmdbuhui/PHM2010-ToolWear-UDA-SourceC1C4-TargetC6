"""Conditional Operator Discrepancy from MathAI-LAB/COD, MPI3D_main.py.

The source implementation calls this metric with source labels and detached target
predictions. The caller owns that choice and the optional four random columns.
"""

import torch


def mul_guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    """Official five-bandwidth Gaussian kernel, including detached bandwidth."""
    n_source = source.size(0)
    total = torch.cat((source, target), dim=0)
    n = total.size(0)
    distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    bandwidth = distance.detach().sum() / (n * n - n)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = sum(torch.exp(-distance / (bandwidth * kernel_mul ** i))
                  for i in range(kernel_num))
    return (kernels[:n_source, :n_source], kernels[n_source:, n_source:],
            kernels[:n_source, n_source:], kernels[n_source:, :n_source])


def cod_metric(fea_s, fea_t, prob_s, prob_t, epsilon=5e-2):
    """Literal CMMD + CKB arithmetic from official ``COD_Metric``."""
    ns, nt = fea_s.size(0), fea_t.size(0)
    eye_s = torch.eye(ns, device=fea_s.device, dtype=fea_s.dtype)
    eye_t = torch.eye(nt, device=fea_t.device, dtype=fea_t.dtype)
    hs = eye_s - torch.ones_like(eye_s) / ns
    ht = eye_t - torch.ones_like(eye_t) / nt
    k_zss, k_ztt, k_zst, k_zts = mul_guassian_kernel(fea_s, fea_t)
    k_yss, k_ytt, _, k_yts = mul_guassian_kernel(prob_s, prob_t)

    inv_yss = torch.inverse(k_yss.mm(k_yss) + epsilon * eye_s).mm(k_yss)
    inv_ytt = torch.inverse(k_ytt.mm(k_ytt) + epsilon * eye_t).mm(k_ytt)
    ct = k_ytt.mm(inv_ytt).mm(k_ztt).mm(inv_ytt)
    cs = k_yss.mm(inv_yss).mm(k_zss).mm(inv_yss)
    cst = k_yts.mm(inv_yss).mm(k_zst).mm(inv_ytt)
    cmmd = -torch.sqrt(cs.trace() + ct.trace() + 2 * cst.trace())

    g_ys = hs.mm(k_yss).mm(hs)
    g_yt = ht.mm(k_ytt).mm(ht)
    g_zs = hs.mm(k_zss).mm(hs)
    g_zt = ht.mm(k_ztt).mm(ht)
    inv_s = torch.inverse(epsilon * ns * eye_s + g_ys)
    inv_t = torch.inverse(epsilon * nt * eye_t + g_yt)
    rs = epsilon * g_zs.mm(inv_s)
    rt = epsilon * g_zt.mm(inv_t)
    bs = ns * epsilon * inv_s
    bt = nt * epsilon * inv_t
    bs = (bs + bs.t()) / 2
    bt = (bt + bt.t()) / 2
    ss, us = torch.linalg.eigh(bs)
    st, ut = torch.linalg.eigh(bt)
    # Match the official NaN guards and 1e-4 eigenvalue shift.
    if torch.isnan(ss[0]):
        ss = ss.clone()
        ss[0] = 0
    if torch.isnan(st[0]):
        st = st.clone()
        st[0] = 0
    ssn = torch.diag((torch.maximum(ss, torch.zeros_like(ss)) + 1e-4).sqrt())
    stn = torch.diag((torch.maximum(st, torch.zeros_like(st)) + 1e-4).sqrt())
    hcs = hs.mm(us.mm(ssn))
    hct = ht.mm(ut.mm(stn))
    nuclear = hct.t().mm(k_zts).mm(hcs)
    singular = torch.linalg.svdvals(nuclear)
    ckb = rs.trace() + rt.trace() - 2 * singular[:-1].sum() / ((ns * nt) ** 0.5)
    return cmmd + ckb
