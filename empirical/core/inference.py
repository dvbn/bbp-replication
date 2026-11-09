"""Row influences and the signed two-way (country, family) cluster covariance.

Per country g at the slope root (B within-centred and column-scaled):
    q = B beta, mu = exp(q), w = mu^(2-tau), W = sum_g w, t = y mu^(1-tau),
    a = sum_g t / W, rho = t - a w, psi_g = sum_g B rho, bbar_g = sum_g B w / W.
Cancelled row score:  (B_i - bbar_g) rho_i
    = d psi_g / d y_i * (y_i - a_g mu_i)   (a_g only as plug-in for residuals).
Joint row score:      a_g^(1-tau) (B_i - bbar_g + (1-tau) psi_g / (a_g W_g)) rho_i
    = beta block of the explicit country-dummy observed-Jacobian (Schur)
      influence; the psi_g term comes from the derivative of the country score
      in the country effect and vanishes only at tau = 1.
Both row scores sum within country to the group's equation contribution.
Influence phi_i = H^{-1} score_i with H the actual (non-symmetric) negative
Jacobian of the same equation, solved through the root's LU factor.
Covariance: country meat + family meat - intersection meat (CGM, signed;
never eigenvalue-clipped).  CR0 is the main version; CR1 multiplies each meat
by G_k/(G_k-1) and is exported only as a labelled sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

Z95 = float(norm.ppf(0.975))


def exp_row_scores(panel, theta, tau, kind):
    """(n, p) row scores in sample-row order for the cancelled or joint equation."""
    n = sum(c.n_obs for c in panel.cohorts)
    out = np.full((n, panel.d), np.nan)
    tau = float(tau)
    for c, st in zip(panel.cohorts, panel.states(theta, tau)):
        bbar = c.apply_t(st.pi)  # sum_g B w / W
        dev = c.bloc - bbar[:, None, :]
        if kind == "C":
            sc = dev * st.rho[:, :, None]
        else:
            if tau != 1.0:
                dev = dev + ((1.0 - tau) * st.psi_loc / (st.a * st.W)[:, None])[:, None, :]
            sc = (st.a ** (1.0 - tau))[:, None, None] * dev * st.rho[:, :, None]
        out[c.rows.ravel()] = sc.reshape(-1, panel.d)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite row score")
    return out


def cluster_meat(phi, codes):
    """Sum outer products of cluster sums; count nonempty clusters only."""
    inv = np.unique(codes, return_inverse=True)[1].ravel()
    G = int(inv.max() + 1)
    sums = np.zeros((G, phi.shape[1]))
    np.add.at(sums, inv, phi)
    return sums.T @ sums, G


@dataclass
class TwoWay:
    cr0: np.ndarray
    cr1: np.ndarray
    meats: dict  # name -> (meat, G)


def two_way(phi, country, family, inter):
    m = {k: cluster_meat(phi, v) for k, v in (("country", country), ("family", family), ("intersection", inter))}
    cr0 = m["country"][0] + m["family"][0] - m["intersection"][0]
    adj = {k: G / (G - 1.0) if G > 1 else np.nan for k, (_, G) in m.items()}
    cr1 = adj["country"] * m["country"][0] + adj["family"] * m["family"][0] - adj["intersection"] * m["intersection"][0]
    return TwoWay(cr0, cr1, m)


def summarize(theta, V_scaled, scale, label):
    """Original-unit coefficients, SEs and normal 95% intervals; a nonpositive
    signed variance yields NaN SE with an explicit status, never a clip."""
    beta = theta / scale
    V = V_scaled / np.outer(scale, scale)
    var = np.diag(V)
    ok = np.isfinite(var) & (var > 0)
    se = np.where(ok, np.sqrt(np.where(ok, var, 1.0)), np.nan)
    status = np.where(ok, "ok", f"{label}_variance_nonpositive")
    return beta, V, se, beta - Z95 * se, beta + Z95 * se, status


def eig_summary(V):
    ev = np.linalg.eigvalsh(0.5 * (V + V.T))
    return {"min_eigenvalue": float(ev.min()), "max_eigenvalue": float(ev.max()),
            "n_negative_eigenvalues": int((ev < 0).sum())}
