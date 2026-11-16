"""Weidner-Zylkin (2021) PPML-only analytical bias, adjusted covariance and
split-panel jackknife, adapted to the all-pairs (diagonal-including) design.

Provenance.  The formulas follow WZ (2021, JIE 132:103513) main Sec. 3.4 and
Supplement A.2-A.3, A.7 as recorded in the independent pyiols reference
(`pyiols/src/pyiols/published/weidner_zylkin.py`, `docs/math/weidner_zylkin.md`),
which is a dense OFF-diagonal oracle that rejects diagonal pairs.  This module is
a new local adaptation: diagonal pairs (i, i) are ordinary pairs whose origin
and destination blocks are distinct, so every exporter/importer group simply
contains its diagonal pair.  No code is copied from the authors' Stata `.ado`.
Exact parity with `ppml_fe_bias` on diagonal-including panels is NOT
established here and requires host Stata runs (status: pending).

At the tau=1 cancelled root theta (== joint PPML root), with fitted pair means
lam_g = a_g mu_g, L_g = sum lam, p_g = lam/L:
  profile score  S_g = y_g - p_g sum y_g      (== rho_g at tau=1 exactly)
  Hbar_g = diag(lam) - lam lam'/L             (our Chat_g at tau=1)
  obs_g  = sum y_g / L_g                      (observed-total multiplier)
  xtilde = lam-weighted residual of x on all FE (pair-constant shifts irrelevant)
  A_s    = sum xtilde' Hbar xtilde            (= Schur complement of H at beta)
  G_g[x] = obs_g ( -diag(lam o z) + ((lam o z) lam' + lam (lam o z)')/L ),
           z = x - p'x
  exporter i:  v_i = - sum_j obs S' H_i^+ Hbar x~ + 1/2 tr[(sum_j G[x~]) H_i^+ V_i H_i^+]
               H_i = sum_j Hbar, V_i = sum_j S S', H_i^+ = F (F'H_i F)^{-1} F'
  importer j:  w_j analogously.
  bias_WZ = N/(N-1) * (sum v + sum w) / A_s      (finite-sample factor stored
            separately: wz_bias_no_factor omits N/(N-1)).
Adjusted covariance (one-sided, not HC3):
  Lev_g = Hbar_g (x~ A_s^{-1} x~' + d_g A_D^{-1} d_g') = Hbar_g B_g H^{-1} B_g'
  V_adj = P/(P-1) A_s^{-2} sum x~' (I - Lev_g)^{-1} S S' x~,
  V_unadj = P/(P-1) A_s^{-2} sum (x~'S)^2   (= P/(P-1) x our pair sandwich).
The WZ bias is NOT forced to equal the generic manuscript correction; both are
stored and their difference is a study output.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve


@dataclass
class WZResult:
    bias: float
    bias_no_factor: float
    finite_factor: float
    var_unadjusted: float
    var_adjusted: float
    se_unadjusted: float
    se_adjusted: float
    A_s: float
    exporter_terms: np.ndarray
    importer_terms: np.ndarray
    min_leverage_singular: float
    status_bias: str
    status_cov: str


def _profile_objects(design, y, ev):
    st = ev.state
    if abs(st.tau - 1.0) > 1e-14:
        raise ValueError("WZ adapter is PPML-only (tau = 1)")
    lam = st.a[:, None] * st.mu
    L = lam.sum(axis=1)
    p = lam / L[:, None]
    ysum = y.sum(axis=1)
    obs = ysum / L
    S = y - p * ysum[:, None]
    Hbar = -lam[:, :, None] * lam[:, None, :] / L[:, None, None]
    idx = np.arange(design.T)
    Hbar[:, idx, idx] += lam
    return lam, L, p, obs, S, Hbar


def partialled_regressor(design, H):
    """xtilde_g = xc_g - D_g delta, delta = A_D^{-1} A_Dx, and A_s."""
    A_D = H[1:, 1:]
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            lu = lu_factor(A_D)
        except LinAlgWarning as exc:  # exactly singular nuisance information
            raise np.linalg.LinAlgError(str(exc)) from exc
    if np.any(np.diag(lu[0]) == 0):
        raise np.linalg.LinAlgError("singular nuisance information A_D")
    delta = lu_solve(lu, H[1:, 0])
    A_s = float(H[0, 0] - H[0, 1:] @ delta)
    theta_delta = np.concatenate(([0.0], delta))
    xt = design.xc - design.index(theta_delta)
    return xt, A_s


def _group_terms(design, group, ngroups, S, Hbar, obs, lam, L, p, xt):
    T = design.T
    F = design.F
    Hsum = np.zeros((ngroups, T, T))
    np.add.at(Hsum, group, Hbar)
    Vsum = np.zeros((ngroups, T, T))
    np.add.at(Vsum, group, S[:, :, None] * S[:, None, :])
    red = np.einsum("tk,ats,sl->akl", F, Hsum, F)
    # positive definiteness of the identified time-contrast information
    np.linalg.cholesky(red)  # raises LinAlgError if not PD
    Hdag = np.einsum("tk,akl,sl->ats", F, np.linalg.inv(red), F)
    Hx = obs[:, None] * np.einsum("gts,gs->gt", Hbar, xt)
    first_g = np.einsum("gt,gts,gs->g", S, Hdag[group], Hx)
    first = np.bincount(group, weights=first_g, minlength=ngroups)
    z = xt - np.sum(p * xt, axis=1, keepdims=True)
    lz = lam * z
    Gg = (lz[:, :, None] * lam[:, None, :] + lam[:, :, None] * lz[:, None, :]) / L[:, None, None]
    idx = np.arange(T)
    Gg[:, idx, idx] -= lz
    Gg *= obs[:, None, None]
    Gsum = np.zeros((ngroups, T, T))
    np.add.at(Gsum, group, Gg)
    M = np.matmul(np.matmul(Hdag, Vsum), Hdag)
    second = 0.5 * np.einsum("ats,ast->a", Gsum, M)
    return -first + second


def wz_correction(design, y, ev, Hinv, n_countries=None):
    """WZ bias, unadjusted and adjusted variances at the tau=1 root."""
    lam, L, p, obs, S, Hbar = _profile_objects(design, y, ev)
    xt, A_s = partialled_regressor(design, ev.H)
    N = int(n_countries) if n_countries is not None else max(design.No, design.Nd)
    status_bias = "ok"
    try:
        v = _group_terms(design, design.pair_o, design.No, S, Hbar, obs, lam, L, p, xt)
        w = _group_terms(design, design.pair_d, design.Nd, S, Hbar, obs, lam, L, p, xt)
        raw = float((v.sum() + w.sum()) / A_s)
        factor = N / (N - 1.0)
        bias = factor * raw
    except np.linalg.LinAlgError:
        v = w = np.full(1, np.nan)
        raw = bias = factor = np.nan
        status_bias = "failed"
    G = design.G
    ff = G / (G - 1.0)
    infl = np.sum(xt * S, axis=1)
    var_un = ff * float(np.sum(infl * infl)) / A_s**2
    P = design.project_local(design.gather_mat(Hinv))
    comp = np.eye(design.T)[None, :, :] - np.matmul(Hbar, P)
    sv = np.linalg.svd(comp, compute_uv=False)
    min_sv = float(np.min(sv[:, -1] / sv[:, 0]))
    status_cov = "ok"
    if min_sv <= 100 * design.T * np.finfo(float).eps:
        var_adj = np.nan
        status_cov = "failed"
    else:
        adj_ss = np.linalg.solve(comp, S[:, :, None] * S[:, None, :])
        var_adj = ff * float(np.einsum("gt,gts,gs->", xt, adj_ss, xt)) / A_s**2
    se_un = float(np.sqrt(var_un)) if var_un > 0 else np.nan
    se_adj = float(np.sqrt(var_adj)) if np.isfinite(var_adj) and var_adj > 0 else np.nan
    if status_cov == "ok" and not np.isfinite(se_adj):
        status_cov = "failed"  # nonpositive adjusted variance; no PSD repair
    return WZResult(bias, raw, factor, var_un, var_adj, se_un, se_adj, A_s, v, w,
                    min_sv, status_bias, status_cov)


def spj_quarters(cell):
    """Four WZ split-panel quarters: origin positions <= n/2 or > n/2 crossed
    with destination positions <= n/2 or > n/2 (WZ: split_i = i <= n/2)."""
    No, Nd, _ = cell.shape
    o_lo, o_hi = np.arange(No // 2), np.arange(No // 2, No)
    d_lo, d_hi = np.arange(Nd // 2), np.arange(Nd // 2, Nd)
    return (
        ("split_i&split_j", o_lo, d_lo),
        ("split_i&!split_j", o_lo, d_hi),
        ("!split_i&split_j", o_hi, d_lo),
        ("!split_i&!split_j", o_hi, d_hi),
    )


def spj_combine(beta_full, beta_quarters):
    """b_spj = 2 b - mean of the four quarter estimates (WZ table scripts)."""
    return 2.0 * beta_full - float(np.mean(beta_quarters))
