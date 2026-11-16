"""Grouped (pair-level) evaluation of the cancelled and profiled-joint equations.

Notation (design_report.tex, Sec. 3-4 and App. A-B), per pair g with T rows:
    q = B_g theta,  mu = exp(q),  w = mu^(2-tau),  r = y/mu,  t = w r,
    W = sum w,  pi = w/W,  T_g = sum t,  a_g = T_g/W,
    rho_g = w (r - a_g) = t - pi T_g,       psi_g = B_g' rho_g,
    S = sum_g psi_g   (raw, unnormalised sums throughout).
Observed Jacobian (eq. jacobian), generally NONsymmetric:
    -D psi_g = B_g' Chat_g B_g,  Chat_g = diag(wc) - pi wc',
    wc = w ((tau-1) r + (2-tau) a_g) = (tau-1) t + (2-tau) a_g w.
Profiled joint (eq. profiled-joint): S_J = sum_g a_g^(1-tau) psi_g with
    D a_g = (1-tau) psi_g / W - a_g bbar_g,   bbar_g = B_g' pi,
    -D(a^(1-tau) psi) = a^(1-tau)(-D psi) - (1-tau) a^(-tau) psi (D a)'.
All B_g products use the local (G, T, K) blocks of RetainedDesign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroupState:
    tau: float
    q: np.ndarray
    mu: np.ndarray
    w: np.ndarray
    r: np.ndarray
    t: np.ndarray
    W: np.ndarray
    pi: np.ndarray
    Tg: np.ndarray
    a: np.ndarray
    rho: np.ndarray
    psi_loc: np.ndarray


def evaluate(design, y, theta, tau):
    """Pair-level state at theta; y is (G, T) aligned with the design pairs."""
    tau = float(tau)
    kappa = 2.0 - tau
    q = design.index(theta)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mu = np.exp(q)
        # pi by a max-shifted softmax of kappa*q; W recovered from the same shift
        lw = kappa * q
        m = lw.max(axis=1, keepdims=True)
        e = np.exp(lw - m)
        se = e.sum(axis=1)
        pi = e / se[:, None]
        W = np.exp(m[:, 0]) * se
        w = np.exp(lw)
        r = y * np.exp(-q)
        t = y * np.exp((1.0 - tau) * q)
        Tg = t.sum(axis=1)
        a = Tg / W
        rho = t - pi * Tg[:, None]
    psi_loc = design.apply_t(rho)
    return GroupState(tau, q, mu, w, r, t, W, pi, Tg, a, rho, psi_loc)


def score(design, st):
    return design.scatter_vec(st.psi_loc)


def chat(st):
    """Chat_g = diag(wc) - pi wc' (G, T, T), so that -D psi_g = B_g' Chat_g B_g."""
    wc = (st.tau - 1.0) * st.t + (2.0 - st.tau) * st.a[:, None] * st.w
    C = -st.pi[:, :, None] * wc[:, None, :]
    idx = np.arange(C.shape[1])
    C[:, idx, idx] += wc
    return C


def neg_jacobian_local(design, st):
    """Local blocks of H = -DS: B'diag(wc)B - (B'pi)(B'wc)'."""
    wc = (st.tau - 1.0) * st.t + (2.0 - st.tau) * st.a[:, None] * st.w
    bw = design.bloc * wc[:, :, None]
    Hloc = np.matmul(design.bloc.transpose(0, 2, 1), bw)
    bbar = design.apply_t(st.pi)
    bwc = design.apply_t(wc)
    Hloc -= bbar[:, :, None] * bwc[:, None, :]
    return Hloc


def score_scale_local(design, st, factor=None):
    """Local |B_g|'(t + a_g w): the absolute size of the summands of each score
    coordinate.  max_k |S_k| / scale_k is the unit-free scaled residual."""
    mag = st.t + st.a[:, None] * st.w
    loc = np.einsum("gtk,gt->gk", np.abs(design.bloc), mag)
    if factor is not None:
        loc = loc * factor[:, None]
    return loc


@dataclass
class Evaluation:
    S: np.ndarray
    H: np.ndarray | None  # -DS (d x d), None when only the score was requested
    scale: np.ndarray
    state: GroupState
    factor: np.ndarray | None  # a_g^(1-tau) for the joint equation, else None


def cancelled_eval(design, y, theta, tau, jacobian=True):
    st = evaluate(design, y, theta, tau)
    S = design.scatter_vec(st.psi_loc)
    scale = design.scatter_vec(score_scale_local(design, st))
    H = design.scatter_mat(neg_jacobian_local(design, st)) if jacobian else None
    return Evaluation(S, H, scale, st, None)


def joint_factor(st):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return st.a ** (1.0 - st.tau)


def joint_eval(design, y, theta, tau, jacobian=True):
    """Profiled joint equation S_J = sum_g a_g^(1-tau) psi_g and -DS_J."""
    st = evaluate(design, y, theta, tau)
    f = joint_factor(st)
    S = design.scatter_vec(f[:, None] * st.psi_loc)
    scale = design.scatter_vec(score_scale_local(design, st, f))
    H = None
    if jacobian:
        Hloc = f[:, None, None] * neg_jacobian_local(design, st)
        bbar = design.apply_t(st.pi)
        Da = (1.0 - st.tau) * st.psi_loc / st.W[:, None] - st.a[:, None] * bbar
        coef = (1.0 - st.tau) * st.a ** (-st.tau)
        Hloc -= coef[:, None, None] * st.psi_loc[:, :, None] * Da[:, None, :]
        H = design.scatter_mat(Hloc)
    return Evaluation(S, H, scale, st, f)


# ---- objectives (only where the cancelled equation has a convex potential) --
def cancelled_objective(design, y, theta, tau):
    """Q with grad Q = -S, convex in theta, for tau in {1, 2} only.

    tau=2: Q = sum y exp(-q)  (B pair-centred, so bbar_g = 0 and psi_g = B_g' r_g).
    tau=1: Q = -sum_g [sum_t y q - (sum_t y) log sum_t exp(q)]  (conditional
           multinomial / PPML pair-profile log-likelihood, negated).
    """
    q = design.index(theta)
    if tau == 2:
        with np.errstate(over="raise", invalid="raise"):
            return float(np.sum(y * np.exp(-q)))
    if tau == 1:
        m = q.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(q - m).sum(axis=1))
        return float(-np.sum(y * q) + np.sum(y.sum(axis=1) * lse))
    raise ValueError("cancelled equation has no scalar potential for this tau")


# ---- adjoint kernels gamma_g, Gamma_g (App. B, eq. gamma-entries) ------------
def adjoint_kernels(design, st, adj):
    """gamma_g (G,T) and Gamma_g (G,T,T): gradient and Hessian w.r.t. q_g of
    F_g(q) = adj' psi_g(q) with the adjoint vector adj held fixed.

        s = B_g adj,  u~ = s - pi's,  kappa = 2 - tau,
        gamma_k = u~_k [(kappa-1) t_k - kappa T_g pi_k]
        Gamma_kl = delta_kl u~_k [(kappa-1)^2 t_k - kappa^2 T_g pi_k]
                   - kappa(kappa-1) (t_k pi_l u~_l + t_l pi_k u~_k)
                   + kappa^2 T_g pi_k pi_l (u~_k + u~_l).
    """
    kappa = 2.0 - st.tau
    s = design.index(adj)
    ut = s - np.sum(st.pi * s, axis=1, keepdims=True)
    Tg = st.Tg[:, None]
    gamma = ut * ((kappa - 1.0) * st.t - kappa * Tg * st.pi)
    pu = st.pi * ut
    Gam = -kappa * (kappa - 1.0) * (
        st.t[:, :, None] * pu[:, None, :] + pu[:, :, None] * st.t[:, None, :]
    )
    Gam += kappa**2 * Tg[:, :, None] * (
        pu[:, :, None] * st.pi[:, None, :] + st.pi[:, :, None] * pu[:, None, :]
    )
    idx = np.arange(st.t.shape[1])
    Gam[:, idx, idx] += ut * ((kappa - 1.0) ** 2 * st.t - kappa**2 * Tg * st.pi)
    return gamma, Gam


# ---- direct derivatives of rho (independent route used by tests/oracle) ----
def drho(st, z):
    """First directional derivative of rho_g(q) along z (G,T)."""
    kappa = 2.0 - st.tau
    dt = (kappa - 1.0) * st.t * z
    zbar = np.sum(st.pi * z, axis=1, keepdims=True)
    dpi = kappa * st.pi * (z - zbar)
    dT = dt.sum(axis=1, keepdims=True)
    return dt - dpi * st.Tg[:, None] - st.pi * dT


def d2rho(st, z):
    """Second directional derivative of rho_g(q) along z (G,T)."""
    kappa = 2.0 - st.tau
    dt = (kappa - 1.0) * st.t * z
    d2t = (kappa - 1.0) ** 2 * st.t * z * z
    zbar = np.sum(st.pi * z, axis=1, keepdims=True)
    zc = z - zbar
    var = np.sum(st.pi * zc * zc, axis=1, keepdims=True)
    dpi = kappa * st.pi * zc
    d2pi = kappa**2 * st.pi * (zc * zc - var)
    dT = dt.sum(axis=1, keepdims=True)
    d2T = d2t.sum(axis=1, keepdims=True)
    return d2t - d2pi * st.Tg[:, None] - 2.0 * dpi * dT - st.pi * d2T
