"""Pair sandwich and the tensor-free manuscript coefficient correction.

With H = -DS at the root (raw sums), target row L = e_1 (beta), adjoint
a = H^{-T} L' (so L H^{-1} x = a'x), v_g = H^{-1} psi_g and K = sum_g psi_g psi_g':

  b = L H^{-1} sum_g ( D psi_g v_g + 1/2 D^2 S[v_g, v_g] )            (eq. correction)
    = sum_g gamma_g' P_g rho_g + 1/2 sum_g < Gamma_g, B_g V B_g' >_F    (eq. correction-blocks)

with P_g = B_g H^{-1} B_g', V = H^{-1} K H^{-T}.  The second term contracts each
pair's Gamma_g with the FULL covariance V (curvature of the full equation along
every v_h), not only the pair's own v_g v_g'.  gamma_g, Gamma_g are the q-gradient
and q-Hessian of adj' psi_g with adj held fixed, so no d x d x d tensor or n x d
design is formed: only d x d inverses/covariances and (2T-1)^2 local slices.
Corrected coefficient: beta_bc = beta - b (both inner signs positive).
Target variance: sum_g (a' psi_g)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..design.operators import adjoint_kernels


@dataclass
class InferenceResult:
    se: float
    status: str
    resid_target_over_se: float
    adj: np.ndarray
    schur_beta: float
    schur_beta_over_xx: float
    target_resid: float = np.nan  # |a'S| at the root


def target_row(d):
    L = np.zeros(d)
    L[0] = 1.0
    return L


def pair_sandwich(design, ev, fac, inference_rel=1e-6, factor=None):
    """Target pair-clustered sandwich from the same root and equation.

    ``factor`` = a_g^(1-tau) for the profiled joint equation (own sandwich of
    S_J); None for the cancelled equation.  The SE is withheld (NaN) when the
    inference-accuracy check |a'S| <= inference_rel * SE fails.
    """
    d = design.d
    L = target_row(d)
    adj = fac.solve_t(L)
    contrib = ev.state.psi_loc if factor is None else factor[:, None] * ev.state.psi_loc
    infl = np.sum(design.gather_vec(adj) * contrib, axis=1)  # a' psi_g per pair
    var = float(np.sum(infl * infl))
    se = float(np.sqrt(var)) if var > 0 and np.isfinite(var) else np.nan
    hinv_e = fac.solve(L)
    schur = 1.0 / float(hinv_e[0])
    schur_xx = schur / float(ev.H[0, 0]) if ev.H[0, 0] != 0 else np.nan
    resid = abs(float(adj @ ev.S))
    if not np.isfinite(se):
        return InferenceResult(np.nan, "failed", np.nan, adj, schur, schur_xx, resid)
    ratio = resid / se
    status = "ok" if ratio <= inference_rel else "inference_check_failed"
    return InferenceResult(se if status == "ok" else np.nan, status, ratio, adj, schur, schur_xx, resid)


@dataclass
class CorrectionResult:
    bias: float
    term1: float
    term2: float
    Hinv: np.ndarray
    V: np.ndarray
    P_local: np.ndarray  # (G, T, T) B_g H^{-1} B_g'


def manuscript_correction(design, ev, fac, adj):
    """Tensor-free b with both terms stored separately."""
    st = ev.state
    Hinv = fac.inverse()
    psi = st.psi_loc
    K = design.scatter_mat(psi[:, :, None] * psi[:, None, :])
    V = Hinv @ K @ Hinv.T
    gamma, Gam = adjoint_kernels(design, st, adj)
    P = design.project_local(design.gather_mat(Hinv))
    term1 = float(np.einsum("gt,gts,gs->", gamma, P, st.rho))
    BVB = design.project_local(design.gather_mat(V))
    term2 = 0.5 * float(np.sum(Gam * BVB))
    bias = term1 + term2
    if not np.isfinite(bias):
        raise FloatingPointError("non-finite correction")
    return CorrectionResult(bias, term1, term2, Hinv, V, P)
