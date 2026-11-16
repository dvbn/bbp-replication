"""Experimental higher-order covariance for scalar-index group equations.

Each independent group contributes x_g f_g(x_g' theta). All derivatives
are of f_g with respect to its scalar index, evaluated at the same root.
Scores and derivatives are UNNORMALIZED sums. The target is l' theta.

This implements the weight-derivative formula in the checked research note,
not a package default and not a claim of coverage improvement. It returns
the raw adjusted variance, including negative values; it never clips it.
Dense group kernels restrict this pilot to modest group counts.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def covariance_refinement(X, f, j, k, u, target=None, *, max_groups=3000):
    """Return ordinary variance, empirical coefficient bias and adjustment.

Let a_g, b_gh, c_ghh be target first/second/third derivatives of the root
of sum_g w_g x_g f_g=0 with respect to UNNORMALIZED weights at w=1.
Then bias=.5 sum_g b_gg, V0=sum_g a_g^2 and
Delta=2 sum_g a_g b_gg + sum_gh a_g c_ghh + .5 sum_gh b_gh^2.
The candidate covariance of the empirically bias-corrected coefficient is
V2=V0-Delta. No deletion multiplier or further division by n is applied.
"""
    X = np.asarray(X, dtype=float)
    f, j, k, u = [np.asarray(v, dtype=float) for v in (f, j, k, u)]
    if X.ndim != 2 or not all(v.shape == (len(X),) for v in (f, j, k, u)):
        raise ValueError("Incompatible group design and derivative shapes")
    if len(X) > max_groups:
        raise ValueError("Dense prototype group limit exceeded; do not silently subsample")
    if not all(np.isfinite(v).all() for v in (X, f, j, k, u)):
        raise ValueError("Nonfinite design or derivative")
    if target is None:
        target = np.eye(X.shape[1])[0]
    target = np.asarray(target, dtype=float)
    if target.shape != (X.shape[1],) or not np.isfinite(target).all():
        raise ValueError("Invalid target vector")
    H = -(X.T * j) @ X
    factor = cho_factor(H, lower=True, check_finite=False)
    XR = cho_solve(factor, X.T, check_finite=False).T
    P = XR @ X.T
    z = X @ cho_solve(factor, target, check_finite=False)
    diagP = np.diag(P)
    a = z * f

    # P diag(w) P using the coefficient dimension, not a G-by-G product.
    def sandwich_kernel(w):
        return (XR @ ((X.T * w) @ X)) @ XR.T

    b = ((z * j)[:, None] * P) * f[None, :]
    b = b + b.T + (f[:, None] * sandwich_kernel(z * k)) * f[None, :]
    bdiag = np.diag(b)
    bias = 0.5 * float(np.sum(bdiag))
    d_linear_second = 2.0 * float(a @ bdiag)
    d_second_second = 0.5 * float(np.sum(b * b))

    # Sum_g a_g c_ghh is the third weight derivative in the frozen a
    # direction and twice in direction h. All sums over h include h=g.
    e = P @ (f * a)
    vdiag = (P * P) @ (f * f)
    r = P @ (2 * j * diagP * f + k * vdiag)
    M = sandwich_kernel(a * j + k * e) * f[None, :]
    M += P * (j * e)[None, :]
    mixed = np.sum((P * f[None, :]) * M, axis=1)
    third_terms = (
        a * k * vdiag
        + 2 * k * e * diagP * f
        + a * j * r
        + 2 * j * np.diag(M)
        + k * e * r
        + 2 * k * mixed
        + u * e * vdiag
    )
    d_linear_third = float(z @ third_terms)
    delta = d_linear_second + d_linear_third + d_second_second
    variance = float(a @ a)
    adjusted = variance - delta
    return {
        "variance_raw": variance,
        "variance_adjusted_raw": adjusted,
        "coefficient_bias": bias,
        "delta": delta,
        "delta_linear_second": d_linear_second,
        "delta_linear_third": d_linear_third,
        "delta_second_second": d_second_second,
        "target_score_error": float(z @ f),
        "max_score": float(np.max(np.abs(X.T @ f))),
        "positive_adjusted_variance": bool(np.isfinite(adjusted) and adjusted > 0),
        "groups": len(X),
        "dimension": X.shape[1],
    }
