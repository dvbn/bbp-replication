"""Original-coordinate distances between solver roots on identical simulated data."""

from __future__ import annotations
import math
import numpy as np
METRICS = ("coefficient_inf", "coefficient_rms", "beta_d_abs_difference", "theta_abs_difference", "fitted_index_rms")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def finite_float(x):
    x = float(x)
    return x if math.isfinite(x) else None

def stable_rms(values):
    """RMS without intermediate squaring overflow; None means unavailable."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        return None
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0 else finite_float(scale * np.sqrt(np.mean((values / scale) ** 2)))

def vector_status(beta, k):
    if beta is None:
        return 'absent'
    require(len(beta) == k, 'Coefficient length differs from frozen design')
    return 'stored_finite' if np.isfinite(beta).all() else 'stored_nonfinite'

def pair_metrics(X, beta_a, beta_b, treatment_column):
    """Original-coordinate differences, with distinct unavailable states.

    RMS index is ||X(beta_a-beta_b)||_2/sqrt(n), not the difference between
    rounded fitted predictions. Targets are separately expm1-transformed.
    """
    n, k = X.shape
    sa, sb = vector_status(beta_a, k), vector_status(beta_b, k)
    result = dict(vector_status_a=sa, vector_status_b=sb,
                  finite_pair=int(sa == sb == 'stored_finite'),
                  **{name: None for name in METRICS})
    detail = {'target_status_a': 'unavailable', 'target_status_b': 'unavailable',
              'arithmetic_status': 'unavailable_vector'}
    if not result['finite_pair']:
        return result, detail
    a, b = np.asarray(beta_a, dtype=float), np.asarray(beta_b, dtype=float)
    with np.errstate(all='ignore'):
        delta = a - b
        index_delta = X @ delta
        ta, tb = np.expm1(a[treatment_column]), np.expm1(b[treatment_column])
        result.update(coefficient_inf=finite_float(np.max(np.abs(delta))),
                      coefficient_rms=stable_rms(delta),
                      beta_d_abs_difference=finite_float(abs(delta[treatment_column])),
                      theta_abs_difference=(finite_float(abs(ta - tb))
                                            if np.isfinite(ta) and np.isfinite(tb) else None),
                      fitted_index_rms=stable_rms(index_delta))
    detail.update(target_status_a='finite' if np.isfinite(ta) else 'overflow',
                  target_status_b='finite' if np.isfinite(tb) else 'overflow',
                  arithmetic_status=('finite_metrics' if all(result[m] is not None for m in METRICS)
                                     else 'nonfinite_metric'),
                  beta_d_a=float(a[treatment_column]), beta_d_b=float(b[treatment_column]),
                  theta_a=finite_float(ta), theta_b=finite_float(tb))
    return result, detail
