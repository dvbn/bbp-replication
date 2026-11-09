"""Overflow-safe scientific moments retained from the historical reporting code."""

import math
from decimal import Decimal, localcontext

def quantile(values, p):
    """Inverse empirical CDF (type 1), not interpolated quantiles."""
    return values[max(0, math.ceil(p * len(values)) - 1)] if values else None

def finite_float(value):
    result = float(value)
    return result if math.isfinite(result) else None

def numeric_summary(values, truth=0.0):
    """Finite-only moments; no deletion of a large but finite observation.

    Exact integer sums/squares work on the stored binary floats, including
    mixtures of subnormal and near-maximum values. Decimal is used only for
    final division/square roots, never for cancellation-prone centering.
    If a mathematical summary
    itself exceeds float range its field is null and explicitly listed; this
    is not confused with input expm1 overflow or a missing observation.
    """
    if any(not math.isfinite(v) for v in values) or not math.isfinite(truth):
        raise ValueError("numeric_summary requires finite values and truth")
    result = dict(n=len(values), truth=truth, mean=None, bias=None, sd=None,
                  rmse=None, mcse_bias=None, minimum=None, median=None,
                  p95=None, maximum=None, arithmetic_overflow_fields=[])
    if not values:
        return result
    ordered = sorted(values)
    ratios = [float(v).as_integer_ratio() for v in values]
    tn, td = float(truth).as_integer_ratio()
    # Every binary-float denominator is a power of two. A common integer
    # scale therefore represents all inputs/truth without rounding.
    scale = max(td, max(den for _, den in ratios))
    units = [num * (scale // den) for num, den in ratios]
    m, t = len(units), tn * (scale // td)
    total, squares = sum(units), sum(v * v for v in units)
    bias_numerator = total - m * t
    variance_numerator = m * squares - total * total
    mse_numerator = squares - 2 * t * total + m * t * t
    if variance_numerator < 0 or m * mse_numerator != bias_numerator ** 2 + variance_numerator:
        raise ValueError("Exact Bias/SD/RMSE identity failed")
    with localcontext() as ctx:
        ctx.prec = 80
        def ratio(num, den):
            return Decimal(num) / Decimal(den)
        calculated = dict(mean=ratio(total, m * scale), bias=ratio(bias_numerator, m * scale),
            sd=ratio(variance_numerator, m * (m - 1) * scale ** 2).sqrt() if m > 1 else None,
            rmse=ratio(mse_numerator, m * scale ** 2).sqrt(),
            mcse_bias=ratio(variance_numerator, m * m * (m - 1) * scale ** 2).sqrt() if m > 1 else None)
        for field, val in calculated.items():
            if val is not None:
                result[field] = finite_float(val)
                if result[field] is None:
                    result["arithmetic_overflow_fields"].append(field)
    result.update(minimum=ordered[0], median=quantile(ordered, .5),
                  p95=quantile(ordered, .95), maximum=ordered[-1])
    return result

def scalar_targets(beta, treatment_column=1):
    """Target policy of the historical scalar reader, without storage validation."""
    if beta is None:
        return None, None, "missing_vector"
    if not all(math.isfinite(float(v)) for v in beta):
        return None, None, "nonfinite_vector"
    beta_d = float(beta[treatment_column])
    try:
        theta = math.expm1(beta_d)
    except OverflowError:
        theta = None
    if theta is not None and not math.isfinite(theta):
        theta = None
    return beta_d, theta, "overflow" if theta is None else "finite"

