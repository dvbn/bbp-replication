from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from statistics import NormalDist
from typing import Any

import numpy as np

from ..results import IOLSResults
from .contracts import SolverResponse

_MIN_TAU_GAP = 1.0e-10


def _normal_stats(
    names: Sequence[str],
    params: Mapping[str, float],
    bse: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    dist = NormalDist()
    tvalues: dict[str, float] = {}
    pvalues: dict[str, float] = {}
    for name in names:
        estimate = float(params[name])
        se_value = float(bse[name])
        if np.isfinite(se_value) and se_value > 0.0:
            tvalue = estimate / se_value
            pvalue = 2.0 * (1.0 - dist.cdf(abs(tvalue)))
        else:
            tvalue = float("nan")
            pvalue = float("nan")
        tvalues[str(name)] = float(tvalue)
        pvalues[str(name)] = float(pvalue)
    return tvalues, pvalues


def _selection_delta_payload(
    *,
    names: Sequence[str],
    selected_params: Mapping[str, float],
    pilot_params: Mapping[str, float],
    selected_covariance: Sequence[Sequence[float]] | None,
    tau_selected: float,
    tau_pilot: float,
    tau_selected_se: float | None,
) -> tuple[np.ndarray | None, dict[str, float] | None, dict[str, Any]]:
    details: dict[str, Any] = {
        "selection_inference_mode_requested": "propagate",
        "selection_inference_mode": "propagate",
        "coefficient_inference_conditioning": (
            "delta_propagated_tau_uncertainty_no_cross_term"
        ),
        "tau_uncertainty_method": "pilot_selected_secant_delta_no_cross_term",
        "tau_uncertainty_cross_term": "not_estimated",
        "tau_uncertainty_cross_term_reason": (
            "joint_beta_tau_influence_not_available_on_this_route"
        ),
        "path_uncertainty_propagated": False,
    }
    if tau_selected_se is None or not np.isfinite(float(tau_selected_se)):
        return None, None, {
            **details,
            "tau_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": "tau_selected_se_unavailable",
        }
    tau_gap = float(tau_selected) - float(tau_pilot)
    if not np.isfinite(tau_gap) or abs(tau_gap) <= _MIN_TAU_GAP:
        return None, None, {
            **details,
            "tau_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": "selected_tau_matches_pilot",
        }
    if selected_covariance is None:
        return None, None, {
            **details,
            "tau_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": (
                "conditional_covariance_unavailable"
            ),
        }
    names_tuple = tuple(str(name) for name in names)
    covariance = np.asarray(selected_covariance, dtype=float)
    if covariance.shape != (len(names_tuple), len(names_tuple)) or not np.all(
        np.isfinite(covariance)
    ):
        return None, None, {
            **details,
            "tau_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": (
                "conditional_covariance_nonfinite"
            ),
        }
    try:
        selected_beta = np.asarray(
            [float(selected_params[name]) for name in names_tuple],
            dtype=float,
        )
        pilot_beta = np.asarray(
            [float(pilot_params[name]) for name in names_tuple],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None, None, {
            **details,
            "tau_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": "selection_params_unavailable",
        }
    if not (np.all(np.isfinite(selected_beta)) and np.all(np.isfinite(pilot_beta))):
        return None, None, {
            **details,
            "tau_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": "selection_params_nonfinite",
        }
    derivative = (selected_beta - pilot_beta) / tau_gap
    adjusted = covariance + np.outer(derivative, derivative) * (
        float(tau_selected_se) ** 2
    )
    adjusted = 0.5 * (adjusted + adjusted.T)
    if not np.all(np.isfinite(adjusted)):
        return None, None, {
            **details,
            "tau_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": (
                "selection_adjusted_covariance_nonfinite"
            ),
        }
    diag = np.diag(adjusted)
    bse_values = np.full(len(names_tuple), np.nan, dtype=float)
    nonnegative = diag >= -1.0e-14
    bse_values[nonnegative] = np.sqrt(np.maximum(diag[nonnegative], 0.0))
    bse = {
        name: float(value)
        for name, value in zip(names_tuple, bse_values, strict=True)
    }
    return adjusted, bse, {
        **details,
        "tau_uncertainty_propagated": True,
        "path_uncertainty_propagated": False,
        "selection_adjusted_covariance_status": "computed_first_order_no_cross_term",
        "selection_adjusted_covariance_reason": None,
        "selection_adjusted_covariance_scope": (
            "selected_tau_uncertainty_no_cross_term"
        ),
        "selection_adjusted_covariance_formula": (
            "V_conditional + dbeta_dtau dbeta_dtau' Var(tau_selected)"
        ),
        "selection_adjusted_covariance_warning": (
            "first-order delta propagation omits the beta/tau influence "
            "cross-covariance; it is not a full joint selection sandwich"
        ),
        "selection_adjusted_covariance_tau_gap": float(tau_gap),
        "selection_adjusted_covariance_tau_se": float(tau_selected_se),
        "selection_adjusted_covariance_derivative": tuple(
            float(value) for value in derivative
        ),
    }


def apply_selection_delta_to_solver_response(
    response: SolverResponse,
    *,
    pilot_response: SolverResponse,
    tau_pilot: float,
    tau_selected: float,
    tau_selected_se: float | None,
) -> SolverResponse:
    names = tuple(response.params.keys())
    adjusted, bse, details = _selection_delta_payload(
        names=names,
        selected_params=response.params,
        pilot_params=pilot_response.params,
        selected_covariance=response.covariance,
        tau_selected=tau_selected,
        tau_pilot=tau_pilot,
        tau_selected_se=tau_selected_se,
    )
    if adjusted is None or bse is None:
        return replace(
            response,
            solver_details={**dict(response.solver_details), **details},
            tau_details={**dict(response.tau_details), **details},
            diagnostic_details={**dict(response.diagnostic_details), **details},
        )
    tvalues, pvalues = _normal_stats(names, response.params, bse)
    covariance = tuple(tuple(float(value) for value in row) for row in adjusted)
    return replace(
        response,
        bse=bse,
        tvalues=tvalues,
        pvalues=pvalues,
        covariance=covariance,
        solver_details={**dict(response.solver_details), **details},
        tau_details={**dict(response.tau_details), **details},
        diagnostic_details={**dict(response.diagnostic_details), **details},
    )


def apply_selection_delta_to_iols_results(
    result: IOLSResults,
    *,
    pilot: IOLSResults,
    tau_pilot: float,
    tau_selected: float,
    tau_selected_se: float | None,
) -> IOLSResults:
    names = tuple(result.params.keys())
    adjusted, bse, details = _selection_delta_payload(
        names=names,
        selected_params=result.params,
        pilot_params=pilot.params,
        selected_covariance=result.cov_params(),
        tau_selected=tau_selected,
        tau_pilot=tau_pilot,
        tau_selected_se=tau_selected_se,
    )
    if adjusted is None or bse is None:
        result.solver_info = {**dict(result.solver_info), **details}
        result.tau_info = {**dict(result.tau_info), **details}
        result.diagnostics = {**dict(result.diagnostics), **details}
        return result
    tvalues, pvalues = _normal_stats(names, result.params, bse)
    result.bse = bse
    result.tvalues = tvalues
    result.pvalues = pvalues
    result._covariance = tuple(tuple(float(value) for value in row) for row in adjusted)
    result.solver_info = {**dict(result.solver_info), **details}
    result.tau_info = {**dict(result.tau_info), **details}
    result.diagnostics = {**dict(result.diagnostics), **details}
    return result
