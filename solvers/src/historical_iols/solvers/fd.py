from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .._numeric import FLOAT64_LOG_MAX, exp_with_float64_overflow_guard
from ..diagnostics import diagnose_fe_support
from ..preprocess import (
    partitioned_research_contract,
    rank_reveal_qr,
    scope_only_contract,
    solve_ols_qr_dropcollinear,
)
from .contracts import SolverRequest, SolverResponse
from .fe import _demean_manyway_checked as _core_demean_manyway_checked
from .fe import demean_manyway as _core_demean_manyway
from .selection_covariance import apply_selection_delta_to_solver_response

_TAU_PATH_STEPS = 20
# iOLS covariance bread guard. IV Algorithm 10 uses a looser guard because its
# Jacobian-corrected bread has a different scale and is screened separately.
_BREAD_CONDITION_LIMIT = 1e12
_LEVERAGE_FLOOR = 1.0e-8
_INSTABILITY_BREAD_CONDITION_WARNING = 1e8
_INSTABILITY_MAX_ABS_LINEAR = 15.0
_INSTABILITY_LINEAR_RANGE = 20.0
_INSTABILITY_WEIGHT_FLOOR_FRACTION = 0.80
_SELECTION_BOUNDARY_TOL = 1e-9
_SELECTION_BOUNDARY_ABS_LINEAR_DELTA = 1.5
_SELECTION_BOUNDARY_LINEAR_RANGE_DELTA = 3.0
_TAU_UNCERTAINTY_NOT_IMPLEMENTED_REASON = "tau_uncertainty_propagation_not_implemented"
_TAU_UNCERTAINTY_AUTO_WARNING = (
    "tau_inference='auto' requested selection-adjusted covariance, but "
    "tau-selection uncertainty propagation is not implemented yet. Reporting "
    "coefficient inference conditional on tau_fit."
)
_FD_ONEWAY_FULLCANCEL_DEFAULT_PATH_TARGETS = (1.0, 2.0)
_FD_PARTITIONED_FULLCANCEL_DEFAULT_PATH_TARGETS = (1.0, 2.0)
_FD_PARTITIONED_MIXED_DEFAULT_PATH_TARGETS = (1.0, 2.0)
_VECTOR_GROUP_LOGSUMEXP_FLAG = "PYIOLS_SPEED_VECTOR_GROUP_LOGSUMEXP"
_PROFILE_CANCEL_CONSTANT_TERMS_FLAG = "PYIOLS_SPEED_PROFILE_CANCEL_CONSTANT_TERMS"
_COMPACT_PUBLIC_RESULTS_FLAG = "PYIOLS_SPEED_COMPACT_PUBLIC_RESULTS"
_TRUST_FULL_RANK_FLAG = "PYIOLS_SPEED_TRUST_FULL_RANK"


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compact_public_results_enabled() -> bool:
    return _env_flag_enabled(_COMPACT_PUBLIC_RESULTS_FLAG)


def _speed_trust_full_rank_enabled() -> bool:
    return _env_flag_enabled(_TRUST_FULL_RANK_FLAG)


def _compact_public_sequence(values: Any) -> tuple[float, ...] | np.ndarray:
    arr = np.asarray(values, dtype=float)
    if _compact_public_results_enabled():
        return arr
    return tuple(float(value) for value in arr)


def _compact_public_covariance(
    covariance: Any | None,
) -> tuple[tuple[float, ...], ...] | np.ndarray | None:
    if covariance is None:
        return None
    arr = np.asarray(covariance, dtype=float)
    if _compact_public_results_enabled():
        return arr
    return tuple(tuple(float(value) for value in row) for row in arr)


def _compact_instability_metrics() -> dict[str, Any]:
    return {
        "instability_linear_min": None,
        "instability_linear_max": None,
        "instability_linear_range": None,
        "instability_max_abs_linear": None,
        "instability_score_inf_norm": None,
        "instability_score_rms": None,
        "instability_bread_weight_zero_fraction": None,
        "instability_bread_weight_max_share": None,
        "instability_metrics_policy": "skipped_by_speed_compact_public_results",
    }


def _compact_instability_warning() -> dict[str, Any]:
    warning = _post_estimation_instability_warning(
        metrics=_compact_instability_metrics(),
    )
    warning["instability_status"] = "skipped_by_speed_compact_public_results"
    return warning
FD_PARTITIONED_THREE_PLUS_MIXED_NON_HC0_BLOCKED_MESSAGE = (
    "Public fd_partitioned/mixed with three-plus FE admits "
    "fixed/selected/path tau covariance for iid, HC0-HC3, and one-way "
    "cluster. Other covariance requests remain blocked before fit pending "
    "mixed covariance hardening."
)
FD_PARTITIONED_THREE_PLUS_FULLCANCEL_NON_HC0_BLOCKED_MESSAGE = (
    "Public fd_partitioned/fullcancel with three-plus cancelled FE admits "
    "fixed/selected tau covariance for iid, HC0-HC3, and one-way cluster. "
    "Other covariance requests remain blocked before fit pending many-way "
    "fullcancel covariance hardening."
)
FD_PARTITIONED_CLUSTER_BLOCKED_MESSAGE = (
    "Public fd_partitioned cluster covariance is admitted only for fullcancel "
    "package routes and mixed package routes with one clustering dimension. "
    "Other partitioned cluster requests remain blocked before fit."
)
FD_PARTITIONED_MIXED_PATH_BLOCKED_MESSAGE = (
    "Public fd_partitioned/mixed tau='path' is admitted only for iid and "
    "HC0-HC3 covariance on exactly two FE, and iid or HC0-HC3 covariance on "
    "three-plus FE, plus one-way cluster covariance on mixed package routes. "
    "Unsupported many-FE mixed covariance requests remain blocked before fit."
)
FD_PARTITIONED_THREE_PLUS_FULLCANCEL_PATH_BLOCKED_MESSAGE = (
    "Public fd_partitioned/fullcancel tau='path' is admitted for exactly two "
    "cancelled FE dimensions, and for three-plus-FE fullcancel iid, HC0-HC3, "
    "or one-way cluster path covariance only. Other partitioned path requests "
    "remain blocked before fit."
)


def _as_dense_group_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.size and int(np.max(array)) <= np.iinfo(np.int32).max:
        return array.astype(np.int32, copy=False)
    return array


def _require_fixed_tau(request: SolverRequest) -> float:
    if request.tau_target is None:
        raise RuntimeError("FD solver execution requires a fixed tau target.")
    return float(request.tau_target)


def _normal_stats(
    names: tuple[str, ...],
    beta: np.ndarray,
    se: np.ndarray,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    params = {name: float(value) for name, value in zip(names, beta, strict=True)}
    bse = {name: float(value) for name, value in zip(names, se, strict=True)}
    tvalues: dict[str, float] = {}
    pvalues: dict[str, float] = {}
    dist = NormalDist()
    for name in names:
        se_value = bse[name]
        if np.isfinite(se_value) and se_value > 0:
            tvalue = params[name] / se_value
            pvalue = 2.0 * (1.0 - dist.cdf(abs(tvalue)))
        else:
            tvalue = float("nan")
            pvalue = float("nan")
        tvalues[name] = float(tvalue)
        pvalues[name] = float(pvalue)
    return params, bse, tvalues, pvalues


def _prepend_routing_log(
    preprocess_log: tuple[dict[str, Any], ...],
    request: SolverRequest,
) -> tuple[dict[str, Any], ...]:
    routing_info = request.routing_diagnostics
    if not isinstance(routing_info, Mapping):
        return preprocess_log
    log_step = routing_info.get("log_step")
    if not isinstance(log_step, Mapping):
        return preprocess_log
    return (dict(log_step),) + preprocess_log


def _selection_variant_from_group(
    group_i: np.ndarray | None,
) -> tuple[str | None, int | None, int | None]:
    if group_i is None:
        return None, None, None
    mapped = _as_dense_group_array(
        np.unique(np.asarray(group_i), return_inverse=True)[1]
    )
    counts = np.bincount(mapped)
    if counts.size == 0:
        return None, None, 0
    median_t = int(np.median(counts))
    if median_t >= 3:
        return "moment_gmm", median_t, int(counts.size)
    if median_t == 2:
        return "moment_t2_approx", median_t, int(counts.size)
    return "unsupported_short_dimension", median_t, int(counts.size)


def _partition_warning_from_request(request: SolverRequest) -> str | None:
    routing_info = request.routing_diagnostics
    if routing_info is None or routing_info.get("status") != "warn":
        return None
    return str(routing_info.get("message"))


def _fd_oneway_inference_warning(
    *,
    request: SolverRequest,
    group_i: np.ndarray | None,
    se_type: str,
) -> dict[str, Any]:
    if group_i is None:
        return {}
    _, short_dimension, n_groups = _selection_variant_from_group(group_i)
    if short_dimension != 2:
        return {}
    se_key = se_type.lower()
    if se_key == "cluster":
        message = (
            "One-FE T=2 FD inference is provisional. Stage 5 baseline Monte "
            "Carlo shows cluster is the preferred public route, but nominal "
            "intervals can still undercover on some Gamma surfaces."
        )
    elif se_key == "hc3":
        message = (
            "One-FE T=2 FD inference is provisional. Stage 5 baseline Monte "
            "Carlo shows HC3 is the least-bad non-cluster option, but "
            "nominal intervals still materially undercover on some surfaces; "
            "prefer cluster when admissible."
        )
    else:
        message = (
            "One-FE T=2 FD inference is provisional. Stage 5 baseline Monte "
            "Carlo shows robust/HC1/HC2 intervals materially "
            "undercover; prefer cluster when admissible and HC3 is the "
            "least-bad non-cluster option."
        )
    return {
        "inference_warning": message,
        "warning_messages": (message,),
        "inference_warning_codes": ("fd_oneway_t2_inference_provisional",),
        "recommended_vcov": "cluster",
        "preferred_noncluster_vcov": "HC3",
        "short_dimension": short_dimension,
        "n_fe_groups": n_groups,
        "tau_route_kind": request.tau_mode,
    }


def _merge_warning_details(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    messages: list[str] = []
    codes: list[str] = []
    for payload in payloads:
        for key, value in payload.items():
            if key in {
                "inference_warning",
                "warning_messages",
                "inference_warning_codes",
            }:
                continue
            merged[key] = value
        single = payload.get("inference_warning")
        if single is not None:
            messages.append(str(single))
        multiple = payload.get("warning_messages")
        if multiple is not None:
            if isinstance(multiple, str):
                messages.append(multiple)
            else:
                messages.extend(str(value) for value in multiple)
        payload_codes = payload.get("inference_warning_codes")
        if payload_codes is not None:
            if isinstance(payload_codes, str):
                codes.append(payload_codes)
            else:
                codes.extend(str(value) for value in payload_codes)
    if messages:
        ordered_messages = tuple(dict.fromkeys(messages))
        merged["inference_warning"] = ordered_messages[0]
        merged["warning_messages"] = ordered_messages
    if codes:
        merged["inference_warning_codes"] = tuple(dict.fromkeys(codes))
    return merged


def _post_estimation_instability_metrics(
    *,
    endog: np.ndarray,
    linear: np.ndarray | tuple[float, ...],
    tau: float,
) -> dict[str, Any]:
    linear_arr = np.asarray(linear, dtype=float)
    finite_mask = np.isfinite(linear_arr)
    metrics: dict[str, Any] = {
        "instability_linear_min": None,
        "instability_linear_max": None,
        "instability_linear_range": None,
        "instability_max_abs_linear": None,
        "instability_score_inf_norm": None,
        "instability_score_rms": None,
        "instability_bread_weight_zero_fraction": None,
        "instability_bread_weight_max_share": None,
    }
    if not np.any(finite_mask):
        return metrics
    finite_linear = linear_arr[finite_mask]
    finite_endog = np.asarray(endog, dtype=float)[finite_mask]
    metrics.update(
        {
            "instability_linear_min": float(np.min(finite_linear)),
            "instability_linear_max": float(np.max(finite_linear)),
            "instability_linear_range": float(
                np.max(finite_linear) - np.min(finite_linear)
            ),
            "instability_max_abs_linear": float(np.max(np.abs(finite_linear))),
        }
    )
    stable_terms = _stable_manyway_weight_terms(
        endog=finite_endog,
        linear=finite_linear,
        tau=tau,
    )
    if stable_terms is None:
        return metrics
    _, score, bread_weight = stable_terms
    metrics["instability_score_inf_norm"] = float(np.max(np.abs(score)))
    metrics["instability_score_rms"] = float(np.sqrt(np.mean(score**2)))
    metrics["instability_bread_weight_zero_fraction"] = float(
        np.mean(bread_weight <= 0.0)
    )
    weight_sum = float(np.sum(bread_weight))
    if weight_sum > 0.0:
        metrics["instability_bread_weight_max_share"] = float(
            np.max(bread_weight) / weight_sum
        )
    return metrics


def _post_estimation_instability_warning(
    *,
    metrics: Mapping[str, Any],
    bread_condition: float | None = None,
) -> dict[str, Any]:
    threshold_details = {
        "instability_max_abs_linear_threshold": _INSTABILITY_MAX_ABS_LINEAR,
        "instability_linear_range_threshold": _INSTABILITY_LINEAR_RANGE,
        "instability_bread_weight_zero_fraction_threshold": (
            _INSTABILITY_WEIGHT_FLOOR_FRACTION
        ),
        "instability_bread_condition_warning_threshold": (
            _INSTABILITY_BREAD_CONDITION_WARNING
        ),
    }
    reasons: list[str] = []
    max_abs_linear = metrics.get("instability_max_abs_linear")
    if (
        max_abs_linear is not None
        and float(max_abs_linear) > _INSTABILITY_MAX_ABS_LINEAR
    ):
        reasons.append("linear_predictor_magnitude")
    linear_range = metrics.get("instability_linear_range")
    if linear_range is not None and float(linear_range) > _INSTABILITY_LINEAR_RANGE:
        reasons.append("linear_predictor_range")
    weight_floor_fraction = metrics.get("instability_bread_weight_zero_fraction")
    if (
        weight_floor_fraction is not None
        and float(weight_floor_fraction) >= _INSTABILITY_WEIGHT_FLOOR_FRACTION
    ):
        reasons.append("bread_weight_zero_concentration")
    if (
        bread_condition is not None
        and float(bread_condition) > _INSTABILITY_BREAD_CONDITION_WARNING
    ):
        reasons.append("ill_conditioned_bread")
    if not reasons:
        return {
            "instability_status": "ok",
            "instability_reasons": (),
            "instability_withheld_inference": False,
            **threshold_details,
        }
    message = (
        "Post-estimation instability screen fired "
        f"({', '.join(reasons)}); inference is withheld for this fit."
    )
    return {
        "instability_status": "withheld",
        "instability_reasons": tuple(reasons),
        "instability_withheld_inference": True,
        "inference_warning": message,
        "warning_messages": (message,),
        "inference_warning_codes": ("post_estimation_instability",),
        **threshold_details,
    }


_INFERENCE_WITHHELD_MESSAGES = {
    "kernel_not_converged": "the package kernel did not converge",
    "no_estimable_slope_parameters": (
        "there are no non-constant slope parameters for covariance estimation"
    ),
    "covariance_unavailable": (
        "the covariance calculation failed its finite or conditioning checks"
    ),
    "retained_index_shape_mismatch": "the retained-index vector does not match nobs",
    "nonfinite_retained_index": "the retained index contains nonfinite values",
    "fullcancel_weight_terms_unavailable": (
        "cancelled-FE covariance weights or ratios are nonfinite"
    ),
    "weighted_within_transform_nonfinite": (
        "the cancelled-FE weighted-within transform produced nonfinite values"
    ),
    "fullcancel_score_terms_nonfinite": (
        "the cancelled-FE score or bread terms are nonfinite"
    ),
    "ill_conditioned_bread": "the covariance bread is singular or ill-conditioned",
    "cluster_variable_missing": "cluster covariance was requested without clusters",
    "cluster_shape_mismatch": "the cluster vector length does not match nobs",
    "too_few_clusters": "cluster covariance needs at least two clusters",
    "nonfinite_covariance": "the covariance calculation produced nonfinite values",
    "post_estimation_instability": "the post-estimation instability screen fired",
}


def _withheld_inference_warning(
    *,
    route: str,
    reason: str,
) -> dict[str, Any]:
    detail = _INFERENCE_WITHHELD_MESSAGES.get(reason, reason.replace("_", " "))
    message = (
        f"{route} inference withheld: {detail}. Coefficients are reported, "
        "but standard errors, p-values, confidence intervals, and covariance "
        "are withheld for this fit."
    )
    return {
        "inference_warning": message,
        "warning_messages": (message,),
        "inference_warning_codes": (f"{route}_inference_withheld_{reason}",),
    }


def _fd_oneway_fullcancel_path_covariance_source(se_type: str) -> str:
    if se_type == "iid":
        return "package_fullcancel_iid_expected_bread"
    return f"package_fullcancel_expected_bread_{se_type}"


def _as_public_covariance(
    covariance: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in covariance)


def _path_record_compact_beta(item: Mapping[str, Any]) -> np.ndarray | None:
    beta_raw = item.get("beta")
    if beta_raw is None:
        beta_raw = item.get("beta_reduced")
    if beta_raw is None:
        return None
    return np.asarray(beta_raw, dtype=float)


def _expand_partitioned_path_coefficients(
    item: dict[str, Any],
    *,
    beta_compact: np.ndarray,
    kept_x_idx: np.ndarray,
    original_param_count: int,
    param_names: tuple[str, ...],
) -> bool:
    if beta_compact.shape[0] != int(kept_x_idx.size):
        return False
    beta_full = np.full(original_param_count, np.nan, dtype=float)
    beta_full[np.asarray(kept_x_idx, dtype=int)] = beta_compact
    item["beta"] = tuple(float(value) for value in beta_full)
    if len(param_names) == original_param_count:
        item["params"] = {
            name: float(value)
            for name, value in zip(param_names, beta_full, strict=True)
        }
    return True


def _strip_tau_path_private_state(
    path_details: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(path_details)
    path = tuple(updated.get("path", ()))
    if path:
        records: list[dict[str, Any]] = []
        for record in path:
            item = dict(record)
            item.pop("_linear_predictor", None)
            records.append(item)
        updated["path"] = tuple(records)
    return updated


def _mixed_compressed_path_diagnostics(
    path_details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in path_details.items()
        if str(key).startswith("mixed_compressed_fullcancel_")
    }


def _strip_public_path_private_state(
    path_details: dict[str, Any],
) -> dict[str, Any]:
    return _strip_tau_path_private_state(path_details)


def _fullcancel_path_covariance_not_computed_reason(
    reason: str | None,
) -> str:
    if reason is None:
        return "path_covariance_not_computed_public_fd_oneway_fullcancel"
    return f"path_covariance_{reason}_public_fd_oneway_fullcancel"


def _attach_public_fd_oneway_fullcancel_path_covariance(
    *,
    path_details: dict[str, Any],
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    param_names: tuple[str, ...],
    se_type: str,
    cluster_var: Any | None,
) -> dict[str, Any]:
    path = tuple(path_details.get("path", ()))
    if not path:
        return _strip_tau_path_private_state(path_details)

    slope_idx = np.flatnonzero(~_constant_column_mask(exog))
    updated = dict(path_details)
    records: list[dict[str, Any]] = []
    coefficient_records = 0
    computed_records = 0
    failure_reasons: list[str] = []
    if slope_idx.size == 0:
        reason = (
            "path_covariance_no_estimable_slope_parameters_public_fd_oneway_fullcancel"
        )
        for record in path:
            item = dict(record)
            item.pop("_linear_predictor", None)
            if bool(item.get("converged", False)) and item.get("beta") is not None:
                item["path_inference_status"] = "not_computed"
                item["path_inference_reason"] = reason
            records.append(item)
        updated["path"] = tuple(records)
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = reason
        return updated

    for record in path:
        item = dict(record)
        linear_raw = item.pop("_linear_predictor", None)
        if not bool(item.get("converged", False)):
            records.append(item)
            continue
        beta_raw = item.get("beta")
        if beta_raw is None:
            beta_raw = item.get("beta_reduced")
        if beta_raw is None:
            records.append(item)
            continue
        coefficient_records += 1
        beta_path = np.asarray(beta_raw, dtype=float)
        if beta_path.shape[0] != exog.shape[1]:
            reason = "path_covariance_beta_shape_mismatch_public_fd_oneway_fullcancel"
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue
        linear_path = np.asarray(linear_raw, dtype=float)
        if (
            linear_path.ndim != 1
            or linear_path.shape[0] != endog.shape[0]
            or not np.all(np.isfinite(linear_path))
        ):
            linear_path = exog @ beta_path
        if linear_path.shape[0] != endog.shape[0] or not np.all(
            np.isfinite(linear_path)
        ):
            reason = (
                "path_covariance_retained_index_unavailable_public_fd_oneway_fullcancel"
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue
        se_slope, covariance_slope, covariance_info = _fullcancel_iols_covariance(
            endog=endog,
            exog=exog[:, slope_idx],
            beta=beta_path[slope_idx],
            groups=groups,
            tau=float(item["tau"]),
            se_type=se_type,
            cluster_var=cluster_var,
            linear=linear_path,
        )
        if covariance_info.get("inference_status") == "withheld":
            reason = _fullcancel_path_covariance_not_computed_reason(
                None
                if covariance_info.get("inference_reason") is None
                else str(covariance_info.get("inference_reason"))
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue
        if not (
            np.all(np.isfinite(se_slope)) and np.all(np.isfinite(covariance_slope))
        ):
            reason = "path_covariance_nonfinite_public_fd_oneway_fullcancel"
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue

        se_full = np.full(exog.shape[1], np.nan, dtype=float)
        covariance_full = np.full((exog.shape[1], exog.shape[1]), np.nan, dtype=float)
        se_full[slope_idx] = np.asarray(se_slope, dtype=float)
        covariance_full[np.ix_(slope_idx, slope_idx)] = np.asarray(
            covariance_slope,
            dtype=float,
        )
        item["std_errors"] = tuple(float(value) for value in se_full)
        item["bse"] = {
            name: float(value) for name, value in zip(param_names, se_full, strict=True)
        }
        item["covariance"] = _as_public_covariance(covariance_full)
        item["path_inference_status"] = (
            "computed"
            if np.all(np.isfinite(se_full))
            else "computed_with_nonfinite_components"
        )
        item["path_inference_reason"] = None
        item["path_covariance_source"] = _fd_oneway_fullcancel_path_covariance_source(
            se_type
        )
        item["path_covariance_scope"] = f"one_fe_fullcancel_{se_type}_retained_index"
        item["path_covariance_fixed_effect_policy"] = "cancelled_fe_not_recovered"
        computed_records += 1
        records.append(item)

    updated["path"] = tuple(records)
    updated["path_covariance_scope"] = f"one_fe_fullcancel_{se_type}_retained_index"
    updated["path_covariance_fixed_effect_policy"] = "cancelled_fe_not_recovered"
    if not coefficient_records:
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = (
            "path_covariance_no_converged_path_coefficients_public_fd_oneway_fullcancel"
        )
    elif computed_records == coefficient_records:
        updated["path_inference_status"] = "computed"
        updated["path_inference_reason"] = None
    elif computed_records:
        updated["path_inference_status"] = "partial"
        updated["path_inference_reason"] = (
            "path_covariance_partially_computed_public_fd_oneway_fullcancel"
        )
    else:
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = (
            failure_reasons[0]
            if failure_reasons
            else "path_covariance_not_computed_public_fd_oneway_fullcancel"
        )
    return updated


def _partitioned_fullcancel_path_covariance_source(se_type: str) -> str:
    if se_type == "iid":
        return "package_partitioned_fullcancel_iid_expected_bread"
    return f"package_partitioned_fullcancel_expected_bread_{se_type}"


def _partitioned_fullcancel_path_covariance_not_computed_reason(
    reason: str | None,
) -> str:
    if reason is None:
        return "path_covariance_not_computed_public_fd_partitioned_fullcancel"
    return f"path_covariance_{reason}_public_fd_partitioned_fullcancel"


def _partitioned_mixed_path_covariance_source(se_type: str) -> str:
    return f"package_partitioned_mixed_{se_type}"


def _partitioned_mixed_path_covariance_not_computed_reason(
    reason: str | None,
) -> str:
    if reason is None:
        return "path_covariance_not_computed_public_fd_partitioned_mixed"
    return f"path_covariance_{reason}_public_fd_partitioned_mixed"


def _attach_public_fd_partitioned_fullcancel_path_covariance(
    *,
    path_details: dict[str, Any],
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    groups: tuple[np.ndarray, ...],
    cluster_var: Any | None,
    kept_x_idx: np.ndarray,
    compact_slope_idx: np.ndarray,
    original_slope_idx: np.ndarray,
    original_param_count: int,
    param_names: tuple[str, ...],
    se_type: str,
) -> dict[str, Any]:
    path = tuple(path_details.get("path", ()))
    if not path:
        return _strip_tau_path_private_state(path_details)

    updated = dict(path_details)
    records: list[dict[str, Any]] = []
    coefficient_records = 0
    computed_records = 0
    failure_reasons: list[str] = []
    scope_prefix = "two_fe" if len(groups) == 2 else "three_plus_fe"
    scope = f"{scope_prefix}_fullcancel_{se_type}_retained_index"

    if compact_slope_idx.size == 0:
        reason = (
            "path_covariance_no_estimable_slope_parameters_public_fd_partitioned_"
            "fullcancel"
        )
        for record in path:
            item = dict(record)
            item.pop("_linear_predictor", None)
            beta_path = _path_record_compact_beta(item)
            if bool(item.get("converged", False)) and beta_path is not None:
                _expand_partitioned_path_coefficients(
                    item,
                    beta_compact=beta_path,
                    kept_x_idx=kept_x_idx,
                    original_param_count=original_param_count,
                    param_names=param_names,
                )
                item["path_inference_status"] = "not_computed"
                item["path_inference_reason"] = reason
            records.append(item)
        updated["path"] = tuple(records)
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = reason
        updated["path_covariance_scope"] = scope
        updated["path_covariance_fixed_effect_policy"] = "cancelled_fe_not_recovered"
        return updated

    for record in path:
        item = dict(record)
        linear_raw = item.pop("_linear_predictor", None)
        if not bool(item.get("converged", False)):
            records.append(item)
            continue
        beta_path = _path_record_compact_beta(item)
        if beta_path is None:
            records.append(item)
            continue
        coefficient_records += 1
        if beta_path.shape[0] != exog_pre.shape[1]:
            reason = (
                "path_covariance_beta_shape_mismatch_public_fd_partitioned_fullcancel"
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue
        _expand_partitioned_path_coefficients(
            item,
            beta_compact=beta_path,
            kept_x_idx=kept_x_idx,
            original_param_count=original_param_count,
            param_names=param_names,
        )

        linear_path = np.asarray(linear_raw, dtype=float)
        if (
            linear_path.ndim != 1
            or linear_path.shape[0] != endog_pre.shape[0]
            or not np.all(np.isfinite(linear_path))
        ):
            linear_path = exog_pre @ beta_path
        if linear_path.shape[0] != endog_pre.shape[0] or not np.all(
            np.isfinite(linear_path)
        ):
            reason = (
                "path_covariance_retained_index_unavailable_public_fd_partitioned_"
                "fullcancel"
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue

        se_slope, covariance_slope, covariance_info = _fullcancel_iols_covariance(
            endog=endog_pre,
            exog=exog_pre[:, compact_slope_idx],
            beta=beta_path[compact_slope_idx],
            groups=groups,
            tau=float(item["tau"]),
            se_type=se_type,
            cluster_var=cluster_var,
            linear=linear_path,
        )
        if covariance_info.get("inference_status") == "withheld":
            reason = _partitioned_fullcancel_path_covariance_not_computed_reason(
                None
                if covariance_info.get("inference_reason") is None
                else str(covariance_info.get("inference_reason"))
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue
        if not (
            np.all(np.isfinite(se_slope)) and np.all(np.isfinite(covariance_slope))
        ):
            reason = "path_covariance_nonfinite_public_fd_partitioned_fullcancel"
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue

        se_full = np.full(original_param_count, np.nan, dtype=float)
        covariance_full = np.full(
            (original_param_count, original_param_count),
            np.nan,
            dtype=float,
        )
        for compact_i, original_i in enumerate(original_slope_idx):
            se_full[int(original_i)] = se_slope[compact_i]
            for compact_j, original_j in enumerate(original_slope_idx):
                covariance_full[int(original_i), int(original_j)] = covariance_slope[
                    compact_i,
                    compact_j,
                ]

        item["std_errors"] = tuple(float(value) for value in se_full)
        item["bse"] = {
            name: float(value) for name, value in zip(param_names, se_full, strict=True)
        }
        item["covariance"] = _as_public_covariance(covariance_full)
        item["path_inference_status"] = (
            "computed"
            if np.all(np.isfinite(se_full))
            else "computed_with_nonfinite_components"
        )
        item["path_inference_reason"] = None
        item["path_covariance_source"] = _partitioned_fullcancel_path_covariance_source(
            se_type
        )
        item["path_covariance_scope"] = scope
        item["path_covariance_fixed_effect_policy"] = "cancelled_fe_not_recovered"
        computed_records += 1
        records.append(item)

    updated["path"] = tuple(records)
    updated["path_covariance_scope"] = scope
    updated["path_covariance_fixed_effect_policy"] = "cancelled_fe_not_recovered"
    if not coefficient_records:
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = (
            "path_covariance_no_converged_path_coefficients_public_"
            "fd_partitioned_fullcancel"
        )
    elif computed_records == coefficient_records:
        updated["path_inference_status"] = "computed"
        updated["path_inference_reason"] = None
    elif computed_records:
        updated["path_inference_status"] = "partial"
        updated["path_inference_reason"] = (
            "path_covariance_partially_computed_public_fd_partitioned_fullcancel"
        )
    else:
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = (
            failure_reasons[0]
            if failure_reasons
            else "path_covariance_not_computed_public_fd_partitioned_fullcancel"
        )
    return updated


def _attach_public_fd_partitioned_mixed_path_covariance(
    *,
    path_details: dict[str, Any],
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_blocks_pre: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    kept_x_idx: np.ndarray,
    intercept_idx: int | None,
    original_param_count: int,
    param_names: tuple[str, ...],
    se_type: str,
    cluster_var: Any | None,
) -> dict[str, Any]:
    path = tuple(path_details.get("path", ()))
    if not path:
        return _strip_public_path_private_state(path_details)

    updated = dict(path_details)
    records: list[dict[str, Any]] = []
    coefficient_records = 0
    computed_records = 0
    failure_reasons: list[str] = []
    scope_prefix = (
        "two_fe" if len(cancel_fe) + len(recover_fe) == 2 else "three_plus_fe"
    )
    scope = f"{scope_prefix}_mixed_{se_type}_retained_index_plus_recovered_fe"
    fixed_effect_policy = (
        "recovered_fe_included_cancelled_fe_profiled_for_covariance_only"
    )
    intercept_position = _intercept_position(kept_x_idx, intercept_idx)
    _, original_slope_idx = _slope_positions(kept_x_idx, intercept_position)

    if original_slope_idx.size == 0:
        reason = (
            "path_covariance_no_estimable_slope_parameters_public_fd_partitioned_mixed"
        )
        for record in path:
            item = dict(record)
            item.pop("_linear_predictor", None)
            beta_path = _path_record_compact_beta(item)
            if bool(item.get("converged", False)) and beta_path is not None:
                _expand_partitioned_path_coefficients(
                    item,
                    beta_compact=beta_path,
                    kept_x_idx=kept_x_idx,
                    original_param_count=original_param_count,
                    param_names=param_names,
                )
                item["path_inference_status"] = "not_computed"
                item["path_inference_reason"] = reason
            records.append(item)
        updated["path"] = tuple(records)
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = reason
        updated["path_covariance_scope"] = scope
        updated["path_covariance_fixed_effect_policy"] = fixed_effect_policy
        return updated

    cancel_groups = tuple(
        np.asarray(fe_blocks_pre[name], dtype=int) for name in cancel_fe
    )
    for record in path:
        item = dict(record)
        base_linear_raw = item.pop("_linear_predictor", None)
        has_coefficients = bool(item.get("converged", False)) and (
            item.get("path_coefficient_status") == "available"
            or item.get("beta") is not None
            or item.get("beta_reduced") is not None
        )
        if not has_coefficients:
            records.append(item)
            continue
        coefficient_records += 1
        beta_path = _path_record_compact_beta(item)
        if beta_path is not None:
            _expand_partitioned_path_coefficients(
                item,
                beta_compact=beta_path,
                kept_x_idx=kept_x_idx,
                original_param_count=original_param_count,
                param_names=param_names,
            )

        if base_linear_raw is None:
            reason = (
                "path_covariance_retained_index_unavailable_public_fd_partitioned_mixed"
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue

        base_linear = np.asarray(base_linear_raw, dtype=float)
        if base_linear.shape[0] != endog_pre.shape[0] or not np.all(
            np.isfinite(base_linear)
        ):
            reason = (
                "path_covariance_retained_index_unavailable_public_fd_partitioned_mixed"
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue

        _, covariance_linear = _profile_cancel_fe_from_base(
            endog=endog_pre,
            base_linear=base_linear,
            groups=cancel_groups,
            tau=float(item["tau"]),
        )
        se_full, covariance_full, mixed_info = _partitioned_mixed_manyway_covariance(
            endog_pre=endog_pre,
            exog_pre=exog_pre,
            fe_blocks_pre=fe_blocks_pre,
            cancel_fe=cancel_fe,
            recover_fe=recover_fe,
            kept_x_idx=kept_x_idx,
            intercept_idx=intercept_idx,
            original_param_count=original_param_count,
            tau=float(item["tau"]),
            profiled_linear_kept=covariance_linear,
            se_type=se_type,
            cluster_var=cluster_var,
        )
        slope_se = se_full[original_slope_idx]
        if (
            mixed_info.get("status") != "supported"
            or not np.all(np.isfinite(slope_se) & (slope_se > 0.0))
            or not np.all(
                np.isfinite(
                    covariance_full[np.ix_(original_slope_idx, original_slope_idx)]
                )
            )
        ):
            reason = _partitioned_mixed_path_covariance_not_computed_reason(
                None
                if mixed_info.get("reason") is None
                else str(mixed_info.get("reason"))
            )
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = reason
            failure_reasons.append(reason)
            records.append(item)
            continue

        item["std_errors"] = tuple(float(value) for value in se_full)
        item["bse"] = {
            name: float(value) for name, value in zip(param_names, se_full, strict=True)
        }
        item["covariance"] = _as_public_covariance(covariance_full)
        item["path_inference_status"] = (
            "computed"
            if np.all(np.isfinite(se_full))
            else "computed_with_nonfinite_components"
        )
        item["path_inference_reason"] = None
        item["path_covariance_source"] = (
            "package_partitioned_mixed_manyway_cluster"
            if se_type == "cluster" and scope_prefix == "three_plus_fe"
            else _partitioned_mixed_path_covariance_source(se_type)
        )
        item["path_covariance_scope"] = scope
        item["path_covariance_fixed_effect_policy"] = fixed_effect_policy
        computed_records += 1
        records.append(item)

    updated["path"] = tuple(records)
    updated["path_covariance_scope"] = scope
    updated["path_covariance_fixed_effect_policy"] = fixed_effect_policy
    if not coefficient_records:
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = (
            "path_covariance_no_converged_path_coefficients_public_fd_partitioned_mixed"
        )
    elif computed_records == coefficient_records:
        updated["path_inference_status"] = "computed"
        updated["path_inference_reason"] = None
    elif computed_records:
        updated["path_inference_status"] = "partial"
        updated["path_inference_reason"] = (
            "path_covariance_partially_computed_public_fd_partitioned_mixed"
        )
    else:
        updated["path_inference_status"] = "not_computed"
        updated["path_inference_reason"] = (
            failure_reasons[0]
            if failure_reasons
            else "path_covariance_not_computed_public_fd_partitioned_mixed"
        )
    return updated


def _warning_messages_from_details(details: Mapping[str, Any]) -> tuple[str, ...]:
    messages: list[str] = []
    single = details.get("inference_warning")
    if single is not None:
        messages.append(str(single))
    multiple = details.get("warning_messages")
    if multiple is not None:
        if isinstance(multiple, str):
            messages.append(multiple)
        else:
            messages.extend(str(value) for value in multiple)
    return tuple(dict.fromkeys(messages))


def _append_warning_message(
    details: Mapping[str, Any],
    *,
    message: str,
    code: str,
) -> dict[str, Any]:
    merged = dict(details)
    merged["warning_messages"] = tuple(
        dict.fromkeys(_warning_messages_from_details(details) + (message,))
    )
    existing_codes = details.get("selection_warning_codes")
    if existing_codes is None:
        codes: tuple[str, ...] = ()
    elif isinstance(existing_codes, str):
        codes = (existing_codes,)
    else:
        codes = tuple(str(value) for value in existing_codes)
    merged["selection_warning_codes"] = tuple(dict.fromkeys(codes + (code,)))
    return merged


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _selection_boundary_hit(
    tau_selected: float,
    tau_range: tuple[float, float],
) -> bool:
    return (
        abs(float(tau_selected) - float(tau_range[0])) <= _SELECTION_BOUNDARY_TOL
        or abs(float(tau_selected) - float(tau_range[1])) <= _SELECTION_BOUNDARY_TOL
    )


def _selection_sanity_metadata(
    *,
    request: SolverRequest,
    pilot_response: SolverResponse,
    selected_response: SolverResponse | None,
    tau_selected: float,
) -> dict[str, Any]:
    if selected_response is None:
        return {
            "selection_boundary_selected": _selection_boundary_hit(
                tau_selected, request.tau_range
            ),
            "selection_selected_withheld_inference": False,
            "selection_selected_max_abs_linear_delta_vs_pilot": None,
            "selection_selected_linear_range_delta_vs_pilot": None,
            "selection_selected_score_rms_ratio_to_pilot": None,
        }
    pilot_diag = pilot_response.diagnostic_details
    selected_diag = selected_response.diagnostic_details
    pilot_score_rms = _finite_float(pilot_diag.get("instability_score_rms"))
    selected_score_rms = _finite_float(selected_diag.get("instability_score_rms"))
    if (
        pilot_score_rms is not None
        and pilot_score_rms > 0.0
        and selected_score_rms is not None
    ):
        score_ratio = selected_score_rms / pilot_score_rms
    else:
        score_ratio = None
    selected_max_abs_linear = _finite_float(
        selected_diag.get("instability_max_abs_linear")
    )
    pilot_max_abs_linear = _finite_float(pilot_diag.get("instability_max_abs_linear"))
    max_abs_linear_delta = (
        None
        if selected_max_abs_linear is None or pilot_max_abs_linear is None
        else selected_max_abs_linear - pilot_max_abs_linear
    )
    selected_linear_range = _finite_float(selected_diag.get("instability_linear_range"))
    pilot_linear_range = _finite_float(pilot_diag.get("instability_linear_range"))
    linear_range_delta = (
        None
        if selected_linear_range is None or pilot_linear_range is None
        else selected_linear_range - pilot_linear_range
    )
    return {
        "selection_boundary_selected": _selection_boundary_hit(
            tau_selected, request.tau_range
        ),
        "selection_selected_withheld_inference": bool(
            selected_diag.get("instability_withheld_inference")
        )
        or str(selected_diag.get("instability_status")) == "withheld",
        "selection_selected_max_abs_linear_delta_vs_pilot": max_abs_linear_delta,
        "selection_selected_linear_range_delta_vs_pilot": linear_range_delta,
        "selection_selected_score_rms_ratio_to_pilot": score_ratio,
    }


def _selection_fallback_reason(
    metadata: Mapping[str, Any],
) -> str | None:
    if bool(metadata.get("selection_selected_withheld_inference")):
        return "selected_instability_fallback_to_pilot"
    if not bool(metadata.get("selection_boundary_selected")):
        return None
    max_abs_linear_delta = _finite_float(
        metadata.get("selection_selected_max_abs_linear_delta_vs_pilot")
    )
    linear_range_delta = _finite_float(
        metadata.get("selection_selected_linear_range_delta_vs_pilot")
    )
    if (
        max_abs_linear_delta is not None
        and max_abs_linear_delta > _SELECTION_BOUNDARY_ABS_LINEAR_DELTA
    ) or (
        linear_range_delta is not None
        and linear_range_delta > _SELECTION_BOUNDARY_LINEAR_RANGE_DELTA
    ):
        return "boundary_extreme_refit_fallback_to_pilot"
    return None


def _selection_fallback_message(
    *,
    reason: str,
    tau_selected: float,
    tau_pilot: float,
) -> str:
    if reason == "selected_instability_fallback_to_pilot":
        return (
            "Selected tau refit at "
            f"tau={tau_selected:.6g} triggered the package instability screen; "
            f"returning the tau={tau_pilot:.6g} pilot fit instead."
        )
    return (
        "Selected tau refit landed on the tau-range boundary at "
        f"tau={tau_selected:.6g} and made the linear predictor materially more "
        f"extreme than the tau={tau_pilot:.6g} pilot; returning the pilot fit "
        "instead."
    )


def _selection_inference_contract(request: SolverRequest) -> dict[str, object]:
    requested = str(request.tau_inference)
    if requested == "propagate":
        return {
            "selection_inference_mode_requested": "propagate",
            "selection_inference_mode": "propagate",
            "coefficient_inference_conditioning": (
                "delta_propagated_tau_uncertainty_no_cross_term"
            ),
            "tau_uncertainty_propagated": False,
            "tau_uncertainty_method": "pilot_selected_secant_delta_no_cross_term",
            "tau_uncertainty_cross_term": "not_estimated",
            "path_uncertainty_propagated": False,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": (
                "selection_adjusted_covariance_requires_selected_refit"
            ),
        }
    if requested == "auto":
        return {
            "selection_inference_mode_requested": "auto",
            "selection_inference_mode": "conditional",
            "coefficient_inference_conditioning": "conditional_on_tau_fit",
            "tau_uncertainty_propagated": False,
            "tau_uncertainty_method": None,
            "selection_adjusted_covariance_status": "conditional_fallback",
            "selection_adjusted_covariance_reason": (
                _TAU_UNCERTAINTY_NOT_IMPLEMENTED_REASON
            ),
        }
    return {
        "selection_inference_mode_requested": "conditional",
        "selection_inference_mode": "conditional",
        "coefficient_inference_conditioning": "conditional_on_tau_fit",
        "tau_uncertainty_propagated": False,
        "tau_uncertainty_method": None,
        "selection_adjusted_covariance_status": "not_requested",
        "selection_adjusted_covariance_reason": None,
    }


def _annotate_selection_response(
    *,
    request: SolverRequest,
    response: SolverResponse,
    selection_method: str,
    selection_status: str,
    tau_pilot: float,
    tau_selected: float,
    tau_selected_se: float | None,
    sanity_status: str,
    sanity_reason: str | None,
    sanity_metadata: Mapping[str, Any],
    warning_message: str | None = None,
    short_dim_median_t: int | None = None,
    n_groups_used: int | None = None,
    selection_metadata: Mapping[str, Any] | None = None,
) -> SolverResponse:
    selector_metadata = {} if selection_metadata is None else dict(selection_metadata)
    inference_metadata = _selection_inference_contract(request)
    solver_details = dict(response.solver_details)
    solver_details.update(
        {
            "selection_method": selection_method,
            "selection_status": selection_status,
            "selection_se_tau": tau_selected_se,
            "selection_sanity_status": sanity_status,
            "selection_sanity_reason": sanity_reason,
            "selection_sanity_rule": (
                "fallback_on_selected_instability_or_boundary_extreme_refit"
            ),
            "selection_boundary_abs_linear_delta_threshold": (
                _SELECTION_BOUNDARY_ABS_LINEAR_DELTA
            ),
            "selection_boundary_linear_range_delta_threshold": (
                _SELECTION_BOUNDARY_LINEAR_RANGE_DELTA
            ),
        }
    )
    solver_details.update(selector_metadata)
    solver_details.update(inference_metadata)
    tau_details = dict(response.tau_details)
    tau_details.update(
        {
            "tau_fit": float(response.tau),
            "tau_pilot": tau_pilot,
            "tau_selected": tau_selected,
            "tau_selected_se": tau_selected_se,
            "selection_method": selection_method,
            "selection_status": selection_status,
            "selection_sanity_status": sanity_status,
            "selection_sanity_reason": sanity_reason,
        }
    )
    if short_dim_median_t is not None:
        tau_details["short_dim_median_t"] = int(short_dim_median_t)
    if n_groups_used is not None:
        tau_details["n_groups_used"] = int(n_groups_used)
    tau_details.update(selector_metadata)
    tau_details.update(inference_metadata)
    diagnostic_details = dict(response.diagnostic_details)
    diagnostic_details.update(sanity_metadata)
    diagnostic_details.update(
        {
            "selection_method": selection_method,
            "selection_status": selection_status,
            "selection_sanity_status": sanity_status,
            "selection_sanity_reason": sanity_reason,
            "selection_sanity_rule": (
                "fallback_on_selected_instability_or_boundary_extreme_refit"
            ),
            "selection_boundary_abs_linear_delta_threshold": (
                _SELECTION_BOUNDARY_ABS_LINEAR_DELTA
            ),
            "selection_boundary_linear_range_delta_threshold": (
                _SELECTION_BOUNDARY_LINEAR_RANGE_DELTA
            ),
        }
    )
    diagnostic_details.update(selector_metadata)
    diagnostic_details.update(inference_metadata)
    if warning_message is not None:
        diagnostic_details = _append_warning_message(
            diagnostic_details,
            message=warning_message,
            code="selected_tau_fallback_to_pilot",
        )
    if request.tau_inference == "auto":
        diagnostic_details = _append_warning_message(
            diagnostic_details,
            message=_TAU_UNCERTAINTY_AUTO_WARNING,
            code="tau_uncertainty_auto_conditional_fallback",
        )
    return replace(
        response,
        solver_details=solver_details,
        tau_details=tau_details,
        diagnostic_details=diagnostic_details,
    )


def _apply_selected_tau_sanity(
    *,
    request: SolverRequest,
    pilot_response: SolverResponse,
    selected_response: SolverResponse,
    selection_method: str,
    tau_pilot: float,
    tau_selected: float,
    tau_selected_se: float | None,
    short_dim_median_t: int | None = None,
    n_groups_used: int | None = None,
    selection_metadata: Mapping[str, Any] | None = None,
) -> SolverResponse:
    sanity_metadata = _selection_sanity_metadata(
        request=request,
        pilot_response=pilot_response,
        selected_response=selected_response,
        tau_selected=tau_selected,
    )
    reason = _selection_fallback_reason(sanity_metadata)
    if reason is None:
        annotated = _annotate_selection_response(
            request=request,
            response=selected_response,
            selection_method=selection_method,
            selection_status="selected",
            tau_pilot=tau_pilot,
            tau_selected=tau_selected,
            tau_selected_se=tau_selected_se,
            sanity_status="accepted_selected",
            sanity_reason=None,
            sanity_metadata=sanity_metadata,
            short_dim_median_t=short_dim_median_t,
            n_groups_used=n_groups_used,
            selection_metadata=selection_metadata,
        )
        if request.tau_inference == "propagate":
            return apply_selection_delta_to_solver_response(
                annotated,
                pilot_response=pilot_response,
                tau_pilot=tau_pilot,
                tau_selected=tau_selected,
                tau_selected_se=tau_selected_se,
            )
        return annotated
    return _annotate_selection_response(
        request=request,
        response=pilot_response,
        selection_method=selection_method,
        selection_status=reason,
        tau_pilot=tau_pilot,
        tau_selected=tau_selected,
        tau_selected_se=tau_selected_se,
        sanity_status="pilot_retained",
        sanity_reason=reason,
        sanity_metadata=sanity_metadata,
        warning_message=_selection_fallback_message(
            reason=reason,
            tau_selected=tau_selected,
            tau_pilot=tau_pilot,
        ),
        short_dim_median_t=short_dim_median_t,
        n_groups_used=n_groups_used,
        selection_metadata=selection_metadata,
    )


def _fd_partition_warning(
    request: SolverRequest,
    *,
    partition_hint: str | None,
) -> str | None:
    messages: list[str] = []
    request_warning = _partition_warning_from_request(request)
    if request_warning is not None:
        messages.append(request_warning)
    if partition_hint is not None:
        messages.append(str(partition_hint))
    if not messages:
        return None
    return " ".join(messages)


def _fd_oneway_linear_predictor(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    group_i: np.ndarray | None,
    tau: float,
    profile_fe,
) -> np.ndarray:
    linear = exog @ beta
    if group_i is None:
        return linear

    unique_groups, mapped = np.unique(group_i, return_inverse=True)
    alpha_i, _ = profile_fe(
        endog,
        exog,
        beta,
        mapped,
        None,
        len(unique_groups),
        0,
        tau=tau,
    )
    return linear + alpha_i[mapped]


def _full_length_linear_predictor(
    retained_index: np.ndarray | None,
    kept_obs_mask: np.ndarray,
) -> tuple[float, ...]:
    if retained_index is None:
        return tuple(float("nan") for _ in range(kept_obs_mask.shape[0]))
    retained = np.asarray(retained_index, dtype=float)
    kept_count = int(np.sum(kept_obs_mask))
    if retained.shape[0] != kept_count:
        raise RuntimeError(
            "The fullcancel fd_partitioned route expected retained_index to have "
            "one entry per kept observation after package preprocessing."
        )
    linear = np.full(kept_obs_mask.shape[0], np.nan, dtype=float)
    linear[kept_obs_mask] = retained
    return tuple(float(value) for value in linear)


def _full_length_fitted(
    linear_predictor: tuple[float, ...],
    kept_obs_mask: np.ndarray,
) -> tuple[float, ...]:
    if not linear_predictor:
        return tuple(float("nan") for _ in range(kept_obs_mask.shape[0]))
    return tuple(
        float(value) for value in exp_with_float64_overflow_guard(linear_predictor)
    )


def _full_length_residuals(
    *,
    endog: np.ndarray,
    fitted: tuple[float, ...],
    kept_obs_mask: np.ndarray,
) -> tuple[float, ...]:
    if not fitted:
        return ()
    resid = np.full(kept_obs_mask.shape[0], np.nan, dtype=float)
    fitted_arr = np.asarray(fitted, dtype=float)
    resid[kept_obs_mask] = endog[kept_obs_mask] - fitted_arr[kept_obs_mask]
    return tuple(float(value) for value in resid)


def _demean_1way_weighted(
    values: np.ndarray,
    group: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    group_arr = np.asarray(group)
    n_groups = int(group_arr.max()) + 1 if group_arr.size else 0
    if array.ndim == 2 and array.shape[1] == 0:
        return array.copy()
    if weights is None:
        counts = np.bincount(group_arr, minlength=n_groups).astype(float)
        if array.ndim == 1:
            sums = np.bincount(group_arr, weights=array, minlength=n_groups)
            return array - (sums / np.maximum(counts, 1.0))[group_arr]
        out = np.empty_like(array)
        for column in range(array.shape[1]):
            sums = np.bincount(
                group_arr,
                weights=array[:, column],
                minlength=n_groups,
            )
            out[:, column] = (
                array[:, column] - (sums / np.maximum(counts, 1.0))[group_arr]
            )
        return out

    wsum = np.bincount(group_arr, weights=weights, minlength=n_groups)
    valid_group = wsum > 1e-15
    if array.ndim == 1:
        sums = np.bincount(group_arr, weights=weights * array, minlength=n_groups)
        means = np.zeros_like(wsum, dtype=float)
        means[valid_group] = sums[valid_group] / wsum[valid_group]
        return array - means[group_arr]
    out = np.empty_like(array)
    for column in range(array.shape[1]):
        sums = np.bincount(
            group_arr,
            weights=weights * array[:, column],
            minlength=n_groups,
        )
        means = np.zeros_like(wsum, dtype=float)
        means[valid_group] = sums[valid_group] / wsum[valid_group]
        out[:, column] = array[:, column] - means[group_arr]
    return out


def _demean_manyway_weighted(
    values: np.ndarray,
    groups: tuple[np.ndarray, ...],
    *,
    weights: np.ndarray | None = None,
    tol: float = 1e-12,
    maxiter: int = 5000,
    use_cpp: bool = True,
) -> np.ndarray:
    if use_cpp:
        cpp_result = _demean_manyway_weighted_cpp(
            values,
            groups,
            weights=weights,
            tol=tol,
            maxiter=maxiter,
        )
        if cpp_result is not None:
            _, converged = _core_demean_manyway_checked(
                cpp_result,
                groups,
                weights=weights,
                tol=tol,
                maxiter=1,
            )
            if not converged:
                raise RuntimeError("manyway_fe_demean_not_converged")
            return cpp_result
    return _core_demean_manyway(
        values,
        groups,
        weights=weights,
        tol=tol,
        maxiter=maxiter,
    )


def _single_group_weighted_centered(
    values: np.ndarray,
    groups: tuple[np.ndarray, ...],
    *,
    weights: np.ndarray,
) -> np.ndarray | None:
    if len(groups) != 1:
        return None
    group = np.asarray(groups[0])
    if group.size and not np.all(group == group[0]):
        return None
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return None
    arr = np.asarray(values, dtype=float)
    means = (arr * weights[:, None]).sum(axis=0) / weight_sum
    return arr - means


def _nan_inference_arrays(
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full(int(size), np.nan, dtype=float),
        np.full((int(size), int(size)), np.nan, dtype=float),
    )


def _cpp_covariance_requested() -> bool:
    import os

    disable = os.environ.get("PYIOLS_DISABLE_CPP")
    if disable is not None and disable.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "cpp",
    }:
        return False
    return True


def _cpp_covariance_policy() -> str:
    import os

    disable = os.environ.get("PYIOLS_DISABLE_CPP")
    if disable is not None and disable.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "cpp",
    }:
        return "disabled_by_env"
    value = os.environ.get("PYIOLS_USE_CPP")
    if value is not None and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "cpp",
    }:
        return "default_on_explicit_env"
    return "default_on"


_CPP_COVARIANCE_LAST_ERROR: str | None = None


def _clear_cpp_covariance_last_error() -> None:
    global _CPP_COVARIANCE_LAST_ERROR
    _CPP_COVARIANCE_LAST_ERROR = None


def _set_cpp_covariance_last_error(reason: str) -> None:
    global _CPP_COVARIANCE_LAST_ERROR
    _CPP_COVARIANCE_LAST_ERROR = str(reason)


def _cpp_covariance_exception_reason(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 160:
        message = message[:157] + "..."
    if not message:
        return f"cpp_exception:{type(exc).__name__}"
    return f"cpp_exception:{type(exc).__name__}:{message}"


def _load_cpp_covariance_module():
    try:
        from historical_iols import _kernel_cpp  # type: ignore[attr-defined]
    except ImportError:
        return None
    if not hasattr(_kernel_cpp, "sandwich_covariance"):
        return None
    return _kernel_cpp


def _cpp_covariance_fallback_reason(
    required_symbols: tuple[str, ...] = ("sandwich_covariance",),
) -> str | None:
    if _CPP_COVARIANCE_LAST_ERROR:
        return _CPP_COVARIANCE_LAST_ERROR
    if not _cpp_covariance_requested():
        return "disabled_by_env"
    try:
        from historical_iols import _kernel_cpp  # type: ignore[attr-defined]
    except ImportError:
        return "extension_unavailable"
    missing = tuple(
        symbol for symbol in required_symbols if not hasattr(_kernel_cpp, symbol)
    )
    if missing:
        return "missing_symbol:" + ",".join(missing)
    return None


def _covariance_backend_metadata(
    *,
    backend: str,
    status: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "covariance_backend": backend,
        "covariance_backend_policy": _cpp_covariance_policy(),
        "covariance_backend_status": status,
        "covariance_backend_fallback_reason": fallback_reason,
    }


def _sandwich_covariance_cpp(
    *,
    design: np.ndarray,
    bread_weight: np.ndarray,
    score: np.ndarray,
    se_type: str,
    df_denom: int,
    cluster_var: Any | None = None,
    symmetrize: bool = False,
) -> dict[str, Any] | None:
    _clear_cpp_covariance_last_error()
    if not _cpp_covariance_requested():
        return None
    cpp_kernel = _load_cpp_covariance_module()
    if cpp_kernel is None:
        return None
    design_arr = np.asarray(design, dtype=float)
    bread_weight_arr = np.asarray(bread_weight, dtype=float)
    score_arr = np.asarray(score, dtype=float)
    cluster_codes = None
    n_clusters = None
    if se_type == "cluster" and cluster_var is not None:
        cluster_arr = np.asarray(cluster_var)
        if cluster_arr.shape[0] != design_arr.shape[0]:
            _set_cpp_covariance_last_error("cpp_cluster_shape_mismatch")
            return None
        _, mapped = np.unique(cluster_arr, return_inverse=True)
        cluster_codes = _as_dense_group_array(mapped).astype(np.int64, copy=False)
        n_clusters = int(mapped.max()) + 1 if mapped.size else 0
    try:
        raw = cpp_kernel.sandwich_covariance(
            design_arr,
            bread_weight_arr,
            score_arr,
            se_type,
            int(max(df_denom, 1)),
            cluster_codes,
            n_clusters,
        )
    except Exception as exc:
        reason = _cpp_covariance_exception_reason(exc)
        if "manyway_fe_demean_not_converged" in reason:
            raise RuntimeError("manyway_fe_demean_not_converged") from exc
        _set_cpp_covariance_last_error(reason)
        return None
    covariance = np.asarray(raw["covariance"], dtype=float)
    if symmetrize:
        covariance = 0.5 * (covariance + covariance.T)
    return {
        "status": str(raw["status"]),
        "reason": None if raw["reason"] is None else str(raw["reason"]),
        "se": np.asarray(raw["se"], dtype=float),
        "covariance": covariance,
        "bread_condition": raw["bread_condition"],
        "n_clusters": raw["n_clusters"],
    }


def _dense_group_bundle_for_cpp(
    groups: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    dense_groups: list[np.ndarray] = []
    counts: list[int] = []
    for group in groups:
        group_arr = np.asarray(group)
        if (
            np.issubdtype(group_arr.dtype, np.integer)
            and group_arr.size
            and int(np.min(group_arr)) >= 0
            and int(np.max(group_arr)) <= max(group_arr.size * 4, 1024)
        ):
            dense = _as_dense_group_array(group_arr).astype(np.int64, copy=False)
            dense_groups.append(dense)
            counts.append(int(np.max(dense)) + 1)
            continue
        _, mapped = np.unique(group_arr, return_inverse=True)
        dense = _as_dense_group_array(mapped).astype(np.int64, copy=False)
        dense_groups.append(dense)
        counts.append(int(mapped.max()) + 1 if mapped.size else 0)
    return tuple(dense_groups), tuple(counts)


def _compact_group_bundle_for_cpp(
    groups: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    dense_groups: list[np.ndarray] = []
    counts: list[int] = []
    for group in groups:
        _, mapped = np.unique(np.asarray(group), return_inverse=True)
        dense = _as_dense_group_array(mapped).astype(np.int64, copy=False)
        dense_groups.append(dense)
        counts.append(int(mapped.max()) + 1 if mapped.size else 0)
    return tuple(dense_groups), tuple(counts)


def _demean_manyway_weighted_cpp(
    values: np.ndarray,
    groups: tuple[np.ndarray, ...],
    *,
    weights: np.ndarray | None,
    tol: float,
    maxiter: int,
) -> np.ndarray | None:
    _clear_cpp_covariance_last_error()
    if not _cpp_covariance_requested():
        return None
    cpp_kernel = _load_cpp_covariance_module()
    if cpp_kernel is None or not hasattr(cpp_kernel, "demean_manyway"):
        return None
    groups_cpp, counts_cpp = _dense_group_bundle_for_cpp(groups)
    try:
        return np.asarray(
            cpp_kernel.demean_manyway(
                np.asarray(values, dtype=float),
                groups_cpp,
                counts_cpp,
                None if weights is None else np.asarray(weights, dtype=float),
                float(tol),
                int(maxiter),
            ),
            dtype=float,
        )
    except Exception as exc:
        _set_cpp_covariance_last_error(_cpp_covariance_exception_reason(exc))
        return None


def _mixed_covariance_projection_cpp(
    *,
    x_slopes: np.ndarray,
    cancel_groups: tuple[np.ndarray, ...],
    recover_groups: tuple[np.ndarray, ...],
    demean_weights: np.ndarray,
    bread_weight: np.ndarray,
    score: np.ndarray,
    se_type: str,
    cluster_var: Any | None = None,
    tol: float = 1e-12,
    maxiter: int = 5000,
) -> dict[str, Any] | None:
    _clear_cpp_covariance_last_error()
    if not _cpp_covariance_requested():
        return None
    cpp_kernel = _load_cpp_covariance_module()
    if cpp_kernel is None or not hasattr(cpp_kernel, "mixed_covariance_projection"):
        return None
    x_arr = np.asarray(x_slopes, dtype=float)
    if x_arr.ndim != 2:
        _set_cpp_covariance_last_error("cpp_mixed_projection_x_shape_mismatch")
        return None
    cancel_cpp, cancel_counts = _compact_group_bundle_for_cpp(cancel_groups)
    recover_cpp, recover_counts = _compact_group_bundle_for_cpp(recover_groups)
    cancel_df = sum(max(int(count) - 1, 0) for count in cancel_counts)
    recover_df = sum(max(int(count) - 1, 0) for count in recover_counts)
    df_denom = max(
        int(x_arr.shape[0]) - int(x_arr.shape[1]) - cancel_df - recover_df,
        1,
    )
    cluster_codes = None
    n_clusters = None
    if se_type == "cluster" and cluster_var is not None:
        cluster_arr = np.asarray(cluster_var)
        if cluster_arr.shape[0] != x_arr.shape[0]:
            _set_cpp_covariance_last_error("cpp_cluster_shape_mismatch")
            return None
        _, mapped = np.unique(cluster_arr, return_inverse=True)
        cluster_codes = _as_dense_group_array(mapped).astype(np.int64, copy=False)
        n_clusters = int(mapped.max()) + 1 if mapped.size else 0
    try:
        raw = cpp_kernel.mixed_covariance_projection(
            x_arr,
            cancel_cpp,
            cancel_counts,
            recover_cpp,
            recover_counts,
            np.asarray(demean_weights, dtype=float),
            np.asarray(bread_weight, dtype=float),
            np.asarray(score, dtype=float),
            se_type,
            int(df_denom),
            cluster_codes,
            n_clusters,
            float(tol),
            int(maxiter),
        )
    except Exception as exc:
        _set_cpp_covariance_last_error(_cpp_covariance_exception_reason(exc))
        return None
    covariance = np.asarray(raw["covariance"], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    return {
        "status": str(raw["status"]),
        "reason": None if raw["reason"] is None else str(raw["reason"]),
        "se": np.asarray(raw["se"], dtype=float),
        "covariance": covariance,
        "bread_condition": raw["bread_condition"],
        "n_clusters": raw["n_clusters"],
        "df_denom": int(df_denom),
        "recover_columns": int(raw["recover_columns"]),
        "slope_columns": int(raw["slope_columns"]),
        "projection_method": str(raw["projection_method"]),
    }


def _dummy_droplast_sparse(
    group: np.ndarray,
    n_groups: int,
) -> sparse.csc_matrix:
    if n_groups <= 1:
        return sparse.csc_matrix((group.shape[0], 0), dtype=float)
    mask = group < (n_groups - 1)
    rows = np.flatnonzero(mask)
    cols = group[mask]
    data = np.ones(rows.shape[0], dtype=float)
    return sparse.csc_matrix(
        (data, (rows, cols)),
        shape=(group.shape[0], n_groups - 1),
        dtype=float,
    )


def _sparse_drop_one_fe_design(
    *,
    x: np.ndarray,
    remapped_groups: tuple[tuple[np.ndarray, int], ...],
) -> sparse.csc_matrix:
    matrices: list[sparse.csc_matrix] = []
    if x.shape[1]:
        matrices.append(sparse.csc_matrix(x))
    for inverse, n_levels in remapped_groups:
        if n_levels > 1:
            matrices.append(_dummy_droplast_sparse(inverse, n_levels))
    if not matrices:
        return sparse.csc_matrix((x.shape[0], 0), dtype=float)
    return sparse.hstack(matrices, format="csc", dtype=float)


def _sparse_design_rank_from_svd(
    design: sparse.csc_matrix,
    *,
    tol: float,
) -> int:
    max_rank = min(int(design.shape[0]), int(design.shape[1]))
    if max_rank == 0:
        return 0
    rank_tol = max(float(tol), 0.0)
    if max_rank == 1:
        norm = float(np.sqrt(design.multiply(design).sum()))
        if not np.isfinite(norm):
            raise np.linalg.LinAlgError("nonfinite sparse design norm")
        return int(norm > rank_tol)
    maxiter = max(10_000, 20 * max_rank)
    try:
        largest = sparse_linalg.svds(
            design,
            k=max_rank - 1,
            which="LM",
            return_singular_vectors=False,
            tol=0.0,
            maxiter=maxiter,
            rng=np.random.default_rng(0),
        )
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        raise np.linalg.LinAlgError("sparse SVD rank failed") from exc
    if not np.all(np.isfinite(largest)):
        raise np.linalg.LinAlgError("nonfinite sparse SVD singular values")
    rank = int(np.sum(largest > rank_tol))
    if rank < max_rank - 1:
        return rank
    try:
        smallest = sparse_linalg.svds(
            design,
            k=1,
            which="SM",
            return_singular_vectors=False,
            tol=0.0,
            maxiter=maxiter,
            rng=np.random.default_rng(0),
        )
    except (RuntimeError, np.linalg.LinAlgError) as exc:
        raise np.linalg.LinAlgError(
            "sparse SVD smallest singular value failed"
        ) from exc
    if not np.all(np.isfinite(smallest)):
        raise np.linalg.LinAlgError("nonfinite sparse SVD smallest singular value")
    return max_rank if float(smallest[0]) > rank_tol else max_rank - 1


def _covariance_rank_diagnostics(
    *,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tol: float = 1e-10,
    max_dense_columns: int = 5000,
    max_dense_elements: int = 5_000_000,
    max_sparse_columns: int = 2000,
    max_sparse_work_elements: int = 4_000_000,
) -> dict[str, Any]:
    x = np.asarray(exog, dtype=float)
    if x.ndim != 2:
        raise ValueError("exog must be two-dimensional")
    n_obs, k_params = x.shape
    remapped_groups: list[tuple[np.ndarray, int]] = []
    dof_fe_additive = 0
    for group in groups:
        group_arr = np.asarray(group)
        if group_arr.shape[0] != n_obs:
            raise ValueError("all FE groups must have one entry per observation")
        _, inverse = np.unique(group_arr, return_inverse=True)
        n_levels = int(inverse.max()) + 1 if inverse.size else 0
        remapped_groups.append((_as_dense_group_array(inverse), n_levels))
        dof_fe_additive += max(n_levels - 1, 0)

    additive_model_rank = int(k_params + dof_fe_additive)
    additive_df_resid = max(int(n_obs) - additive_model_rank, 1)
    total_columns = additive_model_rank
    total_elements = int(n_obs) * max(total_columns, 1)
    base = {
        "rank_status": "withheld",
        "rank_method": "dense_drop_one_fe_rank_not_run",
        "rank_intercept_convention": "caller_estimable_x_plus_drop_one_per_fe_block",
        "n_effective": int(n_obs),
        "k_params": int(k_params),
        "dof_fe_additive": int(dof_fe_additive),
        "dof_model_rank": None,
        "dof_x_rank": None,
        "dof_absorbed_rank": None,
        "df_resid_exact": None,
        "df_resid_additive": int(additive_df_resid),
        "dof_model_rank_additive": int(additive_model_rank),
    }

    x_rank: int
    model_rank: int | None = None
    rank_method: str | None = None
    if total_columns <= int(max_dense_columns) and total_elements <= int(
        max_dense_elements
    ):
        matrices: list[np.ndarray] = []
        if k_params:
            matrices.append(x)
        for inverse, n_levels in remapped_groups:
            if n_levels > 1:
                matrices.append(_dummy_droplast(inverse, n_levels))
        if matrices:
            design = np.column_stack(matrices)
        else:
            design = np.zeros((n_obs, 0), dtype=float)
        try:
            x_rank = int(np.linalg.matrix_rank(x, tol=tol)) if k_params else 0
            model_rank = int(np.linalg.matrix_rank(design, tol=tol))
            rank_method = "dense_drop_one_fe_rank"
        except np.linalg.LinAlgError:
            base["rank_method"] = "dense_drop_one_fe_rank_failed"
            return base
    else:
        try:
            x_rank = int(np.linalg.matrix_rank(x, tol=tol)) if k_params else 0
        except np.linalg.LinAlgError:
            base["rank_method"] = "x_rank_failed"
            return base
        if total_columns > int(max_sparse_columns) or total_columns**2 > int(
            max_sparse_work_elements
        ):
            base["rank_method"] = "sparse_svd_drop_one_fe_rank_withheld_size_limit"
            base["rank_sparse_columns"] = int(total_columns)
            base["rank_sparse_work_elements"] = int(total_columns**2)
            return base
        sparse_design = _sparse_drop_one_fe_design(
            x=x,
            remapped_groups=tuple(remapped_groups),
        )
        try:
            model_rank = _sparse_design_rank_from_svd(
                sparse_design,
                tol=tol,
            )
            rank_method = "sparse_svd_drop_one_fe_rank"
        except np.linalg.LinAlgError:
            base["rank_method"] = "sparse_svd_drop_one_fe_rank_failed"
            return base
        base["rank_sparse_columns"] = int(total_columns)
        base["rank_sparse_work_elements"] = int(total_columns**2)

    if model_rank is None or rank_method is None:
        return base
    absorbed_rank = max(model_rank - x_rank, 0)
    base.update(
        {
            "rank_status": "exact",
            "rank_method": rank_method,
            "dof_model_rank": int(model_rank),
            "dof_x_rank": int(x_rank),
            "dof_absorbed_rank": int(absorbed_rank),
            "df_resid_exact": max(int(n_obs) - int(model_rank), 1),
        }
    )
    return base


def _constant_column_mask(exog: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    x = np.asarray(exog, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    spread = np.ptp(x, axis=0)
    nonzero = np.max(np.abs(x), axis=0) > tol
    return (spread <= tol) & nonzero


def _mixed_hc0_status(
    *,
    supported: bool,
    reason: str | None = None,
    bread_weight_zero_fraction: float | None = None,
    bread_condition: float | None = None,
    n_clusters: int | None = None,
    covariance_backend: str = "not_applicable",
    covariance_backend_status: str | None = None,
    covariance_backend_fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "supported" if supported else "withheld",
        "reason": reason,
        "bread_weight_zero_fraction": bread_weight_zero_fraction,
        "w_obs_floor_fraction": None,
        "bread_condition": bread_condition,
        "n_clusters": n_clusters,
        **_covariance_backend_metadata(
            backend=covariance_backend,
            status=covariance_backend_status
            or ("available" if supported else "withheld"),
            fallback_reason=covariance_backend_fallback_reason,
        ),
    }


def _stable_manyway_weight_terms(
    *,
    endog: np.ndarray,
    linear: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    y = np.asarray(endog, dtype=float)
    linear_arr = np.asarray(linear, dtype=float)
    if y.shape != linear_arr.shape:
        return None
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(linear_arr)):
        return None

    log_a = (2.0 - tau) * linear_arr
    positive = y > 0.0
    log_score_pos = np.full(linear_arr.shape, -np.inf, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_score_pos[positive] = (
            np.log(y[positive]) + (1.0 - tau) * linear_arr[positive]
        )

    finite_candidates = [
        values[np.isfinite(values)]
        for values in (log_a, log_score_pos)
        if np.any(np.isfinite(values))
    ]
    shift = 0.0
    if finite_candidates:
        shift = max(0.0, float(max(np.max(values) for values in finite_candidates)))

    with np.errstate(under="ignore"):
        a_scaled = np.exp(log_a - shift)
        score_pos_scaled = np.zeros_like(linear_arr, dtype=float)
        valid_score = positive & np.isfinite(log_score_pos)
        score_pos_scaled[valid_score] = np.exp(log_score_pos[valid_score] - shift)

    score = score_pos_scaled - a_scaled
    bread_weight = a_scaled

    if not (
        np.all(np.isfinite(a_scaled))
        and np.all(np.isfinite(score))
        and np.all(np.isfinite(bread_weight))
    ):
        return None
    return a_scaled, score, bread_weight


def _as_cov1_observed_weight_from_score(
    *,
    expected_weight: np.ndarray,
    score: np.ndarray,
    tau: float,
) -> np.ndarray:
    return np.asarray(expected_weight, dtype=float) + (float(tau) - 1.0) * np.asarray(
        score, dtype=float
    )


def _as_cov1_matrix_condition(matrix: np.ndarray) -> float | None:
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return None
    return None if not np.isfinite(condition) else condition


def _as_cov1_min_eigenvalue(matrix: np.ndarray) -> float | None:
    try:
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        return None
    if eigenvalues.size == 0 or not np.all(np.isfinite(eigenvalues)):
        return None
    return float(np.min(eigenvalues))


def _as_cov1_rank(matrix: np.ndarray) -> int | None:
    try:
        return int(np.linalg.matrix_rank(matrix))
    except np.linalg.LinAlgError:
        return None


def _as_cov1_hc0_covariance(
    *,
    bread: np.ndarray,
    score_design: np.ndarray,
) -> np.ndarray | None:
    if bread.shape[0] == 0:
        return None
    condition = _as_cov1_matrix_condition(bread)
    if condition is None or condition > _BREAD_CONDITION_LIMIT:
        return None
    min_eigenvalue = _as_cov1_min_eigenvalue(bread)
    if min_eigenvalue is None or min_eigenvalue <= 0.0:
        return None
    try:
        bread_inv = np.linalg.inv(0.5 * (bread + bread.T))
    except np.linalg.LinAlgError:
        return None
    meat = score_design.T @ score_design
    covariance = bread_inv @ meat @ bread_inv
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        return None
    return covariance


def _as_cov1_bread_diagnostics_from_design(
    *,
    route: str,
    tau: float,
    design: np.ndarray,
    expected_weight: np.ndarray,
    observed_weight: np.ndarray,
    score: np.ndarray,
    observed_weight_definition: str,
) -> dict[str, Any]:
    design_arr = np.asarray(design, dtype=float)
    expected = np.asarray(expected_weight, dtype=float)
    observed = np.asarray(observed_weight, dtype=float)
    score_arr = np.asarray(score, dtype=float)
    size = int(design_arr.shape[1]) if design_arr.ndim == 2 else 0
    if (
        design_arr.ndim != 2
        or expected.shape[0] != design_arr.shape[0]
        or observed.shape[0] != design_arr.shape[0]
        or score_arr.shape[0] != design_arr.shape[0]
    ):
        return {
            "status": "unavailable",
            "reason": "shape_mismatch",
            "route": route,
            "tau": float(tau),
            "parameter_count": size,
        }
    if not (
        np.all(np.isfinite(design_arr))
        and np.all(np.isfinite(expected))
        and np.all(np.isfinite(observed))
        and np.all(np.isfinite(score_arr))
    ):
        return {
            "status": "unavailable",
            "reason": "nonfinite_inputs",
            "route": route,
            "tau": float(tau),
            "parameter_count": size,
        }

    expected_bread = (design_arr * expected[:, None]).T @ design_arr
    observed_bread = (design_arr * observed[:, None]).T @ design_arr
    score_design = design_arr * score_arr[:, None]
    if not (
        np.all(np.isfinite(expected_bread))
        and np.all(np.isfinite(observed_bread))
        and np.all(np.isfinite(score_design))
    ):
        return {
            "status": "unavailable",
            "reason": "nonfinite_bread",
            "route": route,
            "tau": float(tau),
            "parameter_count": size,
        }

    expected_weight_sum = float(np.sum(expected))
    observed_abs_sum = float(np.sum(np.abs(observed)))
    expected_covariance = _as_cov1_hc0_covariance(
        bread=expected_bread,
        score_design=score_design,
    )
    observed_covariance = _as_cov1_hc0_covariance(
        bread=observed_bread,
        score_design=score_design,
    )
    se_ratio_mean: float | None = None
    if expected_covariance is not None and observed_covariance is not None:
        expected_se = np.sqrt(np.maximum(np.diag(expected_covariance), 0.0))
        observed_se = np.sqrt(np.maximum(np.diag(observed_covariance), 0.0))
        valid = (
            np.isfinite(expected_se) & np.isfinite(observed_se) & (expected_se > 0.0)
        )
        if np.any(valid):
            se_ratio_mean = float(np.mean(observed_se[valid] / expected_se[valid]))

    observed_min_eigenvalue = _as_cov1_min_eigenvalue(observed_bread)
    observed_condition = _as_cov1_matrix_condition(observed_bread)
    observed_admissible_no_floor = (
        observed_min_eigenvalue is not None
        and observed_min_eigenvalue > 0.0
        and observed_condition is not None
        and observed_condition <= _BREAD_CONDITION_LIMIT
        and bool(np.all(observed > 0.0))
    )
    return {
        "status": "computed",
        "reason": None,
        "route": route,
        "tau": float(tau),
        "parameter_count": size,
        "observed_weight_definition": observed_weight_definition,
        "no_floor_policy": "diagnostic_only_no_floor",
        "expected_bread": expected_bread,
        "observed_bread": observed_bread,
        "expected_hc0_covariance": expected_covariance,
        "observed_hc0_covariance": observed_covariance,
        "expected_bread_condition": _as_cov1_matrix_condition(expected_bread),
        "observed_bread_condition": observed_condition,
        "expected_bread_min_eigenvalue": _as_cov1_min_eigenvalue(expected_bread),
        "observed_bread_min_eigenvalue": observed_min_eigenvalue,
        "expected_bread_rank": _as_cov1_rank(expected_bread),
        "observed_bread_rank": _as_cov1_rank(observed_bread),
        "expected_weight_zero_fraction": float(np.mean(expected <= 0.0)),
        "observed_weight_negative_fraction": float(np.mean(observed < 0.0)),
        "observed_weight_nonpositive_fraction": float(np.mean(observed <= 0.0)),
        "observed_weight_min": float(np.min(observed)),
        "observed_weight_max": float(np.max(observed)),
        "expected_weight_max_share": (
            None
            if expected_weight_sum <= 0.0
            else float(np.max(expected) / expected_weight_sum)
        ),
        "observed_weight_max_abs_share": (
            None
            if observed_abs_sum <= 0.0
            else float(np.max(np.abs(observed)) / observed_abs_sum)
        ),
        "observed_bread_would_require_floor": bool(np.any(observed <= 0.0)),
        "observed_bread_admissible_no_floor": bool(observed_admissible_no_floor),
        "expected_hc0_covariance_available": expected_covariance is not None,
        "observed_hc0_covariance_available": observed_covariance is not None,
        "observed_to_expected_hc0_se_ratio_mean": se_ratio_mean,
    }


def _manyway_iols_bread_diagnostics(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau: float,
    linear: np.ndarray | None = None,
    route: str = "manyway",
) -> dict[str, Any]:
    size = int(np.asarray(beta, dtype=float).shape[0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        if linear is None:
            _, linear_arr = _profile_fe_manyway(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                tau=tau,
            )
        else:
            linear_arr = np.asarray(linear, dtype=float)
        stable_terms = _stable_manyway_weight_terms(
            endog=endog,
            linear=linear_arr,
            tau=tau,
        )
        if stable_terms is None:
            return {
                "status": "unavailable",
                "reason": "stable_weight_terms_unavailable",
                "route": route,
                "tau": float(tau),
                "parameter_count": size,
            }
        a_scaled, score, bread_weight = stable_terms
        design = _demean_manyway_weighted(
            np.asarray(exog, dtype=float),
            groups,
            weights=a_scaled,
        )
    observed_weight = _as_cov1_observed_weight_from_score(
        expected_weight=bread_weight,
        score=score,
        tau=tau,
    )
    return _as_cov1_bread_diagnostics_from_design(
        route=route,
        tau=tau,
        design=design,
        expected_weight=bread_weight,
        observed_weight=observed_weight,
        score=score,
        observed_weight_definition="a_plus_tau_minus_one_times_score",
    )


def _fullcancel_iols_bread_diagnostics(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau: float,
    linear: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(endog, dtype=float)
    x = np.asarray(exog, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    size = int(beta_arr.shape[0])
    if size == 0:
        return {
            "status": "unavailable",
            "reason": "no_estimable_slope_parameters",
            "route": "fullcancel",
            "tau": float(tau),
            "parameter_count": size,
        }
    q = x @ beta_arr if linear is None else np.asarray(linear, dtype=float)
    if q.shape != y.shape or not np.all(np.isfinite(q)):
        return {
            "status": "unavailable",
            "reason": "retained_index_unavailable",
            "route": "fullcancel",
            "tau": float(tau),
            "parameter_count": size,
        }
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weights = exp_with_float64_overflow_guard((2.0 - float(tau)) * q)
        ratio = y * exp_with_float64_overflow_guard(-q)
    if (
        not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(ratio))
        or not np.any(weights > 0.0)
    ):
        return {
            "status": "unavailable",
            "reason": "fullcancel_weight_terms_unavailable",
            "route": "fullcancel",
            "tau": float(tau),
            "parameter_count": size,
        }
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        design = _demean_manyway_weighted(x, groups, weights=weights)
        ratio_demeaned = _demean_manyway_weighted(ratio, groups, weights=weights)
    score = weights * ratio_demeaned
    observed_weight = _as_cov1_observed_weight_from_score(
        expected_weight=weights,
        score=score,
        tau=tau,
    )
    return _as_cov1_bread_diagnostics_from_design(
        route="fullcancel",
        tau=tau,
        design=design,
        expected_weight=weights,
        observed_weight=observed_weight,
        score=score,
        observed_weight_definition="fullcancel_route_score_identity_comparator",
    )


def _fullcancel_covariance_unavailable(
    size: int,
    *,
    reason: str,
    **details: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    se, covariance = _nan_inference_arrays(size)
    info: dict[str, Any] = {
        "inference_status": "withheld",
        "inference_reason": reason,
        "inference_reasons": (reason,),
        "fullcancel_covariance_status": "withheld",
        "fullcancel_covariance_reason": reason,
        **_covariance_backend_metadata(
            backend="not_applicable",
            status="withheld",
            fallback_reason=reason,
        ),
    }
    info.update(details)
    return se, covariance, info


def _fullcancel_iols_covariance(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau: float,
    se_type: str,
    cluster_var: Any | None = None,
    df_resid_override: int | None = None,
    linear: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if se_type not in {"iid", "hc0", "hc1", "hc2", "hc3", "cluster"}:
        raise NotImplementedError(
            "fullcancel iOLS covariance currently supports iid, HC0-HC3, "
            "and one-way cluster covariance."
        )
    y = np.asarray(endog, dtype=float)
    x = np.asarray(exog, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    size = int(beta_arr.shape[0])
    if size == 0:
        return _fullcancel_covariance_unavailable(
            size,
            reason="no_estimable_slope_parameters",
        )
    if linear is None:
        q = x @ beta_arr
    else:
        q = np.asarray(linear, dtype=float)
    if q.shape != y.shape:
        return _fullcancel_covariance_unavailable(
            size,
            reason="retained_index_shape_mismatch",
        )
    if not np.all(np.isfinite(q)):
        return _fullcancel_covariance_unavailable(
            size,
            reason="nonfinite_retained_index",
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weights = exp_with_float64_overflow_guard((2.0 - float(tau)) * q)
        ratio = y * exp_with_float64_overflow_guard(-q)
    if (
        not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(ratio))
        or not np.any(weights > 0.0)
    ):
        return _fullcancel_covariance_unavailable(
            size,
            reason="fullcancel_weight_terms_unavailable",
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x_demeaned = _demean_manyway_weighted(x, groups, weights=weights)
        ratio_demeaned = _demean_manyway_weighted(ratio, groups, weights=weights)
        if not np.all(np.isfinite(x_demeaned)) or not np.all(
            np.isfinite(ratio_demeaned)
        ):
            return _fullcancel_covariance_unavailable(
                size,
                reason="weighted_within_transform_nonfinite",
            )
        bread_weight_zero_fraction = float(np.mean(weights <= 0.0))
        bread_weight_sum = float(np.sum(weights))
        bread_weight_max_share = (
            None
            if bread_weight_sum <= 0.0
            else float(np.max(weights) / bread_weight_sum)
        )
        n_obs = int(y.shape[0])
        k_params = int(beta_arr.shape[0])
        fe_df = sum(max(int(np.unique(group).size) - 1, 0) for group in groups)
        df_denom = (
            max(int(df_resid_override), 1)
            if df_resid_override is not None
            else max(n_obs - k_params - fe_df, 1)
        )
        score = weights * ratio_demeaned
        cpp_covariance = _sandwich_covariance_cpp(
            design=x_demeaned,
            bread_weight=weights,
            score=score,
            se_type=se_type,
            df_denom=df_denom,
            cluster_var=cluster_var,
        )
        if cpp_covariance is not None:
            bread_cond_cpp = cpp_covariance["bread_condition"]
            bread_cond_value = None if bread_cond_cpp is None else float(bread_cond_cpp)
            if cpp_covariance["status"] != "available":
                reason = str(cpp_covariance["reason"])
                if reason in {
                    "nonfinite_sandwich_inputs",
                    "nonfinite_bread",
                    "nonfinite_meat",
                }:
                    reason = "fullcancel_score_terms_nonfinite"
                elif reason == "bread_inversion_failed":
                    reason = "ill_conditioned_bread"
                return _fullcancel_covariance_unavailable(
                    size,
                    reason=reason,
                    **_covariance_backend_metadata(
                        backend="cpp",
                        status="withheld",
                        fallback_reason=None,
                    ),
                    fullcancel_bread_condition=bread_cond_value,
                    fullcancel_bread_weight_zero_fraction=(bread_weight_zero_fraction),
                    fullcancel_bread_weight_max_share=bread_weight_max_share,
                    **(
                        {}
                        if cpp_covariance["n_clusters"] is None
                        else {"n_clusters": int(cpp_covariance["n_clusters"])}
                    ),
                )
            return (
                cpp_covariance["se"],
                cpp_covariance["covariance"],
                {
                    "inference_status": "available",
                    "inference_reason": None,
                    "inference_reasons": (),
                    "fullcancel_covariance_status": "available",
                    "fullcancel_covariance_reason": None,
                    "fullcancel_bread_condition": bread_cond_value,
                    "fullcancel_bread_weight_zero_fraction": bread_weight_zero_fraction,
                    "fullcancel_bread_weight_max_share": bread_weight_max_share,
                    **_covariance_backend_metadata(
                        backend="cpp",
                        status="available",
                        fallback_reason=None,
                    ),
                },
            )
        bread = (x_demeaned * weights[:, None]).T @ x_demeaned
        score_X = x_demeaned * (weights * ratio_demeaned)[:, None]
        if not np.all(np.isfinite(bread)) or not np.all(np.isfinite(score_X)):
            return _fullcancel_covariance_unavailable(
                size,
                reason="fullcancel_score_terms_nonfinite",
                fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
                fullcancel_bread_weight_max_share=bread_weight_max_share,
            )

    try:
        bread_cond = float(np.linalg.cond(bread))
    except np.linalg.LinAlgError:
        return _fullcancel_covariance_unavailable(
            size,
            reason="ill_conditioned_bread",
            fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
            fullcancel_bread_weight_max_share=bread_weight_max_share,
        )
    if (not np.isfinite(bread_cond)) or bread_cond > _BREAD_CONDITION_LIMIT:
        return _fullcancel_covariance_unavailable(
            size,
            reason="ill_conditioned_bread",
            fullcancel_bread_condition=None
            if not np.isfinite(bread_cond)
            else bread_cond,
            fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
            fullcancel_bread_weight_max_share=bread_weight_max_share,
        )
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        return _fullcancel_covariance_unavailable(
            size,
            reason="ill_conditioned_bread",
            fullcancel_bread_condition=bread_cond,
            fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
            fullcancel_bread_weight_max_share=bread_weight_max_share,
        )

    if se_type == "iid":
        phi = float(np.sum(weights * ratio_demeaned**2) / df_denom)
        covariance = phi * bread_inv
    elif se_type == "hc1":
        meat = score_X.T @ score_X
        covariance = bread_inv @ (meat * (n_obs / df_denom)) @ bread_inv
    elif se_type in {"hc2", "hc3"}:
        leverage = weights * np.sum((x_demeaned @ bread_inv) * x_demeaned, axis=1)
        leverage = np.clip(leverage, 0.0, 1.0 - _LEVERAGE_FLOOR)
        adjustment = 1.0 / (1.0 - leverage)
        if se_type == "hc3":
            adjustment = adjustment**2
        score_X_adj = score_X * np.sqrt(adjustment)[:, None]
        meat = score_X_adj.T @ score_X_adj
        covariance = bread_inv @ meat @ bread_inv
    elif se_type == "cluster":
        if cluster_var is None:
            return _fullcancel_covariance_unavailable(
                size,
                reason="cluster_variable_missing",
                fullcancel_bread_condition=bread_cond,
                fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
                fullcancel_bread_weight_max_share=bread_weight_max_share,
            )
        cluster_arr = np.asarray(cluster_var)
        if cluster_arr.shape[0] != n_obs:
            return _fullcancel_covariance_unavailable(
                size,
                reason="cluster_shape_mismatch",
                fullcancel_bread_condition=bread_cond,
                fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
                fullcancel_bread_weight_max_share=bread_weight_max_share,
            )
        _, mapped = np.unique(cluster_arr, return_inverse=True)
        n_clusters = int(mapped.max()) + 1 if mapped.size else 0
        if n_clusters < 2:
            return _fullcancel_covariance_unavailable(
                size,
                reason="too_few_clusters",
                fullcancel_bread_condition=bread_cond,
                fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
                fullcancel_bread_weight_max_share=bread_weight_max_share,
                n_clusters=n_clusters,
            )
        cluster_scores = np.zeros((n_clusters, size), dtype=float)
        for column in range(size):
            cluster_scores[:, column] = np.bincount(
                mapped,
                weights=score_X[:, column],
                minlength=n_clusters,
            )
        meat = cluster_scores.T @ cluster_scores
        covariance = (
            bread_inv
            @ (meat * (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / df_denom))
            @ bread_inv
        )
    else:
        meat = score_X.T @ score_X
        covariance = bread_inv @ meat @ bread_inv

    if not np.all(np.isfinite(covariance)):
        return _fullcancel_covariance_unavailable(
            size,
            reason="nonfinite_covariance",
            fullcancel_bread_condition=bread_cond,
            fullcancel_bread_weight_zero_fraction=bread_weight_zero_fraction,
            fullcancel_bread_weight_max_share=bread_weight_max_share,
        )
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return (
        se,
        covariance,
        {
            "inference_status": "available",
            "inference_reason": None,
            "inference_reasons": (),
            "fullcancel_covariance_status": "available",
            "fullcancel_covariance_reason": None,
            "fullcancel_bread_condition": bread_cond,
            "fullcancel_bread_weight_zero_fraction": bread_weight_zero_fraction,
            "fullcancel_bread_weight_max_share": bread_weight_max_share,
            **_covariance_backend_metadata(
                backend="python_reference",
                status="fallback_python_reference",
                fallback_reason=_cpp_covariance_fallback_reason()
                or "cpp_covariance_unavailable",
            ),
        },
    )


def _group_logsumexp(
    values: np.ndarray,
    group: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    if _env_flag_enabled(_VECTOR_GROUP_LOGSUMEXP_FLAG):
        group_arr = np.asarray(group, dtype=np.int64)
        values_arr = np.asarray(values, dtype=float)
        out = np.full(int(n_groups), -np.inf, dtype=float)
        if n_groups <= 0 or values_arr.size == 0:
            return out
        if group_arr.shape == values_arr.shape:
            valid_group = (group_arr >= 0) & (group_arr < int(n_groups))
            if not bool(np.any(valid_group)):
                return out
            valid_codes = group_arr[valid_group]
            valid_values = values_arr[valid_group]
            maxima = np.full(int(n_groups), -np.inf, dtype=float)
            np.maximum.at(maxima, valid_codes, valid_values)
            posinf = np.isposinf(maxima)
            out[posinf] = np.inf
            finite_groups = np.isfinite(maxima)
            if bool(np.any(finite_groups)):
                finite_rows = finite_groups[valid_codes]
                sums = np.zeros(int(n_groups), dtype=float)
                np.add.at(
                    sums,
                    valid_codes[finite_rows],
                    np.exp(
                        valid_values[finite_rows]
                        - maxima[valid_codes[finite_rows]]
                    ),
                )
                populated = finite_groups & (sums > 0.0)
                out[populated] = maxima[populated] + np.log(sums[populated])
            return out

    out = np.full(n_groups, -np.inf, dtype=float)
    for level in range(n_groups):
        subset = values[group == level]
        if subset.size == 0:
            continue
        max_value = np.max(subset)
        if np.isposinf(max_value):
            out[level] = max_value
            continue
        if not np.isfinite(max_value):
            continue
        out[level] = max_value + np.log(np.sum(np.exp(subset - max_value)))
    return out


def _profile_fe_manyway(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau: float,
    tol: float = 1e-12,
    maxiter: int = 2000,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    xb = np.asarray(exog, dtype=float) @ np.asarray(beta, dtype=float)
    if len(groups) == 0:
        return (), xb

    endog_arr = np.asarray(endog, dtype=float)
    offset = 0.0
    effects = [
        np.zeros(int(group.max()) + 1 if group.size else 0, dtype=float)
        for group in groups
    ]

    for _ in range(maxiter):
        previous_linear = xb + offset
        for effect, group in zip(effects, groups, strict=True):
            if effect.size:
                previous_linear += effect[group]

        for index, group in enumerate(groups):
            partial = xb + offset
            for other_index, other_group in enumerate(groups):
                if other_index == index:
                    continue
                other_effect = effects[other_index]
                if other_effect.size:
                    partial += other_effect[other_group]

            log_numerator_terms = np.full(partial.shape[0], -np.inf, dtype=float)
            positive_mask = endog_arr > 0.0
            log_numerator_terms[positive_mask] = (
                np.log(endog_arr[positive_mask]) + (1.0 - tau) * partial[positive_mask]
            )
            log_denominator_terms = (2.0 - tau) * partial
            log_numerator = _group_logsumexp(
                log_numerator_terms,
                group,
                effects[index].size,
            )
            log_denominator = _group_logsumexp(
                log_denominator_terms,
                group,
                effects[index].size,
            )
            updated = np.zeros_like(effects[index])
            valid = np.isfinite(log_numerator) & np.isfinite(log_denominator)
            updated[valid] = log_numerator[valid] - log_denominator[valid]
            if updated.size:
                shift = updated[0]
                updated = updated - shift
                offset += shift
            effects[index] = updated

        linear = xb + offset
        for effect, group in zip(effects, groups, strict=True):
            if effect.size:
                linear += effect[group]
        if not np.all(np.isfinite(linear)):
            raise RuntimeError("manyway_fe_profile_not_converged")
        if np.max(np.abs(linear - previous_linear)) < tol:
            return tuple(effects), linear

    raise RuntimeError("manyway_fe_profile_not_converged")


def _profile_cancel_fe_from_base(
    *,
    endog: np.ndarray,
    base_linear: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau: float,
    tol: float = 1e-12,
    maxiter: int = 2000,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    base = np.asarray(base_linear, dtype=float)
    if len(groups) == 0:
        return (), base.copy()

    endog_arr = np.asarray(endog, dtype=float)
    offset = 0.0
    effects = [
        np.zeros(int(group.max()) + 1 if group.size else 0, dtype=float)
        for group in groups
    ]

    positive_mask = endog_arr > 0.0
    log_positive_endog = np.log(endog_arr[positive_mask])
    use_constant_profile_terms = _env_flag_enabled(
        _PROFILE_CANCEL_CONSTANT_TERMS_FLAG
    )
    tau_value = float(tau)
    tau_is_one = tau_value == 1.0
    tau_is_two = tau_value == 2.0
    constant_numerators: list[np.ndarray | None] = [None] * len(groups)
    constant_denominators: list[np.ndarray | None] = [None] * len(groups)
    if use_constant_profile_terms:
        for index, group in enumerate(groups):
            n_group = int(effects[index].size)
            if tau_is_one:
                constant_numerators[index] = _group_logsumexp(
                    log_positive_endog,
                    np.asarray(group, dtype=int)[positive_mask],
                    n_group,
                )
            if tau_is_two:
                counts = np.bincount(np.asarray(group, dtype=int), minlength=n_group)
                denominator = np.full(n_group, -np.inf, dtype=float)
                populated = counts > 0
                denominator[populated] = np.log(counts[populated].astype(float))
                constant_denominators[index] = denominator

    for _ in range(maxiter):
        previous_linear = base + offset
        for effect, group in zip(effects, groups, strict=True):
            if effect.size:
                previous_linear += effect[group]

        for index, group in enumerate(groups):
            partial = base + offset
            for other_index, other_group in enumerate(groups):
                if other_index == index:
                    continue
                other_effect = effects[other_index]
                if other_effect.size:
                    partial += other_effect[other_group]

            log_numerator_terms = np.full(partial.shape[0], -np.inf, dtype=float)
            log_numerator_terms[positive_mask] = (
                log_positive_endog + (1.0 - tau) * partial[positive_mask]
            )
            log_numerator = (
                constant_numerators[index]
                if constant_numerators[index] is not None
                else _group_logsumexp(
                    log_numerator_terms,
                    group,
                    effects[index].size,
                )
            )
            log_denominator_terms = (2.0 - tau) * partial
            log_denominator = (
                constant_denominators[index]
                if constant_denominators[index] is not None
                else _group_logsumexp(
                    log_denominator_terms,
                    group,
                    effects[index].size,
                )
            )
            updated = np.zeros_like(effects[index])
            valid = np.isfinite(log_numerator) & np.isfinite(log_denominator)
            updated[valid] = log_numerator[valid] - log_denominator[valid]
            if updated.size:
                shift = updated[0]
                updated = updated - shift
                offset += shift
            effects[index] = updated

        linear = base + offset
        for effect, group in zip(effects, groups, strict=True):
            if effect.size:
                linear += effect[group]
        if np.max(np.abs(linear - previous_linear)) < tol:
            return tuple(effects), linear

    linear = base + offset
    for effect, group in zip(effects, groups, strict=True):
        if effect.size:
            linear += effect[group]
    return tuple(effects), linear


def _manyway_iols_covariance(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau: float,
    se_type: str,
    cluster_var: Any | None = None,
    df_resid_override: int | None = None,
    linear: np.ndarray | None = None,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if se_type not in {"iid", "hc0", "hc1", "hc2", "hc3", "cluster"}:
        raise NotImplementedError(
            "manyway iOLS covariance currently supports iid, HC0-HC3, "
            "and one-way cluster covariance."
        )
    size = np.asarray(beta, dtype=float).shape[0]

    def _finish(
        se: np.ndarray,
        covariance: np.ndarray,
        *,
        backend: str,
        status: str,
        fallback_reason: str | None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not return_info:
            return se, covariance
        return (
            se,
            covariance,
            _covariance_backend_metadata(
                backend=backend,
                status=status,
                fallback_reason=fallback_reason,
            ),
        )

    def _unavailable(
        reason: str,
        *,
        backend: str = "not_applicable",
        status: str = "withheld",
        fallback_reason: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        se_nan, covariance_nan = _nan_inference_arrays(size)
        return _finish(
            se_nan,
            covariance_nan,
            backend=backend,
            status=status,
            fallback_reason=fallback_reason or reason,
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        if linear is None:
            _, linear_arr = _profile_fe_manyway(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                tau=tau,
            )
        else:
            linear_arr = np.asarray(linear, dtype=float)
        stable_terms = _stable_manyway_weight_terms(
            endog=endog,
            linear=linear_arr,
            tau=tau,
        )
        if stable_terms is None:
            return _unavailable("stable_weight_terms_unavailable")
        a_scaled, score, bread_weight = stable_terms
        exog_arr = np.asarray(exog, dtype=float)
        Xa = _single_group_weighted_centered(
            exog_arr,
            groups,
            weights=a_scaled,
        )
        if Xa is None:
            Xa = _demean_manyway_weighted(
                exog_arr,
                groups,
                weights=a_scaled,
            )
        if not np.all(np.isfinite(Xa)):
            return _unavailable("weighted_within_transform_nonfinite")
        n_obs = int(np.asarray(endog).shape[0])
        k_params = int(np.asarray(beta).shape[0])
        single_group_arr = np.asarray(groups[0]) if len(groups) == 1 else None
        if (
            single_group_arr is not None
            and single_group_arr.size
            and np.all(single_group_arr == single_group_arr[0])
        ):
            fe_df = 0
        else:
            fe_df = sum(max(int(np.unique(group).size) - 1, 0) for group in groups)
        df_denom = (
            max(int(df_resid_override), 1)
            if df_resid_override is not None
            else max(n_obs - k_params - fe_df, 1)
        )
        cpp_covariance = _sandwich_covariance_cpp(
            design=Xa,
            bread_weight=bread_weight,
            score=score,
            se_type=se_type,
            df_denom=df_denom,
            cluster_var=cluster_var,
        )
        if cpp_covariance is not None:
            if cpp_covariance["status"] == "available":
                return _finish(
                    cpp_covariance["se"],
                    cpp_covariance["covariance"],
                    backend="cpp",
                    status="available",
                    fallback_reason=None,
                )
            return _unavailable(
                str(cpp_covariance["reason"]),
                backend="cpp",
                fallback_reason=None,
            )
        bread = (Xa * bread_weight[:, None]).T @ Xa
        score_X = Xa * score[:, None]
        if not np.all(np.isfinite(bread)) or not np.all(np.isfinite(score_X)):
            return _unavailable("nonfinite_sandwich_inputs")
        python_fallback_reason = (
            _cpp_covariance_fallback_reason() or "cpp_covariance_unavailable"
        )

    try:
        bread_cond = np.linalg.cond(bread)
    except np.linalg.LinAlgError:
        return _unavailable(
            "ill_conditioned_bread",
            backend="python_reference",
            status="fallback_python_reference",
            fallback_reason=python_fallback_reason,
        )
    if (not np.isfinite(bread_cond)) or bread_cond > _BREAD_CONDITION_LIMIT:
        return _unavailable(
            "ill_conditioned_bread",
            backend="python_reference",
            status="fallback_python_reference",
            fallback_reason=python_fallback_reason,
        )
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        return _unavailable(
            "ill_conditioned_bread",
            backend="python_reference",
            status="fallback_python_reference",
            fallback_reason=python_fallback_reason,
        )

    if se_type == "iid":
        positive_weight = bread_weight > 0.0
        if not np.any(positive_weight):
            return _unavailable(
                "no_positive_bread_weight",
                backend="python_reference",
                status="fallback_python_reference",
                fallback_reason=python_fallback_reason,
            )
        phi = float(
            np.sum((score[positive_weight] ** 2) / bread_weight[positive_weight])
            / df_denom
        )
        covariance = phi * bread_inv
        if not np.all(np.isfinite(covariance)):
            return _unavailable(
                "nonfinite_covariance",
                backend="python_reference",
                status="fallback_python_reference",
                fallback_reason=python_fallback_reason,
            )
        se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return _finish(
            se,
            covariance,
            backend="python_reference",
            status="fallback_python_reference",
            fallback_reason=python_fallback_reason,
        )

    if se_type == "hc1":
        meat = score_X.T @ score_X
        meat = meat * (n_obs / df_denom)
    elif se_type in {"hc2", "hc3"}:
        leverage = bread_weight * np.sum((Xa @ bread_inv) * Xa, axis=1)
        leverage = np.clip(leverage, 0.0, 1.0 - _LEVERAGE_FLOOR)
        if se_type == "hc2":
            adjustment = 1.0 / (1.0 - leverage)
        else:
            adjustment = 1.0 / (1.0 - leverage) ** 2
        score_X_adj = score_X * np.sqrt(adjustment)[:, None]
        meat = score_X_adj.T @ score_X_adj
    elif se_type == "cluster":
        if cluster_var is None:
            return _unavailable(
                "cluster_variable_missing",
                backend="python_reference",
                status="fallback_python_reference",
                fallback_reason=python_fallback_reason,
            )
        cluster_arr = np.asarray(cluster_var)
        if cluster_arr.shape[0] != n_obs:
            return _unavailable(
                "cluster_shape_mismatch",
                backend="python_reference",
                status="fallback_python_reference",
                fallback_reason=python_fallback_reason,
            )
        _, mapped = np.unique(cluster_arr, return_inverse=True)
        n_clusters = int(mapped.max()) + 1 if mapped.size else 0
        if n_clusters < 2:
            return _unavailable(
                "too_few_clusters",
                backend="python_reference",
                status="fallback_python_reference",
                fallback_reason=python_fallback_reason,
            )
        cluster_scores = np.zeros((n_clusters, size), dtype=float)
        for column in range(size):
            cluster_scores[:, column] = np.bincount(
                mapped,
                weights=score_X[:, column],
                minlength=n_clusters,
            )
        meat = cluster_scores.T @ cluster_scores
        meat = meat * (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / df_denom)
    else:
        meat = score_X.T @ score_X

    covariance = bread_inv @ meat @ bread_inv
    if not np.all(np.isfinite(covariance)):
        return _unavailable(
            "nonfinite_covariance",
            backend="python_reference",
            status="fallback_python_reference",
            fallback_reason=python_fallback_reason,
        )
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return _finish(
        se,
        covariance,
        backend="python_reference",
        status="fallback_python_reference",
        fallback_reason=python_fallback_reason,
    )


def _manyway_iols_iid_covariance_df_comparison(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau: float,
    linear: np.ndarray | None = None,
) -> dict[str, Any]:
    size = np.asarray(beta, dtype=float).shape[0]
    rank_info = _covariance_rank_diagnostics(exog=exog, groups=groups)

    def _unavailable(reason: str) -> dict[str, Any]:
        nan_se, nan_cov = _nan_inference_arrays(size)
        return {
            "status": "unavailable",
            "reason": reason,
            "rank_info": rank_info,
            "additive_df_resid": rank_info.get("df_resid_additive"),
            "additive_se": nan_se,
            "additive_covariance": nan_cov,
            "exact_df_resid": rank_info.get("df_resid_exact"),
            "exact_se": nan_se.copy(),
            "exact_covariance": nan_cov.copy(),
            "exact_available": False,
        }

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        if linear is None:
            _, linear_arr = _profile_fe_manyway(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                tau=tau,
            )
        else:
            linear_arr = np.asarray(linear, dtype=float)
        stable_terms = _stable_manyway_weight_terms(
            endog=endog,
            linear=linear_arr,
            tau=tau,
        )
        if stable_terms is None:
            return _unavailable("stable_weight_terms_unavailable")
        a_scaled, score, bread_weight = stable_terms
        Xa = _demean_manyway_weighted(
            np.asarray(exog, dtype=float),
            groups,
            weights=a_scaled,
        )
        if not np.all(np.isfinite(Xa)):
            return _unavailable("nonfinite_absorbed_design")
        bread = (Xa * bread_weight[:, None]).T @ Xa
        if not np.all(np.isfinite(bread)):
            return _unavailable("nonfinite_bread")

    try:
        bread_cond = np.linalg.cond(bread)
    except np.linalg.LinAlgError:
        return _unavailable("bread_condition_failed")
    if (not np.isfinite(bread_cond)) or bread_cond > _BREAD_CONDITION_LIMIT:
        return _unavailable("ill_conditioned_bread")
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        return _unavailable("bread_inversion_failed")

    positive_weight = bread_weight > 0.0
    if not np.any(positive_weight):
        return _unavailable("no_positive_bread_weight")
    numerator = float(
        np.sum((score[positive_weight] ** 2) / bread_weight[positive_weight])
    )
    additive_df = int(rank_info["df_resid_additive"])
    additive_covariance = (numerator / max(additive_df, 1)) * bread_inv
    if not np.all(np.isfinite(additive_covariance)):
        return _unavailable("nonfinite_additive_covariance")
    additive_se = np.sqrt(np.maximum(np.diag(additive_covariance), 0.0))

    exact_available = (
        rank_info.get("rank_status") == "exact"
        and rank_info.get("df_resid_exact") is not None
    )
    exact_df = None if not exact_available else int(rank_info["df_resid_exact"])
    if exact_available:
        exact_covariance = (numerator / max(int(exact_df), 1)) * bread_inv
        if not np.all(np.isfinite(exact_covariance)):
            return _unavailable("nonfinite_exact_covariance")
        exact_se = np.sqrt(np.maximum(np.diag(exact_covariance), 0.0))
    else:
        exact_se, exact_covariance = _nan_inference_arrays(size)

    return {
        "status": "computed",
        "reason": None,
        "rank_info": rank_info,
        "bread_condition": float(bread_cond),
        "pearson_numerator": numerator,
        "additive_df_resid": additive_df,
        "additive_se": additive_se,
        "additive_covariance": additive_covariance,
        "exact_df_resid": exact_df,
        "exact_se": exact_se,
        "exact_covariance": exact_covariance,
        "exact_available": bool(exact_available),
    }


def _dummy_droplast(group: np.ndarray, n_groups: int) -> np.ndarray:
    if n_groups <= 1:
        return np.zeros((len(group), 0), dtype=float)
    design = np.zeros((len(group), n_groups - 1), dtype=float)
    mask = group < (n_groups - 1)
    rows = np.nonzero(mask)[0]
    if len(rows) > 0:
        design[rows, group[rows]] = 1.0
    return design


def _build_recover_design(
    fe_blocks: dict[str, np.ndarray],
    recover_fe: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, dict[str, int | slice | np.ndarray]]]:
    matrices: list[np.ndarray] = []
    meta: dict[str, dict[str, int | slice | np.ndarray]] = {}
    start = 0
    n_obs = len(next(iter(fe_blocks.values()))) if fe_blocks else 0
    for name in recover_fe:
        values = np.asarray(fe_blocks[name])
        unique, inverse = np.unique(values, return_inverse=True)
        inverse = _as_dense_group_array(inverse)
        design = _dummy_droplast(inverse, len(unique))
        stop = start + design.shape[1]
        meta[name] = {
            "levels": inverse,
            "n_levels": len(unique),
            "slice": slice(start, stop),
        }
        matrices.append(design)
        start = stop
    if matrices:
        return np.column_stack(matrices), meta
    return np.zeros((n_obs, 0), dtype=float), meta


def _intercept_position(
    kept_x_idx: np.ndarray,
    intercept_idx: int | None,
) -> int | None:
    if intercept_idx is None:
        return None
    matches = np.flatnonzero(np.asarray(kept_x_idx, dtype=int) == int(intercept_idx))
    if matches.size == 0:
        return None
    return int(matches[0])


def _slope_positions(
    kept_x_idx: np.ndarray,
    intercept_position: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    compact_positions = np.arange(len(kept_x_idx), dtype=int)
    if intercept_position is not None:
        compact_positions = compact_positions[compact_positions != intercept_position]
    return compact_positions, np.asarray(kept_x_idx, dtype=int)[compact_positions]


def _remap_group(
    values: np.ndarray,
    kept_obs_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    kept_values = np.asarray(values)[kept_obs_mask]
    _, mapped = np.unique(kept_values, return_inverse=True)
    mapped_dense = _as_dense_group_array(mapped)
    return mapped_dense, int(mapped.max()) + 1 if mapped.size else 0


def _profiled_partitioned_linear_predictor(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    kept_obs_mask: np.ndarray,
    tau: float,
    profile_fe,
) -> tuple[float, ...]:
    if len(cancel_fe) + len(recover_fe) != 2:
        raise NotImplementedError(
            "The fd_partitioned solver layer only profiles two-FE fitted values."
        )
    if not recover_fe:
        raise RuntimeError(
            "Fullcancel fd_partitioned fits should use retained_index-based "
            "linear predictors, not profile_fe reconstruction."
        )

    cancel_name = cancel_fe[0]
    recover_name = recover_fe[0]

    group_i, n_i = _remap_group(fe_blocks[cancel_name], kept_obs_mask)
    group_t, n_t = _remap_group(fe_blocks[recover_name], kept_obs_mask)
    endog_kept = endog[kept_obs_mask]
    exog_kept = exog[kept_obs_mask]
    alpha_i, gamma_t = profile_fe(
        endog_kept,
        exog_kept,
        beta,
        group_i,
        group_t,
        n_i,
        n_t,
        tau=tau,
    )
    linear = np.full(kept_obs_mask.shape[0], np.nan, dtype=float)
    linear[kept_obs_mask] = exog_kept @ beta + alpha_i[group_i] + gamma_t[group_t]
    return tuple(float(value) for value in linear)


def _profiled_partitioned_generic_linear_predictor(
    *,
    kept_obs_mask: np.ndarray,
    profiled_linear_kept: np.ndarray,
) -> tuple[float, ...]:
    linear = np.full(kept_obs_mask.shape[0], np.nan, dtype=float)
    linear[kept_obs_mask] = np.asarray(profiled_linear_kept, dtype=float)
    return tuple(float(value) for value in linear)


def _partitioned_mixed_manyway_covariance(
    *,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_blocks_pre: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    kept_x_idx: np.ndarray,
    intercept_idx: int | None,
    original_param_count: int,
    tau: float,
    profiled_linear_kept: np.ndarray,
    se_type: str,
    cluster_var: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if se_type not in {"iid", "hc0", "hc1", "hc2", "hc3", "cluster"}:
        raise NotImplementedError(
            "mixed covariance currently supports iid, HC0-HC3, and one-way cluster."
        )

    def _nan_result(
        reason: str,
        *,
        bread_weight_zero_fraction: float | None = None,
        bread_condition: float | None = None,
        n_clusters: int | None = None,
        covariance_backend: str = "not_applicable",
        covariance_backend_status: str = "withheld",
        covariance_backend_fallback_reason: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        nan_se, nan_cov = _nan_inference_arrays(int(original_param_count))
        return (
            nan_se,
            nan_cov,
            _mixed_hc0_status(
                supported=False,
                reason=reason,
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=bread_condition,
                n_clusters=n_clusters,
                covariance_backend=covariance_backend,
                covariance_backend_status=covariance_backend_status,
                covariance_backend_fallback_reason=(
                    covariance_backend_fallback_reason or reason
                ),
            ),
        )

    intercept_position = _intercept_position(kept_x_idx, intercept_idx)
    compact_slope_idx, original_slope_idx = _slope_positions(
        kept_x_idx,
        intercept_position,
    )
    x_slopes = np.asarray(exog_pre, dtype=float)[:, compact_slope_idx]
    cancel_groups = tuple(np.asarray(fe_blocks_pre[name]) for name in cancel_fe)
    recover_groups = tuple(np.asarray(fe_blocks_pre[name]) for name in recover_fe)
    linear = np.asarray(profiled_linear_kept, dtype=float)
    stable_terms = _stable_manyway_weight_terms(
        endog=endog_pre,
        linear=linear,
        tau=tau,
    )
    if stable_terms is None:
        return _nan_result("mixed_hc0_weight_terms_unavailable")
    a_scaled, score, bread_weight = stable_terms
    bread_weight_zero_fraction = float(np.mean(bread_weight <= 0.0))

    cpp_projection = _mixed_covariance_projection_cpp(
        x_slopes=x_slopes,
        cancel_groups=cancel_groups,
        recover_groups=recover_groups,
        demean_weights=a_scaled,
        bread_weight=bread_weight,
        score=score,
        se_type=se_type,
        cluster_var=cluster_var,
    )
    if cpp_projection is not None:
        bread_cond_cpp = cpp_projection["bread_condition"]
        bread_cond_value = None if bread_cond_cpp is None else float(bread_cond_cpp)
        n_clusters_cpp = cpp_projection["n_clusters"]
        n_clusters_value = None if n_clusters_cpp is None else int(n_clusters_cpp)
        if cpp_projection["status"] != "available":
            reason = str(cpp_projection["reason"])
            reason_map = {
                "nonfinite_sandwich_inputs": "mixed_hc0_nonfinite_sandwich_inputs",
                "nonfinite_bread": "mixed_hc0_nonfinite_sandwich_inputs",
                "nonfinite_meat": "mixed_hc0_nonfinite_meat",
                "ill_conditioned_bread": "mixed_hc0_ill_conditioned_bread",
                "bread_inversion_failed": "mixed_hc0_bread_inversion_failed",
                "nonfinite_covariance": "mixed_hc0_nonfinite_covariance",
                "no_positive_bread_weight": "mixed_iid_no_positive_bread_weight",
                "no_estimable_parameters": "mixed_hc0_no_estimable_slopes",
                "shape_mismatch": "mixed_hc0_nonfinite_sandwich_inputs",
            }
            return _nan_result(
                reason_map.get(reason, reason),
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=bread_cond_value,
                n_clusters=n_clusters_value,
                covariance_backend="cpp",
                covariance_backend_fallback_reason=None,
            )

        slope_covariance = np.asarray(cpp_projection["covariance"], dtype=float)
        slope_se = np.sqrt(np.maximum(np.diag(slope_covariance), 0.0))
        original_size = int(original_param_count)
        public_se = np.full(original_size, np.nan, dtype=float)
        public_covariance = np.full((original_size, original_size), np.nan, dtype=float)
        public_se[original_slope_idx] = slope_se
        for i, orig_i in enumerate(original_slope_idx):
            for j, orig_j in enumerate(original_slope_idx):
                public_covariance[int(orig_i), int(orig_j)] = slope_covariance[i, j]
        return (
            public_se,
            public_covariance,
            _mixed_hc0_status(
                supported=True,
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=bread_cond_value,
                n_clusters=n_clusters_value if se_type == "cluster" else None,
                covariance_backend="cpp",
                covariance_backend_status="available",
                covariance_backend_fallback_reason=None,
            ),
        )

    python_projection_fallback_reason = (
        _cpp_covariance_fallback_reason(
            ("sandwich_covariance", "mixed_covariance_projection")
        )
        or "cpp_mixed_projection_unavailable"
    )
    recover_design, _ = _build_recover_design(fe_blocks_pre, recover_fe)
    design = np.column_stack([x_slopes, recover_design])
    design_absorbed = _demean_manyway_weighted(
        design,
        cancel_groups,
        weights=a_scaled,
        use_cpp=False,
    )
    if not np.all(np.isfinite(design_absorbed)):
        return _nan_result(
            "mixed_hc0_nonfinite_absorbed_design",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_projection_fallback_reason,
        )
    slope_design = design_absorbed[:, : x_slopes.shape[1]]
    recover_absorbed = design_absorbed[:, x_slopes.shape[1] :]
    root_w = np.sqrt(bread_weight)
    if not np.all(np.isfinite(root_w)):
        return _nan_result(
            "mixed_hc0_nonfinite_weight_root",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_projection_fallback_reason,
        )
    slope_resid = slope_design.copy()
    if recover_absorbed.shape[1] > 0:
        weighted_recover = recover_absorbed * root_w[:, None]
        if not np.all(np.isfinite(weighted_recover)):
            return _nan_result(
                "mixed_hc0_nonfinite_weighted_recover_design",
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                covariance_backend="python_reference",
                covariance_backend_status="fallback_python_reference",
                covariance_backend_fallback_reason=(python_projection_fallback_reason),
            )
        for column in range(slope_design.shape[1]):
            try:
                gamma, _ = solve_ols_qr_dropcollinear(
                    weighted_recover,
                    slope_design[:, column] * root_w,
                    tol=1e-10,
                )
            except (np.linalg.LinAlgError, ValueError):
                return _nan_result(
                    "mixed_hc0_recover_projection_failed",
                    bread_weight_zero_fraction=bread_weight_zero_fraction,
                    covariance_backend="python_reference",
                    covariance_backend_status="fallback_python_reference",
                    covariance_backend_fallback_reason=(
                        python_projection_fallback_reason
                    ),
                )
            slope_resid[:, column] = slope_design[:, column] - recover_absorbed @ gamma
    weighted_slope_design = slope_resid * root_w[:, None]
    if not np.all(np.isfinite(weighted_slope_design)):
        return _nan_result(
            "mixed_hc0_nonfinite_weighted_slope_design",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_projection_fallback_reason,
        )
    try:
        kept_columns, _, _ = rank_reveal_qr(
            weighted_slope_design,
            tol=1e-10,
            protected=tuple(range(slope_design.shape[1])),
        )
    except ValueError:
        return _nan_result(
            "mixed_hc0_rank_reveal_failed",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_projection_fallback_reason,
        )
    if len(kept_columns) == 0:
        return _nan_result(
            "mixed_hc0_no_estimable_slopes",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_projection_fallback_reason,
        )
    slope_kept = slope_resid[:, kept_columns]
    cancel_df = sum(max(int(np.unique(group).size) - 1, 0) for group in cancel_groups)
    recover_df = int(recover_design.shape[1])
    n_obs = int(endog_pre.shape[0])
    df_denom = max(
        n_obs - int(len(kept_columns)) - cancel_df - recover_df,
        1,
    )
    cpp_covariance = _sandwich_covariance_cpp(
        design=slope_kept,
        bread_weight=bread_weight,
        score=score,
        se_type=se_type,
        df_denom=df_denom,
        cluster_var=cluster_var,
        symmetrize=True,
    )
    if cpp_covariance is not None:
        bread_cond_cpp = cpp_covariance["bread_condition"]
        bread_cond_value = None if bread_cond_cpp is None else float(bread_cond_cpp)
        if cpp_covariance["status"] != "available":
            reason = str(cpp_covariance["reason"])
            reason_map = {
                "nonfinite_sandwich_inputs": "mixed_hc0_nonfinite_sandwich_inputs",
                "nonfinite_bread": "mixed_hc0_nonfinite_sandwich_inputs",
                "nonfinite_meat": "mixed_hc0_nonfinite_meat",
                "ill_conditioned_bread": "mixed_hc0_ill_conditioned_bread",
                "bread_inversion_failed": "mixed_hc0_bread_inversion_failed",
                "nonfinite_covariance": "mixed_hc0_nonfinite_covariance",
                "no_positive_bread_weight": "mixed_iid_no_positive_bread_weight",
            }
            return _nan_result(
                reason_map.get(reason, reason),
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=bread_cond_value,
                n_clusters=(
                    None
                    if cpp_covariance["n_clusters"] is None
                    else int(cpp_covariance["n_clusters"])
                ),
                covariance_backend="cpp",
                covariance_backend_fallback_reason=None,
            )
        covariance_kept = cpp_covariance["covariance"]
        slope_covariance = np.full(
            (x_slopes.shape[1], x_slopes.shape[1]),
            np.nan,
            dtype=float,
        )
        slope_covariance[np.ix_(kept_columns, kept_columns)] = covariance_kept
        slope_se = np.sqrt(np.maximum(np.diag(slope_covariance), 0.0))

        original_size = int(original_param_count)
        public_se = np.full(original_size, np.nan, dtype=float)
        public_covariance = np.full((original_size, original_size), np.nan, dtype=float)
        public_se[original_slope_idx] = slope_se
        for i, orig_i in enumerate(original_slope_idx):
            for j, orig_j in enumerate(original_slope_idx):
                public_covariance[int(orig_i), int(orig_j)] = slope_covariance[i, j]
        return (
            public_se,
            public_covariance,
            _mixed_hc0_status(
                supported=True,
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=bread_cond_value,
                n_clusters=(
                    None
                    if se_type != "cluster"
                    else int(np.unique(np.asarray(cluster_var)).size)
                ),
                covariance_backend="cpp",
                covariance_backend_status="available",
                covariance_backend_fallback_reason=None,
            ),
        )
    python_sandwich_fallback_reason = (
        _cpp_covariance_fallback_reason() or python_projection_fallback_reason
    )
    bread = (slope_kept * bread_weight[:, None]).T @ slope_kept
    score_design = slope_kept * score[:, None]
    if not np.all(np.isfinite(bread)) or not np.all(np.isfinite(score_design)):
        return _nan_result(
            "mixed_hc0_nonfinite_sandwich_inputs",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_sandwich_fallback_reason,
        )
    try:
        bread_sym = 0.5 * (bread + bread.T)
        bread_cond = np.linalg.cond(bread_sym)
        if (not np.isfinite(bread_cond)) or bread_cond > _BREAD_CONDITION_LIMIT:
            return _nan_result(
                "mixed_hc0_ill_conditioned_bread",
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=float(bread_cond),
                covariance_backend="python_reference",
                covariance_backend_status="fallback_python_reference",
                covariance_backend_fallback_reason=python_sandwich_fallback_reason,
            )
        bread_inv = np.linalg.inv(bread_sym)
    except np.linalg.LinAlgError:
        return _nan_result(
            "mixed_hc0_bread_inversion_failed",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_sandwich_fallback_reason,
        )
    meat = score_design.T @ score_design
    if not np.all(np.isfinite(meat)):
        return _nan_result(
            "mixed_hc0_nonfinite_meat",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            bread_condition=float(bread_cond),
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_sandwich_fallback_reason,
        )
    if se_type == "iid":
        positive_weight = bread_weight > 0.0
        if not np.any(positive_weight):
            return _nan_result(
                "mixed_iid_no_positive_bread_weight",
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=float(bread_cond),
                covariance_backend="python_reference",
                covariance_backend_status="fallback_python_reference",
                covariance_backend_fallback_reason=python_sandwich_fallback_reason,
            )
        phi = float(
            np.sum((score[positive_weight] ** 2) / bread_weight[positive_weight])
            / df_denom
        )
        covariance_kept = phi * bread_inv
    elif se_type == "hc1":
        covariance_kept = bread_inv @ (meat * (n_obs / df_denom)) @ bread_inv.T
    elif se_type in {"hc2", "hc3"}:
        leverage = bread_weight * np.sum(
            (slope_kept @ bread_inv) * slope_kept,
            axis=1,
        )
        leverage = np.clip(leverage, 0.0, 1.0 - _LEVERAGE_FLOOR)
        if se_type == "hc2":
            adjustment = 1.0 / (1.0 - leverage)
        else:
            adjustment = 1.0 / (1.0 - leverage) ** 2
        score_design_adj = score_design * np.sqrt(adjustment)[:, None]
        meat_adj = score_design_adj.T @ score_design_adj
        covariance_kept = bread_inv @ meat_adj @ bread_inv.T
    elif se_type == "cluster":
        if cluster_var is None:
            return _nan_result(
                "cluster_variable_missing",
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=float(bread_cond),
                covariance_backend="python_reference",
                covariance_backend_status="fallback_python_reference",
                covariance_backend_fallback_reason=python_sandwich_fallback_reason,
            )
        cluster_arr = np.asarray(cluster_var)
        if cluster_arr.shape[0] != n_obs:
            return _nan_result(
                "cluster_shape_mismatch",
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=float(bread_cond),
                covariance_backend="python_reference",
                covariance_backend_status="fallback_python_reference",
                covariance_backend_fallback_reason=python_sandwich_fallback_reason,
            )
        _, mapped = np.unique(cluster_arr, return_inverse=True)
        n_clusters = int(mapped.max()) + 1 if mapped.size else 0
        if n_clusters < 2:
            return _nan_result(
                "too_few_clusters",
                bread_weight_zero_fraction=bread_weight_zero_fraction,
                bread_condition=float(bread_cond),
                n_clusters=n_clusters,
                covariance_backend="python_reference",
                covariance_backend_status="fallback_python_reference",
                covariance_backend_fallback_reason=python_sandwich_fallback_reason,
            )
        cluster_scores = np.zeros((n_clusters, len(kept_columns)), dtype=float)
        for column in range(len(kept_columns)):
            cluster_scores[:, column] = np.bincount(
                mapped,
                weights=score_design[:, column],
                minlength=n_clusters,
            )
        meat_cluster = cluster_scores.T @ cluster_scores
        covariance_kept = (
            bread_inv
            @ (
                meat_cluster
                * (n_clusters / max(n_clusters - 1, 1))
                * ((n_obs - 1) / df_denom)
            )
            @ bread_inv.T
        )
    else:
        covariance_kept = bread_inv @ meat @ bread_inv.T
    covariance_kept = 0.5 * (covariance_kept + covariance_kept.T)
    if not np.all(np.isfinite(covariance_kept)):
        return _nan_result(
            "mixed_hc0_nonfinite_covariance",
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            bread_condition=float(bread_cond),
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_sandwich_fallback_reason,
        )

    slope_covariance = np.full(
        (x_slopes.shape[1], x_slopes.shape[1]),
        np.nan,
        dtype=float,
    )
    slope_covariance[np.ix_(kept_columns, kept_columns)] = covariance_kept
    slope_se = np.sqrt(np.maximum(np.diag(slope_covariance), 0.0))

    original_size = int(original_param_count)
    public_se = np.full(original_size, np.nan, dtype=float)
    public_covariance = np.full((original_size, original_size), np.nan, dtype=float)
    public_se[original_slope_idx] = slope_se
    for i, orig_i in enumerate(original_slope_idx):
        for j, orig_j in enumerate(original_slope_idx):
            public_covariance[int(orig_i), int(orig_j)] = slope_covariance[i, j]
    return (
        public_se,
        public_covariance,
        _mixed_hc0_status(
            supported=True,
            bread_weight_zero_fraction=bread_weight_zero_fraction,
            bread_condition=float(bread_cond),
            n_clusters=(
                None
                if se_type != "cluster"
                else int(np.unique(np.asarray(cluster_var)).size)
            ),
            covariance_backend="python_reference",
            covariance_backend_status="fallback_python_reference",
            covariance_backend_fallback_reason=python_sandwich_fallback_reason,
        ),
    )


def _partitioned_mixed_manyway_bread_diagnostics(
    *,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_blocks_pre: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    kept_x_idx: np.ndarray,
    intercept_idx: int | None,
    tau: float,
    profiled_linear_kept: np.ndarray,
) -> dict[str, Any]:
    intercept_position = _intercept_position(kept_x_idx, intercept_idx)
    compact_slope_idx, _ = _slope_positions(kept_x_idx, intercept_position)
    if compact_slope_idx.size == 0:
        return {
            "status": "unavailable",
            "reason": "no_estimable_slope_parameters",
            "route": "mixed",
            "tau": float(tau),
            "parameter_count": 0,
        }
    x_slopes = np.asarray(exog_pre, dtype=float)[:, compact_slope_idx]
    recover_design, _ = _build_recover_design(fe_blocks_pre, recover_fe)
    cancel_groups = tuple(np.asarray(fe_blocks_pre[name]) for name in cancel_fe)
    stable_terms = _stable_manyway_weight_terms(
        endog=endog_pre,
        linear=np.asarray(profiled_linear_kept, dtype=float),
        tau=tau,
    )
    if stable_terms is None:
        return {
            "status": "unavailable",
            "reason": "mixed_weight_terms_unavailable",
            "route": "mixed",
            "tau": float(tau),
            "parameter_count": int(compact_slope_idx.size),
        }
    a_scaled, score, bread_weight = stable_terms
    design = np.column_stack([x_slopes, recover_design])
    design_absorbed = _demean_manyway_weighted(
        design,
        cancel_groups,
        weights=a_scaled,
        use_cpp=False,
    )
    if not np.all(np.isfinite(design_absorbed)):
        return {
            "status": "unavailable",
            "reason": "mixed_nonfinite_absorbed_design",
            "route": "mixed",
            "tau": float(tau),
            "parameter_count": int(compact_slope_idx.size),
        }
    slope_design = design_absorbed[:, : x_slopes.shape[1]]
    recover_absorbed = design_absorbed[:, x_slopes.shape[1] :]
    root_w = np.sqrt(bread_weight)
    if not np.all(np.isfinite(root_w)):
        return {
            "status": "unavailable",
            "reason": "mixed_nonfinite_weight_root",
            "route": "mixed",
            "tau": float(tau),
            "parameter_count": int(compact_slope_idx.size),
        }
    slope_resid = slope_design.copy()
    if recover_absorbed.shape[1] > 0:
        weighted_recover = recover_absorbed * root_w[:, None]
        if not np.all(np.isfinite(weighted_recover)):
            return {
                "status": "unavailable",
                "reason": "mixed_nonfinite_weighted_recover_design",
                "route": "mixed",
                "tau": float(tau),
                "parameter_count": int(compact_slope_idx.size),
            }
        for column in range(slope_design.shape[1]):
            try:
                gamma, _ = solve_ols_qr_dropcollinear(
                    weighted_recover,
                    slope_design[:, column] * root_w,
                    tol=1e-10,
                )
            except (np.linalg.LinAlgError, ValueError):
                return {
                    "status": "unavailable",
                    "reason": "mixed_recover_projection_failed",
                    "route": "mixed",
                    "tau": float(tau),
                    "parameter_count": int(compact_slope_idx.size),
                }
            slope_resid[:, column] = slope_design[:, column] - recover_absorbed @ gamma
    weighted_slope_design = slope_resid * root_w[:, None]
    try:
        kept_columns, _, _ = rank_reveal_qr(
            weighted_slope_design,
            tol=1e-10,
            protected=tuple(range(slope_design.shape[1])),
        )
    except ValueError:
        return {
            "status": "unavailable",
            "reason": "mixed_rank_reveal_failed",
            "route": "mixed",
            "tau": float(tau),
            "parameter_count": int(compact_slope_idx.size),
        }
    if len(kept_columns) == 0:
        return {
            "status": "unavailable",
            "reason": "mixed_no_estimable_slopes",
            "route": "mixed",
            "tau": float(tau),
            "parameter_count": int(compact_slope_idx.size),
        }
    observed_weight = _as_cov1_observed_weight_from_score(
        expected_weight=bread_weight,
        score=score,
        tau=tau,
    )
    diagnostics = _as_cov1_bread_diagnostics_from_design(
        route="mixed",
        tau=tau,
        design=slope_resid[:, kept_columns],
        expected_weight=bread_weight,
        observed_weight=observed_weight,
        score=score,
        observed_weight_definition="mixed_profiled_route_score_identity_comparator",
    )
    diagnostics["mixed_kept_slope_columns"] = tuple(
        int(value) for value in kept_columns
    )
    return diagnostics


def _partitioned_mixed_manyway_hc0(
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _partitioned_mixed_manyway_covariance(**kwargs, se_type="hc0")


def solve_fd_oneway(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    group_i: np.ndarray | None,
    fe_name: str | None,
    se_type: str,
    cluster_var: Any,
    anderson_m: int,
    fe_blocks: Mapping[str, Any] | None,
    collinearity_mode: str,
    separation_mode: str,
    iols_taupath=None,
    profile_fe=None,
    research_root: str | Path | None = None,
) -> SolverResponse:
    if request.tau_mode == "path":
        if iols_taupath is not None:
            raise RuntimeError("Public fd_oneway tau='path' is package-kernel only.")
        return _solve_fd_oneway_package(
            request=request,
            endog=endog,
            exog=exog,
            group_i=group_i,
            fe_name=fe_name,
            se_type=se_type,
            cluster_var=cluster_var,
            anderson_m=anderson_m,
            fe_blocks=fe_blocks,
            collinearity_mode=collinearity_mode,
            separation_mode=separation_mode,
        )

    tau_target = _require_fixed_tau(request)
    if iols_taupath is None:
        return _solve_fd_oneway_package(
            request=request,
            endog=endog,
            exog=exog,
            group_i=group_i,
            fe_name=fe_name,
            se_type=se_type,
            cluster_var=cluster_var,
            anderson_m=anderson_m,
            fe_blocks=fe_blocks,
            collinearity_mode=collinearity_mode,
            separation_mode=separation_mode,
        )
    if profile_fe is None or research_root is None:
        raise RuntimeError(
            "Research fd_oneway solve requires profile_fe and research_root."
        )
    anchor_se_type = "hc0" if se_type == "iid" else se_type
    beta, se, converged = iols_taupath(
        endog,
        exog,
        tau_target=tau_target,
        group_i=group_i,
        group_t=None,
        n_steps=_TAU_PATH_STEPS,
        anderson_m=anderson_m,
        se_type=anchor_se_type,
        cluster_var=cluster_var,
    )
    return _build_fd_oneway_response(
        request=request,
        endog=endog,
        exog=exog,
        group_i=group_i,
        fe_name=fe_name,
        se_type=se_type,
        tau_fit=tau_target,
        beta=np.asarray(beta, dtype=float),
        se=np.asarray(se, dtype=float),
        converged=bool(converged),
        profile_fe=profile_fe,
        research_root=research_root,
        fe_blocks=fe_blocks,
        collinearity_mode=collinearity_mode,
        separation_mode=separation_mode,
        solver_details_extra={
            "research_entrypoint": "estimators.iols_taupath",
            "anderson_m": anderson_m,
            "se_type": se_type,
            "research_anchor_se_type": anchor_se_type,
            "tau_path_steps": _TAU_PATH_STEPS,
        },
        tau_details_extra={"tau_fit": tau_target},
    )


def _build_fd_oneway_response(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    group_i: np.ndarray | None,
    fe_name: str | None,
    se_type: str,
    tau_fit: float,
    beta: np.ndarray,
    se: np.ndarray,
    converged: bool,
    profile_fe,
    research_root: str | Path,
    fe_blocks: Mapping[str, Any] | None,
    collinearity_mode: str,
    separation_mode: str,
    solver_details_extra: Mapping[str, Any],
    tau_details_extra: Mapping[str, Any],
) -> SolverResponse:
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    linear = _fd_oneway_linear_predictor(
        endog=endog,
        exog=exog,
        beta=beta,
        group_i=group_i,
        tau=tau_fit,
        profile_fe=profile_fe,
    )
    fitted = exp_with_float64_overflow_guard(linear)
    fe_support = diagnose_fe_support(endog=endog, fe_blocks=fe_blocks)
    inference_warning = _fd_oneway_inference_warning(
        request=request,
        group_i=group_i,
        se_type=se_type,
    )
    instability_metrics = _post_estimation_instability_metrics(
        endog=endog,
        linear=linear,
        tau=tau_fit,
    )
    instability_warning = _post_estimation_instability_warning(
        metrics=instability_metrics,
    )
    warning_details = _merge_warning_details(
        inference_warning,
        instability_warning,
    )
    covariance = None
    covariance_source = "research_anchor"
    if se_type == "iid" and np.all(np.isfinite(beta)):
        groups: tuple[np.ndarray, ...]
        if group_i is None:
            groups = (np.zeros(endog.shape[0], dtype=np.int32),)
        else:
            groups = (
                _as_dense_group_array(np.unique(group_i, return_inverse=True)[1]),
            )
        slope_idx = np.flatnonzero(~_constant_column_mask(exog))
        se = np.full_like(beta, np.nan, dtype=float)
        covariance = np.full((beta.size, beta.size), np.nan, dtype=float)
        if slope_idx.size > 0:
            se_slope, covariance_slope = _manyway_iols_covariance(
                endog=endog,
                exog=exog[:, slope_idx],
                beta=beta[slope_idx],
                groups=groups,
                tau=tau_fit,
                se_type="iid",
                linear=linear,
            )
            se[slope_idx] = se_slope
            for compact_i, original_i in enumerate(slope_idx):
                for compact_j, original_j in enumerate(slope_idx):
                    covariance[int(original_i), int(original_j)] = covariance_slope[
                        compact_i,
                        compact_j,
                    ]
        covariance_source = "package_iid_expected_bread"
    if bool(instability_warning["instability_withheld_inference"]):
        se = np.full_like(se, np.nan, dtype=float)
        covariance = tuple(
            tuple(float("nan") for _ in range(se.size)) for _ in range(se.size)
        )
        covariance_source = "post_estimation_instability_withheld"
    estimable_param_count = int(np.isfinite(se).sum())
    params, bse, tvalues, pvalues = _normal_stats(request.param_names, beta, se)

    fe_df = 0
    if group_i is not None:
        fe_df = max(int(np.unique(group_i).size) - 1, 0)

    preprocess_contract = scope_only_contract(
        step_name="fd_oneway_scope_check",
        n_obs=int(endog.shape[0]),
        status="pass",
        fe_blocks=fe_blocks,
        step_details={
            "fe_dim_case": request.fe_dim_case,
            "fe_topology": request.fe_topology,
        },
        singleton_checked=group_i is not None,
        has_singletons=False,
        fe_name=fe_name,
        collinearity_mode=collinearity_mode,
        collinearity_status="not_applied",
    )
    preprocess_log = _prepend_routing_log(
        preprocess_contract.preprocess_log(),
        request,
    )

    return SolverResponse(
        request=request,
        params=params,
        bse=bse,
        pvalues=pvalues,
        tvalues=tvalues,
        fittedvalues=tuple(float(value) for value in fitted),
        resid_response=tuple(float(value) for value in (endog - fitted)),
        nobs=int(endog.shape[0]),
        df_model=estimable_param_count,
        df_resid=max(int(endog.shape[0]) - estimable_param_count - fe_df, 0),
        family=request.family,
        tau=tau_fit,
        vcov_type=request.vcov_hc_type or request.vcov_type,
        converged=bool(converged),
        fe_method=request.fe_method,
        fe_partition=request.fe_topology,
        partition_warning=_partition_warning_from_request(request),
        singleton_info=preprocess_contract.singleton.as_result_mapping(),
        separation_info={"mode": separation_mode, "status": "not_applied"},
        collinearity_info=preprocess_contract.collinearity.as_result_mapping(),
        preprocess_log=preprocess_log,
        solver_details={
            "research_root": str(Path(research_root)),
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
            "covariance_source": covariance_source,
            **dict(solver_details_extra),
        },
        tau_details=dict(tau_details_extra),
        diagnostic_details={
            "linear_predictor": tuple(float(value) for value in linear),
            "df_resid_formula": "nobs - n_estimable_params - fe_df",
            "n_estimable_params": estimable_param_count,
            "fe_df": fe_df,
            "fe_support": fe_support.as_mapping(),
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
            "research_root": str(Path(research_root)),
            **instability_metrics,
            **warning_details,
        },
        covariance=covariance,
    )


def _solve_fd_oneway_package(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    group_i: np.ndarray | None,
    fe_name: str | None,
    se_type: str,
    cluster_var: Any,
    anderson_m: int,
    fe_blocks: Mapping[str, Any] | None,
    collinearity_mode: str,
    separation_mode: str,
) -> SolverResponse:
    from .kernel import IOLSKernelControl, IOLSKernelSpec, solve_iols_kernel

    is_fullcancel = group_i is not None
    if request.tau_mode == "path" and not is_fullcancel:
        raise NotImplementedError(
            "Public fd_oneway tau='path' is currently admitted only for "
            "one-FE fullcancel fits."
        )
    tau_target = None if request.tau_mode == "path" else _require_fixed_tau(request)
    tau_path_targets = (
        _FD_ONEWAY_FULLCANCEL_DEFAULT_PATH_TARGETS if request.tau_mode == "path" else ()
    )
    groups: tuple[np.ndarray, ...]
    cancel_fe_blocks: dict[str, np.ndarray] = {}
    if is_fullcancel:
        mapped_group = _as_dense_group_array(
            np.unique(np.asarray(group_i), return_inverse=True)[1]
        )
        groups = (mapped_group,)
        cancel_fe_blocks[fe_name or "fe_0"] = np.asarray(group_i)
        topology = "fullcancel"
    else:
        groups = (np.zeros(endog.shape[0], dtype=np.int32),)
        topology = "noFE"
    kernel_result = solve_iols_kernel(
        IOLSKernelSpec(
            endog=endog,
            exog=exog,
            topology=topology,
            tau_target=tau_target,
            tau_path_targets=tuple(float(value) for value in tau_path_targets),
            param_names=request.param_names,
            cancel_fe_blocks=cancel_fe_blocks,
            vcov_request="none",
            control=IOLSKernelControl(
                **request.control.kernel_control_kwargs(anderson_m=anderson_m),
            ),
        )
    )
    beta = np.asarray(kernel_result.beta, dtype=float)
    effective_tau = float(kernel_result.tau_fit)
    linear_raw = np.asarray(kernel_result.linear_predictor, dtype=float)
    if linear_raw.ndim != 1:
        raise RuntimeError(
            "The fd_oneway package route expected a one-dimensional "
            "linear_predictor from the package kernel."
        )
    if linear_raw.shape[0] == endog.shape[0]:
        linear = linear_raw
    elif not kernel_result.converged:
        linear = np.full(endog.shape[0], np.nan, dtype=float)
    else:
        raise RuntimeError(
            "The fd_oneway package route expected the package kernel "
            "linear_predictor to have one entry per observation."
        )
    fitted = exp_with_float64_overflow_guard(linear)
    se = np.full(beta.shape, np.nan, dtype=float)
    covariance: np.ndarray | tuple[tuple[float, ...], ...] | None = np.full(
        (beta.size, beta.size),
        np.nan,
        dtype=float,
    )
    if is_fullcancel:
        covariance_source = f"package_fullcancel_expected_bread_{se_type}"
        if se_type == "iid":
            covariance_source = "package_fullcancel_iid_expected_bread"
    else:
        covariance_source = f"package_expected_bread_{se_type}"
        if se_type == "iid":
            covariance_source = "package_iid_expected_bread"
    covariance_info: dict[str, Any] = {
        "inference_status": "available",
        "inference_reason": None,
        "inference_reasons": (),
    }

    if kernel_result.converged and np.all(np.isfinite(beta)) and linear.size:
        if _speed_trust_full_rank_enabled():
            slope_idx = np.arange(exog.shape[1], dtype=int)
        else:
            slope_idx = np.flatnonzero(~_constant_column_mask(exog))
        if slope_idx.size > 0:
            if is_fullcancel:
                se_slope, covariance_slope, covariance_info = (
                    _fullcancel_iols_covariance(
                        endog=endog,
                        exog=exog[:, slope_idx],
                        beta=beta[slope_idx],
                        groups=groups,
                        tau=effective_tau,
                        se_type=se_type,
                        cluster_var=cluster_var,
                        linear=linear,
                    )
                )
            else:
                se_slope, covariance_slope, covariance_backend_info = (
                    _manyway_iols_covariance(
                        endog=endog,
                        exog=exog[:, slope_idx],
                        beta=beta[slope_idx],
                        groups=groups,
                        tau=effective_tau,
                        se_type=se_type,
                        cluster_var=cluster_var,
                        linear=linear,
                        return_info=True,
                    )
                )
                covariance_info.update(covariance_backend_info)
                if not np.all(np.isfinite(covariance_slope)):
                    covariance_info = {
                        **covariance_info,
                        "inference_status": "withheld",
                        "inference_reason": "covariance_unavailable",
                        "inference_reasons": ("covariance_unavailable",),
                    }
            se[slope_idx] = se_slope
            covariance = np.full((beta.size, beta.size), np.nan, dtype=float)
            for compact_i, original_i in enumerate(slope_idx):
                for compact_j, original_j in enumerate(slope_idx):
                    covariance[int(original_i), int(original_j)] = covariance_slope[
                        compact_i,
                        compact_j,
                    ]
            if covariance_info["inference_status"] == "withheld":
                covariance_source = (
                    "package_fullcancel_expected_bread_withheld"
                    if is_fullcancel
                    else "package_expected_bread_withheld"
                )
        else:
            covariance_info = {
                "inference_status": "withheld",
                "inference_reason": "no_estimable_slope_parameters",
                "inference_reasons": ("no_estimable_slope_parameters",),
            }
            covariance_source = (
                "package_fullcancel_expected_bread_withheld"
                if is_fullcancel
                else "package_expected_bread_withheld"
            )
    else:
        covariance_info = {
            "inference_status": "withheld",
            "inference_reason": "kernel_not_converged",
            "inference_reasons": ("kernel_not_converged",),
        }
        covariance_source = (
            "package_fullcancel_expected_bread_withheld"
            if is_fullcancel
            else "package_expected_bread_withheld"
        )

    fe_support = diagnose_fe_support(endog=endog, fe_blocks=fe_blocks)
    inference_warning = _fd_oneway_inference_warning(
        request=request,
        group_i=group_i,
        se_type=se_type,
    )
    if _compact_public_results_enabled():
        instability_metrics = _compact_instability_metrics()
        instability_warning = _compact_instability_warning()
    else:
        instability_metrics = _post_estimation_instability_metrics(
            endog=endog,
            linear=linear,
            tau=effective_tau,
        )
        instability_warning = _post_estimation_instability_warning(
            metrics=instability_metrics,
        )
    covariance_warning: dict[str, Any] = {}
    if covariance_info["inference_status"] == "withheld":
        covariance_warning = _withheld_inference_warning(
            route="fullcancel" if is_fullcancel else "fd_oneway",
            reason=str(covariance_info["inference_reason"]),
        )
    warning_details = _merge_warning_details(
        inference_warning,
        covariance_warning,
        instability_warning,
    )
    if bool(instability_warning["instability_withheld_inference"]):
        reasons = tuple(covariance_info.get("inference_reasons", ()))
        covariance_info = {
            **covariance_info,
            "inference_status": "withheld",
            "inference_reason": covariance_info.get("inference_reason")
            or "post_estimation_instability",
            "inference_reasons": tuple(
                dict.fromkeys(reasons + ("post_estimation_instability",))
            ),
        }
        se = np.full_like(se, np.nan, dtype=float)
        covariance = tuple(
            tuple(float("nan") for _ in range(se.size)) for _ in range(se.size)
        )
        covariance_source = "post_estimation_instability_withheld"
    estimable_param_count = int(np.isfinite(se).sum())
    params, bse, tvalues, pvalues = _normal_stats(request.param_names, beta, se)

    fe_df = 0
    if group_i is not None:
        fe_df = max(int(np.unique(group_i).size) - 1, 0)

    preprocess_contract = scope_only_contract(
        step_name="fd_oneway_scope_check",
        n_obs=int(endog.shape[0]),
        status="pass",
        fe_blocks=fe_blocks,
        step_details={
            "fe_dim_case": request.fe_dim_case,
            "fe_topology": request.fe_topology,
        },
        singleton_checked=group_i is not None,
        has_singletons=False,
        fe_name=fe_name,
        collinearity_mode=collinearity_mode,
        collinearity_status="not_applied",
    )
    preprocess_log = _prepend_routing_log(
        preprocess_contract.preprocess_log(),
        request,
    )
    path_details = dict(kernel_result.solver_details.get("tau_path", {}))
    if is_fullcancel:
        path_details = _attach_public_fd_oneway_fullcancel_path_covariance(
            path_details=path_details,
            endog=endog,
            exog=exog,
            groups=groups,
            param_names=request.param_names,
            se_type=se_type,
            cluster_var=cluster_var,
        )
    else:
        path_details = _strip_public_path_private_state(path_details)
    path_inference_status = str(
        path_details.get("path_inference_status", "not_computed")
    )
    raw_path_inference_reason = path_details.get("path_inference_reason")
    path_inference_reason = (
        None if raw_path_inference_reason is None else str(raw_path_inference_reason)
    )
    if path_inference_status == "computed":
        path_inference_reason = None
    package_route = (
        "p0_s15_fd_oneway_path_fullcancel_package_kernel"
        if is_fullcancel and request.tau_mode == "path"
        else "p0_s4f_fd_oneway_fixed_fullcancel_package_kernel"
        if is_fullcancel
        else "p0_s4d_fd_oneway_fixed_package_kernel"
    )
    linear_predictor_source = (
        "package_fullcancel_retained_index"
        if is_fullcancel
        else "package_kernel_linear_state"
    )
    prediction_details: dict[str, Any] = {}
    if is_fullcancel:
        prediction_details = {
            "prediction_target": "retained_index",
            "prediction_includes_fixed_effects": False,
            "prediction_fixed_effect_policy": "cancelled_fe_not_recovered",
            "prediction_warning_code": ("fullcancel_predictions_exclude_cancelled_fe"),
            "prediction_warning": (
                "Predicted values for this fullcancel fit are exp(X beta) "
                "and do not include the cancelled fixed effect."
            ),
        }

    return SolverResponse(
        request=request,
        params=params,
        bse=bse,
        pvalues=pvalues,
        tvalues=tvalues,
        fittedvalues=_compact_public_sequence(fitted),
        resid_response=_compact_public_sequence(endog - fitted),
        nobs=int(endog.shape[0]),
        df_model=estimable_param_count,
        df_resid=max(int(endog.shape[0]) - estimable_param_count - fe_df, 0),
        family=request.family,
        tau=effective_tau,
        vcov_type=request.vcov_hc_type or request.vcov_type,
        converged=bool(kernel_result.converged),
        fe_method=request.fe_method,
        fe_partition=request.fe_topology,
        partition_warning=_partition_warning_from_request(request),
        singleton_info=preprocess_contract.singleton.as_result_mapping(),
        separation_info={"mode": separation_mode, "status": "not_applied"},
        collinearity_info=preprocess_contract.collinearity.as_result_mapping(),
        preprocess_log=preprocess_log,
        solver_details={
            **dict(kernel_result.solver_details),
            "tau_path": path_details,
            "research_backend": "not_used",
            "research_entrypoint": "not_used",
            "research_root": None,
            "package_route": package_route,
            "requested_vcov": request.vcov_hc_type or request.vcov_type,
            "returned_vcov": (request.vcov_hc_type or request.vcov_type).upper(),
            "clustered": se_type == "cluster",
            "anderson_m": anderson_m,
            "se_type": se_type,
            "covariance_source": covariance_source,
            "linear_predictor_source": linear_predictor_source,
            "path_inference_status": path_inference_status,
            "path_inference_reason": path_inference_reason,
            **covariance_info,
            **prediction_details,
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
        },
        tau_details={
            "tau_fit": effective_tau,
            "tau_target": tau_target,
            "tau_path_targets": tuple(float(value) for value in tau_path_targets),
            "path_status": path_details.get("path_status"),
            "tau_attained": path_details.get("tau_attained", effective_tau),
            "path_mode": path_details.get("path_mode"),
            "tau_path": path_details,
        },
        diagnostic_details={
            "research_backend": "not_used",
            "linear_predictor": _compact_public_sequence(linear),
            "df_resid_formula": "nobs - n_estimable_params - fe_df",
            "n_estimable_params": estimable_param_count,
            "fe_df": fe_df,
            "fe_support": fe_support.as_mapping(),
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
            "linear_predictor_source": linear_predictor_source,
            "path_inference_status": path_inference_status,
            "path_inference_reason": path_inference_reason,
            **covariance_info,
            **prediction_details,
            **instability_metrics,
            **warning_details,
        },
        covariance=_compact_public_covariance(covariance),
    )


def solve_fd_oneway_selected(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    group_i: np.ndarray | None,
    fe_name: str | None,
    se_type: str,
    cluster_var: Any,
    anderson_m: int,
    moment_tau_estimate,
    nofe_tau_estimate,
    fe_blocks: Mapping[str, Any] | None,
    collinearity_mode: str,
    separation_mode: str,
) -> SolverResponse:
    tau_pilot = 2.0
    if group_i is None:
        return _solve_fd_oneway_nofe_selected(
            request=request,
            endog=endog,
            exog=exog,
            fe_name=fe_name,
            se_type=se_type,
            cluster_var=cluster_var,
            anderson_m=anderson_m,
            nofe_tau_estimate=nofe_tau_estimate,
            fe_blocks=fe_blocks,
            collinearity_mode=collinearity_mode,
            separation_mode=separation_mode,
            tau_pilot=tau_pilot,
        )

    selection_variant, short_dim_median_t, n_groups = _selection_variant_from_group(
        group_i
    )
    if selection_variant == "unsupported_short_dimension":
        raise NotImplementedError(
            "Packet H only exposes tau='select' when the selected short FE "
            "dimension has size at least 2."
        )

    pilot_response = _solve_fd_oneway_selected_package_fit(
        request=request,
        endog=endog,
        exog=exog,
        group_i=group_i,
        fe_name=fe_name,
        se_type=se_type,
        cluster_var=cluster_var,
        anderson_m=anderson_m,
        tau_fit=tau_pilot,
        fe_blocks=fe_blocks,
        collinearity_mode=collinearity_mode,
        separation_mode=separation_mode,
    )
    pilot_beta = np.asarray(
        [pilot_response.params[name] for name in request.param_names],
        dtype=float,
    )
    if not pilot_response.converged:
        return _annotate_selection_response(
            request=request,
            response=pilot_response,
            selection_method=selection_variant,
            selection_status="pilot_not_converged",
            tau_pilot=tau_pilot,
            tau_selected=tau_pilot,
            tau_selected_se=None,
            sanity_status="pilot_retained",
            sanity_reason=None,
            sanity_metadata=_selection_sanity_metadata(
                request=request,
                pilot_response=pilot_response,
                selected_response=None,
                tau_selected=tau_pilot,
            ),
            short_dim_median_t=short_dim_median_t,
            n_groups_used=n_groups,
        )

    tau_hat, se_tau, n_groups_used = moment_tau_estimate(
        endog,
        exog,
        pilot_beta,
        group_i,
        tau_lo=request.tau_range[0],
        tau_hi=request.tau_range[1],
        n_grid=request.tau_grid_size,
    )

    selection_status = "selected"
    tau_selected = tau_pilot

    if np.isfinite(tau_hat):
        tau_selected = float(
            np.clip(tau_hat, request.tau_range[0], request.tau_range[1])
        )
        if abs(tau_selected - tau_pilot) >= 0.05:
            selected_response = _solve_fd_oneway_selected_package_fit(
                request=request,
                endog=endog,
                exog=exog,
                group_i=group_i,
                fe_name=fe_name,
                se_type=se_type,
                cluster_var=cluster_var,
                anderson_m=anderson_m,
                tau_fit=tau_selected,
                fe_blocks=fe_blocks,
                collinearity_mode=collinearity_mode,
                separation_mode=separation_mode,
            )
            if selected_response.converged:
                return _apply_selected_tau_sanity(
                    request=request,
                    pilot_response=pilot_response,
                    selected_response=selected_response,
                    selection_method=selection_variant,
                    tau_pilot=tau_pilot,
                    tau_selected=tau_selected,
                    tau_selected_se=(
                        None if not np.isfinite(se_tau) else float(se_tau)
                    ),
                    short_dim_median_t=short_dim_median_t,
                    n_groups_used=int(n_groups_used),
                )
            else:
                selection_status = "refit_not_converged_fallback_to_pilot"
        else:
            selection_status = "pilot_within_tolerance"
    else:
        selection_status = "moment_selector_nan_fallback_to_pilot"

    return _annotate_selection_response(
        request=request,
        response=pilot_response,
        selection_method=selection_variant,
        selection_status=selection_status,
        tau_pilot=tau_pilot,
        tau_selected=tau_selected,
        tau_selected_se=None if not np.isfinite(se_tau) else float(se_tau),
        sanity_status="pilot_retained",
        sanity_reason=None,
        sanity_metadata=_selection_sanity_metadata(
            request=request,
            pilot_response=pilot_response,
            selected_response=None,
            tau_selected=tau_selected,
        ),
        short_dim_median_t=short_dim_median_t,
        n_groups_used=int(n_groups_used),
    )


def _solve_fd_oneway_nofe_selected(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_name: str | None,
    se_type: str,
    cluster_var: Any,
    anderson_m: int,
    nofe_tau_estimate,
    fe_blocks: Mapping[str, Any] | None,
    collinearity_mode: str,
    separation_mode: str,
    tau_pilot: float,
) -> SolverResponse:
    selection_method = "nofe_power_variance_gmm_identity"
    pilot_response = _solve_fd_oneway_selected_package_fit(
        request=request,
        endog=endog,
        exog=exog,
        group_i=None,
        fe_name=fe_name,
        se_type=se_type,
        cluster_var=cluster_var,
        anderson_m=anderson_m,
        tau_fit=tau_pilot,
        fe_blocks=fe_blocks,
        collinearity_mode=collinearity_mode,
        separation_mode=separation_mode,
    )
    if not pilot_response.converged:
        return _annotate_selection_response(
            request=request,
            response=pilot_response,
            selection_method=selection_method,
            selection_status="pilot_not_converged",
            tau_pilot=tau_pilot,
            tau_selected=tau_pilot,
            tau_selected_se=None,
            sanity_status="pilot_retained",
            sanity_reason=None,
            sanity_metadata=_selection_sanity_metadata(
                request=request,
                pilot_response=pilot_response,
                selected_response=None,
                tau_selected=tau_pilot,
            ),
            n_groups_used=int(endog.shape[0]),
            selection_metadata={
                "selection_selector_status": "not_run",
                "selection_selector_reason": "pilot_not_converged",
                "selection_se_status": "unavailable",
                "selection_se_reason": "pilot_not_converged",
            },
        )

    linear_predictor = np.asarray(
        pilot_response.diagnostic_details.get("linear_predictor", ()),
        dtype=float,
    )
    tau_hat, se_tau, n_used, selection_metadata = nofe_tau_estimate(
        endog,
        linear_predictor,
        tau_lo=request.tau_range[0],
        tau_hi=request.tau_range[1],
        n_grid=request.tau_grid_size,
    )

    selection_status = "selected"
    tau_selected = tau_pilot
    tau_selected_se = None if not np.isfinite(se_tau) else float(se_tau)

    if np.isfinite(tau_hat):
        tau_selected = float(
            np.clip(tau_hat, request.tau_range[0], request.tau_range[1])
        )
        if abs(tau_selected - tau_pilot) >= 0.05:
            selected_response = _solve_fd_oneway_selected_package_fit(
                request=request,
                endog=endog,
                exog=exog,
                group_i=None,
                fe_name=fe_name,
                se_type=se_type,
                cluster_var=cluster_var,
                anderson_m=anderson_m,
                tau_fit=tau_selected,
                fe_blocks=fe_blocks,
                collinearity_mode=collinearity_mode,
                separation_mode=separation_mode,
            )
            if selected_response.converged:
                return _apply_selected_tau_sanity(
                    request=request,
                    pilot_response=pilot_response,
                    selected_response=selected_response,
                    selection_method=selection_method,
                    tau_pilot=tau_pilot,
                    tau_selected=tau_selected,
                    tau_selected_se=tau_selected_se,
                    n_groups_used=int(n_used),
                    selection_metadata=selection_metadata,
                )
            selection_status = "refit_not_converged_fallback_to_pilot"
        else:
            selection_status = "pilot_within_tolerance"
    else:
        selection_status = "selector_nan_fallback_to_pilot"

    return _annotate_selection_response(
        request=request,
        response=pilot_response,
        selection_method=selection_method,
        selection_status=selection_status,
        tau_pilot=tau_pilot,
        tau_selected=tau_selected,
        tau_selected_se=tau_selected_se,
        sanity_status="pilot_retained",
        sanity_reason=None,
        sanity_metadata=_selection_sanity_metadata(
            request=request,
            pilot_response=pilot_response,
            selected_response=None,
            tau_selected=tau_selected,
        ),
        n_groups_used=int(n_used),
        selection_metadata=selection_metadata,
    )


def _solve_fd_oneway_selected_package_fit(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    group_i: np.ndarray | None,
    fe_name: str | None,
    se_type: str,
    cluster_var: Any,
    anderson_m: int,
    tau_fit: float,
    fe_blocks: Mapping[str, Any] | None,
    collinearity_mode: str,
    separation_mode: str,
) -> SolverResponse:
    fixed_request = replace(
        request,
        tau_mode="fixed",
        tau_target=float(tau_fit),
    )
    response = solve_fd_oneway(
        request=fixed_request,
        endog=endog,
        exog=exog,
        group_i=group_i,
        fe_name=fe_name,
        se_type=se_type,
        cluster_var=cluster_var,
        anderson_m=anderson_m,
        fe_blocks=fe_blocks,
        collinearity_mode=collinearity_mode,
        separation_mode=separation_mode,
    )
    fixed_package_route = response.solver_details.get("package_route")
    nofe_route = group_i is None
    solver_details = {
        **dict(response.solver_details),
        "package_route": (
            "p0_s6_fd_oneway_nofe_tau_select_package_kernel"
            if nofe_route
            else "p0_s4g_fd_oneway_tau_select_package_kernel"
        ),
        "fixed_tau_package_route": fixed_package_route,
        "selection_backend": (
            "package_nofe_power_variance_tau_estimate"
            if nofe_route
            else "package_moment_tau_estimate"
        ),
        "selection_refit_backend": "package_fd_oneway_fixed_tau",
        "research_backend": "not_used",
        "research_entrypoint": "not_used",
        "research_root": None,
    }
    diagnostic_details = {
        **dict(response.diagnostic_details),
        "selection_backend": (
            "package_nofe_power_variance_tau_estimate"
            if nofe_route
            else "package_moment_tau_estimate"
        ),
        "selection_refit_backend": "package_fd_oneway_fixed_tau",
        "research_backend": "not_used",
    }
    return replace(
        response,
        request=request,
        solver_details=solver_details,
        diagnostic_details=diagnostic_details,
    )


def solve_fd_partitioned_fullcancel_package(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    cluster_var: Any | None,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    separation_mode: str,
) -> SolverResponse:
    from .kernel import IOLSKernelControl, IOLSKernelSpec, solve_iols_kernel
    from .partitioned import make_partitioned_fe_spec, prepare_partitioned_fe_workspace

    if request.tau_mode == "path":
        if len(cancel_fe) != 2:
            if len(cancel_fe) > 2 and se_type in {
                "iid",
                "hc0",
                "hc1",
                "hc2",
                "hc3",
                "cluster",
            }:
                pass
            else:
                raise NotImplementedError(
                    FD_PARTITIONED_THREE_PLUS_FULLCANCEL_PATH_BLOCKED_MESSAGE
                )
        tau_target = None
        tau_path_targets = _FD_PARTITIONED_FULLCANCEL_DEFAULT_PATH_TARGETS
    else:
        tau_target = _require_fixed_tau(request)
        tau_path_targets = ()
    partition_spec = make_partitioned_fe_spec(
        fe_pre,
        cancel_fe=cancel_fe,
        recover_fe=(),
    )
    workspace = prepare_partitioned_fe_workspace(
        endog_pre,
        exog_pre,
        partition_spec,
        tau_select_fe=tau_select_fe,
        preprocessed=True,
        preprocess_meta=prep_meta,
    )
    kernel_result = solve_iols_kernel(
        IOLSKernelSpec(
            endog=endog_pre,
            exog=exog_pre,
            topology="fullcancel",
            tau_target=tau_target,
            tau_path_targets=tuple(float(value) for value in tau_path_targets),
            param_names=tuple(
                request.param_names[index]
                for index in np.asarray(prep_meta.kept_x_idx, dtype=int)
            ),
            cancel_fe_blocks={name: fe_pre[name] for name in cancel_fe},
            vcov_request="none",
            preprocess_meta=prep_meta,
            control=IOLSKernelControl(
                **request.control.kernel_control_kwargs(anderson_m=anderson_m),
            ),
        )
    )

    kept_x_idx = np.asarray(prep_meta.kept_x_idx, dtype=int)
    intercept_position = _intercept_position(kept_x_idx, prep_meta.intercept_idx)
    compact_slope_idx, original_slope_idx = _slope_positions(
        kept_x_idx,
        intercept_position,
    )
    beta_kept = np.asarray(kernel_result.beta, dtype=float)
    beta = np.full(exog.shape[1], np.nan, dtype=float)
    beta[kept_x_idx] = beta_kept
    linear_kept = np.asarray(kernel_result.linear_predictor, dtype=float)
    linear_kept_for_public: np.ndarray | None = (
        linear_kept
        if linear_kept.shape == endog_pre.shape and np.all(np.isfinite(linear_kept))
        else None
    )
    linear = _full_length_linear_predictor(
        linear_kept_for_public,
        np.asarray(prep_meta.kept_obs_mask, dtype=bool),
    )
    fitted = _full_length_fitted(
        linear,
        np.asarray(prep_meta.kept_obs_mask, dtype=bool),
    )
    groups = tuple(np.asarray(fe_pre[name], dtype=int) for name in cancel_fe)
    cluster_pre = (
        None
        if cluster_var is None
        else np.asarray(cluster_var)[np.asarray(prep_meta.kept_obs_mask, dtype=bool)]
    )

    se = np.full(exog.shape[1], np.nan, dtype=float)
    covariance = np.full((exog.shape[1], exog.shape[1]), np.nan, dtype=float)
    covariance_info: dict[str, Any] = {
        "inference_status": "available",
        "inference_reason": None,
        "inference_reasons": (),
    }
    covariance_source = (
        "package_partitioned_fullcancel_iid_expected_bread"
        if se_type == "iid"
        else f"package_partitioned_fullcancel_expected_bread_{se_type}"
    )
    inference_warning: dict[str, Any] = {}
    if len(cancel_fe) > 2 and se_type == "iid":
        message = (
            "Public three-plus-FE fullcancel iid inference uses an additive "
            "absorbed-FE degrees-of-freedom approximation. Nested or collinear "
            "FE can make intervals anti-conservative; treat this many-FE iid "
            "route as provisional until exact absorbed-rank accounting lands."
        )
        inference_warning = {
            "inference_warning": message,
            "warning_messages": (message,),
            "inference_warning_codes": (
                "fd_partitioned_manyway_iid_fe_df_approximate",
            ),
        }
    elif len(cancel_fe) > 2 and se_type == "hc0":
        message = (
            "Public three-plus-FE fullcancel HC0 inference uses the package "
            "retained-index sandwich covariance with weighted-within cancelled "
            "FE. Leverage and finite-sample corrections for this many-FE "
            "surface are not yet hardened; treat HC0 many-FE fullcancel "
            "intervals as provisional."
        )
        inference_warning = {
            "inference_warning": message,
            "warning_messages": (message,),
            "inference_warning_codes": ("fd_partitioned_manyway_hc0_provisional",),
        }
    elif len(cancel_fe) > 2 and se_type in {"hc1", "hc2", "hc3"}:
        message = (
            "Public three-plus-FE fullcancel HC1/HC2/HC3 inference uses the "
            "package retained-index sandwich covariance with weighted-within "
            "cancelled FE and leverage corrections. Coverage evidence for "
            "this many-FE leverage-corrected surface is still scoped; treat "
            "HC1-HC3 many-FE fullcancel intervals as provisional."
        )
        inference_warning = {
            "inference_warning": message,
            "warning_messages": (message,),
            "inference_warning_codes": ("fd_partitioned_manyway_hc1_hc3_provisional",),
        }
    elif len(cancel_fe) > 2 and se_type == "cluster":
        message = (
            "Public three-plus-FE fullcancel cluster inference uses one-way "
            "cluster sums of the package retained-index weighted-within "
            "scores. Coverage evidence for this many-FE cluster surface is "
            "still scoped; treat many-way fullcancel cluster intervals as "
            "provisional."
        )
        inference_warning = {
            "inference_warning": message,
            "warning_messages": (message,),
            "inference_warning_codes": ("fd_partitioned_manyway_cluster_provisional",),
        }
    if (
        kernel_result.converged
        and np.all(np.isfinite(beta_kept))
        and compact_slope_idx.size > 0
        and linear_kept_for_public is not None
    ):
        se_slope, covariance_slope, covariance_info = _fullcancel_iols_covariance(
            endog=endog_pre,
            exog=exog_pre[:, compact_slope_idx],
            beta=beta_kept[compact_slope_idx],
            groups=groups,
            tau=float(kernel_result.tau_fit),
            se_type=se_type,
            cluster_var=cluster_pre,
            linear=linear_kept_for_public,
        )
        for compact_i, original_i in enumerate(original_slope_idx):
            se[int(original_i)] = se_slope[compact_i]
            for compact_j, original_j in enumerate(original_slope_idx):
                covariance[int(original_i), int(original_j)] = covariance_slope[
                    compact_i,
                    compact_j,
                ]
        if covariance_info["inference_status"] == "withheld":
            covariance_source = (
                f"package_partitioned_fullcancel_expected_bread_{se_type}_withheld"
            )
    else:
        reason = (
            "no_estimable_slope_parameters"
            if compact_slope_idx.size == 0
            else "kernel_not_converged"
        )
        covariance_info = {
            "inference_status": "withheld",
            "inference_reason": reason,
            "inference_reasons": (reason,),
        }
        covariance_source = (
            f"package_partitioned_fullcancel_expected_bread_{se_type}_withheld"
        )

    dropped_x_idx = np.asarray(prep_meta.dropped_x_idx, dtype=int)
    duplicate_constant_dropped = bool(
        np.asarray(
            getattr(prep_meta, "constant_dropped_x_idx", ()),
            dtype=int,
        ).size
        > 0
    )
    regressor_collinearity_dropped = bool(
        np.asarray(
            getattr(prep_meta, "rank_dropped_x_idx", ()),
            dtype=int,
        ).size
        > 0
    )
    preprocess_status = (
        "warn"
        if int(prep_meta.n_obs_out) != int(prep_meta.n_obs_in) or dropped_x_idx.size > 0
        else "ok"
    )
    diagnostics = SimpleNamespace(
        status=preprocess_status,
        notes=(),
        partition_hint=workspace.partition_hint,
        preferred_cancel_fe=workspace.preferred_cancel_fe,
        preferred_recover_fe=workspace.preferred_recover_fe,
        preferred_tau_select_fe=workspace.preferred_tau_select_fe,
        duplicate_constant_dropped=duplicate_constant_dropped,
        regressor_collinearity_dropped=regressor_collinearity_dropped,
        transformed_collinearity_dropped=False,
    )
    dropped_columns = tuple(request.param_names[index] for index in dropped_x_idx)
    preprocess_contract = partitioned_research_contract(
        prep_meta=prep_meta,
        diagnostics=diagnostics,
        partition_case=workspace.partition_case,
        research_backend="not_used",
        dropped_columns=dropped_columns,
    )
    preprocess_log = _prepend_routing_log(
        preprocess_contract.preprocess_log(),
        request,
    )
    fe_support = diagnose_fe_support(endog=endog_pre, fe_blocks=fe_pre)
    instability_metrics = _post_estimation_instability_metrics(
        endog=endog,
        linear=linear,
        tau=float(kernel_result.tau_fit),
    )
    instability_warning = _post_estimation_instability_warning(
        metrics=instability_metrics,
    )
    covariance_warning: dict[str, Any] = {}
    if covariance_info["inference_status"] == "withheld":
        covariance_warning = _withheld_inference_warning(
            route="fullcancel",
            reason=str(covariance_info["inference_reason"]),
        )
    warning_details = _merge_warning_details(
        inference_warning,
        covariance_warning,
        instability_warning,
    )
    if bool(instability_warning["instability_withheld_inference"]):
        reasons = tuple(covariance_info.get("inference_reasons", ()))
        covariance_info = {
            **covariance_info,
            "inference_status": "withheld",
            "inference_reason": covariance_info.get("inference_reason")
            or "post_estimation_instability",
            "inference_reasons": tuple(
                dict.fromkeys(reasons + ("post_estimation_instability",))
            ),
        }
        se = np.full_like(se, np.nan, dtype=float)
        covariance = np.full((exog.shape[1], exog.shape[1]), np.nan, dtype=float)
        covariance_source = "post_estimation_instability_withheld"

    params, bse, tvalues, pvalues = _normal_stats(request.param_names, beta, se)
    estimable_param_count = int(
        kept_x_idx.size - (1 if intercept_position is not None else 0)
    )
    path_details = dict(kernel_result.solver_details.get("tau_path", {}))
    if request.tau_mode == "path":
        path_details = _attach_public_fd_partitioned_fullcancel_path_covariance(
            path_details=path_details,
            endog_pre=endog_pre,
            exog_pre=exog_pre,
            groups=groups,
            cluster_var=cluster_pre,
            kept_x_idx=kept_x_idx,
            compact_slope_idx=compact_slope_idx,
            original_slope_idx=original_slope_idx,
            original_param_count=exog.shape[1],
            param_names=request.param_names,
            se_type=se_type,
        )
    else:
        path_details = _strip_tau_path_private_state(path_details)
    path_inference_status = str(
        path_details.get("path_inference_status", "not_computed")
    )
    raw_path_inference_reason = path_details.get("path_inference_reason")
    path_inference_reason = (
        None if raw_path_inference_reason is None else str(raw_path_inference_reason)
    )
    if path_inference_status == "computed":
        path_inference_reason = None
    prediction_details = {
        "prediction_target": "retained_index",
        "prediction_includes_fixed_effects": False,
        "prediction_fixed_effect_policy": "cancelled_fe_not_recovered",
        "prediction_warning_code": "fullcancel_predictions_exclude_cancelled_fe",
        "prediction_warning": (
            "Predicted values for this fullcancel fit are exp(X beta) and do "
            "not include the cancelled fixed effects."
        ),
    }

    return SolverResponse(
        request=request,
        params=params,
        bse=bse,
        pvalues=pvalues,
        tvalues=tvalues,
        fittedvalues=fitted,
        resid_response=_full_length_residuals(
            endog=endog,
            fitted=fitted,
            kept_obs_mask=prep_meta.kept_obs_mask,
        ),
        nobs=int(prep_meta.n_obs_out),
        df_model=estimable_param_count,
        df_resid=max(int(prep_meta.n_obs_out) - estimable_param_count, 0),
        family=request.family,
        tau=float(kernel_result.tau_fit),
        vcov_type=request.vcov_hc_type or request.vcov_type,
        converged=bool(kernel_result.converged),
        fe_method=request.fe_method,
        fe_partition=request.fe_topology,
        partition_warning=_fd_partition_warning(
            request,
            partition_hint=workspace.partition_hint,
        ),
        dropped_columns=preprocess_contract.sample.dropped_columns,
        dropped_observations=preprocess_contract.sample.dropped_observations,
        singleton_info=preprocess_contract.singleton.as_result_mapping(
            include_rounds=True
        ),
        separation_info={"mode": separation_mode, "status": "not_applied"},
        collinearity_info=preprocess_contract.collinearity.as_result_mapping(
            include_drop_flags=True
        ),
        preprocess_log=preprocess_log,
        solver_details={
            **dict(kernel_result.solver_details),
            "tau_path": path_details,
            "research_backend": "not_used",
            "research_entrypoint": "not_used",
            "research_root": None,
            "package_route": (
                "p0_s18_fd_partitioned_path_fullcancel_manyway_iid_package_kernel"
                if request.tau_mode == "path"
                and len(cancel_fe) > 2
                and se_type == "iid"
                else "p0_s20_fd_partitioned_path_fullcancel_manyway_hc0_package_kernel"
                if request.tau_mode == "path"
                and len(cancel_fe) > 2
                and se_type == "hc0"
                else (
                    "p0_s22_fd_partitioned_path_fullcancel_manyway_"
                    "hc1_hc3_package_kernel"
                )
                if request.tau_mode == "path"
                and len(cancel_fe) > 2
                and se_type in {"hc1", "hc2", "hc3"}
                else (
                    "p0_s23_fd_partitioned_path_fullcancel_manyway_"
                    "cluster_package_kernel"
                )
                if request.tau_mode == "path"
                and len(cancel_fe) > 2
                and se_type == "cluster"
                else "p0_s17_fd_partitioned_path_fullcancel_cluster_package_kernel"
                if request.tau_mode == "path" and se_type == "cluster"
                else "p0_s16_fd_partitioned_path_fullcancel_package_kernel"
                if request.tau_mode == "path"
                else (
                    "p0_s23_fd_partitioned_fixed_fullcancel_manyway_"
                    "cluster_package_kernel"
                )
                if se_type == "cluster" and len(cancel_fe) > 2
                else "p0_s19_fd_partitioned_fixed_fullcancel_cluster_package_kernel"
                if se_type == "cluster" and len(cancel_fe) == 2
                else "p0_s20_fd_partitioned_fixed_fullcancel_manyway_hc0_package_kernel"
                if se_type == "hc0" and len(cancel_fe) > 2
                else (
                    "p0_s22_fd_partitioned_fixed_fullcancel_manyway_"
                    "hc1_hc3_package_kernel"
                )
                if se_type in {"hc1", "hc2", "hc3"} and len(cancel_fe) > 2
                else "p0_s4h_fd_partitioned_fixed_fullcancel_package_kernel"
            ),
            "partition_case": workspace.partition_case,
            "covariance_source": covariance_source,
            "tau_select_fe": workspace.tau_select_fe,
            "anderson_m": anderson_m,
            "se_type": se_type,
            "requested_vcov": request.vcov_hc_type or request.vcov_type,
            "returned_vcov": (request.vcov_hc_type or request.vcov_type).upper(),
            "clustered": se_type == "cluster",
            "linear_predictor_source": "package_fullcancel_retained_index",
            "path_inference_status": path_inference_status,
            "path_inference_reason": path_inference_reason,
            **covariance_info,
            **prediction_details,
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
        },
        runtime_info={"runtime_profile": None},
        tau_details={
            "tau_fit": float(kernel_result.tau_fit),
            "tau_target": tau_target,
            "tau_path_targets": tuple(float(value) for value in tau_path_targets),
            "tau_select_fe": workspace.tau_select_fe,
            "path_status": path_details.get("path_status"),
            "tau_attained": path_details.get(
                "tau_attained",
                float(kernel_result.tau_fit),
            ),
            "path_mode": path_details.get("path_mode"),
            "tau_path": path_details,
        },
        diagnostic_details={
            "research_backend": "not_used",
            "partition_case": workspace.partition_case,
            "covariance_source": covariance_source,
            "status": diagnostics.status,
            "notes": diagnostics.notes,
            "partition_hint": diagnostics.partition_hint,
            "preferred_cancel_fe": diagnostics.preferred_cancel_fe,
            "preferred_recover_fe": diagnostics.preferred_recover_fe,
            "preferred_tau_select_fe": diagnostics.preferred_tau_select_fe,
            "linear_predictor": linear,
            "linear_predictor_source": "package_fullcancel_retained_index",
            "path_inference_status": path_inference_status,
            "path_inference_reason": path_inference_reason,
            "df_resid_formula": "nobs_out - n_estimable_params",
            "n_estimable_params": estimable_param_count,
            "fe_support": fe_support.as_mapping(),
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
            **covariance_info,
            **prediction_details,
            **instability_metrics,
            **warning_details,
        },
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
    )


def solve_fd_partitioned_fullcancel_selected_package(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    cluster_var: Any | None,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    partitioned_moment_tau_estimate,
    separation_mode: str,
) -> SolverResponse:
    if tau_select_fe is None:
        raise NotImplementedError(
            "Public partitioned fullcancel tau selection requires tau_select "
            "to identify the cancelled FE dimension used for moment selection."
        )
    selection_variant, short_dim_median_t, n_groups = _selection_variant_from_group(
        np.asarray(fe_pre[tau_select_fe], dtype=int)
    )
    if selection_variant == "unsupported_short_dimension":
        raise NotImplementedError(
            "Packet H only exposes tau='select' when the selected short FE "
            "dimension has size at least 2."
        )
    selection_method = (
        "partitioned_moment_gmm"
        if selection_variant == "moment_gmm"
        else "partitioned_moment_t2_approx"
    )

    tau_pilot = 2.0
    pilot_response = _solve_fd_partitioned_fullcancel_selected_package_fit(
        request=request,
        endog=endog,
        exog=exog,
        fe_blocks=fe_blocks,
        cancel_fe=cancel_fe,
        tau_select_fe=tau_select_fe,
        anderson_m=anderson_m,
        se_type=se_type,
        cluster_var=cluster_var,
        endog_pre=endog_pre,
        exog_pre=exog_pre,
        fe_pre=fe_pre,
        prep_meta=prep_meta,
        tau_fit=tau_pilot,
        separation_mode=separation_mode,
    )
    if not pilot_response.converged:
        return _annotate_selection_response(
            request=request,
            response=pilot_response,
            selection_method=selection_method,
            selection_status="pilot_not_converged",
            tau_pilot=tau_pilot,
            tau_selected=tau_pilot,
            tau_selected_se=None,
            sanity_status="pilot_retained",
            sanity_reason=None,
            sanity_metadata=_selection_sanity_metadata(
                request=request,
                pilot_response=pilot_response,
                selected_response=None,
                tau_selected=tau_pilot,
            ),
            short_dim_median_t=short_dim_median_t,
            n_groups_used=n_groups,
        )

    kept_obs_mask = np.asarray(prep_meta.kept_obs_mask, dtype=bool)
    pilot_linear = np.asarray(
        pilot_response.diagnostic_details["linear_predictor"],
        dtype=float,
    )
    tau_hat, se_tau, n_groups_used = partitioned_moment_tau_estimate(
        endog_pre,
        pilot_linear[kept_obs_mask],
        np.asarray(fe_pre[tau_select_fe], dtype=int),
        tau_lo=request.tau_range[0],
        tau_hi=request.tau_range[1],
        n_grid=request.tau_grid_size,
    )

    selection_status = "selected"
    tau_selected = tau_pilot
    if np.isfinite(tau_hat):
        tau_selected = float(
            np.clip(tau_hat, request.tau_range[0], request.tau_range[1])
        )
        if abs(tau_selected - tau_pilot) >= 0.05:
            selected_response = _solve_fd_partitioned_fullcancel_selected_package_fit(
                request=request,
                endog=endog,
                exog=exog,
                fe_blocks=fe_blocks,
                cancel_fe=cancel_fe,
                tau_select_fe=tau_select_fe,
                anderson_m=anderson_m,
                se_type=se_type,
                cluster_var=cluster_var,
                endog_pre=endog_pre,
                exog_pre=exog_pre,
                fe_pre=fe_pre,
                prep_meta=prep_meta,
                tau_fit=tau_selected,
                separation_mode=separation_mode,
            )
            if selected_response.converged:
                return _apply_selected_tau_sanity(
                    request=request,
                    pilot_response=pilot_response,
                    selected_response=selected_response,
                    selection_method=selection_method,
                    tau_pilot=tau_pilot,
                    tau_selected=tau_selected,
                    tau_selected_se=(
                        None if not np.isfinite(se_tau) else float(se_tau)
                    ),
                    short_dim_median_t=short_dim_median_t,
                    n_groups_used=int(n_groups_used),
                )
            selection_status = "refit_not_converged_fallback_to_pilot"
        else:
            selection_status = "pilot_within_tolerance"
    else:
        selection_status = "moment_selector_nan_fallback_to_pilot"

    return _annotate_selection_response(
        request=request,
        response=pilot_response,
        selection_method=selection_method,
        selection_status=selection_status,
        tau_pilot=tau_pilot,
        tau_selected=tau_selected,
        tau_selected_se=None if not np.isfinite(se_tau) else float(se_tau),
        sanity_status="pilot_retained",
        sanity_reason=None,
        sanity_metadata=_selection_sanity_metadata(
            request=request,
            pilot_response=pilot_response,
            selected_response=None,
            tau_selected=tau_selected,
        ),
        short_dim_median_t=short_dim_median_t,
        n_groups_used=int(n_groups_used),
    )


def _solve_fd_partitioned_fullcancel_selected_package_fit(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    cluster_var: Any | None,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    tau_fit: float,
    separation_mode: str,
) -> SolverResponse:
    fixed_request = replace(
        request,
        tau_mode="fixed",
        tau_target=float(tau_fit),
    )
    response = solve_fd_partitioned_fullcancel_package(
        request=fixed_request,
        endog=endog,
        exog=exog,
        fe_blocks=fe_blocks,
        cancel_fe=cancel_fe,
        tau_select_fe=tau_select_fe,
        anderson_m=anderson_m,
        se_type=se_type,
        cluster_var=cluster_var,
        endog_pre=endog_pre,
        exog_pre=exog_pre,
        fe_pre=fe_pre,
        prep_meta=prep_meta,
        separation_mode=separation_mode,
    )
    fixed_package_route = response.solver_details.get("package_route")
    solver_details = {
        **dict(response.solver_details),
        "package_route": (
            "p0_s23_fd_partitioned_tau_select_fullcancel_manyway_cluster_package_kernel"
            if se_type == "cluster" and len(cancel_fe) > 2
            else "p0_s19_fd_partitioned_tau_select_fullcancel_cluster_package_kernel"
            if se_type == "cluster" and len(cancel_fe) == 2
            else (
                "p0_s20_fd_partitioned_tau_select_fullcancel_manyway_hc0_package_kernel"
            )
            if se_type == "hc0" and len(cancel_fe) > 2
            else (
                "p0_s22_fd_partitioned_tau_select_fullcancel_manyway_hc1_hc3_"
                "package_kernel"
            )
            if se_type in {"hc1", "hc2", "hc3"} and len(cancel_fe) > 2
            else "p0_s4i_fd_partitioned_tau_select_fullcancel_package_kernel"
        ),
        "fixed_tau_package_route": fixed_package_route,
        "selection_backend": "package_partitioned_moment_tau_estimate",
        "selection_refit_backend": "package_fd_partitioned_fullcancel_fixed_tau",
        "research_backend": "not_used",
        "research_entrypoint": "not_used",
        "research_root": None,
    }
    diagnostic_details = {
        **dict(response.diagnostic_details),
        "selection_backend": "package_partitioned_moment_tau_estimate",
        "selection_refit_backend": "package_fd_partitioned_fullcancel_fixed_tau",
        "research_backend": "not_used",
    }
    return replace(
        response,
        request=request,
        solver_details=solver_details,
        diagnostic_details=diagnostic_details,
    )


def solve_fd_partitioned_mixed_package(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    cluster_var: Any | None,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    separation_mode: str,
) -> SolverResponse:
    from .kernel import IOLSKernelControl, IOLSKernelSpec, solve_iols_kernel
    from .partitioned import make_partitioned_fe_spec, prepare_partitioned_fe_workspace

    if not recover_fe:
        raise RuntimeError("Partitioned mixed package route requires recover_fe.")
    fe_count = len(cancel_fe) + len(recover_fe)
    if fe_count > 2 and se_type not in {
        "iid",
        "hc0",
        "hc1",
        "hc2",
        "hc3",
        "cluster",
    }:
        raise NotImplementedError(
            FD_PARTITIONED_THREE_PLUS_MIXED_NON_HC0_BLOCKED_MESSAGE
        )
    if fe_count == 2 and se_type not in {
        "iid",
        "hc0",
        "hc1",
        "hc2",
        "hc3",
        "cluster",
    }:
        raise NotImplementedError(FD_PARTITIONED_MIXED_PATH_BLOCKED_MESSAGE)
    if request.tau_mode == "path":
        tau_target = None
        tau_path_targets = _FD_PARTITIONED_MIXED_DEFAULT_PATH_TARGETS
    else:
        tau_target = _require_fixed_tau(request)
        tau_path_targets = ()
    partition_spec = make_partitioned_fe_spec(
        fe_pre,
        cancel_fe=cancel_fe,
        recover_fe=recover_fe,
    )
    workspace = prepare_partitioned_fe_workspace(
        endog_pre,
        exog_pre,
        partition_spec,
        tau_select_fe=tau_select_fe,
        preprocessed=True,
        preprocess_meta=prep_meta,
    )
    kernel_result = solve_iols_kernel(
        IOLSKernelSpec(
            endog=endog_pre,
            exog=exog_pre,
            topology="mixed",
            tau_target=tau_target,
            tau_path_targets=tuple(float(value) for value in tau_path_targets),
            param_names=tuple(
                request.param_names[index]
                for index in np.asarray(prep_meta.kept_x_idx, dtype=int)
            ),
            cancel_fe_blocks={name: fe_pre[name] for name in cancel_fe},
            recover_fe_blocks={name: fe_pre[name] for name in recover_fe},
            vcov_request="none",
            preprocess_meta=prep_meta,
            control=IOLSKernelControl(
                **request.control.kernel_control_kwargs(anderson_m=anderson_m),
            ),
        )
    )

    kept_x_idx = np.asarray(prep_meta.kept_x_idx, dtype=int)
    intercept_position = _intercept_position(kept_x_idx, prep_meta.intercept_idx)
    compact_slope_idx, original_slope_idx = _slope_positions(
        kept_x_idx,
        intercept_position,
    )
    beta_kept = np.asarray(kernel_result.beta, dtype=float)
    beta = np.full(exog.shape[1], np.nan, dtype=float)
    beta[kept_x_idx] = beta_kept

    linear_kept = np.asarray(kernel_result.linear_predictor, dtype=float)
    if linear_kept.shape[0] != endog_pre.shape[0]:
        linear_kept = np.full(endog_pre.shape[0], np.nan, dtype=float)
    prediction_linear = _full_length_linear_predictor(
        linear_kept,
        np.asarray(prep_meta.kept_obs_mask, dtype=bool),
    )
    fitted = _full_length_fitted(
        prediction_linear,
        np.asarray(prep_meta.kept_obs_mask, dtype=bool),
    )

    cancel_groups = tuple(np.asarray(fe_pre[name], dtype=int) for name in cancel_fe)
    cluster_pre = (
        None
        if cluster_var is None
        else np.asarray(cluster_var)[np.asarray(prep_meta.kept_obs_mask, dtype=bool)]
    )
    covariance_linear_kept = np.full(endog_pre.shape[0], np.nan, dtype=float)
    covariance_linear_source = "unavailable"
    if np.all(np.isfinite(linear_kept)):
        _, covariance_linear_kept = _profile_cancel_fe_from_base(
            endog=endog_pre,
            base_linear=linear_kept,
            groups=cancel_groups,
            tau=float(kernel_result.tau_fit),
        )
        covariance_linear_source = "retained_index_plus_profiled_cancel_fe"
    covariance_linear = _full_length_linear_predictor(
        covariance_linear_kept,
        np.asarray(prep_meta.kept_obs_mask, dtype=bool),
    )

    se = np.full(exog.shape[1], np.nan, dtype=float)
    covariance = np.full((exog.shape[1], exog.shape[1]), np.nan, dtype=float)
    mixed_info: dict[str, Any] | None = None
    inference_reason: str | None = None
    covariance_source = f"package_partitioned_mixed_{se_type}"
    if se_type == "cluster" and fe_count > 2:
        covariance_source = "package_partitioned_mixed_manyway_cluster"
    if compact_slope_idx.size == 0:
        inference_reason = "no_estimable_slope_parameters"
    elif not kernel_result.converged:
        inference_reason = (
            kernel_result.iteration.failure_reason or "kernel_not_converged"
        )
    elif not np.all(np.isfinite(beta_kept)) or not np.all(
        np.isfinite(covariance_linear_kept)
    ):
        inference_reason = "covariance_unavailable"
    else:
        se, covariance, mixed_info = _partitioned_mixed_manyway_covariance(
            endog_pre=endog_pre,
            exog_pre=exog_pre,
            fe_blocks_pre=fe_pre,
            cancel_fe=cancel_fe,
            recover_fe=recover_fe,
            kept_x_idx=kept_x_idx,
            intercept_idx=prep_meta.intercept_idx,
            original_param_count=exog.shape[1],
            tau=float(kernel_result.tau_fit),
            profiled_linear_kept=covariance_linear_kept,
            se_type=se_type,
            cluster_var=cluster_pre,
        )
        slope_se = se[original_slope_idx]
        if not np.all(np.isfinite(slope_se) & (slope_se > 0.0)):
            inference_reason = (
                "covariance_unavailable"
                if mixed_info is None or mixed_info["reason"] is None
                else str(mixed_info["reason"])
            )

    covariance_info: dict[str, Any]
    if inference_reason is None:
        covariance_info = {
            "inference_status": "available",
            "inference_reason": None,
            "inference_reasons": (),
        }
    else:
        se = np.full(exog.shape[1], np.nan, dtype=float)
        covariance = np.full((exog.shape[1], exog.shape[1]), np.nan, dtype=float)
        covariance_source = f"package_partitioned_mixed_{se_type}_withheld"
        if se_type == "cluster" and fe_count > 2:
            covariance_source = "package_partitioned_mixed_manyway_cluster_withheld"
        covariance_info = {
            "inference_status": "withheld",
            "inference_reason": inference_reason,
            "inference_reasons": (inference_reason,),
        }
        if mixed_info is None:
            mixed_info = _mixed_hc0_status(
                supported=False,
                reason=inference_reason,
            )

    dropped_x_idx = np.asarray(prep_meta.dropped_x_idx, dtype=int)
    duplicate_constant_dropped = bool(
        np.asarray(
            getattr(prep_meta, "constant_dropped_x_idx", ()),
            dtype=int,
        ).size
        > 0
    )
    regressor_collinearity_dropped = bool(
        np.asarray(
            getattr(prep_meta, "rank_dropped_x_idx", ()),
            dtype=int,
        ).size
        > 0
    )
    preprocess_status = (
        "warn"
        if int(prep_meta.n_obs_out) != int(prep_meta.n_obs_in) or dropped_x_idx.size > 0
        else "ok"
    )
    diagnostics = SimpleNamespace(
        status=preprocess_status,
        notes=(),
        partition_hint=workspace.partition_hint,
        preferred_cancel_fe=workspace.preferred_cancel_fe,
        preferred_recover_fe=workspace.preferred_recover_fe,
        preferred_tau_select_fe=workspace.preferred_tau_select_fe,
        duplicate_constant_dropped=duplicate_constant_dropped,
        regressor_collinearity_dropped=regressor_collinearity_dropped,
        transformed_collinearity_dropped=False,
    )
    dropped_columns = tuple(request.param_names[index] for index in dropped_x_idx)
    preprocess_contract = partitioned_research_contract(
        prep_meta=prep_meta,
        diagnostics=diagnostics,
        partition_case=workspace.partition_case,
        research_backend="not_used",
        dropped_columns=dropped_columns,
    )
    preprocess_log = _prepend_routing_log(
        preprocess_contract.preprocess_log(),
        request,
    )
    inference_warning: dict[str, Any] = {}
    if se_type == "cluster":
        if fe_count > 2:
            message = (
                "Public three-plus-FE fd_partitioned/mixed cluster inference "
                "uses one-way cluster sums of the package mixed retained-index "
                "scores. Recovered FE are included in the retained mixed "
                "design; cancelled FE are profiled only for covariance "
                "weights. Coverage evidence for this many-FE mixed cluster "
                "surface is still scoped; treat many-FE mixed cluster "
                "intervals as provisional."
            )
            warning_code = "fd_partitioned_mixed_manyway_cluster_provisional"
        else:
            message = (
                "Public fd_partitioned/mixed cluster inference uses one-way "
                "cluster sums of the package mixed retained-index scores. "
                "Recovered FE are included in the retained mixed design; "
                "cancelled FE are profiled only for covariance weights. Coverage "
                "evidence for this mixed cluster surface is still scoped; treat "
                "mixed cluster intervals as provisional."
            )
            warning_code = "fd_partitioned_mixed_cluster_provisional"
        inference_warning = {
            "inference_warning": message,
            "warning_messages": (message,),
            "inference_warning_codes": (warning_code,),
        }
    elif fe_count > 2 and se_type in {"hc1", "hc2", "hc3"}:
        message = (
            "Public three-plus-FE mixed HC1/HC2/HC3 inference uses the "
            "package mixed retained-index plus recovered-FE sandwich "
            "covariance with cancelled FE profiled only for covariance "
            "weights. Coverage evidence for this many-FE HC1-HC3 "
            "mixed surface is still scoped; treat HC1-HC3 many-FE mixed "
            "intervals as provisional."
        )
        inference_warning = {
            "inference_warning": message,
            "warning_messages": (message,),
            "inference_warning_codes": (
                "fd_partitioned_mixed_manyway_hc1_hc3_provisional",
            ),
        }
    fe_support = diagnose_fe_support(endog=endog_pre, fe_blocks=fe_pre)
    instability_metrics = _post_estimation_instability_metrics(
        endog=endog,
        linear=covariance_linear,
        tau=float(kernel_result.tau_fit),
    )
    instability_warning = _post_estimation_instability_warning(
        metrics=instability_metrics,
        bread_condition=None if mixed_info is None else mixed_info["bread_condition"],
    )
    covariance_warning: dict[str, Any] = {}
    if covariance_info["inference_status"] == "withheld":
        covariance_warning = _withheld_inference_warning(
            route="mixed",
            reason=str(covariance_info["inference_reason"]),
        )
    warning_details = _merge_warning_details(
        inference_warning,
        covariance_warning,
        instability_warning,
    )
    if bool(instability_warning["instability_withheld_inference"]):
        inference_was_already_withheld = (
            covariance_info.get("inference_status") == "withheld"
        )
        reasons = tuple(covariance_info.get("inference_reasons", ()))
        covariance_info = {
            **covariance_info,
            "inference_status": "withheld",
            "inference_reason": covariance_info.get("inference_reason")
            or "post_estimation_instability",
            "inference_reasons": tuple(
                dict.fromkeys(reasons + ("post_estimation_instability",))
            ),
        }
        se = np.full(exog.shape[1], np.nan, dtype=float)
        covariance = np.full((exog.shape[1], exog.shape[1]), np.nan, dtype=float)
        if not inference_was_already_withheld:
            covariance_source = "post_estimation_instability_withheld"

    params, bse, tvalues, pvalues = _normal_stats(request.param_names, beta, se)
    estimable_param_count = int(
        kept_x_idx.size - (1 if intercept_position is not None else 0)
    )
    recover_levels = sum(
        int(prep_meta.fe_levels_after.get(name, 0)) for name in recover_fe
    )
    fe_df = max(recover_levels - len(recover_fe), 0)
    path_details = dict(kernel_result.solver_details.get("tau_path", {}))
    if request.tau_mode == "path":
        path_details = _attach_public_fd_partitioned_mixed_path_covariance(
            path_details=path_details,
            endog_pre=endog_pre,
            exog_pre=exog_pre,
            fe_blocks_pre=fe_pre,
            cancel_fe=cancel_fe,
            recover_fe=recover_fe,
            kept_x_idx=kept_x_idx,
            intercept_idx=prep_meta.intercept_idx,
            original_param_count=exog.shape[1],
            param_names=request.param_names,
            se_type=se_type,
            cluster_var=cluster_pre,
        )
    else:
        path_details = _strip_tau_path_private_state(path_details)
        reason = "path_covariance_requires_tau_path_mode_public_fd_partitioned_mixed"
        if (
            str(path_details.get("path_inference_status", "not_computed"))
            in {"not_computed", "unavailable"}
            and path_details.get("path_inference_reason") is None
        ):
            records: list[dict[str, Any]] = []
            for record in tuple(path_details.get("path", ())):
                item = dict(record)
                if (
                    str(item.get("path_inference_status", "not_computed"))
                    in {"not_computed", "unavailable"}
                    and item.get("path_inference_reason") is None
                ):
                    item["path_inference_status"] = "not_computed"
                    item["path_inference_reason"] = reason
                records.append(item)
            if records:
                path_details["path"] = tuple(records)
            path_details["path_inference_status"] = "not_computed"
            path_details["path_inference_reason"] = reason
    path_inference_status = str(
        path_details.get("path_inference_status", "not_computed")
    )
    raw_path_inference_reason = path_details.get("path_inference_reason")
    path_inference_reason = (
        None if raw_path_inference_reason is None else str(raw_path_inference_reason)
    )
    if path_inference_status == "computed":
        path_inference_reason = None
    mixed_compressed_details = _mixed_compressed_path_diagnostics(path_details)
    prediction_details = {
        "prediction_target": "retained_index_plus_recovered_fe",
        "prediction_includes_fixed_effects": False,
        "prediction_includes_all_fixed_effects": False,
        "prediction_includes_recovered_fixed_effects": True,
        "prediction_includes_cancelled_fixed_effects": False,
        "prediction_fixed_effect_policy": (
            "recovered_fe_included_cancelled_fe_not_recovered"
        ),
        "prediction_warning_code": "mixed_predictions_exclude_cancelled_fe",
        "prediction_warning": (
            "Predicted values for this mixed fit are exp(X beta plus recovered "
            "fixed effects) and do not include the cancelled fixed effects."
        ),
    }
    kernel_details = dict(kernel_result.solver_details)
    kernel_details.update(
        {
            "anchor_algorithm": ("iols_phase1_phase2_mixed_cancel_recover_design"),
            "mixed_algorithm": (
                "cancelled_fe_weighted_within_with_explicit_recover_design"
            ),
        }
    )

    return SolverResponse(
        request=request,
        params=params,
        bse=bse,
        pvalues=pvalues,
        tvalues=tvalues,
        fittedvalues=fitted,
        resid_response=_full_length_residuals(
            endog=endog,
            fitted=fitted,
            kept_obs_mask=prep_meta.kept_obs_mask,
        ),
        nobs=int(prep_meta.n_obs_out),
        df_model=estimable_param_count,
        df_resid=max(int(prep_meta.n_obs_out) - estimable_param_count - fe_df, 0),
        family=request.family,
        tau=float(kernel_result.tau_fit),
        vcov_type=request.vcov_hc_type or request.vcov_type,
        converged=bool(kernel_result.converged),
        fe_method=request.fe_method,
        fe_partition=request.fe_topology,
        partition_warning=_fd_partition_warning(
            request,
            partition_hint=workspace.partition_hint,
        ),
        dropped_columns=preprocess_contract.sample.dropped_columns,
        dropped_observations=preprocess_contract.sample.dropped_observations,
        singleton_info=preprocess_contract.singleton.as_result_mapping(
            include_rounds=True
        ),
        separation_info={"mode": separation_mode, "status": "not_applied"},
        collinearity_info=preprocess_contract.collinearity.as_result_mapping(
            include_drop_flags=True
        ),
        preprocess_log=preprocess_log,
        solver_details={
            **kernel_details,
            "tau_path": path_details,
            "research_backend": "not_used",
            "research_entrypoint": "not_used",
            "research_root": None,
            "package_route": (
                "p0_s31_fd_partitioned_path_mixed_manyway_cluster_package_kernel"
                if request.tau_mode == "path"
                and se_type == "cluster"
                and len(cancel_fe) + len(recover_fe) > 2
                else "p0_s26_fd_partitioned_path_mixed_cluster_package_kernel"
                if request.tau_mode == "path" and se_type == "cluster"
                else ("p0_s27_fd_partitioned_path_mixed_manyway_hc1_hc3_package_kernel")
                if request.tau_mode == "path"
                and len(cancel_fe) + len(recover_fe) > 2
                and se_type in {"hc1", "hc2", "hc3"}
                else "p0_s25_fd_partitioned_path_mixed_manyway_hc0_package_kernel"
                if request.tau_mode == "path"
                and len(cancel_fe) + len(recover_fe) > 2
                and se_type == "hc0"
                else "p0_s25_fd_partitioned_path_mixed_manyway_iid_package_kernel"
                if request.tau_mode == "path"
                and len(cancel_fe) + len(recover_fe) > 2
                and se_type == "iid"
                else "p0_s25_fd_partitioned_path_mixed_package_kernel"
                if request.tau_mode == "path"
                else "p0_s31_fd_partitioned_fixed_mixed_manyway_cluster_package_kernel"
                if se_type == "cluster" and len(cancel_fe) + len(recover_fe) > 2
                else "p0_s26_fd_partitioned_fixed_mixed_cluster_package_kernel"
                if se_type == "cluster"
                else (
                    "p0_s27_fd_partitioned_fixed_mixed_manyway_hc1_hc3_package_kernel"
                )
                if len(cancel_fe) + len(recover_fe) > 2
                and se_type in {"hc1", "hc2", "hc3"}
                else "p0_s4j_fd_partitioned_fixed_mixed_package_kernel"
            ),
            "partition_case": workspace.partition_case,
            "covariance_source": covariance_source,
            "tau_select_fe": workspace.tau_select_fe,
            "anderson_m": anderson_m,
            "se_type": se_type,
            "requested_vcov": request.vcov_hc_type or request.vcov_type,
            "returned_vcov": (request.vcov_hc_type or request.vcov_type).upper(),
            "clustered": se_type == "cluster",
            "linear_predictor_source": "package_mixed_retained_index",
            "covariance_linear_predictor_source": covariance_linear_source,
            "path_inference_status": path_inference_status,
            "path_inference_reason": path_inference_reason,
            **mixed_compressed_details,
            "mixed_hc0_inference_status": (
                None if mixed_info is None else mixed_info["status"]
            ),
            "mixed_hc0_inference_reason": (
                None if mixed_info is None else mixed_info["reason"]
            ),
            "mixed_covariance_status": (
                None if mixed_info is None else mixed_info["status"]
            ),
            "mixed_covariance_reason": (
                None if mixed_info is None else mixed_info["reason"]
            ),
            "mixed_covariance_se_type": se_type,
            "mixed_covariance_fixed_effect_policy": (
                "recovered_fe_included_cancelled_fe_profiled_for_covariance_only"
            ),
            "mixed_covariance_n_clusters": (
                None if mixed_info is None else mixed_info.get("n_clusters")
            ),
            **covariance_info,
            **prediction_details,
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
        },
        runtime_info={"runtime_profile": None},
        tau_details={
            "tau_fit": float(kernel_result.tau_fit),
            "tau_target": tau_target,
            "tau_path_targets": tuple(float(value) for value in tau_path_targets),
            "tau_select_fe": workspace.tau_select_fe,
            "path_status": path_details.get("path_status"),
            "tau_attained": path_details.get(
                "tau_attained",
                float(kernel_result.tau_fit),
            ),
            "path_mode": path_details.get("path_mode"),
            "tau_path": path_details,
        },
        diagnostic_details={
            "research_backend": "not_used",
            "partition_case": workspace.partition_case,
            "covariance_source": covariance_source,
            "status": diagnostics.status,
            "notes": diagnostics.notes,
            "partition_hint": diagnostics.partition_hint,
            "preferred_cancel_fe": diagnostics.preferred_cancel_fe,
            "preferred_recover_fe": diagnostics.preferred_recover_fe,
            "preferred_tau_select_fe": diagnostics.preferred_tau_select_fe,
            "linear_predictor": prediction_linear,
            "linear_predictor_source": "package_mixed_retained_index",
            "covariance_linear_predictor": covariance_linear,
            "covariance_linear_predictor_source": covariance_linear_source,
            "path_inference_status": path_inference_status,
            "path_inference_reason": path_inference_reason,
            **mixed_compressed_details,
            "df_resid_formula": (
                "nobs_out - n_estimable_params - approximate_recover_fe_df"
            ),
            "approximate_recover_fe_df": fe_df,
            "n_estimable_params": estimable_param_count,
            "fe_support": fe_support.as_mapping(),
            "mixed_hc0_inference_status": (
                None if mixed_info is None else mixed_info["status"]
            ),
            "mixed_hc0_inference_reason": (
                None if mixed_info is None else mixed_info["reason"]
            ),
            "mixed_hc0_bread_weight_zero_fraction": (
                None if mixed_info is None else mixed_info["bread_weight_zero_fraction"]
            ),
            "mixed_hc0_w_obs_floor_fraction": None,
            "mixed_hc0_bread_condition": (
                None if mixed_info is None else mixed_info["bread_condition"]
            ),
            "mixed_covariance_status": (
                None if mixed_info is None else mixed_info["status"]
            ),
            "mixed_covariance_reason": (
                None if mixed_info is None else mixed_info["reason"]
            ),
            "mixed_covariance_se_type": se_type,
            "mixed_covariance_fixed_effect_policy": (
                "recovered_fe_included_cancelled_fe_profiled_for_covariance_only"
            ),
            "mixed_covariance_n_clusters": (
                None if mixed_info is None else mixed_info.get("n_clusters")
            ),
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
            **covariance_info,
            **prediction_details,
            **instability_metrics,
            **warning_details,
        },
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
    )


def solve_fd_partitioned_mixed_selected_package(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    cluster_var: Any | None,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    partitioned_moment_tau_estimate,
    separation_mode: str,
) -> SolverResponse:
    if tau_select_fe is None:
        raise NotImplementedError(
            "Public partitioned mixed tau selection requires tau_select to "
            "identify the cancelled FE dimension used for moment selection."
        )
    selection_variant, short_dim_median_t, n_groups = _selection_variant_from_group(
        np.asarray(fe_pre[tau_select_fe], dtype=int)
    )
    if selection_variant == "unsupported_short_dimension":
        raise NotImplementedError(
            "Packet H only exposes tau='select' when the selected short FE "
            "dimension has size at least 2."
        )
    selection_method = (
        "partitioned_moment_gmm"
        if selection_variant == "moment_gmm"
        else "partitioned_moment_t2_approx"
    )

    tau_pilot = 2.0
    pilot_response = _solve_fd_partitioned_mixed_selected_package_fit(
        request=request,
        endog=endog,
        exog=exog,
        fe_blocks=fe_blocks,
        cancel_fe=cancel_fe,
        recover_fe=recover_fe,
        tau_select_fe=tau_select_fe,
        anderson_m=anderson_m,
        se_type=se_type,
        cluster_var=cluster_var,
        endog_pre=endog_pre,
        exog_pre=exog_pre,
        fe_pre=fe_pre,
        prep_meta=prep_meta,
        tau_fit=tau_pilot,
        separation_mode=separation_mode,
    )
    if not pilot_response.converged:
        return _annotate_selection_response(
            request=request,
            response=pilot_response,
            selection_method=selection_method,
            selection_status="pilot_not_converged",
            tau_pilot=tau_pilot,
            tau_selected=tau_pilot,
            tau_selected_se=None,
            sanity_status="pilot_retained",
            sanity_reason=None,
            sanity_metadata=_selection_sanity_metadata(
                request=request,
                pilot_response=pilot_response,
                selected_response=None,
                tau_selected=tau_pilot,
            ),
            short_dim_median_t=short_dim_median_t,
            n_groups_used=n_groups,
        )

    kept_obs_mask = np.asarray(prep_meta.kept_obs_mask, dtype=bool)
    pilot_linear = np.asarray(
        pilot_response.diagnostic_details["linear_predictor"],
        dtype=float,
    )
    tau_hat, se_tau, n_groups_used = partitioned_moment_tau_estimate(
        endog_pre,
        pilot_linear[kept_obs_mask],
        np.asarray(fe_pre[tau_select_fe], dtype=int),
        tau_lo=request.tau_range[0],
        tau_hi=request.tau_range[1],
        n_grid=request.tau_grid_size,
    )

    selection_status = "selected"
    tau_selected = tau_pilot
    if np.isfinite(tau_hat):
        tau_selected = float(
            np.clip(tau_hat, request.tau_range[0], request.tau_range[1])
        )
        if abs(tau_selected - tau_pilot) >= 0.05:
            selected_response = _solve_fd_partitioned_mixed_selected_package_fit(
                request=request,
                endog=endog,
                exog=exog,
                fe_blocks=fe_blocks,
                cancel_fe=cancel_fe,
                recover_fe=recover_fe,
                tau_select_fe=tau_select_fe,
                anderson_m=anderson_m,
                se_type=se_type,
                cluster_var=cluster_var,
                endog_pre=endog_pre,
                exog_pre=exog_pre,
                fe_pre=fe_pre,
                prep_meta=prep_meta,
                tau_fit=tau_selected,
                separation_mode=separation_mode,
            )
            if selected_response.converged:
                return _apply_selected_tau_sanity(
                    request=request,
                    pilot_response=pilot_response,
                    selected_response=selected_response,
                    selection_method=selection_method,
                    tau_pilot=tau_pilot,
                    tau_selected=tau_selected,
                    tau_selected_se=(
                        None if not np.isfinite(se_tau) else float(se_tau)
                    ),
                    short_dim_median_t=short_dim_median_t,
                    n_groups_used=int(n_groups_used),
                )
            selection_status = "refit_not_converged_fallback_to_pilot"
        else:
            selection_status = "pilot_within_tolerance"
    else:
        selection_status = "moment_selector_nan_fallback_to_pilot"

    return _annotate_selection_response(
        request=request,
        response=pilot_response,
        selection_method=selection_method,
        selection_status=selection_status,
        tau_pilot=tau_pilot,
        tau_selected=tau_selected,
        tau_selected_se=None if not np.isfinite(se_tau) else float(se_tau),
        sanity_status="pilot_retained",
        sanity_reason=None,
        sanity_metadata=_selection_sanity_metadata(
            request=request,
            pilot_response=pilot_response,
            selected_response=None,
            tau_selected=tau_selected,
        ),
        short_dim_median_t=short_dim_median_t,
        n_groups_used=int(n_groups_used),
    )


def _solve_fd_partitioned_mixed_selected_package_fit(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    cluster_var: Any | None,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    tau_fit: float,
    separation_mode: str,
) -> SolverResponse:
    fixed_request = replace(
        request,
        tau_mode="fixed",
        tau_target=float(tau_fit),
    )
    response = solve_fd_partitioned_mixed_package(
        request=fixed_request,
        endog=endog,
        exog=exog,
        fe_blocks=fe_blocks,
        cancel_fe=cancel_fe,
        recover_fe=recover_fe,
        tau_select_fe=tau_select_fe,
        anderson_m=anderson_m,
        se_type=se_type,
        cluster_var=cluster_var,
        endog_pre=endog_pre,
        exog_pre=exog_pre,
        fe_pre=fe_pre,
        prep_meta=prep_meta,
        separation_mode=separation_mode,
    )
    fixed_package_route = response.solver_details.get("package_route")
    solver_details = {
        **dict(response.solver_details),
        "package_route": (
            "p0_s31_fd_partitioned_tau_select_mixed_manyway_cluster_package_kernel"
            if se_type == "cluster" and len(cancel_fe) + len(recover_fe) > 2
            else "p0_s26_fd_partitioned_tau_select_mixed_cluster_package_kernel"
            if se_type == "cluster"
            else (
                "p0_s27_fd_partitioned_tau_select_mixed_manyway_hc1_hc3_package_kernel"
            )
            if len(cancel_fe) + len(recover_fe) > 2 and se_type in {"hc1", "hc2", "hc3"}
            else "p0_s4j_fd_partitioned_tau_select_mixed_package_kernel"
        ),
        "fixed_tau_package_route": fixed_package_route,
        "selection_backend": "package_partitioned_moment_tau_estimate",
        "selection_refit_backend": "package_fd_partitioned_mixed_fixed_tau",
        "research_backend": "not_used",
        "research_entrypoint": "not_used",
        "research_root": None,
    }
    diagnostic_details = {
        **dict(response.diagnostic_details),
        "selection_backend": "package_partitioned_moment_tau_estimate",
        "selection_refit_backend": "package_fd_partitioned_mixed_fixed_tau",
        "research_backend": "not_used",
    }
    return replace(
        response,
        request=request,
        solver_details=solver_details,
        diagnostic_details=diagnostic_details,
    )


def solve_fd_partitioned(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    make_partitioned_fe_spec,
    prepare_partitioned_fe_workspace,
    iols_partitioned_taupath_prepared,
    profile_fe,
    research_root: str | Path,
    separation_mode: str,
) -> SolverResponse:
    tau_target = _require_fixed_tau(request)
    research_spec = make_partitioned_fe_spec(
        fe_pre,
        cancel_fe=cancel_fe,
        recover_fe=recover_fe,
    )
    workspace = prepare_partitioned_fe_workspace(
        endog_pre,
        exog_pre,
        research_spec,
        tau_select_fe=tau_select_fe,
        preprocessed=True,
        preprocess_meta=prep_meta,
    )
    workspace = replace(workspace, K_orig=exog.shape[1])
    result = iols_partitioned_taupath_prepared(
        workspace,
        tau_target,
        n_steps=_TAU_PATH_STEPS,
        anderson_m=anderson_m,
        se_type=se_type,
        return_meta=True,
    )
    return _build_fd_partitioned_response(
        request=request,
        endog=endog,
        exog=exog,
        fe_blocks=fe_blocks,
        cancel_fe=cancel_fe,
        recover_fe=recover_fe,
        tau_target=tau_target,
        se_type=se_type,
        endog_pre=endog_pre,
        exog_pre=exog_pre,
        fe_pre=fe_pre,
        prep_meta=prep_meta,
        result=result,
        profile_fe=profile_fe,
        research_root=research_root,
        separation_mode=separation_mode,
        solver_details_extra={
            "research_entrypoint": "estimators_partitioned.iols_partitioned_taupath",
            "anderson_m": anderson_m,
            "se_type": se_type,
            "tau_path_steps": _TAU_PATH_STEPS,
        },
        tau_details_extra={"tau_target": tau_target},
    )


def _build_fd_partitioned_response(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    tau_target: float,
    se_type: str,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    result: Any,
    profile_fe,
    research_root: str | Path,
    separation_mode: str,
    solver_details_extra: Mapping[str, Any],
    tau_details_extra: Mapping[str, Any],
) -> SolverResponse:
    beta = np.asarray(result.beta, dtype=float)
    se = np.asarray(result.se, dtype=float)
    covariance = None
    covariance_source = "research_anchor"
    mixed_hc0_info: dict[str, Any] | None = None
    generic_mixed_route = bool(recover_fe and (len(cancel_fe) + len(recover_fe) > 2))
    inference_warning: dict[str, Any] = {}
    if not recover_fe and se_type == "iid":
        if len(cancel_fe) > 2:
            message = (
                "Public three-plus-FE fullcancel iid inference uses an "
                "additive absorbed-FE degrees-of-freedom approximation. "
                "Nested or collinear FE can make intervals anti-conservative; "
                "treat this many-FE iid route as provisional until exact "
                "absorbed-rank accounting lands."
            )
            inference_warning = {
                "inference_warning": message,
                "warning_messages": (message,),
                "inference_warning_codes": (
                    "fd_partitioned_manyway_iid_fe_df_approximate",
                ),
            }
        groups = tuple(np.asarray(fe_pre[name]) for name in cancel_fe)
        intercept_position = _intercept_position(
            np.asarray(prep_meta.kept_x_idx, dtype=int),
            prep_meta.intercept_idx,
        )
        compact_slope_idx, original_slope_idx = _slope_positions(
            np.asarray(prep_meta.kept_x_idx, dtype=int),
            intercept_position,
        )
        se = np.full_like(beta, np.nan, dtype=float)
        covariance = np.full((beta.size, beta.size), np.nan, dtype=float)
        if compact_slope_idx.size > 0:
            beta_slope = beta[np.asarray(prep_meta.kept_x_idx, dtype=int)][
                compact_slope_idx
            ]
            se_slope, covariance_slope = _manyway_iols_covariance(
                endog=endog_pre,
                exog=exog_pre[:, compact_slope_idx],
                beta=beta_slope,
                groups=groups,
                tau=tau_target,
                se_type="iid",
            )
            for compact_i, original_i in enumerate(original_slope_idx):
                se[int(original_i)] = se_slope[compact_i]
                for compact_j, original_j in enumerate(original_slope_idx):
                    covariance[int(original_i), int(original_j)] = covariance_slope[
                        compact_i,
                        compact_j,
                    ]
        covariance_source = "package_manyway_iid"
    elif not recover_fe and len(cancel_fe) > 2:
        # Defensive legacy-builder branch. Live public three-plus-FE HC0 uses
        # the package route before this path; unsupported non-HC0 covariance
        # must not inherit usable-looking BSEs from the old anchor.
        se = np.full_like(se, np.nan, dtype=float)
        covariance_source = "research_anchor_na_inference_contract"

    diagnostics = result.diagnostics
    if prep_meta is None or diagnostics is None:
        raise RuntimeError("The partitioned research anchor did not return metadata.")
    diagnostics_payload = (
        dict(vars(diagnostics))
        if hasattr(diagnostics, "__dict__")
        else dict(diagnostics)
    )
    diagnostics_payload.update(
        {
            "duplicate_constant_dropped": bool(
                np.asarray(
                    getattr(prep_meta, "constant_dropped_x_idx", ()),
                    dtype=int,
                ).size
                > 0
            ),
            "regressor_collinearity_dropped": bool(
                np.asarray(
                    getattr(prep_meta, "rank_dropped_x_idx", ()),
                    dtype=int,
                ).size
                > 0
            ),
            "transformed_collinearity_dropped": bool(
                getattr(diagnostics, "transformed_collinearity_dropped", False)
            ),
        }
    )
    diagnostics = SimpleNamespace(**diagnostics_payload)
    kept_x_idx = np.asarray(prep_meta.kept_x_idx, dtype=int)
    has_kept_intercept = bool(
        prep_meta.intercept_idx is not None
        and np.any(kept_x_idx == int(prep_meta.intercept_idx))
    )
    estimable_param_count = int(kept_x_idx.size - int(has_kept_intercept))

    if recover_fe:
        if generic_mixed_route:
            if result.retained_index is None:
                raise RuntimeError(
                    "The partitioned research anchor did not return retained_index "
                    "for the widened mixed route."
                )
            cancel_groups = tuple(np.asarray(fe_pre[name]) for name in cancel_fe)
            _, profiled_linear_kept = _profile_cancel_fe_from_base(
                endog=endog_pre,
                base_linear=np.asarray(result.retained_index, dtype=float),
                groups=cancel_groups,
                tau=tau_target,
            )
            linear = _profiled_partitioned_generic_linear_predictor(
                kept_obs_mask=prep_meta.kept_obs_mask,
                profiled_linear_kept=profiled_linear_kept,
            )
            linear_predictor_source = "retained_index_plus_profiled_cancel_fe"
            se, covariance, mixed_hc0_info = _partitioned_mixed_manyway_covariance(
                endog_pre=endog_pre,
                exog_pre=exog_pre,
                fe_blocks_pre=fe_pre,
                cancel_fe=cancel_fe,
                recover_fe=recover_fe,
                kept_x_idx=kept_x_idx,
                intercept_idx=prep_meta.intercept_idx,
                original_param_count=exog.shape[1],
                tau=tau_target,
                profiled_linear_kept=profiled_linear_kept,
                se_type=se_type,
            )
            slope_indices = [
                index
                for index in range(exog.shape[1])
                if prep_meta.intercept_idx is None or index != prep_meta.intercept_idx
            ]
            slope_se = se[slope_indices]
            if not np.all(np.isfinite(slope_se) & (slope_se > 0.0)):
                # Public mixed many-FE inference is all-or-nothing for now:
                # partial slope covariance would be easy to misread as a
                # complete public inference contract.
                mixed_hc0_info = _mixed_hc0_status(
                    supported=False,
                    reason=(
                        "mixed_hc0_zero_or_nonpositive_se"
                        if mixed_hc0_info is None or mixed_hc0_info["reason"] is None
                        else str(mixed_hc0_info["reason"])
                    ),
                    bread_weight_zero_fraction=(
                        None
                        if mixed_hc0_info is None
                        else mixed_hc0_info["bread_weight_zero_fraction"]
                    ),
                    bread_condition=(
                        None
                        if mixed_hc0_info is None
                        else mixed_hc0_info["bread_condition"]
                    ),
                )
                se = np.full_like(se, np.nan, dtype=float)
                covariance = np.full(
                    (exog.shape[1], exog.shape[1]),
                    np.nan,
                    dtype=float,
                )
                covariance_source = (
                    "package_partitioned_mixed_hc0_withheld"
                    if se_type == "hc0"
                    else f"package_partitioned_mixed_{se_type}_withheld"
                )
            else:
                covariance_source = (
                    "package_partitioned_mixed_hc0"
                    if se_type == "hc0"
                    else f"package_partitioned_mixed_{se_type}"
                )
        else:
            linear = _profiled_partitioned_linear_predictor(
                endog=endog,
                exog=exog,
                beta=beta,
                fe_blocks=fe_blocks,
                cancel_fe=cancel_fe,
                recover_fe=recover_fe,
                kept_obs_mask=prep_meta.kept_obs_mask,
                tau=tau_target,
                profile_fe=profile_fe,
            )
            linear_predictor_source = "profile_fe_reconstruction"
            if se_type == "iid":
                linear_array = np.asarray(linear, dtype=float)
                profiled_linear_kept = linear_array[
                    np.asarray(prep_meta.kept_obs_mask, dtype=bool)
                ]
                se, covariance, mixed_hc0_info = _partitioned_mixed_manyway_covariance(
                    endog_pre=endog_pre,
                    exog_pre=exog_pre,
                    fe_blocks_pre=fe_pre,
                    cancel_fe=cancel_fe,
                    recover_fe=recover_fe,
                    kept_x_idx=kept_x_idx,
                    intercept_idx=prep_meta.intercept_idx,
                    original_param_count=exog.shape[1],
                    tau=tau_target,
                    profiled_linear_kept=profiled_linear_kept,
                    se_type="iid",
                )
                slope_indices = [
                    index
                    for index in range(exog.shape[1])
                    if prep_meta.intercept_idx is None
                    or index != prep_meta.intercept_idx
                ]
                slope_se = se[slope_indices]
                if not np.all(np.isfinite(slope_se) & (slope_se > 0.0)):
                    se = np.full_like(se, np.nan, dtype=float)
                    covariance = np.full(
                        (exog.shape[1], exog.shape[1]),
                        np.nan,
                        dtype=float,
                    )
                    covariance_source = "package_partitioned_mixed_iid_withheld"
                else:
                    covariance_source = "package_partitioned_mixed_iid"
    else:
        linear = _full_length_linear_predictor(
            result.retained_index,
            prep_meta.kept_obs_mask,
        )
        linear_predictor_source = "retained_index"

    fitted = _full_length_fitted(linear, prep_meta.kept_obs_mask)
    instability_metrics = _post_estimation_instability_metrics(
        endog=endog,
        linear=linear,
        tau=tau_target,
    )
    instability_warning = _post_estimation_instability_warning(
        metrics=instability_metrics,
        bread_condition=(
            None if mixed_hc0_info is None else mixed_hc0_info["bread_condition"]
        ),
    )
    if bool(instability_warning["instability_withheld_inference"]):
        se = np.full_like(se, np.nan, dtype=float)
        covariance = np.full(
            (exog.shape[1], exog.shape[1]),
            np.nan,
            dtype=float,
        )
        covariance_source = "post_estimation_instability_withheld"
    params, bse, tvalues, pvalues = _normal_stats(request.param_names, beta, se)
    dropped_columns = tuple(
        request.param_names[index] for index in prep_meta.dropped_x_idx
    )
    preprocess_contract = partitioned_research_contract(
        prep_meta=prep_meta,
        diagnostics=diagnostics,
        partition_case=result.partition_case,
        research_backend=result.backend,
        dropped_columns=dropped_columns,
    )
    fe_support = diagnose_fe_support(endog=endog_pre, fe_blocks=fe_pre)
    dropped_observations = preprocess_contract.sample.dropped_observations
    recover_levels = sum(
        int(prep_meta.fe_levels_after.get(name, 0)) for name in recover_fe
    )
    fe_df = max(recover_levels - len(recover_fe), 0)
    preprocess_log = _prepend_routing_log(
        preprocess_contract.preprocess_log(),
        request,
    )
    warning_details = _merge_warning_details(inference_warning, instability_warning)

    return SolverResponse(
        request=request,
        params=params,
        bse=bse,
        pvalues=pvalues,
        tvalues=tvalues,
        fittedvalues=fitted,
        resid_response=_full_length_residuals(
            endog=endog,
            fitted=fitted,
            kept_obs_mask=prep_meta.kept_obs_mask,
        ),
        nobs=int(prep_meta.n_obs_out),
        df_model=estimable_param_count,
        df_resid=max(int(prep_meta.n_obs_out) - estimable_param_count - fe_df, 0),
        family=request.family,
        tau=tau_target,
        vcov_type=request.vcov_hc_type or request.vcov_type,
        converged=bool(result.converged),
        fe_method=request.fe_method,
        fe_partition=request.fe_topology,
        partition_warning=_fd_partition_warning(
            request,
            partition_hint=diagnostics.partition_hint,
        ),
        dropped_columns=preprocess_contract.sample.dropped_columns,
        dropped_observations=dropped_observations,
        singleton_info=preprocess_contract.singleton.as_result_mapping(
            include_rounds=True
        ),
        separation_info={"mode": separation_mode, "status": "not_applied"},
        collinearity_info=preprocess_contract.collinearity.as_result_mapping(
            include_drop_flags=True
        ),
        preprocess_log=preprocess_log,
        solver_details={
            "research_backend": result.backend,
            "partition_case": result.partition_case,
            "research_root": str(Path(research_root)),
            "covariance_source": covariance_source,
            "tau_select_fe": result.tau_select_fe,
            "mixed_hc0_inference_status": (
                None if mixed_hc0_info is None else mixed_hc0_info["status"]
            ),
            "mixed_hc0_inference_reason": (
                None if mixed_hc0_info is None else mixed_hc0_info["reason"]
            ),
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
            **dict(solver_details_extra),
        },
        runtime_info={
            "runtime_profile": None
            if result.runtime_profile is None
            else result.runtime_profile.__dict__,
        },
        tau_details={
            "tau_select_fe": result.tau_select_fe,
            **dict(tau_details_extra),
        },
        diagnostic_details={
            "research_backend": result.backend,
            "partition_case": result.partition_case,
            "covariance_source": covariance_source,
            "status": diagnostics.status,
            "notes": diagnostics.notes,
            "partition_hint": diagnostics.partition_hint,
            "preferred_cancel_fe": diagnostics.preferred_cancel_fe,
            "preferred_recover_fe": diagnostics.preferred_recover_fe,
            "preferred_tau_select_fe": diagnostics.preferred_tau_select_fe,
            "linear_predictor": linear,
            "exp_guard_policy": "float64_overflow_boundary_only",
            "exp_guard_upper_log": FLOAT64_LOG_MAX,
            "research_root": str(Path(research_root)),
            "df_resid_formula": "nobs_out - n_estimable_params - approximate_fe_df",
            "approximate_fe_df": fe_df,
            "n_estimable_params": estimable_param_count,
            "linear_predictor_source": linear_predictor_source,
            "fe_support": fe_support.as_mapping(),
            "mixed_hc0_inference_status": (
                None if mixed_hc0_info is None else mixed_hc0_info["status"]
            ),
            "mixed_hc0_inference_reason": (
                None if mixed_hc0_info is None else mixed_hc0_info["reason"]
            ),
            "mixed_hc0_bread_weight_zero_fraction": (
                None
                if mixed_hc0_info is None
                else mixed_hc0_info["bread_weight_zero_fraction"]
            ),
            "mixed_hc0_w_obs_floor_fraction": None,
            "mixed_hc0_bread_condition": (
                None if mixed_hc0_info is None else mixed_hc0_info["bread_condition"]
            ),
            **instability_metrics,
            **warning_details,
        },
        covariance=None
        if covariance is None
        else tuple(tuple(float(value) for value in row) for row in covariance),
    )


def solve_fd_partitioned_selected(
    *,
    request: SolverRequest,
    endog: np.ndarray,
    exog: np.ndarray,
    fe_blocks: dict[str, np.ndarray],
    cancel_fe: tuple[str, ...],
    recover_fe: tuple[str, ...],
    tau_select_fe: str | None,
    anderson_m: int,
    se_type: str,
    endog_pre: np.ndarray,
    exog_pre: np.ndarray,
    fe_pre: dict[str, np.ndarray],
    prep_meta: Any,
    make_partitioned_fe_spec,
    prepare_partitioned_fe_workspace,
    iols_partitioned_taupath_prepared,
    partitioned_moment_tau_estimate,
    profile_fe,
    research_root: str | Path,
    separation_mode: str,
) -> SolverResponse:
    if tau_select_fe is None:
        raise NotImplementedError(
            "Packet H requires tau_select to identify the cancelled FE "
            "dimension used for public partitioned tau selection."
        )
    selection_variant, short_dim_median_t, _ = _selection_variant_from_group(
        np.asarray(fe_pre[tau_select_fe], dtype=int)
    )
    if selection_variant == "unsupported_short_dimension":
        raise NotImplementedError(
            "Packet H only exposes tau='select' when the selected short FE "
            "dimension has size at least 2."
        )
    selection_method = (
        "partitioned_moment_gmm"
        if selection_variant == "moment_gmm"
        else "partitioned_moment_t2_approx"
    )
    research_spec = make_partitioned_fe_spec(
        fe_pre,
        cancel_fe=cancel_fe,
        recover_fe=recover_fe,
    )
    workspace = prepare_partitioned_fe_workspace(
        endog_pre,
        exog_pre,
        research_spec,
        tau_select_fe=tau_select_fe,
        preprocessed=True,
        preprocess_meta=prep_meta,
    )
    workspace = replace(workspace, K_orig=exog.shape[1])

    tau_pilot = 2.0
    pilot = iols_partitioned_taupath_prepared(
        workspace,
        tau_pilot,
        n_steps=_TAU_PATH_STEPS,
        anderson_m=anderson_m,
        se_type=se_type,
        return_meta=True,
    )

    if (
        (not pilot.converged)
        or pilot.retained_index is None
        or pilot.cancel_group is None
    ):
        pilot_response = _build_fd_partitioned_response(
            request=request,
            endog=endog,
            exog=exog,
            fe_blocks=fe_blocks,
            cancel_fe=cancel_fe,
            recover_fe=recover_fe,
            tau_target=tau_pilot,
            se_type=se_type,
            endog_pre=endog_pre,
            exog_pre=exog_pre,
            fe_pre=fe_pre,
            prep_meta=prep_meta,
            result=pilot,
            profile_fe=profile_fe,
            research_root=research_root,
            separation_mode=separation_mode,
            solver_details_extra={
                "research_entrypoint": (
                    "estimators_partitioned.partitioned_moment_tau_estimate + "
                    "estimators_partitioned.iols_partitioned_taupath_prepared"
                ),
                "anderson_m": anderson_m,
                "se_type": se_type,
                "tau_path_steps": _TAU_PATH_STEPS,
            },
            tau_details_extra={},
        )
        return _annotate_selection_response(
            request=request,
            response=pilot_response,
            selection_method=selection_method,
            selection_status="pilot_not_converged",
            tau_pilot=tau_pilot,
            tau_selected=tau_pilot,
            tau_selected_se=None,
            sanity_status="pilot_retained",
            sanity_reason=None,
            sanity_metadata=_selection_sanity_metadata(
                request=request,
                pilot_response=pilot_response,
                selected_response=None,
                tau_selected=tau_pilot,
            ),
            short_dim_median_t=short_dim_median_t,
        )
    pilot_response = _build_fd_partitioned_response(
        request=request,
        endog=endog,
        exog=exog,
        fe_blocks=fe_blocks,
        cancel_fe=cancel_fe,
        recover_fe=recover_fe,
        tau_target=tau_pilot,
        se_type=se_type,
        endog_pre=endog_pre,
        exog_pre=exog_pre,
        fe_pre=fe_pre,
        prep_meta=prep_meta,
        result=pilot,
        profile_fe=profile_fe,
        research_root=research_root,
        separation_mode=separation_mode,
        solver_details_extra={
            "research_entrypoint": (
                "estimators_partitioned.partitioned_moment_tau_estimate + "
                "estimators_partitioned.iols_partitioned_taupath_prepared"
            ),
            "anderson_m": anderson_m,
            "se_type": se_type,
            "tau_path_steps": _TAU_PATH_STEPS,
        },
        tau_details_extra={},
    )

    tau_hat, se_tau, _ = partitioned_moment_tau_estimate(
        endog_pre,
        np.asarray(pilot.retained_index, dtype=float),
        pilot.cancel_group,
        tau_lo=request.tau_range[0],
        tau_hi=request.tau_range[1],
        n_grid=request.tau_grid_size,
    )
    tau_selected = tau_pilot
    selection_status = "selected"

    if np.isfinite(tau_hat):
        tau_selected = float(
            np.clip(tau_hat, request.tau_range[0], request.tau_range[1])
        )
        if abs(tau_selected - tau_pilot) >= 0.05:
            final = iols_partitioned_taupath_prepared(
                workspace,
                tau_selected,
                n_steps=_TAU_PATH_STEPS,
                anderson_m=anderson_m,
                se_type=se_type,
                return_meta=True,
            )
            if final.converged:
                selected_response = _build_fd_partitioned_response(
                    request=request,
                    endog=endog,
                    exog=exog,
                    fe_blocks=fe_blocks,
                    cancel_fe=cancel_fe,
                    recover_fe=recover_fe,
                    tau_target=tau_selected,
                    se_type=se_type,
                    endog_pre=endog_pre,
                    exog_pre=exog_pre,
                    fe_pre=fe_pre,
                    prep_meta=prep_meta,
                    result=final,
                    profile_fe=profile_fe,
                    research_root=research_root,
                    separation_mode=separation_mode,
                    solver_details_extra={
                        "research_entrypoint": (
                            "estimators_partitioned.partitioned_moment_tau_estimate + "
                            "estimators_partitioned.iols_partitioned_taupath_prepared"
                        ),
                        "anderson_m": anderson_m,
                        "se_type": se_type,
                        "tau_path_steps": _TAU_PATH_STEPS,
                    },
                    tau_details_extra={},
                )
                return _apply_selected_tau_sanity(
                    request=request,
                    pilot_response=pilot_response,
                    selected_response=selected_response,
                    selection_method=selection_method,
                    tau_pilot=tau_pilot,
                    tau_selected=tau_selected,
                    tau_selected_se=(
                        None if not np.isfinite(se_tau) else float(se_tau)
                    ),
                    short_dim_median_t=short_dim_median_t,
                )
            else:
                selection_status = "refit_not_converged_fallback_to_pilot"
        else:
            selection_status = "pilot_within_tolerance"
    else:
        selection_status = "moment_selector_nan_fallback_to_pilot"

    return _annotate_selection_response(
        request=request,
        response=pilot_response,
        selection_method=selection_method,
        selection_status=selection_status,
        tau_pilot=tau_pilot,
        tau_selected=tau_selected,
        tau_selected_se=None if not np.isfinite(se_tau) else float(se_tau),
        sanity_status="pilot_retained",
        sanity_reason=None,
        sanity_metadata=_selection_sanity_metadata(
            request=request,
            pilot_response=pilot_response,
            selected_response=None,
            tau_selected=tau_selected,
        ),
        short_dim_median_t=short_dim_median_t,
    )
