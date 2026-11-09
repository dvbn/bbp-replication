from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, Final, Literal

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .._diagnostic_metadata import is_known_fe_runtime_failure
from .._numeric import exp_with_float64_overflow_guard
from .fd import _manyway_iols_covariance
from .fe import (
    _demean_manyway_checked,
    demean,
    demean_manyway,
    profile_fe,
    profile_fe_manyway,
)

KernelTopology = Literal["noFE", "fullcancel", "fullrecover", "mixed"]
ScoreValidationPolicy = Literal["diagnostic"]

TAU2_JOINT_HDFE_SUPPORT_CHECK = "tau2_joint_hdfe_boundary_separation"
TAU2_RECOVERED_FE_SUPPORT_CHECK = "tau2_recovered_fe_boundary_separation"
TAU2_BOUNDARY_SEPARATION_CODE = "tau2_boundary_separation"
TAU2_BOUNDARY_SUPPORT_POLICY = "fail_explicit"
TAU2_BOUNDARY_FUTURE_POLICY = (
    "drop_separated_requires_theory_diagnostics_sample_accounting_covariance_mc"
)
TAU2_BOUNDARY_DIRECTION_TOL = 1e-7
MIXED_TAU2_SUPPORT_CONDITIONING_CHECK = "mixed_tau2_recovered_fe_support_conditioning"
MIXED_TAU2_NEAR_BOUNDARY_CODE = (
    "mixed_tau2_recovered_fe_near_boundary_or_ill_conditioned"
)
MIXED_TAU2_SUPPORT_POLICY = "diagnose_then_reclassify_anchor_failures"
FULLRECOVER_JOINT_FE_POLICY = "pure_hdfe_recover_all"
FULLRECOVER_PROFILED_FE_POLICY = "analytical_profile"
KERNEL_BACKEND_PYTHON_REFERENCE: Final = "python_reference"
KERNEL_BACKEND_CPP: Final = "cpp"

KERNEL_FAILURE_CONFIGURED_PATH_PARTIAL: Final = "configured_path_partial"
KERNEL_FAILURE_CRITERION_EXPLOSION: Final = "criterion_explosion"
KERNEL_FAILURE_CRITERION_STAGNATION: Final = "criterion_stagnation"
KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH: Final = "empty_finite_rho_path"
KERNEL_FAILURE_INVALID_PHI_STEP: Final = "invalid_phi_step"
KERNEL_FAILURE_MAX_PATH_EVALUATIONS_REACHED: Final = "max_path_evaluations_reached"
KERNEL_FAILURE_MIN_DTAU_REACHED: Final = "min_dtau_reached"
KERNEL_FAILURE_NONCONTRACTIVE: Final = "noncontractive"
KERNEL_FAILURE_NOT_CONVERGED: Final = "not_converged"
KERNEL_FAILURE_NOT_STARTED: Final = "not_started"
KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE: Final = "phase1_candidate_unusable"
KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED: Final = (
    "phase2_evaluation_budget_exhausted"
)
KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED: Final = (
    KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED
)
KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE: Final = (
    "post_convergence_noncontractive"
)
KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL: Final = "post_convergence_residual"
# runtime_v4 direct route only: the final verification map was invalid.
KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP: Final = "post_convergence_invalid_map"
KERNEL_FAILURE_Q_CONVERGENCE_INVALID: Final = "q_convergence_invalid"
KERNEL_FAILURE_TAU2_ANCHOR_FAILED: Final = "tau2_anchor_failed"
KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO: Final = "tau2_anchor_missing_rho"
TAU_PATH_FRONTIER_RETRY_BUDGET: Final = 4
TRANSFORMED_NORMALIZATION_REL_FLOOR: Final = 1e-12
TRANSFORMED_NORMALIZATION_ABS_BACKSTOP: Final = 1e-300
_FIXED_TAU_FAST_CONTROL_FLAG: Final = "PYIOLS_SPEED_FIXED_TAU_FAST_CONTROL"
_COMPACT_FIXED_TAU_DETAILS_FLAG: Final = "PYIOLS_SPEED_COMPACT_FIXED_TAU_DETAILS"
_SKIP_FULLRECOVER_BOUNDARY_CHECK_FLAG: Final = (
    "PYIOLS_SPEED_SKIP_FULLRECOVER_BOUNDARY_CHECK"
)
_TRUST_FE_LABELS_FLAG: Final = "PYIOLS_SPEED_TRUST_FE_LABELS"
KERNEL_FAILURE_REASONS: Final[frozenset[str]] = frozenset(
    {
        KERNEL_FAILURE_CONFIGURED_PATH_PARTIAL,
        KERNEL_FAILURE_CRITERION_EXPLOSION,
        KERNEL_FAILURE_CRITERION_STAGNATION,
        KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH,
        KERNEL_FAILURE_INVALID_PHI_STEP,
        KERNEL_FAILURE_MAX_PATH_EVALUATIONS_REACHED,
        KERNEL_FAILURE_MIN_DTAU_REACHED,
        KERNEL_FAILURE_NONCONTRACTIVE,
        KERNEL_FAILURE_NOT_CONVERGED,
        KERNEL_FAILURE_NOT_STARTED,
        KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE,
        KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED,
        KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE,
        KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL,
        KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP,
        KERNEL_FAILURE_Q_CONVERGENCE_INVALID,
        KERNEL_FAILURE_TAU2_ANCHOR_FAILED,
        KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO,
        MIXED_TAU2_NEAR_BOUNDARY_CODE,
        TAU2_BOUNDARY_SEPARATION_CODE,
    }
)


@dataclass(frozen=True)
class IOLSKernelControl:
    """Numerical controls for the package-owned iOLS continuation kernel.

    `stopcrit` is the per-rho fixed-point iteration cap. Phase 1, Phase 2,
    FE profiling, and FE demeaning default to MATLAB-style high iteration
    ceilings; evidence scripts may pass smaller explicit budgets for smoke and
    optimization checks. `max_path_evaluations` caps attempted tau continuation
    steps, including adaptive bisections.
    """

    anderson_m: int = 5
    eps: float = 1e-6
    stopcrit: int = 10000
    tau_path_steps: int = 20
    min_dtau: float | None = None
    max_path_evaluations: int = 10000
    max_phase1_iterations: int = 10000
    max_phase2_evaluations: int = 50000
    fe_profile_maxiter: int = 10000
    fe_profile_tol: float = 1e-6
    fe_demean_maxiter: int = 10000
    fe_demean_tol: float = 1e-10
    # Starting numerical grids follow iOLS.m defaults; callers may widen or
    # refine them when convergence diagnostics show the default grid is tight.
    delta_log_min: float = -5.0
    delta_log_max: float = 5.0
    delta_grid_size: int = 5
    rho_log_min: float = -7.0
    rho_log_max: float = 14.0
    rho_grid_size: int = 150
    transformed_design_normalization: Literal["none", "scale"] = "none"
    # runtime_v3: Phase 2 restart start policy, read ONLY by the direct no-FE
    # fixed-tau route (legacy tau-path routes ignore it).
    #   "phase1_start"       every rho restart (and the non-converged return)
    #                        uses the Phase 1 start / last-attempt final iterate
    #                        (runtime_v2 behaviour)
    #   "best_score_iterate" every rho restart starts from the Phase 2 iterate
    #                        with the smallest rho-free preconditioned score
    #                        norm (1+rho)*||phi(beta)-beta||_2 seen so far, and
    #                        the non-converged return is that iterate
    direct_restart_start: Literal["phase1_start", "best_score_iterate"] = (
        "phase1_start"
    )


DIRECT_RESTART_START_POLICIES: Final = ("phase1_start", "best_score_iterate")


@dataclass(frozen=True)
class IOLSKernelSpec:
    endog: np.ndarray
    exog: np.ndarray
    topology: KernelTopology
    tau_target: float | None
    tau_path_targets: tuple[float, ...] = ()
    param_names: tuple[str, ...] = ()
    cancel_fe_blocks: Mapping[str, np.ndarray] = field(default_factory=dict)
    recover_fe_blocks: Mapping[str, np.ndarray] = field(default_factory=dict)
    recover_fe_counts: Mapping[str, int] = field(default_factory=dict)
    vcov_request: str = "iid"
    control: IOLSKernelControl = field(default_factory=IOLSKernelControl)
    preprocess_meta: Any | None = None


@dataclass(frozen=True)
class IOLSIterationDiagnostics:
    """Iteration ledger for one kernel solve.

    `iterations` is the final per-rho fixed-point counter, while
    `details["phase2_evaluations"]` records total fixed-point work across rho
    restarts when available.
    """

    converged: bool
    phase: str
    iterations: int
    restarts: int
    final_rho: float | None
    rho_path: tuple[float, ...]
    final_criterion: float | None
    anderson_m_initial: int
    anderson_m_final: int
    failure_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IOLSKernelResult:
    beta: np.ndarray
    se: np.ndarray | None
    covariance: np.ndarray | None
    converged: bool
    tau_fit: float
    linear_predictor: np.ndarray
    fittedvalues: np.ndarray
    topology: KernelTopology
    granular_case: str | None
    iteration: IOLSIterationDiagnostics
    solver_details: Mapping[str, Any]


@dataclass(frozen=True)
class _TauPathSolve:
    beta: np.ndarray
    linear: np.ndarray | None
    tau_fit: float
    converged: bool
    iteration: IOLSIterationDiagnostics
    details: Mapping[str, Any]


@dataclass(frozen=True)
class _ConcentratedSolve:
    beta: np.ndarray
    linear: np.ndarray | None
    iteration: IOLSIterationDiagnostics


@dataclass(frozen=True)
class _FullrecoverInvariantState:
    x_demeaned: np.ndarray
    projection: np.ndarray


@dataclass(frozen=True)
class _TransformedNormalizationState:
    requested: bool
    enabled: bool
    status: str
    basis: str = "std"
    scale: np.ndarray | None = None
    condition_before: float | None = None
    condition_after: float | None = None
    scale_min: float | None = None
    scale_max: float | None = None
    scale_floor: float | None = None
    transformed_rank: int | None = None


def _transformed_normalization_mode(control: IOLSKernelControl) -> str:
    mode = str(getattr(control, "transformed_design_normalization", "none"))
    if mode not in {"none", "scale"}:
        raise ValueError(
            "transformed_design_normalization must be one of 'none' or 'scale'"
        )
    return mode


def _as_float_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _as_float_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array


def _as_dense_group_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.size and int(np.max(array)) <= np.iinfo(np.int32).max:
        return array.astype(np.int32, copy=False)
    return array


def _as_encoded_group_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    return array


def _trusted_dense_group_array(values: np.ndarray) -> tuple[np.ndarray, int] | None:
    if not _truthy_env(os.environ.get(_TRUST_FE_LABELS_FLAG)):
        return None
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        return None
    if array.ndim != 1:
        return None
    if array.size == 0:
        return _as_dense_group_array(array), 0
    min_label = int(np.min(array))
    max_label = int(np.max(array))
    if min_label < 0 or max_label > array.size:
        return None
    return _as_dense_group_array(array), max_label + 1


def _remap_groups(
    blocks: Mapping[str, np.ndarray],
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[int, ...]]:
    names: list[str] = []
    groups: list[np.ndarray] = []
    counts: list[int] = []
    for name, values in blocks.items():
        names.append(str(name))
        trusted = _trusted_dense_group_array(values)
        if trusted is not None:
            mapped, n_groups = trusted
        else:
            _, mapped = np.unique(np.asarray(values), return_inverse=True)
            mapped = _as_dense_group_array(mapped)
            n_groups = int(mapped.max()) + 1 if mapped.size else 0
        groups.append(mapped)
        counts.append(n_groups)
    return tuple(names), tuple(groups), tuple(counts)


def _encoded_groups_from_blocks(
    blocks: Mapping[str, np.ndarray],
    counts_by_name: Mapping[str, int],
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[int, ...]]:
    names: list[str] = []
    groups: list[np.ndarray] = []
    counts: list[int] = []
    for name, values in blocks.items():
        key = str(name)
        if key not in counts_by_name:
            raise ValueError(f"missing encoded fixed-effect count for {key!r}")
        names.append(key)
        groups.append(_as_encoded_group_array(values))
        counts.append(int(counts_by_name[key]))
    return tuple(names), tuple(groups), tuple(counts)


def _split_intercept(
    exog: np.ndarray,
    beta_init: np.ndarray | None,
) -> tuple[np.ndarray, bool, np.ndarray | None]:
    has_intercept = bool(exog.shape[1] > 0 and np.all(exog[:, 0] == 1))
    if has_intercept:
        init = None if beta_init is None else np.asarray(beta_init, dtype=float)[1:]
        return exog[:, 1:], True, init
    return exog, False, beta_init


def _expand_intercept(
    beta: np.ndarray,
    se: np.ndarray | None,
    covariance: np.ndarray | None,
    *,
    original_k: int,
    has_intercept: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if not has_intercept:
        return beta, se, covariance
    beta_full = np.zeros(original_k, dtype=float)
    beta_full[1:] = beta
    se_full = None
    if se is not None:
        se_full = np.full(original_k, np.nan, dtype=float)
        se_full[1:] = se
    cov_full = None
    if covariance is not None:
        cov_full = np.full((original_k, original_k), np.nan, dtype=float)
        cov_full[1:, 1:] = covariance
    return beta_full, se_full, cov_full


def _as_public_vector(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _as_owned_float64_vector(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        array = array.reshape(-1).copy()
    if not array.flags.c_contiguous or not array.flags.owndata:
        array = np.ascontiguousarray(array).copy()
    return array


def _as_public_matrix(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row) for row in np.asarray(values, dtype=float)
    )


def _path_record_linear_predictor(record: Mapping[str, Any]) -> np.ndarray | None:
    linear_raw = record.get("_linear_predictor")
    if linear_raw is None:
        return None
    linear = np.asarray(linear_raw, dtype=float)
    if linear.ndim != 1 or not np.all(np.isfinite(linear)):
        return None
    return linear


def _path_details_with_public_coefficients(
    details: Mapping[str, Any],
    *,
    param_names: tuple[str, ...],
    original_k: int,
    has_intercept: bool,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    vcov_request: str,
    path_inference_allowed: bool = True,
) -> dict[str, Any]:
    path_details = dict(details)
    records: list[dict[str, Any]] = []
    coefficient_records = 0
    inference_records = 0
    for record in tuple(path_details.get("path", ())):
        item = dict(record)
        beta_reduced_raw = item.get("beta_reduced")
        if beta_reduced_raw is None or not bool(item.get("converged", False)):
            item["path_coefficient_status"] = "unavailable"
            item["path_inference_status"] = "unavailable"
            records.append(item)
            continue

        beta_reduced = np.asarray(beta_reduced_raw, dtype=float)
        beta_full, _, _ = _expand_intercept(
            beta_reduced,
            None,
            None,
            original_k=original_k,
            has_intercept=has_intercept,
        )
        item["beta"] = _as_public_vector(beta_full)
        if len(param_names) == beta_full.shape[0]:
            item["params"] = {
                name: float(value)
                for name, value in zip(param_names, beta_full, strict=True)
            }
        item["path_coefficient_status"] = "available"
        coefficient_records += 1

        if (
            not path_inference_allowed
            or vcov_request.lower() != "iid"
            or beta_reduced.size == 0
            or _uses_joint_hdfe_state(groups)
        ):
            item["path_inference_status"] = "not_computed"
            records.append(item)
            continue

        try:
            se_reduced, covariance_reduced = _manyway_iols_covariance(
                endog=endog,
                exog=exog,
                beta=beta_reduced,
                groups=groups,
                tau=float(item["tau"]),
                se_type="iid",
                linear=_path_record_linear_predictor(item),
            )
        except RuntimeError as exc:
            if not is_known_fe_runtime_failure(exc):
                raise
            item["path_inference_status"] = "not_computed"
            item["path_inference_reason"] = str(exc)
            records.append(item)
            continue
        _, se_full, covariance_full = _expand_intercept(
            beta_reduced,
            se_reduced,
            covariance_reduced,
            original_k=original_k,
            has_intercept=has_intercept,
        )
        if se_full is None or covariance_full is None:
            item["path_inference_status"] = "not_computed"
            records.append(item)
            continue
        item["std_errors"] = _as_public_vector(se_full)
        item["covariance"] = _as_public_matrix(covariance_full)
        if len(param_names) == se_full.shape[0]:
            item["bse"] = {
                name: float(value)
                for name, value in zip(param_names, se_full, strict=True)
            }
        item["path_inference_status"] = (
            "computed"
            if np.all(np.isfinite(se_full))
            else "computed_with_nonfinite_components"
        )
        inference_records += 1
        records.append(item)

    path_details["path"] = tuple(records)
    path_details["path_coefficient_status"] = (
        "available" if coefficient_records else "unavailable"
    )
    if inference_records == coefficient_records and coefficient_records:
        path_details["path_inference_status"] = "computed"
    elif inference_records:
        path_details["path_inference_status"] = "partial"
    else:
        path_details["path_inference_status"] = "not_computed"
    return path_details


def _is_singleton_intercept_group(
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> bool:
    if len(groups) != 1 or tuple(int(count) for count in counts) != (1,):
        return False
    group = np.asarray(groups[0], dtype=int)
    return bool(group.size == 0 or np.all(group == 0))


def _logsumexp_all(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("-inf")
    if np.any(np.isposinf(array)):
        return float("inf")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("-inf")
    max_value = float(np.max(finite))
    return max_value + float(np.log(np.sum(np.exp(finite - max_value))))


def _profile_intercept_adjustment(
    *,
    endog: np.ndarray,
    linear: np.ndarray,
    tau: float,
) -> float:
    endog_arr = np.asarray(endog, dtype=float)
    linear_arr = np.asarray(linear, dtype=float)
    positive = endog_arr > 0.0
    log_numerator_terms = np.full(linear_arr.shape[0], -np.inf, dtype=float)
    log_numerator_terms[positive] = (
        np.log(endog_arr[positive]) + (1.0 - tau) * linear_arr[positive]
    )
    log_denominator_terms = (2.0 - tau) * linear_arr
    log_numerator = _logsumexp_all(log_numerator_terms)
    log_denominator = _logsumexp_all(log_denominator_terms)
    if np.isfinite(log_numerator) and np.isfinite(log_denominator):
        return float(log_numerator - log_denominator)
    return 0.0


def _demean_singleton_intercept(
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array.copy()
    if weights is None:
        return array - np.mean(array, axis=0)

    weight_arr = np.asarray(weights, dtype=float)
    if weight_arr.shape != (array.shape[0],):
        raise ValueError("weights and group must have the same shape")
    weight_sum = float(np.sum(weight_arr))
    if weight_sum <= 1e-15:
        return array.copy()
    if array.ndim == 1:
        mean = float(np.sum(weight_arr * array) / weight_sum)
    else:
        mean = np.sum(weight_arr[:, None] * array, axis=0) / weight_sum
    return array - mean


def _profile_linear(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    maxiter: int = 10000,
    tol: float = 1e-6,
) -> np.ndarray:
    base_linear = exog @ beta
    if len(groups) == 0:
        return base_linear
    if _is_singleton_intercept_group(groups, counts):
        return base_linear + _profile_intercept_adjustment(
            endog=endog,
            linear=base_linear,
            tau=tau,
        )
    if len(groups) <= 2:
        alpha, gamma = profile_fe(
            endog,
            exog,
            beta,
            groups[0],
            groups[1] if len(groups) == 2 else None,
            counts[0],
            counts[1] if len(groups) == 2 else None,
            tau=tau,
            maxiter=maxiter,
            tol=tol,
        )
        linear = base_linear + alpha[groups[0]]
        if gamma is not None:
            linear = linear + gamma[groups[1]]
        return linear
    _, linear = profile_fe_manyway(
        endog,
        exog,
        beta,
        groups,
        tau=tau,
        maxiter=maxiter,
        tol=tol,
    )
    return linear


def _demean_for_groups(
    values: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    *,
    tol: float = 1e-10,
    maxiter: int = 10000,
) -> np.ndarray:
    if len(groups) == 0:
        return np.asarray(values, dtype=float).copy()
    if _is_singleton_intercept_group(groups, counts):
        return _demean_singleton_intercept(values)
    if len(groups) <= 2:
        return demean(
            values,
            groups[0],
            groups[1] if len(groups) == 2 else None,
            counts[0],
            counts[1] if len(groups) == 2 else None,
            tol=tol,
            maxiter=maxiter,
        )
    return demean_manyway(values, groups, n_groups=counts, tol=tol, maxiter=maxiter)


def _weighted_demean_for_groups(
    values: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    weights: np.ndarray,
    *,
    tol: float = 1e-10,
    maxiter: int = 10000,
) -> np.ndarray:
    if len(groups) == 0:
        return np.asarray(values, dtype=float).copy()
    if _is_singleton_intercept_group(groups, counts):
        return _demean_singleton_intercept(values, weights=weights)
    if len(groups) <= 2:
        return demean(
            values,
            groups[0],
            groups[1] if len(groups) == 2 else None,
            counts[0],
            counts[1] if len(groups) == 2 else None,
            weights=weights,
            tol=tol,
            maxiter=maxiter,
        )
    return demean_manyway(
        values,
        groups,
        n_groups=counts,
        weights=weights,
        tol=tol,
        maxiter=maxiter,
    )


def _weighted_demean_for_groups_checked(
    values: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    weights: np.ndarray,
    *,
    tol: float = 1e-10,
    maxiter: int = 10000,
) -> tuple[np.ndarray, bool]:
    if len(groups) == 0:
        return np.asarray(values, dtype=float).copy(), True
    if _is_singleton_intercept_group(groups, counts):
        return _demean_singleton_intercept(values, weights=weights), True
    return _demean_manyway_checked(
        values,
        groups,
        n_groups=counts,
        weights=weights,
        tol=tol,
        maxiter=maxiter,
    )


def _uses_joint_hdfe_state(groups: tuple[np.ndarray, ...]) -> bool:
    return len(groups) >= 2


def _fullrecover_fe_policy(groups: tuple[np.ndarray, ...]) -> str:
    if _uses_joint_hdfe_state(groups):
        return FULLRECOVER_JOINT_FE_POLICY
    return FULLRECOVER_PROFILED_FE_POLICY


def _has_recovered_fe_support_scope(
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> bool:
    if len(groups) == 0:
        return False
    if len(groups) == 1 and counts == (1,):
        group = np.asarray(groups[0], dtype=int)
        return bool(group.size and not np.all(group == 0))
    return True


def _recover_fe_linear(
    values: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    *,
    tol: float = 1e-10,
    maxiter: int = 10000,
) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    if len(groups) == 0:
        return np.zeros_like(values_arr, dtype=float)
    return values_arr - _demean_for_groups(
        values_arr,
        groups,
        counts,
        tol=tol,
        maxiter=maxiter,
    )


def _fullrecover_invariant_state(
    *,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    control: IOLSKernelControl,
) -> _FullrecoverInvariantState:
    x_demeaned = _demean_for_groups(
        exog,
        groups,
        counts,
        tol=float(control.fe_demean_tol),
        maxiter=int(control.fe_demean_maxiter),
    )
    projection = np.linalg.lstsq(
        x_demeaned.T @ x_demeaned,
        x_demeaned.T,
        rcond=None,
    )[0]
    return _FullrecoverInvariantState(
        x_demeaned=x_demeaned,
        projection=projection,
    )


def _gram_condition(values: np.ndarray) -> float | None:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return None
    gram = matrix.T @ matrix
    if not np.all(np.isfinite(gram)):
        return float("inf")
    try:
        return float(np.linalg.cond(gram))
    except np.linalg.LinAlgError:
        return float("inf")


def _transformed_design_for_normalization(
    *,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    control: IOLSKernelControl,
) -> np.ndarray:
    return _demean_for_groups(
        exog,
        groups,
        counts,
        tol=float(control.fe_demean_tol),
        maxiter=int(control.fe_demean_maxiter),
    )


def _build_transformed_normalization_state(
    *,
    topology: KernelTopology,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    control: IOLSKernelControl,
) -> _TransformedNormalizationState:
    mode = _transformed_normalization_mode(control)
    if mode == "none":
        return _TransformedNormalizationState(
            requested=False,
            enabled=False,
            status="disabled",
        )
    if exog.shape[1] == 0:
        return _TransformedNormalizationState(
            requested=True,
            enabled=False,
            status="no_slope_columns",
        )

    transformed = _transformed_design_for_normalization(
        exog=exog,
        groups=groups,
        counts=counts,
        control=control,
    )
    if transformed.shape != exog.shape or not np.all(np.isfinite(transformed)):
        raise ValueError("transformed_design_normalization_failed")

    raw_scale = np.std(transformed, axis=0)
    positive_scale = raw_scale[raw_scale > 0.0]
    max_scale = float(np.max(positive_scale)) if positive_scale.size else 0.0
    scale_floor = max(
        TRANSFORMED_NORMALIZATION_REL_FLOOR * max_scale,
        TRANSFORMED_NORMALIZATION_ABS_BACKSTOP,
    )
    near_zero = (raw_scale > 0.0) & (raw_scale < scale_floor)
    if np.any(near_zero):
        raise ValueError(
            "S_singular: transformed-design scale is below the relative floor"
        )

    scale = np.where(raw_scale == 0.0, 1.0, raw_scale)
    transformed_star = transformed / scale
    return _TransformedNormalizationState(
        requested=True,
        enabled=True,
        status="enabled",
        basis="std",
        scale=np.asarray(scale, dtype=float),
        condition_before=_gram_condition(transformed),
        condition_after=_gram_condition(transformed_star),
        scale_min=float(np.min(scale)) if scale.size else None,
        scale_max=float(np.max(scale)) if scale.size else None,
        scale_floor=float(scale_floor),
        transformed_rank=int(np.linalg.matrix_rank(transformed)),
    )


def _apply_transformed_normalization(
    exog: np.ndarray,
    state: _TransformedNormalizationState,
) -> np.ndarray:
    if not state.enabled or state.scale is None:
        return exog
    return np.asarray(exog, dtype=float) / state.scale


def _back_transform_beta(
    beta: np.ndarray,
    state: _TransformedNormalizationState,
) -> np.ndarray:
    values = np.asarray(beta, dtype=float)
    if not state.enabled or state.scale is None:
        return values
    return values / state.scale


def _back_transform_residual(
    beta: np.ndarray,
    beta_star: np.ndarray,
    state: _TransformedNormalizationState,
) -> float | None:
    if not state.enabled or state.scale is None:
        return None
    canonical = np.asarray(beta, dtype=float)
    expected = np.asarray(beta_star, dtype=float) / state.scale
    if canonical.shape != expected.shape:
        return float("inf")
    return float(np.max(np.abs(canonical - expected))) if canonical.size else 0.0


def _linear_back_transform_residual(
    *,
    exog_canonical: np.ndarray | None,
    exog_star: np.ndarray | None,
    beta: np.ndarray,
    beta_star: np.ndarray,
) -> float | None:
    if exog_canonical is None or exog_star is None:
        return None
    canonical_x = np.asarray(exog_canonical, dtype=float)
    star_x = np.asarray(exog_star, dtype=float)
    canonical_beta = np.asarray(beta, dtype=float)
    star_beta = np.asarray(beta_star, dtype=float)
    if (
        canonical_x.ndim != 2
        or star_x.ndim != 2
        or canonical_beta.ndim != 1
        or star_beta.ndim != 1
        or canonical_x.shape != star_x.shape
        or canonical_x.shape[1] != canonical_beta.shape[0]
        or star_x.shape[1] != star_beta.shape[0]
    ):
        return float("inf")
    canonical_linear = canonical_x @ canonical_beta
    star_linear = star_x @ star_beta
    if canonical_linear.shape != star_linear.shape:
        return float("inf")
    if canonical_linear.size == 0:
        return 0.0
    return float(np.max(np.abs(canonical_linear - star_linear)))


def _normalization_details(
    state: _TransformedNormalizationState,
    *,
    back_transform_residual: float | None = None,
    linear_back_transform_residual: float | None = None,
) -> dict[str, Any]:
    if not state.requested:
        return {}
    details: dict[str, Any] = {
        "transformed_design_normalization": "scale",
        "transformed_design_normalization_enabled": bool(state.enabled),
        "transformed_design_normalization_status": state.status,
        "transformed_design_normalization_basis": state.basis,
        "transformed_design_normalization_centering": "no_mean_shift",
        "transformed_design_scale_statistic": "centered_std_ddof0",
    }
    if state.scale is not None:
        details.update(
            {
                "transformed_design_scale_factors": tuple(
                    float(value) for value in state.scale
                ),
                "transformed_design_scale_min": state.scale_min,
                "transformed_design_scale_max": state.scale_max,
                "transformed_design_scale_floor": state.scale_floor,
                "transformed_design_scale_ratio": (
                    None
                    if not state.scale_min or state.scale_min <= 0.0
                    else float(state.scale_max) / float(state.scale_min)
                ),
                "transformed_design_condition_before": state.condition_before,
                "transformed_design_condition_after": state.condition_after,
                "transformed_design_rank": state.transformed_rank,
            }
        )
    if back_transform_residual is not None:
        details["back_transform_residual"] = float(back_transform_residual)
    if linear_back_transform_residual is not None:
        details["linear_back_transform_residual"] = float(
            linear_back_transform_residual
        )
    return details


def _path_details_back_transformed(
    details: Mapping[str, Any],
    state: _TransformedNormalizationState,
    *,
    exog_canonical: np.ndarray | None = None,
    exog_star: np.ndarray | None = None,
) -> dict[str, Any]:
    if not state.enabled or state.scale is None:
        return dict(details)
    updated = dict(details)
    records: list[Any] = []
    for raw_record in tuple(updated.get("path", ())):
        if not isinstance(raw_record, Mapping):
            records.append(raw_record)
            continue
        record = dict(raw_record)
        beta_star_raw = record.get("beta_reduced")
        if beta_star_raw is not None:
            beta_star = np.asarray(beta_star_raw, dtype=float)
            beta = _back_transform_beta(beta_star, state)
            linear_residual = _linear_back_transform_residual(
                exog_canonical=exog_canonical,
                exog_star=exog_star,
                beta=beta,
                beta_star=beta_star,
            )
            record["beta_star_reduced"] = _as_public_vector(beta_star)
            record["beta_reduced"] = _as_owned_float64_vector(beta)
            diagnostics = dict(record.get("diagnostics", {}))
            diagnostics.update(
                {
                    "back_transform_residual": _back_transform_residual(
                        beta,
                        beta_star,
                        state,
                    ),
                    "transformed_design_normalization_enabled": True,
                    "criterion_scale": "star",
                    "criterion_star_available": True,
                }
            )
            if linear_residual is not None:
                diagnostics["linear_back_transform_residual"] = linear_residual
            record["diagnostics"] = diagnostics
        theta_star_raw = record.get("theta_reduced")
        if theta_star_raw is not None:
            theta_star = np.asarray(theta_star_raw, dtype=float)
            if theta_star.ndim != 1 or theta_star.shape[0] < state.scale.shape[0]:
                raise ValueError(
                    "transformed_design_normalization_theta_shape_mismatch"
                )
            beta_star = theta_star[: state.scale.shape[0]]
            beta = _back_transform_beta(beta_star, state)
            theta = np.concatenate((beta, theta_star[state.scale.shape[0] :]))
            record["theta_star_reduced"] = _as_public_vector(theta_star)
            record["theta_reduced"] = _as_owned_float64_vector(theta)
            record["recover_fe_reduced"] = _as_owned_float64_vector(
                theta[state.scale.shape[0] :]
            )
            diagnostics = dict(record.get("diagnostics", {}))
            diagnostics.update(
                {
                    "theta_back_transform_residual": _back_transform_residual(
                        theta[: state.scale.shape[0]],
                        beta_star,
                        state,
                    ),
                    "mixed_recover_fe_normalization_policy": "unscaled",
                }
            )
            record["diagnostics"] = diagnostics
        records.append(record)
    updated["path"] = tuple(records)
    return updated


def _recovered_fe_tau2_boundary_separation(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> dict[str, Any]:
    """Detect Gamma/tau=2 boundary separation for recovered-FE designs.

    For the tau=2 fullrecover anchor, the finite-objective condition rules out
    directions v in span(X, FE) where v is nonnegative on positive outcomes and
    has negative total sum. Along such a direction, the Gamma/iOLS tau=2
    quasi-log-likelihood is unbounded above, equivalently the minimization
    criterion is unbounded below, so no finite fixed point should be reported.
    """

    y = np.asarray(endog, dtype=float)
    positive = y > 0.0
    nonpositive = ~positive
    positive_count = int(np.sum(positive))
    nonpositive_count = int(np.sum(nonpositive))
    support_check = (
        TAU2_JOINT_HDFE_SUPPORT_CHECK
        if _uses_joint_hdfe_state(groups)
        else TAU2_RECOVERED_FE_SUPPORT_CHECK
    )
    support_scope = "joint_hdfe" if _uses_joint_hdfe_state(groups) else "recovered_fe"
    details: dict[str, Any] = {
        "support_check": support_check,
        "support_check_scope": support_scope,
        "support_policy": TAU2_BOUNDARY_SUPPORT_POLICY,
        "future_support_policy": TAU2_BOUNDARY_FUTURE_POLICY,
        "positive_observations": positive_count,
        "nonpositive_observations": nonpositive_count,
        "separation_detected": False,
        "separation_certificate": (
            "exists v in span(X, recovered FE) with v[y>0] >= 0 and sum(v) <= -1"
        ),
        "direction_tolerance": TAU2_BOUNDARY_DIRECTION_TOL,
    }
    if not _has_recovered_fe_support_scope(groups, counts) or not np.any(nonpositive):
        return details

    n_obs = int(y.shape[0])
    blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(np.asarray(exog, dtype=float))]
    row_index = np.arange(n_obs, dtype=int)
    for group, count in zip(groups, counts, strict=True):
        blocks.append(
            sparse.csr_matrix(
                (
                    np.ones(n_obs, dtype=float),
                    (row_index, np.asarray(group, dtype=int)),
                ),
                shape=(n_obs, int(count)),
            )
        )
    design = sparse.hstack(blocks, format="csr")
    details["support_design_columns"] = int(design.shape[1])

    # Feasibility problem:
    #   v_pos >= 0 and sum_i v_i <= -1, where v = design @ theta.
    # The normalization removes the homogeneous scale indeterminacy.
    positive_design = design[positive]
    total_design = sparse.csr_matrix(design.sum(axis=0))
    a_ub = sparse.vstack((-positive_design, total_design), format="csr")
    b_ub = np.concatenate(
        (
            np.zeros(int(np.sum(positive)), dtype=float),
            np.array([-1.0], dtype=float),
        )
    )
    try:
        result = linprog(
            c=np.zeros(design.shape[1], dtype=float),
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(None, None)] * int(design.shape[1]),
            method="highs",
        )
    except Exception as exc:  # noqa: BLE001 - support check must be diagnostic.
        details.update(
            {
                "support_check_status": "error",
                "support_check_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return details

    details.update(
        {
            "support_check_status": str(result.message),
            "support_check_success": bool(result.success),
        }
    )
    if not result.success:
        return details

    direction = np.asarray(design @ np.asarray(result.x, dtype=float)).ravel()
    positive_direction = direction[positive]
    nonpositive_direction = direction[nonpositive]
    direction_sum = float(np.sum(direction))
    positive_min = (
        float(np.min(positive_direction)) if positive_count > 0 else float("nan")
    )
    tol = TAU2_BOUNDARY_DIRECTION_TOL
    if (positive_count > 0 and positive_min < -tol) or direction_sum > -1.0 + tol:
        details.update(
            {
                "support_check_status": "solution_constraint_violation",
                "support_check_success": False,
                "direction_positive_min": positive_min,
                "direction_sum": direction_sum,
            }
        )
        return details

    negative_rows = np.flatnonzero(direction < -tol)
    positive_binding_rows = np.flatnonzero(positive & (direction <= tol))
    positive_strict_count = int(np.sum(positive_direction > tol))
    positive_binding_count = int(np.sum(positive_direction <= tol))
    nonpositive_negative_count = int(np.sum(nonpositive_direction < -tol))
    nonpositive_positive_count = int(np.sum(nonpositive_direction > tol))
    separation_summary = (
        f"LP certificate found a direction in span(X, recovered FE) that is "
        f"nonnegative on all {positive_count} positive-outcome observations "
        f"and has normalized total direction {direction_sum:.6g}; "
        f"{nonpositive_negative_count} of {nonpositive_count} nonpositive "
        "outcome observations have negative direction."
    )
    warning_scope = (
        "joint-HDFE fullrecover"
        if _uses_joint_hdfe_state(groups)
        else "recovered-FE fullrecover"
    )
    separation_warning = (
        f"Tau=2 {warning_scope} boundary separation detected. "
        f"{separation_summary} Under the current support policy the package "
        "fails explicitly and does not drop observations automatically; "
        "support_policy='drop_separated' is future optional work that requires "
        "theory, diagnostics, sample accounting, covariance, and Monte Carlo "
        "evidence."
    )
    details.update(
        {
            "separation_detected": True,
            "separation_warning_code": TAU2_BOUNDARY_SEPARATION_CODE,
            "separation_warning": separation_warning,
            "warning_messages": (separation_warning,),
            "separation_summary": separation_summary,
            "direction_positive_min": positive_min,
            "direction_positive_max": float(np.max(positive_direction))
            if positive_count > 0
            else float("nan"),
            "direction_nonpositive_min": float(np.min(nonpositive_direction)),
            "direction_nonpositive_max": float(np.max(nonpositive_direction)),
            "direction_sum": direction_sum,
            "direction_positive_binding_count": positive_binding_count,
            "direction_positive_strict_count": positive_strict_count,
            "direction_nonpositive_negative_count": nonpositive_negative_count,
            "direction_nonpositive_positive_count": nonpositive_positive_count,
            "direction_negative_row_count": int(negative_rows.size),
            "direction_negative_row_preview": tuple(
                int(index) for index in negative_rows[:20]
            ),
            "direction_positive_binding_row_preview": tuple(
                int(index) for index in positive_binding_rows[:20]
            ),
        }
    )
    return details


def _initial_linear_state(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    control: IOLSKernelControl,
) -> np.ndarray:
    if not _uses_joint_hdfe_state(groups):
        return _profile_linear(
            endog=endog,
            exog=exog,
            beta=beta,
            groups=groups,
            counts=counts,
            tau=tau,
            maxiter=int(control.fe_profile_maxiter),
            tol=float(control.fe_profile_tol),
        )
    base = exog @ np.asarray(beta, dtype=float)
    target = np.log(np.asarray(endog, dtype=float) + 1.0)
    recovered = _recover_fe_linear(
        target - base,
        groups,
        counts,
        tol=float(control.fe_demean_tol),
        maxiter=int(control.fe_demean_maxiter),
    )
    return base + recovered


def _effective_anderson_m(anderson_m: int, _phase: int, _param: float) -> int:
    return int(anderson_m)


def _adapt_anderson_m(aa_m: int, kappa_val: float) -> int:
    if not np.isfinite(kappa_val):
        return aa_m
    if kappa_val >= 0.95:
        return max(aa_m, 20)
    if kappa_val >= 0.85:
        return max(aa_m, 12)
    if kappa_val >= 0.70:
        return max(aa_m, 8)
    return aa_m


_DIRECT_KAPPA_WINDOW: Final = 10


def _windowed_contraction_rate(history: list[float], window: int = 10) -> float:
    """runtime_v3 direct-route kappa statistic: geometric-mean step contraction.

    Returns (c_last / c_{last-window}) ** (1/window) from the recorded finite
    history (runtime_v4 direct route: max-abs plain-map residuals at the
    current iterate, see DIRECT_PROGRESS_MONITOR_POLICY), i.e. the average
    per-iteration contraction over the last ``window`` iterations. It is
    scale-free, is not dominated by single coordinates whose previous change
    passed through zero, and is reset by construction at every check. NaN when
    fewer than ``window + 1`` values are recorded or they are not usable.
    """
    if window < 1 or len(history) < window + 1:
        return float("nan")
    current = float(history[-1])
    previous = float(history[-1 - window])
    if not (math.isfinite(current) and math.isfinite(previous)):
        return float("nan")
    if previous <= 0.0 or current < 0.0:
        return float("nan")
    return float((current / previous) ** (1.0 / float(window)))


def _direct_progress_trace_fields(
    progress_history: list[float],
    *,
    aa_m_start: int,
    aa_m_end: int,
    aa_accepted: int,
) -> dict[str, Any]:
    """runtime_v4 per-attempt progress-monitor fields (twin of the C++ record).

    The v3 criterion fields keep their meaning (accepted coefficient moves);
    these add the plain-map residual monitor and the Anderson memory in use.
    """
    return {
        "final_progress_residual": (
            float(progress_history[-1]) if progress_history else float("nan")
        ),
        "best_progress_residual": (
            float(min(progress_history)) if progress_history else float("nan")
        ),
        "anderson_m_start": int(aa_m_start),
        "anderson_m_end": int(aa_m_end),
        "anderson_accepted": int(aa_accepted),
    }


def _criterion_stagnated(
    history: list[float],
    *,
    eps: float,
    window: int = 50,
    min_iterations: int = 100,
    min_relative_drop: float = 0.01,
) -> bool:
    if len(history) < max(window * 2, min_iterations):
        return False
    current = float(history[-1])
    previous = float(history[-window])
    if not (np.isfinite(current) and np.isfinite(previous)):
        return False
    if current <= eps * 10.0:
        return False
    relative_drop = (previous - current) / max(abs(previous), eps)
    return bool(relative_drop < min_relative_drop)


def _record_criterion_progress(
    history: list[float],
    criterion: float,
    *,
    phase: int,
    tau: float,
    eps: float,
    max_phase2_evaluations: int,
    last_finite_criterion: float,
    best_finite_criterion: float,
    stagnation_history: list[float] | None = None,
) -> tuple[float, float, str | None]:
    # runtime_v4: the direct route passes its plain-map residual history as
    # stagnation_history; every other caller keeps the criterion history.
    if not np.isfinite(criterion):
        return last_finite_criterion, best_finite_criterion, None
    criterion_float = float(criterion)
    history.append(criterion_float)
    last_finite_criterion = criterion_float
    best_finite_criterion = (
        min(float(best_finite_criterion), criterion_float)
        if np.isfinite(best_finite_criterion)
        else criterion_float
    )
    if phase == 1 and tau > 2.0 and len(history) >= 20:
        criterion_ref = history[-20]
        if (
            np.isfinite(criterion_ref)
            and criterion_ref > eps
            and criterion_float > max(criterion_ref * 100.0, 1e6)
        ):
            return (
                last_finite_criterion,
                best_finite_criterion,
                KERNEL_FAILURE_CRITERION_EXPLOSION,
            )
    no_progress_min_iterations = min(
        max(int(max_phase2_evaluations) // 2500, 20),
        200,
    )
    no_progress_window = min(max(no_progress_min_iterations // 2, 10), 50)
    if phase == 1 and _criterion_stagnated(
        history if stagnation_history is None else stagnation_history,
        eps=eps,
        window=no_progress_window,
        min_iterations=no_progress_min_iterations,
        min_relative_drop=1e-4,
    ):
        return (
            last_finite_criterion,
            best_finite_criterion,
            KERNEL_FAILURE_CRITERION_STAGNATION,
        )
    return last_finite_criterion, best_finite_criterion, None


def _phase2_expensive_failed_step_threshold(control: IOLSKernelControl) -> int:
    budget = max(int(control.max_phase2_evaluations), 1)
    return min(budget, max(budget // 20, 1000))


def _phase2_weighted_demean_maxiter(control: IOLSKernelControl) -> int:
    budget = max(int(control.max_phase2_evaluations), 1)
    # Phase 2 weighted within transforms are bounded by the same evaluation
    # budget that controls fixed-point continuation work.
    bounded = max(budget // 500, 100)
    return max(1, min(int(control.fe_demean_maxiter), bounded))


def _fullcancel_phi_demean_maxiter(
    control: IOLSKernelControl,
    *,
    tau: float,
    phase: int,
) -> int:
    if phase == 1 and abs(float(tau) - 2.0) > 1e-10:
        return _phase2_weighted_demean_maxiter(control)
    return int(control.fe_demean_maxiter)


def _fullcancel_effective_anderson_m(
    control: IOLSKernelControl,
    *,
    tau: float,
    phase: int,
) -> int:
    return _effective_anderson_m(control.anderson_m, phase, 0.0)


def _control_with_anderson_m(
    control: IOLSKernelControl,
    *,
    anderson_m: int,
) -> IOLSKernelControl:
    values = {
        item.name: getattr(control, item.name)
        for item in fields(IOLSKernelControl)
    }
    values["anderson_m"] = int(anderson_m)
    return IOLSKernelControl(**values)


def _fullcancel_tau_path_control(
    control: IOLSKernelControl,
    *,
    tau_target: float,
) -> IOLSKernelControl:
    if float(tau_target) > 2.0 + 1e-10 and int(control.anderson_m) != 0:
        return _control_with_anderson_m(control, anderson_m=0)
    return control


def _tau_path_expensive_failed_step(
    iteration: IOLSIterationDiagnostics,
    *,
    control: IOLSKernelControl,
) -> bool:
    if iteration.converged:
        return False
    if iteration.failure_reason == KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED:
        return True
    if iteration.failure_reason != KERNEL_FAILURE_CRITERION_STAGNATION:
        return False
    phase2_evaluations = int(iteration.details.get("phase2_evaluations", 0))
    return phase2_evaluations >= _phase2_expensive_failed_step_threshold(control)


def _tau_path_score_warning_refinement_step(
    iteration: IOLSIterationDiagnostics,
) -> bool:
    if not iteration.converged:
        return False
    return str(iteration.details.get("score_validation_status", "")) == (
        "diagnostic_warning"
    )


def _focus_rho_path(rho_path: np.ndarray, rho_hint: float | None = None) -> np.ndarray:
    path = np.asarray(rho_path, dtype=float)
    path = path[np.isfinite(path) & (path > 0.0)]
    if path.size == 0:
        return path
    if rho_hint is None or not np.isfinite(rho_hint) or rho_hint <= 0.0:
        return path

    hint = min(max(float(rho_hint), float(path[0])), float(path[-1]))
    focus_mult = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
    ordered: list[float] = []
    seen: set[float] = set()
    focus = np.minimum(np.maximum(hint * focus_mult, path[0]), path[-1])
    for value in np.concatenate([focus, path]):
        finite_value = float(value)
        if not np.isfinite(finite_value) or finite_value <= 0.0 or finite_value in seen:
            continue
        seen.add(finite_value)
        ordered.append(finite_value)
    return np.asarray(ordered, dtype=float)


def _rho_base_for_tau(tau: float, control: IOLSKernelControl) -> np.ndarray:
    # The base grid is tau-independent for now; the argument preserves the
    # signature needed for future tau-adaptive grid policies.
    _ = tau
    grid_size = max(int(control.rho_grid_size), 2)
    log_min = float(control.rho_log_min)
    log_max = float(control.rho_log_max)
    if not (np.isfinite(log_min) and np.isfinite(log_max)):
        raise ValueError("rho log-grid bounds must be finite")
    if log_max <= log_min:
        raise ValueError("rho_log_max must be greater than rho_log_min")
    return np.exp(np.linspace(log_min, log_max, grid_size))


def _delta_base_for_anchor(control: IOLSKernelControl) -> np.ndarray:
    grid_size = max(int(control.delta_grid_size), 0)
    if grid_size == 0:
        return np.array([], dtype=float)
    log_min = float(control.delta_log_min)
    log_max = float(control.delta_log_max)
    if not (np.isfinite(log_min) and np.isfinite(log_max)):
        raise ValueError("delta log-grid bounds must be finite")
    if log_max <= log_min:
        raise ValueError("delta_log_max must be greater than delta_log_min")
    return np.exp(np.linspace(log_min, log_max, grid_size))


def _rho_path_for_linear_state(
    *,
    linear: np.ndarray,
    tau: float,
    rho_hint: float | None,
    control: IOLSKernelControl,
) -> tuple[np.ndarray, dict[str, Any]]:
    linear_arr = np.asarray(linear, dtype=float)
    base = _rho_base_for_tau(tau, control)
    rho_path = np.unique(base[np.isfinite(base) & (base > 0.0)])
    rho_path = _focus_rho_path(rho_path, rho_hint=rho_hint)
    details = {
        "rho_policy": "fixed_log_grid",
        "linear_min": float(np.nanmin(linear_arr)) if linear_arr.size else float("nan"),
        "linear_max": float(np.nanmax(linear_arr)) if linear_arr.size else float("nan"),
        "rho_candidate_count": int(rho_path.size),
        "rho_hint": None if rho_hint is None else float(rho_hint),
        "rho_log_min": float(control.rho_log_min),
        "rho_log_max": float(control.rho_log_max),
        "rho_grid_size": int(control.rho_grid_size),
    }
    if rho_path.size:
        details.update(
            {
                "rho_min": float(rho_path[0]),
                "rho_max": float(rho_path[-1]),
            }
        )
    return rho_path, details


def _phi_step_fullrecover(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    linear: np.ndarray | None,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    projection: np.ndarray,
    phase: int,
    param: float,
    fe_profile_maxiter: int = 10000,
    fe_profile_tol: float = 1e-6,
    fe_demean_maxiter: int = 10000,
    fe_demean_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    joint_hdfe = _uses_joint_hdfe_state(groups)
    if joint_hdfe:
        if linear is None:
            return beta.copy(), None, False
        linear_current = np.asarray(linear, dtype=float)
    else:
        try:
            linear_current = _profile_linear(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                counts=counts,
                tau=tau,
                maxiter=fe_profile_maxiter,
                tol=fe_profile_tol,
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return beta.copy(), linear, False
            raise
    if not np.all(np.isfinite(linear_current)):
        return beta.copy(), linear, False

    mu = exp_with_float64_overflow_guard(linear_current)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u_ratio = endog / mu
        weights = exp_with_float64_overflow_guard((2.0 - tau) * linear_current)
        if phase == 0:
            delta = float(param)
            delta_tilde = delta / np.maximum(weights, 1e-300)
            pseudo = linear_current + np.log1p((u_ratio - 1.0) / (1.0 + delta_tilde))
        else:
            rho = float(param)
            log_denom = np.log1p(rho)
            positive = endog > 0.0
            positive_component = np.zeros_like(linear_current, dtype=float)
            positive_log = np.full(linear_current.shape[0], -np.inf, dtype=float)
            positive_log[positive] = (
                np.log(endog[positive])
                + (1.0 - tau) * linear_current[positive]
                - log_denom
            )
            positive_component[positive] = exp_with_float64_overflow_guard(
                positive_log[positive]
            )
            negative_component = exp_with_float64_overflow_guard(
                (2.0 - tau) * linear_current - log_denom
            )
            pseudo = linear_current + positive_component - negative_component
    if not np.all(np.isfinite(pseudo)):
        return beta.copy(), linear, False
    try:
        demeaned_pseudo = _demean_for_groups(
            pseudo,
            groups,
            counts,
            tol=fe_demean_tol,
            maxiter=fe_demean_maxiter,
        )
    except RuntimeError as exc:
        if is_known_fe_runtime_failure(exc):
            return beta.copy(), linear, False
        raise
    beta_new = projection @ demeaned_pseudo
    if not np.all(np.isfinite(beta_new)):
        return beta.copy(), linear, False
    if not joint_hdfe:
        return np.asarray(beta_new, dtype=float), None, True

    try:
        recovered = _recover_fe_linear(
            pseudo - exog @ beta_new,
            groups,
            counts,
            tol=fe_demean_tol,
            maxiter=fe_demean_maxiter,
        )
    except RuntimeError as exc:
        if is_known_fe_runtime_failure(exc):
            return beta.copy(), linear, False
        raise
    linear_new = exog @ beta_new + recovered
    if not np.all(np.isfinite(linear_new)):
        return beta.copy(), linear, False
    return np.asarray(beta_new, dtype=float), np.asarray(linear_new, dtype=float), True


def _profiled_slope_score_residual(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    x_demeaned: np.ndarray,
    tau: float,
    linear: np.ndarray | None = None,
    fe_profile_maxiter: int = 10000,
    fe_profile_tol: float = 1e-6,
) -> float:
    if linear is None:
        try:
            linear_arr = _profile_linear(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                counts=counts,
                tau=tau,
                maxiter=fe_profile_maxiter,
                tol=fe_profile_tol,
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return float("inf")
            raise
    else:
        linear_arr = np.asarray(linear, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score_obs = endog * exp_with_float64_overflow_guard(
            (1.0 - tau) * linear_arr
        ) - exp_with_float64_overflow_guard((2.0 - tau) * linear_arr)
    if not np.all(np.isfinite(score_obs)):
        return float("inf")
    score = x_demeaned.T @ score_obs
    if not np.all(np.isfinite(score)):
        return float("inf")
    return float(np.max(np.abs(score)) / max(float(endog.shape[0]), 1.0))


def _profiled_fe_score_residual(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    linear: np.ndarray | None = None,
    fe_profile_maxiter: int = 10000,
    fe_profile_tol: float = 1e-6,
) -> float:
    if linear is None:
        try:
            linear_arr = _profile_linear(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                counts=counts,
                tau=tau,
                maxiter=fe_profile_maxiter,
                tol=fe_profile_tol,
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return float("inf")
            raise
    else:
        linear_arr = np.asarray(linear, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score_obs = endog * exp_with_float64_overflow_guard(
            (1.0 - tau) * linear_arr
        ) - exp_with_float64_overflow_guard((2.0 - tau) * linear_arr)
    if not np.all(np.isfinite(score_obs)):
        return float("inf")
    if len(groups) == 0:
        return abs(float(np.sum(score_obs))) / max(float(endog.shape[0]), 1.0)
    residual = 0.0
    for group, count in zip(groups, counts, strict=True):
        sums = np.bincount(group, weights=score_obs, minlength=count)
        if sums.size:
            residual = max(residual, float(np.max(np.abs(sums))))
    return residual / max(float(endog.shape[0]), 1.0)


def _phase2_initialization_usable(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    linear: np.ndarray | None,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    projection: np.ndarray,
    control: IOLSKernelControl,
) -> bool:
    if not np.all(np.isfinite(beta)):
        return False
    if linear is None:
        try:
            linear_current = _initial_linear_state(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                counts=counts,
                tau=tau,
                control=control,
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return False
            raise
    else:
        linear_current = np.asarray(linear, dtype=float)
    if not np.all(np.isfinite(linear_current)):
        return False
    rho_path, _ = _rho_path_for_linear_state(
        linear=linear_current,
        tau=tau,
        rho_hint=None,
        control=control,
    )
    for rho in rho_path:
        _, _, valid = _phi_step_fullrecover(
            endog=endog,
            exog=exog,
            beta=beta,
            linear=linear_current,
            groups=groups,
            counts=counts,
            tau=tau,
            projection=projection,
            phase=1,
            param=float(rho),
            fe_profile_maxiter=int(control.fe_profile_maxiter),
            fe_profile_tol=float(control.fe_profile_tol),
            fe_demean_maxiter=int(control.fe_demean_maxiter),
            fe_demean_tol=float(control.fe_demean_tol),
        )
        if valid:
            return True
    return False


def _max_abs_change(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


_BETA_PATH_RETAINED: Final = 4


def _record_beta_path_entry(
    beta_path: list[np.ndarray],
    beta: np.ndarray,
    *,
    entries_seen: int,
    kappa_row_max: float,
) -> tuple[int, float]:
    """Append to the bounded beta path and fold in the newest kappa ratio row.

    The kappa monitor is nanmax over every ratio row of beta_path[1:], so a
    running max over rows as they are appended gives the same value without
    retaining or rescanning the full history. qhat and the post-convergence
    contraction check only read the last four entries.
    """
    beta_path.append(beta.copy())
    entries_seen += 1
    if entries_seen >= 4:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.abs(beta_path[-1] - beta_path[-2]) / np.abs(
                beta_path[-2] - beta_path[-3]
            )
        row_mean = float(np.nanmean(ratios))
        if not np.isnan(row_mean):
            kappa_row_max = (
                row_mean if np.isnan(kappa_row_max) else max(kappa_row_max, row_mean)
            )
    if len(beta_path) > _BETA_PATH_RETAINED:
        del beta_path[0]
    return entries_seen, kappa_row_max


def _fullcancel_linear_state(exog: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(exog, dtype=float) @ np.asarray(beta, dtype=float)


def _recover_design_from_groups(
    *,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if not groups:
        raise ValueError("mixed topology requires at least one recover FE block")
    n_obs = int(groups[0].shape[0])
    columns: list[np.ndarray] = []
    column_names: list[str] = []
    for name, group, count in zip(names, groups, counts, strict=True):
        group_arr = np.asarray(group, dtype=int)
        if group_arr.shape[0] != n_obs:
            raise ValueError("recover FE blocks must have the same number of rows")
        for level in range(1, int(count)):
            columns.append((group_arr == level).astype(float))
            column_names.append(f"{name}[{level}]")
    if not columns:
        return np.empty((n_obs, 0), dtype=float), ()
    return np.column_stack(columns), tuple(column_names)


def _design_boundary_separation_diagnostics(
    *,
    endog: np.ndarray,
    design: np.ndarray,
    check_name: str,
    scope: str,
) -> dict[str, Any]:
    y = np.asarray(endog, dtype=float)
    design_arr = np.asarray(design, dtype=float)
    positive = y > 0.0
    nonpositive = ~positive
    positive_count = int(np.sum(positive))
    nonpositive_count = int(np.sum(nonpositive))
    details: dict[str, Any] = {
        "support_check": check_name,
        "support_check_scope": scope,
        "positive_observations": positive_count,
        "nonpositive_observations": nonpositive_count,
        "support_design_columns": int(design_arr.shape[1])
        if design_arr.ndim == 2
        else 0,
        "separation_detected": False,
        "direction_tolerance": TAU2_BOUNDARY_DIRECTION_TOL,
    }
    if (
        design_arr.ndim != 2
        or design_arr.shape[0] != y.shape[0]
        or design_arr.shape[1] == 0
    ):
        details["support_check_status"] = "invalid_design"
        return details
    if not np.all(np.isfinite(design_arr)):
        details["support_check_status"] = "nonfinite_design"
        return details
    if not np.any(nonpositive):
        details["support_check_status"] = "no_nonpositive_outcomes"
        details["support_check_success"] = False
        return details

    design_sparse = sparse.csr_matrix(design_arr)
    positive_design = design_sparse[positive]
    total_design = sparse.csr_matrix(design_sparse.sum(axis=0))
    a_ub = sparse.vstack((-positive_design, total_design), format="csr")
    b_ub = np.concatenate(
        (
            np.zeros(positive_count, dtype=float),
            np.array([-1.0], dtype=float),
        )
    )
    try:
        result = linprog(
            c=np.zeros(design_sparse.shape[1], dtype=float),
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(None, None)] * int(design_sparse.shape[1]),
            method="highs",
        )
    except Exception as exc:  # noqa: BLE001 - support check must be diagnostic.
        details.update(
            {
                "support_check_status": "error",
                "support_check_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return details

    details.update(
        {
            "support_check_status": str(result.message),
            "support_check_success": bool(result.success),
        }
    )
    if not result.success:
        return details

    direction = np.asarray(design_sparse @ np.asarray(result.x, dtype=float)).ravel()
    positive_direction = direction[positive]
    nonpositive_direction = direction[nonpositive]
    direction_sum = float(np.sum(direction))
    positive_min = (
        float(np.min(positive_direction)) if positive_count > 0 else float("nan")
    )
    tol = TAU2_BOUNDARY_DIRECTION_TOL
    if (positive_count > 0 and positive_min < -tol) or direction_sum > -1.0 + tol:
        details.update(
            {
                "support_check_status": "solution_constraint_violation",
                "support_check_success": False,
                "direction_positive_min": positive_min,
                "direction_sum": direction_sum,
            }
        )
        return details

    details.update(
        {
            "separation_detected": True,
            "direction_sum": direction_sum,
            "direction_positive_min": positive_min,
            "direction_positive_max": float(np.max(positive_direction)),
            "positive_strict_count": int(np.sum(positive_direction > tol)),
            "positive_binding_count": int(np.sum(positive_direction <= tol)),
            "nonpositive_negative_count": int(np.sum(nonpositive_direction < -tol)),
            "nonpositive_positive_count": int(np.sum(nonpositive_direction > tol)),
        }
    )
    return details


def _prefix_diagnostics(prefix: str, details: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in details.items()}


def _design_rank_diagnostics(
    design: np.ndarray,
    *,
    prefix: str,
) -> dict[str, Any]:
    design_arr = np.asarray(design, dtype=float)
    details: dict[str, Any] = {
        f"{prefix}_rows": int(design_arr.shape[0]) if design_arr.ndim == 2 else 0,
        f"{prefix}_columns": int(design_arr.shape[1]) if design_arr.ndim == 2 else 0,
        f"{prefix}_rank": 0,
        f"{prefix}_condition_number": float("nan"),
        f"{prefix}_min_singular_value": float("nan"),
        f"{prefix}_max_singular_value": float("nan"),
    }
    if (
        design_arr.ndim != 2
        or design_arr.shape[0] == 0
        or design_arr.shape[1] == 0
        or not np.all(np.isfinite(design_arr))
    ):
        return details
    try:
        singular_values = np.linalg.svd(design_arr, compute_uv=False)
    except np.linalg.LinAlgError:
        details[f"{prefix}_svd_status"] = "failed"
        return details
    if singular_values.size == 0:
        return details
    max_sv = float(singular_values[0])
    min_sv = float(singular_values[-1])
    tol = max(design_arr.shape) * np.finfo(float).eps * max(max_sv, 1.0)
    details.update(
        {
            f"{prefix}_rank": int(np.sum(singular_values > tol)),
            f"{prefix}_condition_number": float("inf")
            if min_sv <= 0.0
            else float(max_sv / min_sv),
            f"{prefix}_min_singular_value": min_sv,
            f"{prefix}_max_singular_value": max_sv,
            f"{prefix}_rank_tolerance": float(tol),
            f"{prefix}_svd_status": "computed",
        }
    )
    return details


def _finalize_mixed_support_conditioning_diagnostics(
    diagnostics: dict[str, Any],
    *,
    raw_support: Mapping[str, Any],
    quotient_support: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics.update(_prefix_diagnostics("mixed_raw_recovered", raw_support))
    diagnostics.update(_prefix_diagnostics("mixed_cancel_quotient", quotient_support))
    raw_separated = bool(raw_support.get("separation_detected", False))
    quotient_separated = bool(quotient_support.get("separation_detected", False))
    all_zero_count = int(diagnostics["mixed_recover_all_zero_cell_count"])
    min_positive_count = int(diagnostics["mixed_recover_min_positive_count"])
    quotient_columns = int(diagnostics["mixed_cancel_quotient_design_columns"])
    quotient_rank = int(diagnostics["mixed_cancel_quotient_design_rank"])
    quotient_condition = float(
        diagnostics["mixed_cancel_quotient_design_condition_number"]
    )
    if quotient_separated:
        classification = "cancel_quotient_boundary_separation_candidate"
    elif raw_separated and (all_zero_count > 0 or min_positive_count <= 1):
        classification = "raw_recovered_boundary_with_short_cells"
    elif raw_separated:
        classification = "raw_recovered_boundary"
    elif quotient_columns and quotient_rank < quotient_columns:
        classification = "cancel_quotient_rank_deficient"
    elif np.isfinite(quotient_condition) and quotient_condition > 1e8:
        classification = "cancel_quotient_ill_conditioned"
    else:
        classification = "passed"
    diagnostics["mixed_support_classification"] = classification
    diagnostics["mixed_support_needs_attention"] = classification != "passed"
    if classification != "passed":
        diagnostics["mixed_support_warning"] = (
            "Mixed tau=2 retained-index support/conditioning diagnostics are "
            f"{classification}. This is not a covariance or inference "
            "failure certificate by itself; if the mixed tau=2 anchor also fails, "
            "the package reports a mixed near-boundary/conditioning failure "
            "instead of a generic numerical budget failure."
        )
    return diagnostics


def _mixed_recover_cell_diagnostics(
    *,
    endog: np.ndarray,
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
) -> dict[str, Any]:
    y = np.asarray(endog, dtype=float)
    all_zero_cells: list[str] = []
    one_positive_cells: list[str] = []
    min_positive_count: int | None = None
    min_cell_size: int | None = None
    max_cell_size = 0
    total_cells = 0
    for name, group, count in zip(
        recover_names,
        recover_groups,
        recover_counts,
        strict=True,
    ):
        group_arr = np.asarray(group, dtype=int)
        for level in range(int(count)):
            mask = group_arr == level
            cell_size = int(np.sum(mask))
            if cell_size == 0:
                continue
            total_cells += 1
            positives = int(np.sum(y[mask] > 0.0))
            min_positive_count = (
                positives
                if min_positive_count is None
                else min(min_positive_count, positives)
            )
            min_cell_size = (
                cell_size if min_cell_size is None else min(min_cell_size, cell_size)
            )
            max_cell_size = max(max_cell_size, cell_size)
            cell_name = f"{name}[{level}]"
            if positives == 0:
                all_zero_cells.append(cell_name)
            elif positives == 1:
                one_positive_cells.append(cell_name)
    return {
        "mixed_recover_cell_count": int(total_cells),
        "mixed_recover_min_cell_size": int(min_cell_size or 0),
        "mixed_recover_max_cell_size": int(max_cell_size),
        "mixed_recover_min_positive_count": int(min_positive_count or 0),
        "mixed_recover_all_zero_cell_count": int(len(all_zero_cells)),
        "mixed_recover_one_positive_cell_count": int(len(one_positive_cells)),
        "mixed_recover_all_zero_cells": tuple(all_zero_cells[:20]),
        "mixed_recover_one_positive_cells": tuple(one_positive_cells[:20]),
    }


def _mixed_support_conditioning_diagnostics(
    *,
    endog: np.ndarray,
    mixed_design: np.ndarray,
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    control: IOLSKernelControl,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "mixed_support_check": MIXED_TAU2_SUPPORT_CONDITIONING_CHECK,
        "mixed_support_policy": MIXED_TAU2_SUPPORT_POLICY,
    }
    diagnostics.update(
        _mixed_recover_cell_diagnostics(
            endog=endog,
            recover_names=recover_names,
            recover_groups=recover_groups,
            recover_counts=recover_counts,
        )
    )
    raw_support = _design_boundary_separation_diagnostics(
        endog=endog,
        design=mixed_design,
        check_name="mixed_raw_recovered_retained_boundary_separation",
        scope="raw_retained_index",
    )
    diagnostics.update(
        _design_rank_diagnostics(
            mixed_design,
            prefix="mixed_raw_retained_design",
        )
    )

    try:
        cancel_quotient = _demean_for_groups(
            mixed_design,
            cancel_groups,
            cancel_counts,
            tol=float(control.fe_demean_tol),
            maxiter=int(control.fe_demean_maxiter),
        )
        diagnostics["mixed_cancel_quotient_status"] = "computed"
    except Exception as exc:  # noqa: BLE001 - diagnostic must not mask solver.
        cancel_quotient = np.empty((np.asarray(mixed_design).shape[0], 0))
        diagnostics["mixed_cancel_quotient_status"] = "error"
        diagnostics["mixed_cancel_quotient_error"] = f"{type(exc).__name__}: {exc}"
    quotient_support = _design_boundary_separation_diagnostics(
        endog=endog,
        design=cancel_quotient,
        check_name="mixed_cancel_quotient_boundary_separation",
        scope="cancel_quotient_retained_index",
    )
    diagnostics.update(
        _design_rank_diagnostics(
            cancel_quotient,
            prefix="mixed_cancel_quotient_design",
        )
    )
    return _finalize_mixed_support_conditioning_diagnostics(
        diagnostics,
        raw_support=raw_support,
        quotient_support=quotient_support,
    )


def _mixed_support_conditioning_diagnostics_cpp(
    *,
    endog: np.ndarray,
    mixed_design: np.ndarray,
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    control: IOLSKernelControl,
) -> dict[str, Any]:
    cpp_kernel = _load_cpp_kernel_module()
    if cpp_kernel is None:
        raise ImportError("pyiols._kernel_cpp is not importable")
    raw = cpp_kernel.mixed_support_conditioning_preflight(
        np.ascontiguousarray(endog, dtype=np.float64),
        np.ascontiguousarray(mixed_design, dtype=np.float64),
        cancel_groups=tuple(
            np.asarray(group, dtype=np.int64) for group in cancel_groups
        ),
        cancel_counts=tuple(int(count) for count in cancel_counts),
        recover_groups=tuple(
            np.asarray(group, dtype=np.int64) for group in recover_groups
        ),
        recover_counts=tuple(int(count) for count in recover_counts),
        recover_names=tuple(str(name) for name in recover_names),
        control=_control_to_cpp_mapping(control),
    )
    diagnostics: dict[str, Any] = {
        "mixed_support_check": MIXED_TAU2_SUPPORT_CONDITIONING_CHECK,
        "mixed_support_policy": MIXED_TAU2_SUPPORT_POLICY,
    }
    for key, value in raw.items():
        if key in {
            "mixed_cancel_quotient_design",
            "gil_was_released",
            "input_snapshot_owned",
        }:
            continue
        diagnostics[str(key)] = value
    raw_support = _design_boundary_separation_diagnostics(
        endog=endog,
        design=mixed_design,
        check_name="mixed_raw_recovered_retained_boundary_separation",
        scope="raw_retained_index",
    )
    cancel_quotient = np.asarray(raw["mixed_cancel_quotient_design"], dtype=float)
    quotient_support = _design_boundary_separation_diagnostics(
        endog=endog,
        design=cancel_quotient,
        check_name="mixed_cancel_quotient_boundary_separation",
        scope="cancel_quotient_retained_index",
    )
    return _finalize_mixed_support_conditioning_diagnostics(
        diagnostics,
        raw_support=raw_support,
        quotient_support=quotient_support,
    )


def _mixed_design_matrix(
    *,
    exog: np.ndarray,
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    recover_design, recover_column_names = _recover_design_from_groups(
        groups=recover_groups,
        counts=recover_counts,
        names=recover_names,
    )
    if recover_design.shape[0] != exog.shape[0]:
        raise ValueError("recover FE blocks must align with exog rows")
    if recover_design.shape[1] == 0:
        return np.asarray(exog, dtype=float), recover_column_names
    return np.column_stack([np.asarray(exog, dtype=float), recover_design]), (
        recover_column_names
    )


def _mixed_design_matrix_cpp(
    *,
    exog: np.ndarray,
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    cpp_kernel = _load_cpp_kernel_module()
    if cpp_kernel is None:
        raise ImportError("pyiols._kernel_cpp is not importable")
    raw = cpp_kernel.mixed_design_matrix(
        np.ascontiguousarray(exog, dtype=np.float64),
        recover_groups=tuple(
            np.asarray(group, dtype=np.int64) for group in recover_groups
        ),
        recover_counts=tuple(int(count) for count in recover_counts),
        recover_names=tuple(str(name) for name in recover_names),
    )
    return np.asarray(raw["design"], dtype=float), tuple(
        str(name) for name in raw["recover_column_names"]
    )


def _prepare_mixed_cpp_route(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    control: IOLSKernelControl,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    mixed_design, recover_column_names = _mixed_design_matrix_cpp(
        exog=exog,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        recover_names=recover_names,
    )
    support_diagnostics = _mixed_support_conditioning_diagnostics_cpp(
        endog=endog,
        mixed_design=mixed_design,
        cancel_groups=cancel_groups,
        cancel_counts=cancel_counts,
        recover_names=recover_names,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        control=control,
    )
    return recover_column_names, support_diagnostics


def _slice_mixed_tau_path_details(
    details: Mapping[str, Any],
    *,
    slope_count: int,
    recover_column_names: tuple[str, ...],
    cancel_names: tuple[str, ...],
    recover_names: tuple[str, ...],
    support_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    mixed_details = dict(details)
    records: list[dict[str, Any]] = []
    for record in tuple(mixed_details.get("path", ())):
        item = dict(record)
        theta_raw = item.get("beta_reduced")
        if theta_raw is not None:
            theta = np.asarray(theta_raw, dtype=float)
            item["theta_reduced"] = _as_owned_float64_vector(theta)
            item["beta_reduced"] = _as_owned_float64_vector(theta[:slope_count])
            item["recover_fe_reduced"] = _as_owned_float64_vector(theta[slope_count:])
        diagnostics = dict(item.get("diagnostics", {}))
        diagnostics.update(
            {
                "mixed_algorithm": (
                    "cancelled_fe_weighted_within_with_explicit_recover_design"
                ),
                "mixed_cancel_fe_policy": "weighted_within_cancel",
                "mixed_recover_fe_policy": "explicit_recover_design_projection",
                "mixed_retained_index_policy": "slope_plus_recovered_fe",
                "mixed_slope_count": int(slope_count),
                "mixed_recover_design_columns": int(len(recover_column_names)),
            }
        )
        if abs(float(item.get("tau", float("nan"))) - 2.0) <= 1e-10:
            diagnostics.update(support_diagnostics)
            if not bool(item.get("converged", False)) and bool(
                support_diagnostics.get(
                    "mixed_support_needs_attention",
                    False,
                )
            ):
                original_failure = item.get("failure_reason")
                diagnostics["mixed_anchor_original_failure_reason"] = original_failure
                diagnostics["mixed_anchor_failure_classification"] = (
                    MIXED_TAU2_NEAR_BOUNDARY_CODE
                )
                item["failure_reason"] = MIXED_TAU2_NEAR_BOUNDARY_CODE
        item["diagnostics"] = diagnostics
        records.append(item)
    failure_reason = mixed_details.get("failure_reason")
    if (
        records
        and mixed_details.get("path_status") == "anchor_failed"
        and records[0].get("failure_reason") == MIXED_TAU2_NEAR_BOUNDARY_CODE
    ):
        mixed_details["mixed_anchor_original_failure_reason"] = failure_reason
        mixed_details["failure_reason"] = MIXED_TAU2_NEAR_BOUNDARY_CODE
    mixed_details.update(
        {
            "mixed_algorithm": (
                "cancelled_fe_weighted_within_with_explicit_recover_design"
            ),
            "mixed_cancel_fe_policy": "weighted_within_cancel",
            "mixed_recover_fe_policy": "explicit_recover_design_projection",
            "mixed_retained_index_policy": "slope_plus_recovered_fe",
            "mixed_cancel_fe_names": tuple(cancel_names),
            "mixed_recover_fe_names": tuple(recover_names),
            "mixed_recover_design_columns": int(len(recover_column_names)),
            "mixed_recover_design_column_names": tuple(recover_column_names),
            "mixed_support_check": support_diagnostics.get(
                "mixed_support_check",
                "not_computed",
            ),
            "mixed_support_classification": support_diagnostics.get(
                "mixed_support_classification",
                "not_computed",
            ),
            "mixed_support_needs_attention": bool(
                support_diagnostics.get("mixed_support_needs_attention", False)
            ),
            "path": tuple(records),
        }
    )
    return mixed_details


def _solve_mixed_tau_path(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    cancel_names: tuple[str, ...],
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
) -> _TauPathSolve:
    mixed_design, recover_column_names = _mixed_design_matrix(
        exog=exog,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        recover_names=recover_names,
    )
    support_diagnostics = _mixed_support_conditioning_diagnostics(
        endog=endog,
        mixed_design=mixed_design,
        cancel_groups=cancel_groups,
        cancel_counts=cancel_counts,
        recover_names=recover_names,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        control=control,
    )
    theta_solve = _solve_fullcancel_tau_path(
        endog=endog,
        exog=mixed_design,
        groups=cancel_groups,
        counts=cancel_counts,
        tau_target=tau_target,
        control=control,
    )
    slope_count = int(exog.shape[1])
    beta = np.asarray(theta_solve.beta[:slope_count], dtype=float)
    details = _slice_mixed_tau_path_details(
        theta_solve.details,
        slope_count=slope_count,
        recover_column_names=recover_column_names,
        cancel_names=cancel_names,
        recover_names=recover_names,
        support_diagnostics=support_diagnostics,
    )
    iteration = theta_solve.iteration
    if (
        details.get("path_status") == "anchor_failed"
        and details.get("failure_reason") == MIXED_TAU2_NEAR_BOUNDARY_CODE
    ):
        iteration_details = dict(iteration.details)
        iteration_details.update(support_diagnostics)
        iteration_details["mixed_anchor_original_failure_reason"] = (
            iteration.failure_reason
        )
        iteration_details["mixed_anchor_failure_classification"] = (
            MIXED_TAU2_NEAR_BOUNDARY_CODE
        )
        iteration = IOLSIterationDiagnostics(
            converged=iteration.converged,
            phase=iteration.phase,
            iterations=iteration.iterations,
            restarts=iteration.restarts,
            final_rho=iteration.final_rho,
            rho_path=iteration.rho_path,
            final_criterion=iteration.final_criterion,
            anderson_m_initial=iteration.anderson_m_initial,
            anderson_m_final=iteration.anderson_m_final,
            failure_reason=MIXED_TAU2_NEAR_BOUNDARY_CODE,
            details=iteration_details,
        )
    return _TauPathSolve(
        beta=beta,
        linear=theta_solve.linear,
        tau_fit=theta_solve.tau_fit,
        converged=theta_solve.converged,
        iteration=iteration,
        details=details,
    )


def _solve_mixed_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    cancel_names: tuple[str, ...],
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
) -> _TauPathSolve:
    recover_column_names, support_diagnostics = _prepare_mixed_cpp_route(
        endog=endog,
        exog=exog,
        cancel_groups=cancel_groups,
        cancel_counts=cancel_counts,
        recover_names=recover_names,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        control=control,
    )
    return _solve_mixed_tau_path_cpp_precomputed(
        endog=endog,
        exog=exog,
        cancel_names=cancel_names,
        cancel_groups=cancel_groups,
        cancel_counts=cancel_counts,
        recover_names=recover_names,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        tau_target=tau_target,
        control=control,
        recover_column_names=recover_column_names,
        support_diagnostics=support_diagnostics,
    )


def _solve_mixed_tau_path_cpp_precomputed(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    cancel_names: tuple[str, ...],
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
    recover_column_names: tuple[str, ...],
    support_diagnostics: Mapping[str, Any],
) -> _TauPathSolve:
    theta_solve = _solve_mixed_fullcancel_tau_path_cpp(
        endog=endog,
        exog=exog,
        cancel_groups=cancel_groups,
        cancel_counts=cancel_counts,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        tau_target=tau_target,
        control=control,
    )
    slope_count = int(exog.shape[1])
    beta = np.asarray(theta_solve.beta[:slope_count], dtype=float)
    details = _slice_mixed_tau_path_details(
        theta_solve.details,
        slope_count=slope_count,
        recover_column_names=recover_column_names,
        cancel_names=cancel_names,
        recover_names=recover_names,
        support_diagnostics=support_diagnostics,
    )
    iteration = theta_solve.iteration
    if (
        details.get("path_status") == "anchor_failed"
        and details.get("failure_reason") == MIXED_TAU2_NEAR_BOUNDARY_CODE
    ):
        iteration_details = dict(iteration.details)
        iteration_details.update(support_diagnostics)
        iteration_details["mixed_anchor_original_failure_reason"] = (
            iteration.failure_reason
        )
        iteration_details["mixed_anchor_failure_classification"] = (
            MIXED_TAU2_NEAR_BOUNDARY_CODE
        )
        iteration = IOLSIterationDiagnostics(
            converged=iteration.converged,
            phase=iteration.phase,
            iterations=iteration.iterations,
            restarts=iteration.restarts,
            final_rho=iteration.final_rho,
            rho_path=iteration.rho_path,
            final_criterion=iteration.final_criterion,
            anderson_m_initial=iteration.anderson_m_initial,
            anderson_m_final=iteration.anderson_m_final,
            failure_reason=MIXED_TAU2_NEAR_BOUNDARY_CODE,
            details=iteration_details,
        )
    return _TauPathSolve(
        beta=beta,
        linear=theta_solve.linear,
        tau_fit=theta_solve.tau_fit,
        converged=theta_solve.converged,
        iteration=iteration,
        details=details,
    )


def _solve_mixed_configured_tau_path(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    cancel_names: tuple[str, ...],
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    tau_targets: tuple[float, ...],
    control: IOLSKernelControl,
) -> _TauPathSolve:
    if not tau_targets:
        raise ValueError("tau_path_targets must be non-empty when tau_target is None")
    path_solves = tuple(
        _solve_mixed_tau_path(
            endog=endog,
            exog=exog,
            cancel_names=cancel_names,
            cancel_groups=cancel_groups,
            cancel_counts=cancel_counts,
            recover_names=recover_names,
            recover_groups=recover_groups,
            recover_counts=recover_counts,
            tau_target=float(tau_target),
            control=control,
        )
        for tau_target in tau_targets
    )
    return _merge_configured_tau_paths(path_solves, tau_targets)


def _solve_mixed_configured_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    cancel_names: tuple[str, ...],
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_names: tuple[str, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    tau_targets: tuple[float, ...],
    control: IOLSKernelControl,
) -> _TauPathSolve:
    if not tau_targets:
        raise ValueError("tau_path_targets must be non-empty when tau_target is None")
    recover_column_names, support_diagnostics = _prepare_mixed_cpp_route(
        endog=endog,
        exog=exog,
        cancel_groups=cancel_groups,
        cancel_counts=cancel_counts,
        recover_names=recover_names,
        recover_groups=recover_groups,
        recover_counts=recover_counts,
        control=control,
    )
    path_solves = tuple(
        _solve_mixed_tau_path_cpp_precomputed(
            endog=endog,
            exog=exog,
            cancel_names=cancel_names,
            cancel_groups=cancel_groups,
            cancel_counts=cancel_counts,
            recover_names=recover_names,
            recover_groups=recover_groups,
            recover_counts=recover_counts,
            tau_target=float(tau_target),
            control=control,
            recover_column_names=recover_column_names,
            support_diagnostics=support_diagnostics,
        )
        for tau_target in tau_targets
    )
    return _merge_configured_tau_paths(path_solves, tau_targets)


def _fullcancel_score_residual(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    fe_demean_maxiter: int = 10000,
    fe_demean_tol: float = 1e-10,
    unweighted_x_demeaned: np.ndarray | None = None,
) -> float:
    q = _fullcancel_linear_state(exog, beta)
    if not np.all(np.isfinite(q)):
        return float("inf")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights = exp_with_float64_overflow_guard((2.0 - float(tau)) * q)
        ratio = endog * exp_with_float64_overflow_guard(-q)
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(ratio)):
        return float("inf")
    if abs(float(tau) - 2.0) <= 1e-10:
        x_demeaned = (
            np.asarray(unweighted_x_demeaned, dtype=float)
            if unweighted_x_demeaned is not None
            else _demean_for_groups(
                exog,
                groups,
                counts,
                tol=fe_demean_tol,
                maxiter=fe_demean_maxiter,
            )
        )
        ratio_demeaned = _demean_for_groups(
            ratio,
            groups,
            counts,
            tol=fe_demean_tol,
            maxiter=fe_demean_maxiter,
        )
        if not np.all(np.isfinite(x_demeaned)) or not np.all(
            np.isfinite(ratio_demeaned)
        ):
            return float("inf")
        score = x_demeaned.T @ ratio_demeaned
        if not np.all(np.isfinite(score)):
            return float("inf")
        if score.size == 0:
            return 0.0
        return float(np.max(np.abs(score)) / max(float(endog.shape[0]), 1.0))
    x_demeaned = _weighted_demean_for_groups(
        exog,
        groups,
        counts,
        weights,
        tol=fe_demean_tol,
        maxiter=fe_demean_maxiter,
    )
    ratio_demeaned = _weighted_demean_for_groups(
        ratio,
        groups,
        counts,
        weights,
        tol=fe_demean_tol,
        maxiter=fe_demean_maxiter,
    )
    if not np.all(np.isfinite(x_demeaned)) or not np.all(np.isfinite(ratio_demeaned)):
        return float("inf")
    score = x_demeaned.T @ (weights * ratio_demeaned)
    if not np.all(np.isfinite(score)):
        return float("inf")
    if score.size == 0:
        return 0.0
    return float(np.max(np.abs(score)) / max(float(endog.shape[0]), 1.0))


def _fullcancel_initial_scale(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    control: IOLSKernelControl,
) -> float:
    q = _fullcancel_linear_state(exog, beta)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights = exp_with_float64_overflow_guard((2.0 - float(tau)) * q)
        ratio = endog * exp_with_float64_overflow_guard(-q)
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(ratio)):
        return 1.0
    if abs(float(tau) - 2.0) <= 1e-10:
        try:
            ratio_demeaned = _demean_for_groups(
                ratio,
                groups,
                counts,
                tol=float(control.fe_demean_tol),
                maxiter=int(control.fe_demean_maxiter),
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return 1.0
            raise
        demean_converged = True
    else:
        ratio_demeaned, demean_converged = _weighted_demean_for_groups_checked(
            ratio,
            groups,
            counts,
            weights,
            tol=float(control.fe_demean_tol),
            maxiter=_phase2_weighted_demean_maxiter(control),
        )
    if (
        not demean_converged
        or ratio_demeaned.size == 0
        or not np.all(np.isfinite(ratio_demeaned))
    ):
        return 1.0
    scale = float(np.nanmax(np.abs(ratio_demeaned)))
    if not np.isfinite(scale):
        return 1.0
    return max(scale, 1.0)


def _rho_path_for_fullcancel(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    rho_hint: float | None,
    control: IOLSKernelControl,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = _rho_base_for_tau(tau, control)
    scale = _fullcancel_initial_scale(
        endog=endog,
        exog=exog,
        beta=beta,
        groups=groups,
        counts=counts,
        tau=tau,
        control=control,
    )
    rho_path = np.unique(base[np.isfinite(base) & (base > 0.0)] * scale)
    rho_path = _focus_rho_path(rho_path, rho_hint=rho_hint)
    details = {
        "rho_policy": "scaled_fixed_log_grid",
        "fullcancel_initial_scale": float(scale),
        "rho_candidate_count": int(rho_path.size),
        "rho_hint": None if rho_hint is None else float(rho_hint),
        "rho_log_min": float(control.rho_log_min),
        "rho_log_max": float(control.rho_log_max),
        "rho_grid_size": int(control.rho_grid_size),
    }
    if rho_path.size:
        details.update(
            {
                "rho_min": float(rho_path[0]),
                "rho_max": float(rho_path[-1]),
            }
        )
    return rho_path, details


def _phi_step_fullcancel(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    phase: int,
    param: float,
    fe_demean_maxiter: int = 10000,
    fe_demean_tol: float = 1e-10,
    require_demean_converged: bool = False,
    unweighted_x_demeaned: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    q = _fullcancel_linear_state(exog, beta)
    if not np.all(np.isfinite(q)):
        return beta.copy(), False

    if phase == 0:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = endog * exp_with_float64_overflow_guard(-q)
            log_arg = (ratio - 1.0) / (1.0 + float(param))
            pseudo = q + np.log1p(log_arg)
        if not np.all(np.isfinite(pseudo)):
            return beta.copy(), False
        try:
            x_demeaned = (
                np.asarray(unweighted_x_demeaned, dtype=float)
                if unweighted_x_demeaned is not None
                else _demean_for_groups(
                    exog,
                    groups,
                    counts,
                    tol=fe_demean_tol,
                    maxiter=fe_demean_maxiter,
                )
            )
            y_demeaned = _demean_for_groups(
                pseudo,
                groups,
                counts,
                tol=fe_demean_tol,
                maxiter=fe_demean_maxiter,
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return beta.copy(), False
            raise
        try:
            beta_new = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)[0]
        except np.linalg.LinAlgError:
            return beta.copy(), False
        if not np.all(np.isfinite(beta_new)):
            return beta.copy(), False
        return np.asarray(beta_new, dtype=float), True

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights = exp_with_float64_overflow_guard((2.0 - float(tau)) * q)
        ratio = endog * exp_with_float64_overflow_guard(-q)
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(ratio)):
        return beta.copy(), False

    if abs(float(tau) - 2.0) <= 1e-10:
        try:
            x_demeaned = (
                np.asarray(unweighted_x_demeaned, dtype=float)
                if unweighted_x_demeaned is not None
                else _demean_for_groups(
                    exog,
                    groups,
                    counts,
                    tol=fe_demean_tol,
                    maxiter=fe_demean_maxiter,
                )
            )
            ratio_demeaned = _demean_for_groups(
                ratio,
                groups,
                counts,
                tol=fe_demean_tol,
                maxiter=fe_demean_maxiter,
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return beta.copy(), False
            raise
        if not np.all(np.isfinite(x_demeaned)) or not np.all(
            np.isfinite(ratio_demeaned)
        ):
            return beta.copy(), False
        eta = 1.0 / (1.0 + float(param))
        y_star = x_demeaned @ beta + eta * ratio_demeaned
        try:
            beta_new = np.linalg.lstsq(x_demeaned, y_star, rcond=None)[0]
        except np.linalg.LinAlgError:
            return beta.copy(), False
        if not np.all(np.isfinite(beta_new)):
            return beta.copy(), False
        return np.asarray(beta_new, dtype=float), True

    if require_demean_converged:
        x_demeaned, x_demean_converged = _weighted_demean_for_groups_checked(
            exog,
            groups,
            counts,
            weights,
            tol=fe_demean_tol,
            maxiter=fe_demean_maxiter,
        )
        ratio_demeaned, ratio_demean_converged = _weighted_demean_for_groups_checked(
            ratio,
            groups,
            counts,
            weights,
            tol=fe_demean_tol,
            maxiter=fe_demean_maxiter,
        )
        if not x_demean_converged or not ratio_demean_converged:
            return beta.copy(), False
    else:
        try:
            x_demeaned = _weighted_demean_for_groups(
                exog,
                groups,
                counts,
                weights,
                tol=fe_demean_tol,
                maxiter=fe_demean_maxiter,
            )
            ratio_demeaned = _weighted_demean_for_groups(
                ratio,
                groups,
                counts,
                weights,
                tol=fe_demean_tol,
                maxiter=fe_demean_maxiter,
            )
        except RuntimeError as exc:
            if is_known_fe_runtime_failure(exc):
                return beta.copy(), False
            raise
    if not np.all(np.isfinite(x_demeaned)) or not np.all(np.isfinite(ratio_demeaned)):
        return beta.copy(), False

    eta = 1.0 / (1.0 + float(param))
    sqrt_weights = np.sqrt(weights)
    x_star = x_demeaned * sqrt_weights[:, None]
    y_star = sqrt_weights * (x_demeaned @ beta + eta * ratio_demeaned)
    try:
        beta_new = np.linalg.lstsq(x_star, y_star, rcond=None)[0]
    except np.linalg.LinAlgError:
        return beta.copy(), False
    if not np.all(np.isfinite(beta_new)):
        return beta.copy(), False
    return np.asarray(beta_new, dtype=float), True


def _phase2_initialization_usable_fullcancel(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    control: IOLSKernelControl,
    unweighted_x_demeaned: np.ndarray | None = None,
) -> bool:
    if not np.all(np.isfinite(beta)):
        return False
    rho_path, _ = _rho_path_for_fullcancel(
        endog=endog,
        exog=exog,
        beta=beta,
        groups=groups,
        counts=counts,
        tau=tau,
        rho_hint=None,
        control=control,
    )
    for rho in rho_path:
        _, valid = _phi_step_fullcancel(
            endog=endog,
            exog=exog,
            beta=beta,
            groups=groups,
            counts=counts,
            tau=tau,
            phase=1,
            param=float(rho),
            fe_demean_maxiter=_fullcancel_phi_demean_maxiter(
                control,
                tau=tau,
                phase=1,
            ),
            fe_demean_tol=float(control.fe_demean_tol),
            require_demean_converged=True,
            unweighted_x_demeaned=unweighted_x_demeaned,
        )
        if valid:
            return True
    return False


def _concentrated_fullcancel(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    control: IOLSKernelControl,
    beta_init: np.ndarray | None = None,
    rho_hint: float | None = None,
    score_validation_policy: ScoreValidationPolicy = "diagnostic",
) -> _ConcentratedSolve:
    if score_validation_policy != "diagnostic":
        raise ValueError("score_validation_policy must be 'diagnostic'")
    if beta_init is None and abs(float(tau) - 2.0) > 1e-10:
        raise ValueError(
            "Non-anchor fullcancel path steps require beta_init from the "
            "tau=2 Phase 1 + Phase 2 anchor."
        )

    delta_grid = (
        _delta_base_for_anchor(control)
        if beta_init is None
        else np.array([], dtype=float)
    )
    x_demeaned = _demean_for_groups(
        exog,
        groups,
        counts,
        tol=float(control.fe_demean_tol),
        maxiter=int(control.fe_demean_maxiter),
    )
    if beta_init is None:
        y_init = _demean_for_groups(
            np.log(endog + 1.0),
            groups,
            counts,
            tol=float(control.fe_demean_tol),
            maxiter=int(control.fe_demean_maxiter),
        )
        try:
            beta = np.linalg.lstsq(x_demeaned, y_init, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.full(exog.shape[1], np.nan, dtype=float)
        n_outer = len(delta_grid) + 1
    else:
        beta = np.asarray(beta_init, dtype=float).copy()
        n_outer = 1
    beta_initial = beta.copy()
    phase1_accepted_beta = beta.copy()
    phase1_attempts = 0
    phase1_accepted = 0
    phase1_rejected = 0

    final_diag = IOLSIterationDiagnostics(
        converged=False,
        phase="not_started",
        iterations=0,
        restarts=0,
        final_rho=None,
        rho_path=(),
        final_criterion=None,
        anderson_m_initial=int(control.anderson_m),
        anderson_m_final=int(control.anderson_m),
        failure_reason=KERNEL_FAILURE_NOT_STARTED,
    )

    for outer_index in range(n_outer):
        if np.any(np.isnan(beta)):
            beta = beta_initial.copy()
        beta_outer_initial = beta.copy()

        if outer_index < len(delta_grid):
            phase = 0
            param = float(delta_grid[outer_index])
            eps_current = max(float(control.eps) * 10.0, 1e-5)
            stopcrit_current = min(
                int(control.stopcrit),
                int(control.max_phase1_iterations),
            )
            rho_path = np.array([], dtype=float)
            rho_details: dict[str, Any] = {}
        else:
            phase = 1
            if beta_init is None:
                beta = phase1_accepted_beta.copy()
            beta_outer_initial = beta.copy()
            rho_path, rho_details = _rho_path_for_fullcancel(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                counts=counts,
                tau=tau,
                rho_hint=rho_hint,
                control=control,
            )
            if rho_path.size == 0:
                rho_details.update(
                    {
                        "phase1_attempts": int(phase1_attempts),
                        "phase1_accepted": int(phase1_accepted),
                        "phase1_rejected": int(phase1_rejected),
                    }
                )
                return _ConcentratedSolve(
                    beta=np.asarray(beta, dtype=float),
                    linear=None,
                    iteration=IOLSIterationDiagnostics(
                        converged=False,
                        phase="P2",
                        iterations=0,
                        restarts=0,
                        final_rho=None,
                        rho_path=(),
                        final_criterion=None,
                        anderson_m_initial=int(control.anderson_m),
                        anderson_m_final=int(control.anderson_m),
                        failure_reason=KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH,
                        details=rho_details,
                    ),
                )
            param = float(rho_path[0])
            eps_current = float(control.eps)
            stopcrit_current = int(control.stopcrit)

        restart_index = 0
        verified = False
        phase2_evaluations = 0
        phase2_budget_exhausted = False
        iteration_count = 0
        final_criterion: float | None = None
        aa_m_final = int(control.anderson_m)
        failure_reason = None
        qhat_last = float("nan")
        qhat_invalid_count = 0
        kappa_monitor_last = float("nan")
        kappa_monitor_max = float("nan")
        kappa_monitor_checks = 0
        post_convergence_residual = float("nan")
        post_convergence_kappa_est = float("nan")
        post_convergence_score_residual = float("nan")
        score_tol = float("nan")
        score_validation_status = "not_checked"
        score_warning_codes: list[str] = []
        score_warning_messages: list[str] = []
        score_exceeds_tolerance = False
        last_finite_criterion = float("nan")
        best_finite_criterion = float("nan")

        while not verified:
            k = 1
            criterion = 100.0
            criterion_history: list[float] = []
            kappa_bad_count = 0
            beta_path = [beta.copy()]
            aa_m = _fullcancel_effective_anderson_m(
                control,
                tau=tau,
                phase=phase,
            )
            aa_x_hist: list[np.ndarray] = []
            aa_f_hist: list[np.ndarray] = []

            while (
                criterion > eps_current
                and k < stopcrit_current
                and not np.isnan(criterion)
            ):
                beta0 = beta.copy()
                if phase == 1:
                    phase2_evaluations += 1
                    if phase2_evaluations > int(control.max_phase2_evaluations):
                        criterion = np.nan
                        failure_reason = (
                            KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED
                        )
                        phase2_budget_exhausted = True
                        break
                beta_phi, valid = _phi_step_fullcancel(
                    endog=endog,
                    exog=exog,
                    beta=beta,
                    groups=groups,
                    counts=counts,
                    tau=tau,
                    phase=phase,
                    param=param,
                    fe_demean_maxiter=_fullcancel_phi_demean_maxiter(
                        control,
                        tau=tau,
                        phase=phase,
                    ),
                    fe_demean_tol=float(control.fe_demean_tol),
                    require_demean_converged=phase == 1,
                    unweighted_x_demeaned=x_demeaned,
                )
                if not valid:
                    criterion = np.nan
                    failure_reason = KERNEL_FAILURE_INVALID_PHI_STEP
                    break

                accepted_aa_step = False
                if aa_m > 0 and k > aa_m + 1:
                    f_k = beta_phi - beta
                    aa_x_hist.append(beta.copy())
                    aa_f_hist.append(f_k.copy())
                    if len(aa_f_hist) > aa_m + 1:
                        aa_x_hist.pop(0)
                        aa_f_hist.pop(0)

                    m_cur = len(aa_f_hist) - 1
                    if m_cur >= 1:
                        f_mat = np.column_stack(
                            [
                                aa_f_hist[-1] - aa_f_hist[-(j + 1)]
                                for j in range(1, m_cur + 1)
                            ]
                        )
                        gamma, _, _, _ = np.linalg.lstsq(
                            f_mat,
                            aa_f_hist[-1],
                            rcond=None,
                        )
                        beta_aa = beta_phi.copy()
                        for j in range(m_cur):
                            phi_prev = aa_x_hist[-(j + 2)] + aa_f_hist[-(j + 2)]
                            beta_aa += gamma[j] * (phi_prev - beta_phi)

                        if np.all(np.isfinite(beta_aa)):
                            if phase == 1:
                                phase2_evaluations += 1
                                if phase2_evaluations > int(
                                    control.max_phase2_evaluations
                                ):
                                    criterion = np.nan
                                    failure_reason = (
                                        KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED
                                    )
                                    phase2_budget_exhausted = True
                                    break
                            beta_aa_phi, aa_valid = _phi_step_fullcancel(
                                endog=endog,
                                exog=exog,
                                beta=beta_aa,
                                groups=groups,
                                counts=counts,
                                tau=tau,
                                phase=phase,
                                param=param,
                                fe_demean_maxiter=_fullcancel_phi_demean_maxiter(
                                    control,
                                    tau=tau,
                                    phase=phase,
                                ),
                                fe_demean_tol=float(control.fe_demean_tol),
                                require_demean_converged=phase == 1,
                                unweighted_x_demeaned=x_demeaned,
                            )
                            if aa_valid:
                                res_plain = np.linalg.norm(beta_phi - beta)
                                res_aa = np.linalg.norm(beta_aa_phi - beta_aa)
                                if res_aa < res_plain * 1.1:
                                    beta = beta_aa
                                    beta_path.append(beta.copy())
                                    if k >= 2:
                                        criterion = _max_abs_change(beta, beta0)
                                    accepted_aa_step = True
                elif aa_m > 0:
                    f_k = beta_phi - beta
                    aa_x_hist.append(beta.copy())
                    aa_f_hist.append(f_k.copy())

                if not accepted_aa_step:
                    beta = beta_phi
                    beta_path.append(beta.copy())
                if k >= 2:
                    if not accepted_aa_step:
                        criterion = _max_abs_change(beta, beta0)
                    (
                        last_finite_criterion,
                        best_finite_criterion,
                        criterion_failure,
                    ) = _record_criterion_progress(
                        criterion_history,
                        criterion,
                        phase=phase,
                        tau=tau,
                        eps=eps_current,
                        max_phase2_evaluations=int(control.max_phase2_evaluations),
                        last_finite_criterion=last_finite_criterion,
                        best_finite_criterion=best_finite_criterion,
                    )
                    if criterion_failure is not None:
                        criterion = np.nan
                        failure_reason = criterion_failure

                if (
                    k > 4
                    and len(beta_path) >= 4
                    and not (phase == 1 and abs(tau - 2.0) <= 1e-10)
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        d1 = np.abs(beta_path[-1] - beta_path[-2])
                        d2 = np.abs(beta_path[-2] - beta_path[-3])
                        d3 = np.abs(beta_path[-3] - beta_path[-4])
                        qhat = np.nanmean(np.log(d1 / d2) / np.log(d2 / d3))
                    qhat_last = float(qhat)
                    if np.isinf(np.abs(qhat)) and criterion > eps_current * 10.0:
                        qhat_invalid_count += 1
                        criterion = np.nan
                        failure_reason = KERNEL_FAILURE_Q_CONVERGENCE_INVALID

                if (
                    phase == 1
                    and abs(tau - 2.0) > 1e-10
                    and k >= 30
                    and k % 10 == 0
                    and len(beta_path) >= 3
                ):
                    beta_mat = np.array(beta_path[1:])
                    if beta_mat.shape[0] >= 3:
                        diffs = np.abs(np.diff(beta_mat, axis=0))
                        if diffs.shape[0] >= 2:
                            with np.errstate(divide="ignore", invalid="ignore"):
                                ratios = diffs[1:] / diffs[:-1]
                            kappa_val = np.nanmax(np.nanmean(ratios, axis=1))
                            kappa_monitor_last = float(kappa_val)
                            kappa_monitor_checks += 1
                            if np.isfinite(kappa_val):
                                if np.isfinite(kappa_monitor_max):
                                    kappa_monitor_max = max(
                                        kappa_monitor_max,
                                        float(kappa_val),
                                    )
                                else:
                                    kappa_monitor_max = float(kappa_val)
                            aa_m_new = _adapt_anderson_m(aa_m, kappa_val)
                            if aa_m_new > aa_m:
                                aa_m = aa_m_new
                                kappa_bad_count = 0
                            elif np.isfinite(kappa_val) and kappa_val >= 1.0:
                                kappa_bad_count += 1
                            else:
                                kappa_bad_count = 0
                            if kappa_bad_count >= 3:
                                criterion = np.nan
                                failure_reason = KERNEL_FAILURE_NONCONTRACTIVE

                k += 1
                iteration_count = k

                if np.isnan(criterion):
                    restart_index += 1
                    if (
                        phase == 1
                        and not phase2_budget_exhausted
                        and restart_index < len(rho_path)
                    ):
                        param = float(rho_path[restart_index])
                        criterion = 100.0
                        beta = beta_outer_initial.copy()
                        k = 1
                        criterion_history = []
                        kappa_bad_count = 0
                        beta_path = [beta.copy()]
                        aa_x_hist = []
                        aa_f_hist = []
                    elif phase == 0:
                        break

            converged = not np.isnan(criterion) and criterion <= eps_current
            final_criterion = None if np.isnan(criterion) else float(criterion)
            if converged:
                beta_check, valid = _phi_step_fullcancel(
                    endog=endog,
                    exog=exog,
                    beta=beta,
                    groups=groups,
                    counts=counts,
                    tau=tau,
                    phase=phase,
                    param=param,
                    fe_demean_maxiter=_fullcancel_phi_demean_maxiter(
                        control,
                        tau=tau,
                        phase=phase,
                    ),
                    fe_demean_tol=float(control.fe_demean_tol),
                    require_demean_converged=phase == 1,
                    unweighted_x_demeaned=x_demeaned,
                )
                if valid:
                    post_convergence_residual = _max_abs_change(beta_check, beta)
                    if post_convergence_residual > eps_current * 10.0:
                        converged = False
                        failure_reason = KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL
                if converged and phase == 1:
                    score_residual = _fullcancel_score_residual(
                        endog=endog,
                        exog=exog,
                        beta=beta,
                        groups=groups,
                        counts=counts,
                        tau=tau,
                        fe_demean_maxiter=int(control.fe_demean_maxiter),
                        fe_demean_tol=float(control.fe_demean_tol),
                        unweighted_x_demeaned=x_demeaned,
                    )
                    score_tol = max(eps_current * 10.0, 1e-8)
                    post_convergence_score_residual = float(score_residual)
                    score_exceeds_tolerance = bool(
                        not np.isfinite(score_residual) or score_residual > score_tol
                    )
                    if score_exceeds_tolerance:
                        score_warning_codes.append("post_convergence_score_residual")
                        score_warning_messages.append(
                            "Fullcancel weighted-within score residual exceeds "
                            "tolerance at this tau-path state."
                        )
                        score_validation_status = "diagnostic_warning"
                    elif score_validation_status == "not_checked":
                        score_validation_status = "passed"
                if (
                    converged
                    and len(beta_path) >= 4
                    and not (phase == 1 and abs(tau - 2.0) <= 1e-10)
                ):
                    diffs = [
                        _max_abs_change(beta_path[-j], beta_path[-j - 1])
                        for j in range(1, min(4, len(beta_path)))
                    ]
                    if diffs[1] > 1e-15 and diffs[2] > 1e-15:
                        kappa_est = diffs[0] / diffs[1]
                        post_convergence_kappa_est = float(kappa_est)
                        if kappa_est >= 1.0:
                            converged = False
                            failure_reason = (
                                KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE
                            )

            if (
                not converged
                and phase == 1
                and not phase2_budget_exhausted
                and restart_index < len(rho_path) - 1
            ):
                restart_index = min(
                    restart_index + max(restart_index, 5),
                    len(rho_path) - 1,
                )
                param = float(rho_path[restart_index])
                beta = beta_outer_initial.copy()
            else:
                verified = True

            aa_m_final = aa_m

        diag_details_common: dict[str, Any] = {
            "fullcancel_algorithm": "weighted_within_fullcancel",
            "fullcancel_index_policy": "slope_only_raw_exog",
            "fullcancel_fe_policy": "weighted_within_cancel_all",
            "anderson_policy": "beta_anderson",
            "qhat_last": float(qhat_last),
            "qhat_invalid_count": int(qhat_invalid_count),
            "kappa_monitor_last": float(kappa_monitor_last),
            "kappa_monitor_max": float(kappa_monitor_max),
            "kappa_monitor_checks": int(kappa_monitor_checks),
            "phase2_evaluations": int(phase2_evaluations),
            "max_phase2_evaluations": int(control.max_phase2_evaluations),
            "last_finite_criterion": float(last_finite_criterion),
            "best_finite_criterion": float(best_finite_criterion),
            "post_convergence_residual": float(post_convergence_residual),
            "post_convergence_kappa_est": float(post_convergence_kappa_est),
            "post_convergence_score_residual": float(post_convergence_score_residual),
            "score_tolerance": float(score_tol),
            "score_validation_policy": str(score_validation_policy),
            "score_validation_status": str(score_validation_status),
            "score_warning_codes": tuple(score_warning_codes),
            "score_warning_messages": tuple(score_warning_messages),
            "post_convergence_score_exceeds_tolerance": bool(score_exceeds_tolerance),
        }
        if phase == 0:
            phase1_attempts += 1
            candidate_usable = bool(
                converged
                and _phase2_initialization_usable_fullcancel(
                    endog=endog,
                    exog=exog,
                    beta=beta,
                    groups=groups,
                    counts=counts,
                    tau=tau,
                    control=control,
                    unweighted_x_demeaned=x_demeaned,
                )
            )
            if candidate_usable:
                phase1_accepted += 1
                phase1_accepted_beta = beta.copy()
            else:
                phase1_rejected += 1
                beta = phase1_accepted_beta.copy()
                if converged and failure_reason is None:
                    failure_reason = KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE
            diag_converged = candidate_usable
            diag_details: Mapping[str, Any] = {
                **diag_details_common,
                "phase1_attempts": int(phase1_attempts),
                "phase1_accepted": int(phase1_accepted),
                "phase1_rejected": int(phase1_rejected),
            }
        else:
            if not converged:
                beta = beta_outer_initial.copy()
            rho_details.update(
                {
                    **diag_details_common,
                    "phase1_attempts": int(phase1_attempts),
                    "phase1_accepted": int(phase1_accepted),
                    "phase1_rejected": int(phase1_rejected),
                }
            )
            diag_converged = bool(converged)
            diag_details = rho_details

        final_diag = IOLSIterationDiagnostics(
            converged=bool(diag_converged),
            phase="P1" if phase == 0 else "P2",
            iterations=int(iteration_count),
            restarts=int(restart_index),
            final_rho=None if phase == 0 else float(param),
            rho_path=tuple(float(value) for value in rho_path[: restart_index + 1]),
            final_criterion=final_criterion,
            anderson_m_initial=int(control.anderson_m),
            anderson_m_final=int(aa_m_final),
            failure_reason=None
            if diag_converged
            else failure_reason or KERNEL_FAILURE_NOT_CONVERGED,
            details=diag_details,
        )

    linear = _fullcancel_linear_state(exog, beta) if np.all(np.isfinite(beta)) else None
    return _ConcentratedSolve(
        beta=np.asarray(beta, dtype=float),
        linear=linear,
        iteration=final_diag,
    )


def _concentrated_fullrecover(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    control: IOLSKernelControl,
    beta_init: np.ndarray | None = None,
    linear_init: np.ndarray | None = None,
    rho_hint: float | None = None,
    score_validation_policy: ScoreValidationPolicy = "diagnostic",
    invariant_state: _FullrecoverInvariantState | None = None,
    direct_nofe_fixed_tau: bool = False,
    direct_rho_grid_policy: str = "legacy",
) -> _ConcentratedSolve:
    if score_validation_policy != "diagnostic":
        raise ValueError("score_validation_policy must be 'diagnostic'")
    # rho_grid_hook_v1: opt-in policy is direct-route only.
    if direct_rho_grid_policy not in DIRECT_RHO_GRID_POLICIES:
        raise ValueError(
            f"rho_grid_policy must be one of {DIRECT_RHO_GRID_POLICIES}"
        )
    if direct_rho_grid_policy != "legacy" and not direct_nofe_fixed_tau:
        raise ValueError(
            "rho_grid_policy other than 'legacy' requires direct_nofe_fixed_tau."
        )
    if direct_nofe_fixed_tau and (
        beta_init is not None or not _is_singleton_intercept_group(groups, counts)
    ):
        raise ValueError(
            "Direct fixed-tau Phase 1 + Phase 2 is authorized only for the "
            "noFE singleton-intercept route without beta_init."
        )
    if (
        beta_init is None
        and abs(float(tau) - 2.0) > 1e-10
        and not direct_nofe_fixed_tau
    ):
        raise ValueError(
            "Non-anchor iOLS path steps require beta_init from the tau=2 "
            "Phase 1 + Phase 2 anchor."
        )

    if beta_init is None:
        delta_grid = _delta_base_for_anchor(control)
    else:
        delta_grid = np.array([], dtype=float)
    state = (
        invariant_state
        if invariant_state is not None
        else _fullrecover_invariant_state(
            exog=exog,
            groups=groups,
            counts=counts,
            control=control,
        )
    )
    x_demeaned = state.x_demeaned
    projection = state.projection

    if beta_init is not None:
        beta = beta_init.copy()
        if linear_init is not None:
            linear = np.asarray(linear_init, dtype=float).copy()
        else:
            linear = _initial_linear_state(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                counts=counts,
                tau=tau,
                control=control,
            )
        n_outer = 1
    else:
        y_init = _demean_for_groups(
            np.log(endog + 1.0),
            groups,
            counts,
            tol=float(control.fe_demean_tol),
            maxiter=int(control.fe_demean_maxiter),
        )
        beta = projection @ y_init
        linear = _initial_linear_state(
            endog=endog,
            exog=exog,
            beta=beta,
            groups=groups,
            counts=counts,
            tau=tau,
            control=control,
        )
        n_outer = len(delta_grid) + 1
    beta_initial = beta.copy()
    linear_initial = linear.copy()
    phase1_accepted_beta = beta.copy()
    phase1_accepted_linear = linear.copy()
    phase1_attempts = 0
    phase1_accepted = 0
    phase1_rejected = 0
    # runtime_v2: total Phase 1 map evaluations (plain steps plus Anderson trial
    # maps, summed over every delta attempt). Usability probes and the
    # post-convergence check step are not counted; CPU time covers them.
    phase1_evaluations = 0
    if direct_nofe_fixed_tau:
        # runtime_v3 direct-route state (never read on legacy routes):
        # restart policy, best preconditioned-score iterate over Phase 2,
        # per-attempt trace and the diagnostic-only coordinate kappa maximum.
        restart_from_best = (
            str(control.direct_restart_start) == "best_score_iterate"
        )
        best_score_norm = float("inf")
        best_score_rho = float("nan")
        best_score_beta: np.ndarray | None = None
        phase2_warm_restarts = 0
        phase2_attempt_trace: list[dict[str, Any]] = []
        kappa_coordinate_row_max = float("nan")
        attempt_evaluations_start = 0
        attempt_start_policy = "phase1_start"
        # runtime_v4: final verification maps that came back invalid, by
        # internal phase (0 = Phase 1 delta grid, 1 = Phase 2 rho grid).
        post_convergence_invalid_map_counts = [0, 0]
        # rho_grid_hook_v1: per-policy grid diagnostics (None when not applicable).
        hook_alpha0: float | None = None
        hook_alpha_min: float | None = None
        hook_clipped: bool | None = None
        hook_values: tuple[float, ...] | None = None
        hook_curvature: dict[str, Any] | None = None

    final_diag = IOLSIterationDiagnostics(
        converged=False,
        phase="not_started",
        iterations=0,
        restarts=0,
        final_rho=None,
        rho_path=(),
        final_criterion=None,
        anderson_m_initial=int(control.anderson_m),
        anderson_m_final=int(control.anderson_m),
        failure_reason=KERNEL_FAILURE_NOT_STARTED,
    )

    if beta_init is None and abs(float(tau) - 2.0) <= 1e-10:
        support_details = _recovered_fe_tau2_boundary_separation(
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
        )
        if bool(support_details.get("separation_detected", False)):
            support_details.update(
                {
                    "phase1_attempts": 0,
                    "phase1_accepted": 0,
                    "phase1_rejected": 0,
                    "rho_candidate_count": 0,
                    "fe_state_policy": _fullrecover_fe_policy(groups),
                    "anderson_policy": "not_started_boundary_support_failure",
                    "failure_reason": TAU2_BOUNDARY_SEPARATION_CODE,
                }
            )
            return _ConcentratedSolve(
                beta=np.asarray(beta, dtype=float),
                linear=np.asarray(linear, dtype=float),
                iteration=IOLSIterationDiagnostics(
                    converged=False,
                    phase="support_check",
                    iterations=0,
                    restarts=0,
                    final_rho=None,
                    rho_path=(),
                    final_criterion=None,
                    anderson_m_initial=int(control.anderson_m),
                    anderson_m_final=0,
                    failure_reason=TAU2_BOUNDARY_SEPARATION_CODE,
                    details=support_details,
                ),
            )

    for outer_index in range(n_outer):
        if np.any(np.isnan(beta)):
            beta = beta_initial.copy()
            linear = linear_initial.copy()
        beta_outer_initial = beta.copy()
        linear_outer_initial = linear.copy()

        if outer_index < len(delta_grid):
            phase = 0
            param = float(delta_grid[outer_index])
            eps_current = max(float(control.eps) * 10.0, 1e-5)
            stopcrit_current = min(
                int(control.stopcrit),
                int(control.max_phase1_iterations),
            )
            rho_path = np.array([], dtype=float)
        else:
            phase = 1
            if beta_init is None:
                beta = phase1_accepted_beta.copy()
                linear = phase1_accepted_linear.copy()
            linear_for_rho = np.asarray(linear, dtype=float)
            if direct_nofe_fixed_tau and direct_rho_grid_policy != "legacy":
                # rho_grid_hook_v1: step grid; curvature L0 at the Phase-2 pilot.
                if direct_rho_grid_policy == "curvature_step_grid":
                    hook_curvature = _direct_profiled_curvature_L0(
                        endog=endog,
                        exog=exog,
                        beta=beta,
                        groups=groups,
                        counts=counts,
                        x_demeaned=x_demeaned,
                        tau=tau,
                        control=control,
                    )
                rho_path, rho_details = _direct_policy_rho_path(
                    policy=direct_rho_grid_policy,
                    linear=linear_for_rho,
                    tau=tau,
                    control=control,
                    curvature=hook_curvature,
                )
                hook_alpha0 = float(rho_details.pop("_rho_grid_alpha0"))
                hook_alpha_min = float(rho_details.pop("_rho_grid_alpha_min"))
                hook_clipped = bool(rho_details.pop("_rho_grid_clipped"))
                hook_values = tuple(float(v) for v in rho_path)
            else:
                rho_path, rho_details = _rho_path_for_linear_state(
                    linear=linear_for_rho,
                    tau=tau,
                    rho_hint=rho_hint,
                    control=control,
                )
            if rho_path.size == 0:
                rho_details.update(
                    {
                        "phase1_attempts": int(phase1_attempts),
                        "phase1_accepted": int(phase1_accepted),
                        "phase1_rejected": int(phase1_rejected),
                    }
                )
                if direct_nofe_fixed_tau:
                    rho_details["phase1_evaluations"] = int(phase1_evaluations)
                return _ConcentratedSolve(
                    beta=np.asarray(beta, dtype=float),
                    linear=np.asarray(linear, dtype=float),
                    iteration=IOLSIterationDiagnostics(
                        converged=False,
                        phase="P2",
                        iterations=0,
                        restarts=0,
                        final_rho=None,
                        rho_path=(),
                        final_criterion=None,
                        anderson_m_initial=int(control.anderson_m),
                        anderson_m_final=int(control.anderson_m),
                        failure_reason=KERNEL_FAILURE_EMPTY_FINITE_RHO_PATH,
                        details=rho_details,
                    ),
                )
            param = float(rho_path[0])
            eps_current = float(control.eps)
            stopcrit_current = int(control.stopcrit)
            beta_outer_initial = beta.copy()
            linear_outer_initial = linear.copy()

        restart_index = 0
        verified = False
        phase2_evaluations = 0
        phase2_budget_exhausted = False
        iteration_count = 0
        final_criterion: float | None = None
        aa_m_final = int(control.anderson_m)
        failure_reason = None
        qhat_last = float("nan")
        qhat_invalid_count = 0
        kappa_monitor_last = float("nan")
        kappa_monitor_max = float("nan")
        kappa_monitor_checks = 0
        post_convergence_residual = float("nan")
        post_convergence_kappa_est = float("nan")
        post_convergence_score_residual = float("nan")
        post_convergence_fe_score_residual = float("nan")
        score_tol = float("nan")
        score_validation_status = "not_checked"
        score_warning_codes: list[str] = []
        score_warning_messages: list[str] = []
        score_exceeds_tolerance = False
        fe_score_exceeds_tolerance = False
        last_finite_criterion = float("nan")
        best_finite_criterion = float("nan")

        while not verified:
            k = 1
            criterion = 100.0
            criterion_history: list[float] = []
            kappa_bad_count = 0
            beta_path = [beta.copy()]
            beta_path_entries = 1
            kappa_row_max = float("nan")
            if direct_nofe_fixed_tau and phase == 1:
                attempt_evaluations_start = phase2_evaluations
            joint_hdfe = _uses_joint_hdfe_state(groups)
            aa_m = (
                0
                if joint_hdfe
                else _effective_anderson_m(control.anderson_m, phase, param)
            )
            aa_x_hist: list[np.ndarray] = []
            aa_f_hist: list[np.ndarray] = []
            # runtime_v4 direct-route Phase 2 progress monitor: max-abs
            # plain-map residual at the current iterate, recorded at the same
            # iterations as the criterion. It drives stagnation, kappa and the
            # Anderson memory adaptation; the eps stop still uses `criterion`.
            progress_history: list[float] = []
            plain_residual_maxabs = float("nan")
            attempt_aa_m_start = aa_m
            attempt_aa_accepted = 0

            while (
                criterion > eps_current
                and k < stopcrit_current
                and not np.isnan(criterion)
            ):
                beta0 = beta.copy()
                linear0 = linear.copy()
                if phase == 1:
                    phase2_evaluations += 1
                    if phase2_evaluations > int(control.max_phase2_evaluations):
                        criterion = np.nan
                        failure_reason = (
                            KERNEL_FAILURE_PHASE2_EVALUATION_BUDGET_EXHAUSTED
                        )
                        phase2_budget_exhausted = True
                        break
                else:
                    phase1_evaluations += 1
                beta_phi, linear_phi, valid = _phi_step_fullrecover(
                    endog=endog,
                    exog=exog,
                    beta=beta,
                    linear=linear,
                    groups=groups,
                    counts=counts,
                    tau=tau,
                    projection=projection,
                    phase=phase,
                    param=param,
                    fe_profile_maxiter=int(control.fe_profile_maxiter),
                    fe_profile_tol=float(control.fe_profile_tol),
                    fe_demean_maxiter=int(control.fe_demean_maxiter),
                    fe_demean_tol=float(control.fe_demean_tol),
                )
                if not valid:
                    criterion = np.nan
                    failure_reason = KERNEL_FAILURE_INVALID_PHI_STEP
                    break
                if direct_nofe_fixed_tau and phase == 1:
                    # runtime_v3: rho-free preconditioned score norm of the
                    # current iterate, (1+rho)*||phi(beta)-beta||_2 =
                    # ||P score_obs(beta)||_2; tracks the best Phase 2 iterate.
                    step_score_norm = float(np.linalg.norm(beta_phi - beta)) * (
                        1.0 + float(param)
                    )
                    if np.isfinite(step_score_norm) and (
                        step_score_norm < best_score_norm
                    ):
                        best_score_norm = step_score_norm
                        best_score_rho = float(param)
                        best_score_beta = beta.copy()
                    # runtime_v4: same operands as the plain-step criterion,
                    # so it equals `criterion` whenever no Anderson step is
                    # accepted; no extra map evaluation.
                    plain_residual_maxabs = float(np.max(np.abs(beta_phi - beta)))

                accepted_aa_step = False
                if aa_m > 0 and k > aa_m + 1:
                    f_k = beta_phi - beta
                    aa_x_hist.append(beta.copy())
                    aa_f_hist.append(f_k.copy())
                    if len(aa_f_hist) > aa_m + 1:
                        aa_x_hist.pop(0)
                        aa_f_hist.pop(0)

                    m_cur = len(aa_f_hist) - 1
                    if m_cur >= 1:
                        f_mat = np.column_stack(
                            [
                                aa_f_hist[-1] - aa_f_hist[-(j + 1)]
                                for j in range(1, m_cur + 1)
                            ]
                        )
                        gamma, _, _, _ = np.linalg.lstsq(
                            f_mat,
                            aa_f_hist[-1],
                            rcond=None,
                        )
                        beta_aa = beta_phi.copy()
                        for j in range(m_cur):
                            phi_prev = aa_x_hist[-(j + 2)] + aa_f_hist[-(j + 2)]
                            beta_aa += gamma[j] * (phi_prev - beta_phi)

                        if np.all(np.isfinite(beta_aa)):
                            if phase == 1:
                                phase2_evaluations += 1
                                if phase2_evaluations > int(
                                    control.max_phase2_evaluations
                                ):
                                    criterion = np.nan
                                    failure_reason = (
                                        KERNEL_FAILURE_PHASE2_BUDGET_EXHAUSTED
                                    )
                                    phase2_budget_exhausted = True
                                    break
                            else:
                                phase1_evaluations += 1
                            beta_aa_phi, _, aa_valid = _phi_step_fullrecover(
                                endog=endog,
                                exog=exog,
                                beta=beta_aa,
                                linear=None,
                                groups=groups,
                                counts=counts,
                                tau=tau,
                                projection=projection,
                                phase=phase,
                                param=param,
                                fe_profile_maxiter=int(control.fe_profile_maxiter),
                                fe_profile_tol=float(control.fe_profile_tol),
                                fe_demean_maxiter=int(control.fe_demean_maxiter),
                                fe_demean_tol=float(control.fe_demean_tol),
                            )
                            if aa_valid:
                                res_plain = np.linalg.norm(beta_phi - beta)
                                res_aa = np.linalg.norm(beta_aa_phi - beta_aa)
                                if res_aa < res_plain * 1.1:
                                    beta = beta_aa
                                    (
                                        beta_path_entries,
                                        kappa_row_max,
                                    ) = _record_beta_path_entry(
                                        beta_path,
                                        beta,
                                        entries_seen=beta_path_entries,
                                        kappa_row_max=kappa_row_max,
                                    )
                                    if k >= 2:
                                        criterion = float(np.max(np.abs(beta - beta0)))
                                    accepted_aa_step = True
                                    attempt_aa_accepted += 1
                elif aa_m > 0:
                    f_k = beta_phi - beta
                    aa_x_hist.append(beta.copy())
                    aa_f_hist.append(f_k.copy())

                if not accepted_aa_step:
                    beta = beta_phi
                    if linear_phi is not None:
                        linear = linear_phi
                    elif not joint_hdfe:
                        linear = _initial_linear_state(
                            endog=endog,
                            exog=exog,
                            beta=beta,
                            groups=groups,
                            counts=counts,
                            tau=tau,
                            control=control,
                        )
                    beta_path_entries, kappa_row_max = _record_beta_path_entry(
                        beta_path,
                        beta,
                        entries_seen=beta_path_entries,
                        kappa_row_max=kappa_row_max,
                    )
                if k >= 2:
                    if not accepted_aa_step:
                        beta_criterion = float(np.max(np.abs(beta - beta0)))
                        if joint_hdfe:
                            linear_criterion = float(np.max(np.abs(linear - linear0)))
                            criterion = max(beta_criterion, linear_criterion)
                        else:
                            criterion = beta_criterion
                    direct_progress = direct_nofe_fixed_tau and phase == 1
                    if direct_progress and np.isfinite(plain_residual_maxabs):
                        progress_history.append(plain_residual_maxabs)
                    (
                        last_finite_criterion,
                        best_finite_criterion,
                        criterion_failure,
                    ) = _record_criterion_progress(
                        criterion_history,
                        criterion,
                        phase=phase,
                        tau=tau,
                        eps=eps_current,
                        max_phase2_evaluations=int(control.max_phase2_evaluations),
                        last_finite_criterion=last_finite_criterion,
                        best_finite_criterion=best_finite_criterion,
                        stagnation_history=(
                            progress_history if direct_progress else None
                        ),
                    )
                    if criterion_failure is not None:
                        criterion = np.nan
                        failure_reason = criterion_failure

                if (
                    k > 4
                    and len(beta_path) >= 4
                    and not (phase == 1 and abs(tau - 2.0) <= 1e-10)
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        d1 = np.abs(beta_path[-1] - beta_path[-2])
                        d2 = np.abs(beta_path[-2] - beta_path[-3])
                        d3 = np.abs(beta_path[-3] - beta_path[-4])
                        qhat = np.nanmean(np.log(d1 / d2) / np.log(d2 / d3))
                    qhat_last = float(qhat)
                    if np.isinf(np.abs(qhat)) and criterion > eps_current * 10.0:
                        qhat_invalid_count += 1
                        # runtime_v3 direct route: advisory only (counted,
                        # never a restart); the C++ core has no such guard.
                        if not direct_nofe_fixed_tau:
                            criterion = np.nan
                            failure_reason = KERNEL_FAILURE_Q_CONVERGENCE_INVALID

                if (
                    phase == 1
                    and abs(tau - 2.0) > 1e-10
                    and k >= 30
                    and k % 10 == 0
                    and beta_path_entries >= 4
                ):
                    kappa_val = kappa_row_max
                    if direct_nofe_fixed_tau:
                        # runtime_v3: windowed geometric-mean step contraction
                        # replaces the lifetime max of coordinate-mean ratio
                        # rows (which one near-zero previous coordinate change
                        # poisons for the rest of the attempt). The coordinate
                        # statistic is kept as a diagnostic only.
                        # +inf rows are kept: they are the pollution this
                        # diagnostic exists to show.
                        if not np.isnan(kappa_row_max):
                            kappa_coordinate_row_max = (
                                float(kappa_row_max)
                                if np.isnan(kappa_coordinate_row_max)
                                else max(kappa_coordinate_row_max, float(kappa_row_max))
                            )
                        # runtime_v4: the windowed rate is taken over the
                        # plain-map residual history, not accepted moves.
                        kappa_val = _windowed_contraction_rate(
                            progress_history,
                            window=_DIRECT_KAPPA_WINDOW,
                        )
                    kappa_monitor_last = float(kappa_val)
                    kappa_monitor_checks += 1
                    if np.isfinite(kappa_val):
                        if np.isfinite(kappa_monitor_max):
                            kappa_monitor_max = max(
                                kappa_monitor_max,
                                float(kappa_val),
                            )
                        else:
                            kappa_monitor_max = float(kappa_val)
                    aa_m_new = _adapt_anderson_m(aa_m, kappa_val)
                    if aa_m_new > aa_m:
                        aa_m = aa_m_new
                        kappa_bad_count = 0
                    elif np.isfinite(kappa_val) and kappa_val >= 1.0:
                        kappa_bad_count += 1
                    else:
                        kappa_bad_count = 0
                    if kappa_bad_count >= 3:
                        criterion = np.nan
                        failure_reason = KERNEL_FAILURE_NONCONTRACTIVE

                k += 1
                iteration_count = k

                if np.isnan(criterion):
                    restart_index += 1
                    if (
                        phase == 1
                        and not phase2_budget_exhausted
                        and restart_index < len(rho_path)
                    ):
                        if direct_nofe_fixed_tau:
                            phase2_attempt_trace.append(
                                {
                                    # restart_index was already advanced: the
                                    # attempt that just ended is the previous one.
                                    "attempt": int(restart_index - 1),
                                    "rho": float(param),
                                    "iterations": int(k),
                                    "evaluations": int(
                                        phase2_evaluations - attempt_evaluations_start
                                    ),
                                    "failure_reason": failure_reason,
                                    "final_criterion": (
                                        None
                                        if np.isnan(last_finite_criterion)
                                        else float(last_finite_criterion)
                                    ),
                                    "best_criterion": float(
                                        min(criterion_history)
                                        if criterion_history
                                        else float("nan")
                                    ),
                                    "kappa_last": float(kappa_monitor_last),
                                    "start_policy": attempt_start_policy,
                                    **_direct_progress_trace_fields(
                                        progress_history,
                                        aa_m_start=attempt_aa_m_start,
                                        aa_m_end=aa_m,
                                        aa_accepted=attempt_aa_accepted,
                                    ),
                                }
                            )
                        param = float(rho_path[restart_index])
                        criterion = 100.0
                        beta = beta_outer_initial.copy()
                        linear = linear_outer_initial.copy()
                        if direct_nofe_fixed_tau:
                            attempt_start_policy = "phase1_start"
                            if restart_from_best and best_score_beta is not None:
                                beta = best_score_beta.copy()
                                linear = _initial_linear_state(
                                    endog=endog,
                                    exog=exog,
                                    beta=beta,
                                    groups=groups,
                                    counts=counts,
                                    tau=tau,
                                    control=control,
                                )
                                phase2_warm_restarts += 1
                                attempt_start_policy = "best_score_iterate"
                            attempt_evaluations_start = phase2_evaluations
                            kappa_monitor_last = float("nan")
                        k = 1
                        criterion_history = []
                        kappa_bad_count = 0
                        beta_path = [beta.copy()]
                        beta_path_entries = 1
                        kappa_row_max = float("nan")
                        aa_x_hist = []
                        aa_f_hist = []
                        # runtime_v4: aa_m itself is still carried over (v3).
                        progress_history = []
                        plain_residual_maxabs = float("nan")
                        attempt_aa_m_start = aa_m
                        attempt_aa_accepted = 0
                    elif phase == 0:
                        break

            converged = not np.isnan(criterion) and criterion <= eps_current
            final_criterion = None if np.isnan(criterion) else float(criterion)
            if converged:
                beta_check, linear_check, valid = _phi_step_fullrecover(
                    endog=endog,
                    exog=exog,
                    beta=beta,
                    linear=linear,
                    groups=groups,
                    counts=counts,
                    tau=tau,
                    projection=projection,
                    phase=phase,
                    param=param,
                    fe_profile_maxiter=int(control.fe_profile_maxiter),
                    fe_profile_tol=float(control.fe_profile_tol),
                    fe_demean_maxiter=int(control.fe_demean_maxiter),
                    fe_demean_tol=float(control.fe_demean_tol),
                )
                if valid:
                    residual = float(np.max(np.abs(beta_check - beta)))
                    if joint_hdfe and linear_check is not None:
                        residual = max(
                            residual,
                            float(np.max(np.abs(linear_check - linear))),
                        )
                    post_convergence_residual = float(residual)
                    if residual > eps_current * 10.0:
                        converged = False
                        failure_reason = KERNEL_FAILURE_POST_CONVERGENCE_RESIDUAL
                elif direct_nofe_fixed_tau:
                    # runtime_v4: an invalid final verification map clears
                    # native convergence (v3 kept it). No residual exists, so
                    # an earlier attempt's post-check values are not reported.
                    converged = False
                    failure_reason = KERNEL_FAILURE_POST_CONVERGENCE_INVALID_MAP
                    post_convergence_invalid_map_counts[phase] += 1
                    post_convergence_residual = float("nan")
                    post_convergence_kappa_est = float("nan")
                if converged and phase == 1:
                    score_residual = _profiled_slope_score_residual(
                        endog=endog,
                        exog=exog,
                        beta=beta,
                        groups=groups,
                        counts=counts,
                        x_demeaned=x_demeaned,
                        tau=tau,
                        linear=linear if joint_hdfe else None,
                        fe_profile_maxiter=int(control.fe_profile_maxiter),
                        fe_profile_tol=float(control.fe_profile_tol),
                    )
                    score_tol = max(eps_current * 10.0, 1e-8)
                    post_convergence_score_residual = float(score_residual)
                    score_exceeds_tolerance = bool(
                        not np.isfinite(score_residual) or score_residual > score_tol
                    )
                    if score_exceeds_tolerance:
                        score_warning_codes.append("post_convergence_score_residual")
                        score_warning_messages.append(
                            "Slope score residual exceeds tolerance at this "
                            "tau-path state."
                        )
                        score_validation_status = "diagnostic_warning"
                    elif score_validation_status == "not_checked":
                        score_validation_status = "passed"
                if converged and phase == 1:
                    fe_score_residual = _profiled_fe_score_residual(
                        endog=endog,
                        exog=exog,
                        beta=beta,
                        groups=groups,
                        counts=counts,
                        tau=tau,
                        linear=linear if joint_hdfe else None,
                        fe_profile_maxiter=int(control.fe_profile_maxiter),
                        fe_profile_tol=float(control.fe_profile_tol),
                    )
                    score_tol = max(eps_current * 10.0, 1e-8)
                    post_convergence_fe_score_residual = float(fe_score_residual)
                    fe_score_exceeds_tolerance = bool(
                        not np.isfinite(fe_score_residual)
                        or fe_score_residual > score_tol
                    )
                    if fe_score_exceeds_tolerance:
                        score_warning_codes.append("post_convergence_fe_score_residual")
                        score_warning_messages.append(
                            "Recovered-FE score residual exceeds tolerance at "
                            "this tau-path state."
                        )
                        score_validation_status = "diagnostic_warning"
                    elif score_validation_status == "not_checked":
                        score_validation_status = "passed"
                if (
                    converged
                    and len(beta_path) >= 4
                    and not (phase == 1 and abs(tau - 2.0) <= 1e-10)
                ):
                    diffs = [
                        np.max(np.abs(beta_path[-j] - beta_path[-j - 1]))
                        for j in range(1, min(4, len(beta_path)))
                    ]
                    if diffs[1] > 1e-15 and diffs[2] > 1e-15:
                        kappa_est = diffs[0] / diffs[1]
                        post_convergence_kappa_est = float(kappa_est)
                        if kappa_est >= 1.0:
                            converged = False
                            failure_reason = (
                                KERNEL_FAILURE_POST_CONVERGENCE_NONCONTRACTIVE
                            )

            if direct_nofe_fixed_tau and phase == 1:
                phase2_attempt_trace.append(
                    {
                        # restart_index can equal len(rho_path) when the last
                        # attempt was killed in-loop with no rho left.
                        "attempt": int(min(restart_index, len(rho_path) - 1)),
                        "rho": float(param),
                        # k (not iteration_count) so an attempt that ended on
                        # its first map does not inherit the previous count.
                        "iterations": int(k),
                        "evaluations": int(
                            phase2_evaluations - attempt_evaluations_start
                        ),
                        # a finite criterion above eps here means the per-rho
                        # cap (stopcrit) ended the attempt; failure_reason is
                        # per phase and would be stale.
                        "failure_reason": (
                            None
                            if converged
                            else (
                                "stopcrit_reached"
                                if not np.isnan(criterion) and criterion > eps_current
                                else failure_reason
                            )
                        ),
                        "final_criterion": final_criterion,
                        "best_criterion": float(
                            min(criterion_history)
                            if criterion_history
                            else float("nan")
                        ),
                        "kappa_last": float(kappa_monitor_last),
                        "start_policy": attempt_start_policy,
                        **_direct_progress_trace_fields(
                            progress_history,
                            aa_m_start=attempt_aa_m_start,
                            aa_m_end=aa_m,
                            aa_accepted=attempt_aa_accepted,
                        ),
                    }
                )
            if (
                not converged
                and phase == 1
                and not phase2_budget_exhausted
                and restart_index < len(rho_path) - 1
            ):
                restart_index = min(
                    restart_index + max(restart_index, 5),
                    len(rho_path) - 1,
                )
                param = float(rho_path[restart_index])
                beta = beta_outer_initial.copy()
                linear = linear_outer_initial.copy()
                if direct_nofe_fixed_tau:
                    attempt_start_policy = "phase1_start"
                    if restart_from_best and best_score_beta is not None:
                        beta = best_score_beta.copy()
                        linear = _initial_linear_state(
                            endog=endog,
                            exog=exog,
                            beta=beta,
                            groups=groups,
                            counts=counts,
                            tau=tau,
                            control=control,
                        )
                        phase2_warm_restarts += 1
                        attempt_start_policy = "best_score_iterate"
                    kappa_monitor_last = float("nan")
            else:
                verified = True

            aa_m_final = aa_m

        if phase == 0:
            phase1_attempts += 1
            candidate_usable = bool(
                converged
                and _phase2_initialization_usable(
                    endog=endog,
                    exog=exog,
                    beta=beta,
                    linear=linear,
                    groups=groups,
                    counts=counts,
                    tau=tau,
                    projection=projection,
                    control=control,
                )
            )
            if candidate_usable:
                phase1_accepted += 1
                phase1_accepted_beta = beta.copy()
                phase1_accepted_linear = linear.copy()
            else:
                phase1_rejected += 1
                beta = phase1_accepted_beta.copy()
                linear = phase1_accepted_linear.copy()
                if converged and failure_reason is None:
                    failure_reason = KERNEL_FAILURE_PHASE1_CANDIDATE_UNUSABLE
            diag_converged = candidate_usable
            diag_details: Mapping[str, Any] = {
                "phase1_attempts": int(phase1_attempts),
                "phase1_accepted": int(phase1_accepted),
                "phase1_rejected": int(phase1_rejected),
                "fe_state_policy": _fullrecover_fe_policy(groups),
                "anderson_policy": "disabled_for_joint_hdfe"
                if _uses_joint_hdfe_state(groups)
                else "beta_anderson",
                "qhat_last": float(qhat_last),
                "qhat_invalid_count": int(qhat_invalid_count),
                "kappa_monitor_last": float(kappa_monitor_last),
                "kappa_monitor_max": float(kappa_monitor_max),
                "kappa_monitor_checks": int(kappa_monitor_checks),
                **(
                    {"phase1_evaluations": int(phase1_evaluations)}
                    if direct_nofe_fixed_tau
                    else {}
                ),
                "phase2_evaluations": int(phase2_evaluations),
                "max_phase2_evaluations": int(control.max_phase2_evaluations),
                "last_finite_criterion": float(last_finite_criterion),
                "best_finite_criterion": float(best_finite_criterion),
                "post_convergence_residual": float(post_convergence_residual),
                "post_convergence_kappa_est": float(post_convergence_kappa_est),
                "post_convergence_score_residual": float(
                    post_convergence_score_residual
                ),
                "post_convergence_fe_score_residual": float(
                    post_convergence_fe_score_residual
                ),
                "score_tolerance": float(score_tol),
                "score_validation_policy": str(score_validation_policy),
                "score_validation_status": str(score_validation_status),
                "score_warning_codes": tuple(score_warning_codes),
                "score_warning_messages": tuple(score_warning_messages),
                "post_convergence_score_exceeds_tolerance": bool(
                    score_exceeds_tolerance
                ),
                "post_convergence_fe_score_exceeds_tolerance": bool(
                    fe_score_exceeds_tolerance
                ),
            }
        else:
            if not converged and not direct_nofe_fixed_tau:
                # Legacy tau-path semantics: a failed Phase 2 step hands the
                # anchor/previous state back to the caller.
                beta = beta_outer_initial.copy()
                linear = linear_outer_initial.copy()
            # runtime_v2 direct route: keep the final iterate of the last rho
            # attempt. A native unsuccessful stop (budget, stagnation,
            # noncontractive monitor) is not evidence that the iterate is
            # inaccurate; the external exact-score audit decides, and the
            # Phase 1 start would discard any accurate root already reached.
            if direct_nofe_fixed_tau:
                # Direct-route-only keys keep legacy tau-path detail dicts
                # byte-identical to runtime_v1.
                nonconverged_policy = "last_attempt_final_iterate"
                if restart_from_best:
                    nonconverged_policy = "best_score_iterate"
                    if not converged and best_score_beta is not None:
                        # runtime_v3: hand back the smallest preconditioned
                        # score iterate of the whole Phase 2 (the external
                        # exact-score audit decides; the native flag stays
                        # unconverged).
                        beta = best_score_beta.copy()
                        linear = _initial_linear_state(
                            endog=endog,
                            exog=exog,
                            beta=beta,
                            groups=groups,
                            counts=counts,
                            tau=tau,
                            control=control,
                        )
                rho_details.update(
                    {
                        "nonconverged_iterate_policy": nonconverged_policy,
                        "phase1_evaluations": int(phase1_evaluations),
                        "direct_restart_start": str(control.direct_restart_start),
                        "phase2_attempts": int(len(phase2_attempt_trace)),
                        "phase2_warm_restarts": int(phase2_warm_restarts),
                        "best_score_norm": float(best_score_norm),
                        "best_score_rho": float(best_score_rho),
                        "phase2_attempt_trace": tuple(phase2_attempt_trace),
                        "kappa_coordinate_row_max": float(kappa_coordinate_row_max),
                        "kappa_monitor_policy": DIRECT_KAPPA_MONITOR_POLICY,
                        "qhat_guard_policy": DIRECT_QHAT_GUARD_POLICY_NUMPY,
                        "progress_monitor_policy": DIRECT_PROGRESS_MONITOR_POLICY,
                        "post_check_policy": DIRECT_POST_CHECK_POLICY,
                        "post_convergence_invalid_map_count": int(
                            post_convergence_invalid_map_counts[1]
                        ),
                        "phase1_post_convergence_invalid_map_count": int(
                            post_convergence_invalid_map_counts[0]
                        ),
                        "rho_grid_policy": str(direct_rho_grid_policy),
                        "rho_grid_hook_version": DIRECT_RHO_GRID_HOOK_VERSION,
                        "rho_grid_count": int(len(rho_path)),
                        "rho_grid_values": hook_values,
                        "rho_grid_alpha0": hook_alpha0,
                        "rho_grid_alpha_min": hook_alpha_min,
                        "rho_grid_clipped": hook_clipped,
                        "rho_curvature_L0": (
                            None if hook_curvature is None else hook_curvature["L0"]
                        ),
                        "rho_curvature_lambda_min": (
                            None
                            if hook_curvature is None
                            else hook_curvature["lambda_min"]
                        ),
                        "rho_curvature_wall_ns": (
                            None
                            if hook_curvature is None
                            else int(hook_curvature["wall_ns"])
                        ),
                        "rho_curvature_pilot": (
                            None
                            if hook_curvature is None
                            else "phase1_accepted_beta"
                        ),
                    }
                )
            rho_details.update(
                {
                    "phase1_attempts": int(phase1_attempts),
                    "phase1_accepted": int(phase1_accepted),
                    "phase1_rejected": int(phase1_rejected),
                    "fe_state_policy": _fullrecover_fe_policy(groups),
                    "anderson_policy": "disabled_for_joint_hdfe"
                    if _uses_joint_hdfe_state(groups)
                    else "beta_anderson",
                    "qhat_last": float(qhat_last),
                    "qhat_invalid_count": int(qhat_invalid_count),
                    "kappa_monitor_last": float(kappa_monitor_last),
                    "kappa_monitor_max": float(kappa_monitor_max),
                    "kappa_monitor_checks": int(kappa_monitor_checks),
                    "phase2_evaluations": int(phase2_evaluations),
                    "max_phase2_evaluations": int(control.max_phase2_evaluations),
                    "last_finite_criterion": float(last_finite_criterion),
                    "best_finite_criterion": float(best_finite_criterion),
                    "post_convergence_residual": float(post_convergence_residual),
                    "post_convergence_kappa_est": float(post_convergence_kappa_est),
                    "post_convergence_score_residual": float(
                        post_convergence_score_residual
                    ),
                    "post_convergence_fe_score_residual": float(
                        post_convergence_fe_score_residual
                    ),
                    "score_tolerance": float(score_tol),
                    "score_validation_policy": str(score_validation_policy),
                    "score_validation_status": str(score_validation_status),
                    "score_warning_codes": tuple(score_warning_codes),
                    "score_warning_messages": tuple(score_warning_messages),
                    "post_convergence_score_exceeds_tolerance": bool(
                        score_exceeds_tolerance
                    ),
                    "post_convergence_fe_score_exceeds_tolerance": bool(
                        fe_score_exceeds_tolerance
                    ),
                }
            )
            diag_converged = bool(converged)
            diag_details = rho_details

        final_diag = IOLSIterationDiagnostics(
            converged=bool(diag_converged),
            phase="P1" if phase == 0 else "P2",
            iterations=int(iteration_count),
            restarts=int(restart_index),
            final_rho=None if phase == 0 else float(param),
            rho_path=tuple(float(value) for value in rho_path[: restart_index + 1]),
            final_criterion=final_criterion,
            anderson_m_initial=int(control.anderson_m),
            anderson_m_final=int(aa_m_final),
            failure_reason=None
            if diag_converged
            else failure_reason or KERNEL_FAILURE_NOT_CONVERGED,
            details=diag_details,
        )
        if bool(diag_converged) and not _uses_joint_hdfe_state(groups):
            linear = _initial_linear_state(
                endog=endog,
                exog=exog,
                beta=beta,
                groups=groups,
                counts=counts,
                tau=tau,
                control=control,
            )

    return _ConcentratedSolve(
        beta=np.asarray(beta, dtype=float),
        linear=np.asarray(linear, dtype=float) if np.all(np.isfinite(linear)) else None,
        iteration=final_diag,
    )


def _coerce_concentrated_solve(
    raw: _ConcentratedSolve | tuple[np.ndarray, IOLSIterationDiagnostics],
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau: float,
    control: IOLSKernelControl,
) -> _ConcentratedSolve:
    if isinstance(raw, _ConcentratedSolve):
        return raw
    beta, iteration = raw
    linear = None
    if iteration.converged and np.all(np.isfinite(beta)):
        linear = _initial_linear_state(
            endog=endog,
            exog=exog,
            beta=np.asarray(beta, dtype=float),
            groups=groups,
            counts=counts,
            tau=tau,
            control=control,
        )
    return _ConcentratedSolve(
        beta=np.asarray(beta, dtype=float),
        linear=linear,
        iteration=iteration,
    )


def _tau_path_grid(
    tau_target: float,
    *,
    n_steps: int,
) -> tuple[np.ndarray, float]:
    n_steps_actual = max(1, int(n_steps))
    delta = abs(float(tau_target) - 2.0)
    sign = 1.0 if tau_target > 2.0 else -1.0
    u = np.linspace(0.0, 1.0, n_steps_actual + 1)[1:]
    fracs = 1.0 - (1.0 - u) ** 2
    return 2.0 + sign * delta * fracs, sign


def _tau_path_next_try(
    *,
    tau_goal: float,
    tau_current: float,
    sign: float,
    step_cap: float | None,
) -> float:
    remaining_gap = abs(float(tau_goal) - float(tau_current))
    if step_cap is None or not np.isfinite(step_cap) or float(step_cap) <= 0.0:
        step = remaining_gap
    else:
        step = min(remaining_gap, float(step_cap))
    return float(tau_current + sign * step)


def _tau_path_refined_step_cap(
    *,
    failed_gap: float,
    min_dtau: float,
) -> float:
    return float(max(float(min_dtau), float(failed_gap) / 2.0))


def _combine_tau_path_iteration(
    *,
    converged: bool,
    failure_reason: str | None,
    diagnostics: list[IOLSIterationDiagnostics],
) -> IOLSIterationDiagnostics:
    if not diagnostics:
        return IOLSIterationDiagnostics(
            converged=False,
            phase="tau_path",
            iterations=0,
            restarts=0,
            final_rho=None,
            rho_path=(),
            final_criterion=None,
            anderson_m_initial=0,
            anderson_m_final=0,
            failure_reason=failure_reason or KERNEL_FAILURE_NOT_STARTED,
        )
    final = diagnostics[-1]
    rho_path: list[float] = []
    for diagnostic in diagnostics:
        rho_path.extend(float(value) for value in diagnostic.rho_path)
    return IOLSIterationDiagnostics(
        converged=bool(converged),
        phase="tau_path",
        iterations=sum(int(diagnostic.iterations) for diagnostic in diagnostics),
        restarts=sum(int(diagnostic.restarts) for diagnostic in diagnostics),
        final_rho=final.final_rho,
        rho_path=tuple(rho_path),
        final_criterion=final.final_criterion,
        anderson_m_initial=diagnostics[0].anderson_m_initial,
        anderson_m_final=final.anderson_m_final,
        failure_reason=(
            None
            if converged
            else failure_reason or final.failure_reason or KERNEL_FAILURE_NOT_CONVERGED
        ),
    )


def _solve_fullrecover_tau_path(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
    invariant_state: _FullrecoverInvariantState | None = None,
) -> _TauPathSolve:
    state = (
        invariant_state
        if invariant_state is not None
        else _fullrecover_invariant_state(
            exog=exog,
            groups=groups,
            counts=counts,
            control=control,
        )
    )
    anchor_solve = _coerce_concentrated_solve(
        _concentrated_fullrecover(
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
            tau=2.0,
            control=control,
            beta_init=None,
            rho_hint=None,
            invariant_state=state,
        ),
        endog=endog,
        exog=exog,
        groups=groups,
        counts=counts,
        tau=2.0,
        control=control,
    )
    anchor_beta = anchor_solve.beta
    anchor_linear = anchor_solve.linear
    anchor_diag = anchor_solve.iteration
    diagnostics = [anchor_diag]
    path_records: list[dict[str, Any]] = [
        {
            "tau": 2.0,
            "from_tau": None,
            "converged": bool(anchor_diag.converged),
            "beta_reduced": _as_owned_float64_vector(anchor_beta)
            if anchor_diag.converged
            else None,
            "rho": anchor_diag.final_rho,
            "iterations": int(anchor_diag.iterations),
            "restarts": int(anchor_diag.restarts),
            "anderson_m_initial": int(anchor_diag.anderson_m_initial),
            "anderson_m_final": int(anchor_diag.anderson_m_final),
            "failure_reason": anchor_diag.failure_reason,
            "beta_init_source_tau": None,
            "rho_hint": None,
            "_linear_predictor": _as_owned_float64_vector(anchor_linear)
            if anchor_diag.converged and anchor_linear is not None
            else None,
            "diagnostics": dict(anchor_diag.details),
        }
    ]
    details: dict[str, Any] = {
        "path_status": "target_reached",
        "tau_anchor": 2.0,
        "tau_target": float(tau_target),
        "tau_attained": 2.0,
        "tau_failed": None,
        "n_path_evaluations": 0,
        "n_path_successes": 0,
        "n_path_failures": 0,
        "tau_grid_policy": "quadratic_endpoint_dense_with_adaptive_bisection",
        "score_validation_policy": "diagnostic",
        "path": (),
    }
    if not anchor_diag.converged:
        details.update(
            {
                "path_status": "anchor_failed",
                "tau_failed": 2.0,
                "failure_reason": anchor_diag.failure_reason
                or KERNEL_FAILURE_TAU2_ANCHOR_FAILED,
                "path": tuple(path_records),
            }
        )
        return _TauPathSolve(
            beta=np.full_like(anchor_beta, np.nan, dtype=float),
            linear=None,
            tau_fit=2.0,
            converged=False,
            iteration=anchor_diag,
            details=details,
        )

    if anchor_diag.final_rho is None:
        details.update(
            {
                "path_status": "anchor_failed",
                "tau_failed": 2.0,
                "failure_reason": KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO,
                "path": tuple(path_records),
            }
        )
        return _TauPathSolve(
            beta=np.full_like(anchor_beta, np.nan, dtype=float),
            linear=None,
            tau_fit=2.0,
            converged=False,
            iteration=IOLSIterationDiagnostics(
                converged=False,
                phase=anchor_diag.phase,
                iterations=anchor_diag.iterations,
                restarts=anchor_diag.restarts,
                final_rho=None,
                rho_path=anchor_diag.rho_path,
                final_criterion=anchor_diag.final_criterion,
                anderson_m_initial=anchor_diag.anderson_m_initial,
                anderson_m_final=anchor_diag.anderson_m_final,
                failure_reason=KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO,
            ),
            details=details,
        )

    if abs(float(tau_target) - 2.0) <= 1e-10:
        details["path"] = tuple(path_records)
        return _TauPathSolve(
            beta=anchor_beta,
            linear=anchor_linear,
            tau_fit=2.0,
            converged=True,
            iteration=anchor_diag,
            details=details,
        )

    grid, sign = _tau_path_grid(
        tau_target,
        n_steps=control.tau_path_steps,
    )
    delta = abs(float(tau_target) - 2.0)
    explicit_min_dtau = None if control.min_dtau is None else float(control.min_dtau)
    min_dtau = (
        explicit_min_dtau
        if explicit_min_dtau is not None and explicit_min_dtau > 0.0
        else max(delta / max(float(control.tau_path_steps) * 128.0, 1.0), 1e-3)
    )
    path_tol = max(min_dtau * 0.1, 1e-12)
    use_local_frontier_refinement = sign < 0
    beta_current = anchor_beta
    beta_previous = anchor_beta.copy()
    linear_current = anchor_linear
    linear_previous = None if anchor_linear is None else anchor_linear.copy()
    rho_previous = anchor_diag.final_rho
    tau_current = 2.0
    failure_reason: str | None = None
    failed_tau: float | None = None

    for tau_goal in grid:
        if sign * (float(tau_goal) - tau_current) <= path_tol:
            continue
        frontier_step_cap: float | None = None
        frontier_locked = False
        frontier_expensive_failures = 0
        while sign * (float(tau_goal) - tau_current) > path_tol:
            tau_try = _tau_path_next_try(
                tau_goal=float(tau_goal),
                tau_current=float(tau_current),
                sign=sign,
                step_cap=frontier_step_cap,
            )
            if details["n_path_evaluations"] >= int(control.max_path_evaluations):
                failure_reason = KERNEL_FAILURE_MAX_PATH_EVALUATIONS_REACHED
                failed_tau = tau_try
                break

            tau_from = float(tau_current)
            details["n_path_evaluations"] += 1
            step_solve = _coerce_concentrated_solve(
                _concentrated_fullrecover(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau=tau_try,
                    control=control,
                    beta_init=beta_previous.copy(),
                    linear_init=None
                    if linear_previous is None
                    else linear_previous.copy(),
                    rho_hint=rho_previous,
                    score_validation_policy="diagnostic",
                    invariant_state=state,
                ),
                endog=endog,
                exog=exog,
                groups=groups,
                counts=counts,
                tau=tau_try,
                control=control,
            )
            beta_step = step_solve.beta
            linear_step = step_solve.linear
            step_diag = step_solve.iteration
            diagnostics.append(step_diag)
            path_record = {
                "tau": float(tau_try),
                "from_tau": float(tau_current),
                "step": float(abs(tau_try - tau_current)),
                "converged": bool(step_diag.converged),
                "beta_reduced": _as_owned_float64_vector(beta_step)
                if step_diag.converged
                else None,
                "rho": step_diag.final_rho,
                "iterations": int(step_diag.iterations),
                "restarts": int(step_diag.restarts),
                "anderson_m_initial": int(step_diag.anderson_m_initial),
                "anderson_m_final": int(step_diag.anderson_m_final),
                "failure_reason": step_diag.failure_reason,
                "beta_init_source_tau": float(tau_current),
                "rho_hint": None if rho_previous is None else float(rho_previous),
                "_linear_predictor": _as_owned_float64_vector(linear_step)
                if step_diag.converged and linear_step is not None
                else None,
                "diagnostics": dict(step_diag.details),
            }
            score_warning_refinement = (
                use_local_frontier_refinement
                and _tau_path_score_warning_refinement_step(step_diag)
            )
            if step_diag.converged:
                if score_warning_refinement:
                    details["n_path_failures"] += 1
                    frontier_locked = True
                    frontier_expensive_failures += 1
                    gap = abs(tau_try - tau_current)
                    if gap <= min_dtau * (1.0 + 1e-12):
                        failure_reason = KERNEL_FAILURE_MIN_DTAU_REACHED
                        failed_tau = tau_try
                        break
                    if frontier_expensive_failures > TAU_PATH_FRONTIER_RETRY_BUDGET:
                        failure_reason = (
                            step_diag.failure_reason or KERNEL_FAILURE_NOT_CONVERGED
                        )
                        failed_tau = tau_try
                        break
                    frontier_step_cap = _tau_path_refined_step_cap(
                        failed_gap=gap,
                        min_dtau=min_dtau,
                    )
                    continue
                path_records.append(path_record)
                details["n_path_successes"] += 1
                tau_current = float(tau_try)
                beta_current = beta_step
                beta_previous = beta_step.copy()
                linear_current = linear_step
                linear_previous = None if linear_step is None else linear_step.copy()
                rho_previous = step_diag.final_rho
                if use_local_frontier_refinement and frontier_locked:
                    success_gap = abs(tau_current - tau_from)
                    frontier_step_cap = max(
                        float(min_dtau),
                        min(
                            success_gap,
                            success_gap
                            if frontier_step_cap is None
                            else float(frontier_step_cap),
                        ),
                    )
                else:
                    frontier_step_cap = None
                if sign * (float(tau_goal) - tau_current) <= path_tol:
                    break
                continue

            path_records.append(path_record)
            details["n_path_failures"] += 1
            gap = abs(tau_try - tau_current)
            if gap <= min_dtau * (1.0 + 1e-12):
                failure_reason = (
                    step_diag.failure_reason or KERNEL_FAILURE_MIN_DTAU_REACHED
                )
                failed_tau = tau_try
                break
            if use_local_frontier_refinement and _tau_path_expensive_failed_step(
                step_diag,
                control=control,
            ):
                frontier_locked = True
                frontier_expensive_failures += 1
                if frontier_expensive_failures > TAU_PATH_FRONTIER_RETRY_BUDGET:
                    failure_reason = (
                        step_diag.failure_reason or KERNEL_FAILURE_NOT_CONVERGED
                    )
                    failed_tau = tau_try
                    break
            frontier_step_cap = _tau_path_refined_step_cap(
                failed_gap=gap,
                min_dtau=min_dtau,
            )

        if failure_reason is not None:
            break

    target_reached = failure_reason is None
    details.update(
        {
            "path_status": "target_reached"
            if target_reached
            else "partial_target_not_reached",
            "tau_attained": float(tau_current),
            "tau_failed": None if failed_tau is None else float(failed_tau),
            "tau_gap": float(abs(float(tau_target) - tau_current)),
            "failure_reason": failure_reason,
            "final_rho": None if rho_previous is None else float(rho_previous),
            "min_dtau": float(min_dtau),
            "path": tuple(path_records),
        }
    )
    return _TauPathSolve(
        beta=beta_current,
        linear=linear_current,
        tau_fit=float(tau_current),
        converged=bool(target_reached),
        iteration=_combine_tau_path_iteration(
            converged=target_reached,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
        ),
        details=details,
    )


def _solve_fullcancel_tau_path(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
) -> _TauPathSolve:
    path_control = _fullcancel_tau_path_control(control, tau_target=tau_target)
    anchor_solve = _concentrated_fullcancel(
        endog=endog,
        exog=exog,
        groups=groups,
        counts=counts,
        tau=2.0,
        control=path_control,
        beta_init=None,
        rho_hint=None,
    )
    anchor_beta = anchor_solve.beta
    anchor_linear = anchor_solve.linear
    anchor_diag = anchor_solve.iteration
    diagnostics = [anchor_diag]
    path_records: list[dict[str, Any]] = [
        {
            "tau": 2.0,
            "from_tau": None,
            "converged": bool(anchor_diag.converged),
            "beta_reduced": _as_owned_float64_vector(anchor_beta)
            if anchor_diag.converged
            else None,
            "rho": anchor_diag.final_rho,
            "iterations": int(anchor_diag.iterations),
            "restarts": int(anchor_diag.restarts),
            "anderson_m_initial": int(anchor_diag.anderson_m_initial),
            "anderson_m_final": int(anchor_diag.anderson_m_final),
            "failure_reason": anchor_diag.failure_reason,
            "beta_init_source_tau": None,
            "rho_hint": None,
            "_linear_predictor": _as_owned_float64_vector(anchor_linear)
            if anchor_diag.converged and anchor_linear is not None
            else None,
            "diagnostics": dict(anchor_diag.details),
        }
    ]
    details: dict[str, Any] = {
        "path_status": "target_reached",
        "tau_anchor": 2.0,
        "tau_target": float(tau_target),
        "tau_attained": 2.0,
        "tau_failed": None,
        "n_path_evaluations": 0,
        "n_path_successes": 0,
        "n_path_failures": 0,
        "tau_grid_policy": "quadratic_endpoint_dense_with_adaptive_bisection",
        "score_validation_policy": "diagnostic",
        "fullcancel_algorithm": "weighted_within_fullcancel",
        "fullcancel_index_policy": "slope_only_raw_exog",
        "path": (),
    }
    if not anchor_diag.converged:
        details.update(
            {
                "path_status": "anchor_failed",
                "tau_failed": 2.0,
                "failure_reason": anchor_diag.failure_reason
                or KERNEL_FAILURE_TAU2_ANCHOR_FAILED,
                "path": tuple(path_records),
            }
        )
        return _TauPathSolve(
            beta=np.full_like(anchor_beta, np.nan, dtype=float),
            linear=None,
            tau_fit=2.0,
            converged=False,
            iteration=anchor_diag,
            details=details,
        )

    if anchor_diag.final_rho is None:
        details.update(
            {
                "path_status": "anchor_failed",
                "tau_failed": 2.0,
                "failure_reason": KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO,
                "path": tuple(path_records),
            }
        )
        return _TauPathSolve(
            beta=np.full_like(anchor_beta, np.nan, dtype=float),
            linear=None,
            tau_fit=2.0,
            converged=False,
            iteration=IOLSIterationDiagnostics(
                converged=False,
                phase=anchor_diag.phase,
                iterations=anchor_diag.iterations,
                restarts=anchor_diag.restarts,
                final_rho=None,
                rho_path=anchor_diag.rho_path,
                final_criterion=anchor_diag.final_criterion,
                anderson_m_initial=anchor_diag.anderson_m_initial,
                anderson_m_final=anchor_diag.anderson_m_final,
                failure_reason=KERNEL_FAILURE_TAU2_ANCHOR_MISSING_RHO,
            ),
            details=details,
        )

    if abs(float(tau_target) - 2.0) <= 1e-10:
        details["path"] = tuple(path_records)
        return _TauPathSolve(
            beta=anchor_beta,
            linear=anchor_linear,
            tau_fit=2.0,
            converged=True,
            iteration=anchor_diag,
            details=details,
        )

    grid, sign = _tau_path_grid(
        tau_target,
        n_steps=control.tau_path_steps,
    )
    delta = abs(float(tau_target) - 2.0)
    explicit_min_dtau = None if control.min_dtau is None else float(control.min_dtau)
    min_dtau = (
        explicit_min_dtau
        if explicit_min_dtau is not None and explicit_min_dtau > 0.0
        else max(delta / max(float(control.tau_path_steps) * 128.0, 1.0), 1e-3)
    )
    path_tol = max(min_dtau * 0.1, 1e-12)
    use_local_frontier_refinement = sign < 0
    beta_current = anchor_beta
    beta_previous = anchor_beta.copy()
    linear_current = anchor_linear
    rho_previous = anchor_diag.final_rho
    tau_current = 2.0
    failure_reason: str | None = None
    failed_tau: float | None = None

    for tau_goal in grid:
        if sign * (float(tau_goal) - tau_current) <= path_tol:
            continue
        frontier_step_cap: float | None = None
        frontier_locked = False
        frontier_expensive_failures = 0
        while sign * (float(tau_goal) - tau_current) > path_tol:
            tau_try = _tau_path_next_try(
                tau_goal=float(tau_goal),
                tau_current=float(tau_current),
                sign=sign,
                step_cap=frontier_step_cap,
            )
            if details["n_path_evaluations"] >= int(control.max_path_evaluations):
                failure_reason = KERNEL_FAILURE_MAX_PATH_EVALUATIONS_REACHED
                failed_tau = tau_try
                break

            tau_from = float(tau_current)
            details["n_path_evaluations"] += 1
            step_solve = _concentrated_fullcancel(
                endog=endog,
                exog=exog,
                groups=groups,
                counts=counts,
                tau=tau_try,
                control=path_control,
                beta_init=beta_previous.copy(),
                rho_hint=rho_previous,
                score_validation_policy="diagnostic",
            )
            beta_step = step_solve.beta
            linear_step = step_solve.linear
            step_diag = step_solve.iteration
            diagnostics.append(step_diag)
            path_records.append(
                {
                    "tau": float(tau_try),
                    "from_tau": float(tau_current),
                    "step": float(abs(tau_try - tau_current)),
                    "converged": bool(step_diag.converged),
                    "beta_reduced": _as_owned_float64_vector(beta_step)
                    if step_diag.converged
                    else None,
                    "rho": step_diag.final_rho,
                    "iterations": int(step_diag.iterations),
                    "restarts": int(step_diag.restarts),
                    "anderson_m_initial": int(step_diag.anderson_m_initial),
                    "anderson_m_final": int(step_diag.anderson_m_final),
                    "failure_reason": step_diag.failure_reason,
                    "beta_init_source_tau": float(tau_current),
                    "rho_hint": None if rho_previous is None else float(rho_previous),
                    "_linear_predictor": _as_owned_float64_vector(linear_step)
                    if step_diag.converged and linear_step is not None
                    else None,
                    "diagnostics": dict(step_diag.details),
                }
            )
            if step_diag.converged:
                details["n_path_successes"] += 1
                tau_current = float(tau_try)
                beta_current = beta_step
                beta_previous = beta_step.copy()
                linear_current = linear_step
                rho_previous = step_diag.final_rho
                if use_local_frontier_refinement and frontier_locked:
                    success_gap = abs(tau_current - tau_from)
                    frontier_step_cap = max(
                        float(min_dtau),
                        min(
                            success_gap,
                            success_gap
                            if frontier_step_cap is None
                            else float(frontier_step_cap),
                        ),
                    )
                else:
                    frontier_step_cap = None
                if sign * (float(tau_goal) - tau_current) <= path_tol:
                    break
                continue

            details["n_path_failures"] += 1
            gap = abs(tau_try - tau_current)
            if gap <= min_dtau * (1.0 + 1e-12):
                failure_reason = (
                    step_diag.failure_reason or KERNEL_FAILURE_MIN_DTAU_REACHED
                )
                failed_tau = tau_try
                break
            if use_local_frontier_refinement and _tau_path_expensive_failed_step(
                step_diag,
                control=control,
            ):
                frontier_locked = True
                frontier_expensive_failures += 1
                if frontier_expensive_failures > TAU_PATH_FRONTIER_RETRY_BUDGET:
                    failure_reason = (
                        step_diag.failure_reason or KERNEL_FAILURE_NOT_CONVERGED
                    )
                    failed_tau = tau_try
                    break
            elif _tau_path_expensive_failed_step(
                step_diag,
                control=control,
            ):
                failure_reason = (
                    step_diag.failure_reason or KERNEL_FAILURE_NOT_CONVERGED
                )
                failed_tau = tau_try
                break
            frontier_step_cap = _tau_path_refined_step_cap(
                failed_gap=gap,
                min_dtau=min_dtau,
            )

        if failure_reason is not None:
            break

    target_reached = failure_reason is None
    details.update(
        {
            "path_status": "target_reached"
            if target_reached
            else "partial_target_not_reached",
            "tau_attained": float(tau_current),
            "tau_failed": None if failed_tau is None else float(failed_tau),
            "tau_gap": float(abs(float(tau_target) - tau_current)),
            "failure_reason": failure_reason,
            "final_rho": None if rho_previous is None else float(rho_previous),
            "min_dtau": float(min_dtau),
            "path": tuple(path_records),
        }
    )
    return _TauPathSolve(
        beta=beta_current,
        linear=linear_current,
        tau_fit=float(tau_current),
        converged=bool(target_reached),
        iteration=_combine_tau_path_iteration(
            converged=target_reached,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
        ),
        details=details,
    )


def _merge_configured_tau_paths(
    path_solves: tuple[_TauPathSolve, ...],
    tau_targets: tuple[float, ...],
) -> _TauPathSolve:
    if not path_solves:
        raise ValueError("At least one tau path solve is required")

    first = path_solves[0]
    merged_records: list[dict[str, Any]] = []
    seen_converged_tau: set[float] = set()
    target_details: list[dict[str, Any]] = []
    for tau_target, path_solve in zip(tau_targets, path_solves, strict=True):
        details = path_solve.details
        target_details.append(
            {
                "tau_target": float(tau_target),
                "path_status": details.get("path_status"),
                "tau_attained": details.get("tau_attained"),
                "tau_failed": details.get("tau_failed"),
                "failure_reason": details.get("failure_reason"),
                "n_path_evaluations": details.get("n_path_evaluations", 0),
                "n_path_successes": details.get("n_path_successes", 0),
                "n_path_failures": details.get("n_path_failures", 0),
            }
        )
        for record in tuple(details.get("path", ())):
            item = dict(record)
            item["path_target"] = float(tau_target)
            tau_value = float(item.get("tau", float("nan")))
            if bool(item.get("converged", False)):
                key = round(tau_value, 12)
                if key in seen_converged_tau:
                    continue
                seen_converged_tau.add(key)
            merged_records.append(item)

    first_path = tuple(first.details.get("path", ()))
    anchor_beta_raw = None
    anchor_linear_raw = None
    if first_path:
        anchor_beta_raw = first_path[0].get("beta_reduced")
        anchor_linear_raw = first_path[0].get("_linear_predictor")
    if anchor_beta_raw is None:
        primary_beta = first.beta
    else:
        primary_beta = _as_owned_float64_vector(anchor_beta_raw)
    primary_linear = (
        first.linear
        if anchor_linear_raw is None
        else _as_owned_float64_vector(anchor_linear_raw)
    )

    all_converged = all(path_solve.converged for path_solve in path_solves)
    failure_reasons = tuple(
        str(path_solve.details.get("failure_reason"))
        for path_solve in path_solves
        if path_solve.details.get("failure_reason") is not None
    )
    rho_path: list[float] = []
    for path_solve in path_solves:
        rho_path.extend(float(value) for value in path_solve.iteration.rho_path)
    final_iteration = path_solves[-1].iteration
    iteration = IOLSIterationDiagnostics(
        converged=all_converged,
        phase="tau_path_configured",
        iterations=sum(
            int(path_solve.iteration.iterations) for path_solve in path_solves
        ),
        restarts=sum(int(path_solve.iteration.restarts) for path_solve in path_solves),
        final_rho=final_iteration.final_rho,
        rho_path=tuple(rho_path),
        final_criterion=final_iteration.final_criterion,
        anderson_m_initial=first.iteration.anderson_m_initial,
        anderson_m_final=final_iteration.anderson_m_final,
        failure_reason=None
        if all_converged
        else (
            failure_reasons[0]
            if failure_reasons
            else KERNEL_FAILURE_CONFIGURED_PATH_PARTIAL
        ),
    )
    n_path_evaluations = sum(
        int(path_solve.details.get("n_path_evaluations", 0))
        for path_solve in path_solves
    )
    n_path_successes = sum(
        int(path_solve.details.get("n_path_successes", 0)) for path_solve in path_solves
    )
    n_path_failures = sum(
        int(path_solve.details.get("n_path_failures", 0)) for path_solve in path_solves
    )
    details = {
        "path_status": "path_complete"
        if all_converged
        else "partial_configured_path_not_reached",
        "path_mode": "configured_targets",
        "tau_anchor": 2.0,
        "tau_target": None,
        "tau_targets": tuple(float(value) for value in tau_targets),
        "tau_attained": tuple(
            float(path_solve.details.get("tau_attained", path_solve.tau_fit))
            for path_solve in path_solves
        ),
        "tau_failed": tuple(
            path_solve.details.get("tau_failed") for path_solve in path_solves
        ),
        "failure_reason": None
        if all_converged
        else (
            failure_reasons[0]
            if failure_reasons
            else KERNEL_FAILURE_CONFIGURED_PATH_PARTIAL
        ),
        "target_details": tuple(target_details),
        "n_path_evaluations": int(n_path_evaluations),
        "n_path_successes": int(n_path_successes),
        "n_path_failures": int(n_path_failures),
        "tau_grid_policy": "configured_targets_with_quadratic_endpoint_dense_paths",
        "path": tuple(merged_records),
    }
    for key, value in first.details.items():
        if key.startswith("mixed_") or key in {
            "fullcancel_algorithm",
            "fullcancel_index_policy",
            "fullcancel_fe_policy",
        }:
            details.setdefault(key, value)
    return _TauPathSolve(
        beta=primary_beta,
        linear=primary_linear,
        tau_fit=2.0,
        converged=all_converged,
        iteration=iteration,
        details=details,
    )


def _solve_fullrecover_configured_tau_path(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_targets: tuple[float, ...],
    control: IOLSKernelControl,
) -> _TauPathSolve:
    if not tau_targets:
        raise ValueError("tau_path_targets must be non-empty when tau_target is None")
    state = _fullrecover_invariant_state(
        exog=exog,
        groups=groups,
        counts=counts,
        control=control,
    )
    path_solves = tuple(
        _solve_fullrecover_tau_path(
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
            tau_target=float(tau_target),
            control=control,
            invariant_state=state,
        )
        for tau_target in tau_targets
    )
    return _merge_configured_tau_paths(path_solves, tau_targets)


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "cpp"}


def _fixed_tau_speed_target(tau_target: float | None) -> bool:
    return tau_target in {1.0, 2.0}


def _speed_fixed_tau_control(
    control: IOLSKernelControl,
    *,
    tau_target: float | None,
) -> tuple[IOLSKernelControl, dict[str, Any]]:
    if (
        not _fixed_tau_speed_target(tau_target)
        or not _truthy_env(os.environ.get(_FIXED_TAU_FAST_CONTROL_FLAG))
    ):
        return control, {}
    updated = replace(
        control,
        delta_grid_size=0,
        rho_grid_size=2,
        tau_path_steps=1,
        transformed_design_normalization="none",
    )
    return updated, {
        "speed_fixed_tau_fast_control": True,
        "speed_fixed_tau_fast_control_flag": _FIXED_TAU_FAST_CONTROL_FLAG,
        "speed_fixed_tau_fast_control_policy": (
            "fixed_tau_restricted_grid_skip_transformed_scale"
        ),
        "speed_fixed_tau_fast_control_tau": float(tau_target),
        "speed_fixed_tau_fast_control_delta_grid_size": int(
            updated.delta_grid_size,
        ),
        "speed_fixed_tau_fast_control_rho_grid_size": int(updated.rho_grid_size),
        "speed_fixed_tau_fast_control_tau_path_steps": int(
            updated.tau_path_steps,
        ),
        "speed_fixed_tau_fast_control_transformed_design_normalization": (
            updated.transformed_design_normalization
        ),
    }


def _speed_compact_fixed_tau_details_enabled(tau_target: float | None) -> bool:
    return _fixed_tau_speed_target(tau_target) and _truthy_env(
        os.environ.get(_COMPACT_FIXED_TAU_DETAILS_FLAG),
    )


def _speed_skip_fullrecover_boundary_check_enabled(
    tau_target: float | None,
) -> bool:
    return _fixed_tau_speed_target(tau_target) and _truthy_env(
        os.environ.get(_SKIP_FULLRECOVER_BOUNDARY_CHECK_FLAG),
    )


def _fullrecover_boundary_support_details(
    *,
    topology: KernelTopology,
    tau_target: float | None,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if topology != "fullrecover":
        return None, {}
    if _speed_skip_fullrecover_boundary_check_enabled(tau_target):
        return None, {
            "speed_fullrecover_boundary_check_skipped": True,
            "speed_fullrecover_boundary_check_flag": (
                _SKIP_FULLRECOVER_BOUNDARY_CHECK_FLAG
            ),
            "speed_fullrecover_boundary_check_policy": (
                "skip_tau2_boundary_lp_diagnostic_for_fixed_tau_speed_research"
            ),
        }
    return (
        _recovered_fe_tau2_boundary_separation(
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
        ),
        {},
    )


def _compact_fixed_tau_path_details(
    *,
    raw_details: Mapping[str, Any],
    tau_target: float | None,
    tau_fit: float,
    result_converged: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    status = raw_details.get("path_status")
    if status is None:
        status = "target_reached" if result_converged else "target_failed"
    details: dict[str, Any] = {
        "path_status": status,
        "path_mode": raw_details.get("path_mode", "fixed_tau"),
        "tau_target": tau_target,
        "tau_attained": float(tau_fit),
        "failure_reason": failure_reason,
        "n_path_evaluations": int(raw_details.get("n_path_evaluations", 0) or 0),
        "n_path_successes": int(raw_details.get("n_path_successes", 0) or 0),
        "n_path_failures": int(raw_details.get("n_path_failures", 0) or 0),
        "path": (),
        "path_coefficient_status": "not_materialized_speed_compact_fixed_tau",
        "path_inference_status": "not_computed",
        "path_inference_reason": "speed_compact_fixed_tau_details",
        "speed_compact_fixed_tau_details": True,
        "speed_compact_fixed_tau_details_flag": _COMPACT_FIXED_TAU_DETAILS_FLAG,
    }
    if "tau_anchor" in raw_details:
        details["tau_anchor"] = raw_details["tau_anchor"]
    if raw_details.get("speed_accept_score_warning_path_steps") is not None:
        details["speed_accept_score_warning_path_steps"] = bool(
            raw_details.get("speed_accept_score_warning_path_steps"),
        )
    return details


def _cpp_kernel_policy() -> str:
    if _truthy_env(os.environ.get("PYIOLS_DISABLE_CPP")):
        return "disabled_by_env"
    if _truthy_env(os.environ.get("PYIOLS_USE_CPP")):
        return "default_on_explicit_env"
    return "default_on"


def _cpp_kernel_allowed() -> bool:
    return _cpp_kernel_policy() != "disabled_by_env"


def _load_cpp_kernel_module():
    try:
        from historical_iols import _kernel_cpp  # type: ignore[attr-defined]
    except ImportError:
        return None
    return _kernel_cpp


def _cpp_kernel_fallback_reason(
    required_symbols: tuple[str, ...],
) -> str | None:
    if not _cpp_kernel_allowed():
        return "disabled_by_env"
    cpp_kernel = _load_cpp_kernel_module()
    if cpp_kernel is None:
        return "extension_unavailable"
    missing = tuple(
        symbol for symbol in required_symbols if not hasattr(cpp_kernel, symbol)
    )
    if missing:
        return "missing_symbol:" + ",".join(missing)
    return None


def _cpp_solver_metadata(
    *,
    policy: str,
    status: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "cpp_backend_policy": policy,
        "cpp_backend_status": status,
        "cpp_backend_fallback_reason": fallback_reason,
        "cpp_backend_used": fallback_reason is None and status.startswith("used_"),
    }


def _control_to_cpp_mapping(control: IOLSKernelControl) -> dict[str, Any]:
    return {
        "anderson_m": int(control.anderson_m),
        "eps": float(control.eps),
        "stopcrit": int(control.stopcrit),
        "tau_path_steps": int(control.tau_path_steps),
        "min_dtau": None if control.min_dtau is None else float(control.min_dtau),
        "max_path_evaluations": int(control.max_path_evaluations),
        "max_phase1_iterations": int(control.max_phase1_iterations),
        "max_phase2_evaluations": int(control.max_phase2_evaluations),
        "fe_demean_maxiter": int(control.fe_demean_maxiter),
        "fe_demean_tol": float(control.fe_demean_tol),
        "delta_log_min": float(control.delta_log_min),
        "delta_log_max": float(control.delta_log_max),
        "delta_grid_size": int(control.delta_grid_size),
        "rho_log_min": float(control.rho_log_min),
        "rho_log_max": float(control.rho_log_max),
        "rho_grid_size": int(control.rho_grid_size),
        # runtime_v3: read only by the C++ direct no-FE fixed-tau route.
        "direct_restart_start": str(control.direct_restart_start),
    }


def _iteration_from_cpp_mapping(raw: Mapping[str, Any]) -> IOLSIterationDiagnostics:
    final_rho_raw = raw.get("final_rho")
    final_criterion_raw = raw.get("final_criterion")
    failure_reason_raw = raw.get("failure_reason")
    return IOLSIterationDiagnostics(
        converged=bool(raw.get("converged", False)),
        phase=str(raw.get("phase", "not_started")),
        iterations=int(raw.get("iterations", 0)),
        restarts=int(raw.get("restarts", 0)),
        final_rho=None if final_rho_raw is None else float(final_rho_raw),
        rho_path=tuple(float(value) for value in raw.get("rho_path", ())),
        final_criterion=None
        if final_criterion_raw is None
        else float(final_criterion_raw),
        anderson_m_initial=int(raw.get("anderson_m_initial", 0)),
        anderson_m_final=int(raw.get("anderson_m_final", 0)),
        failure_reason=None if failure_reason_raw is None else str(failure_reason_raw),
        details=dict(raw.get("details", {})),
    )


def _profiled_fullrecover_cpp_supported(
    *,
    spec: IOLSKernelSpec,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_target: float | None,
) -> bool:
    if tau_target is None:
        return False
    return (
        _fullrecover_cpp_fallback_reason(
            spec=spec,
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
        )
        is None
    )


def _fe_profile_controls_override(control: IOLSKernelControl) -> bool:
    return int(control.fe_profile_maxiter) != 10000 or not np.isclose(
        float(control.fe_profile_tol),
        1e-6,
        rtol=0.0,
        atol=0.0,
    )


def _profiled_fullrecover_cpp_fallback_reason(
    *,
    spec: IOLSKernelSpec,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_target: float | None,
) -> str | None:
    if tau_target is None:
        return "fixed_tau_target_required"
    return _fullrecover_cpp_fallback_reason(
        spec=spec,
        endog=endog,
        exog=exog,
        groups=groups,
        counts=counts,
    )


def _fullrecover_cpp_supported(
    *,
    spec: IOLSKernelSpec,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> bool:
    return (
        _fullrecover_cpp_fallback_reason(
            spec=spec,
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
        )
        is None
    )


def _fullrecover_cpp_fallback_reason(
    *,
    spec: IOLSKernelSpec,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> str | None:
    if spec.topology not in {"noFE", "fullrecover"}:
        return "unsupported_topology"
    if not groups or len(groups) != len(counts):
        return "invalid_fe_groups"
    if _uses_joint_hdfe_state(groups) and _fe_profile_controls_override(spec.control):
        return "fe_profile_controls_python_reference_only"
    required_symbol = (
        "solve_fullrecover_tau_path_manyway"
        if _uses_joint_hdfe_state(groups)
        else "solve_fullrecover_tau_path"
    )
    capability_reason = _cpp_kernel_fallback_reason((required_symbol,))
    if capability_reason is not None:
        return capability_reason
    return None


def _nofe_cpp_fallback_reason() -> str | None:
    return _cpp_kernel_fallback_reason(("solve_nofe_tau_path",))


def _nofe_cpp_backend_strategy() -> tuple[str | None, str | None]:
    fallback_reason = _nofe_cpp_fallback_reason()
    if fallback_reason is None:
        return "dedicated_nofe", None
    if fallback_reason == "missing_symbol:solve_nofe_tau_path":
        legacy_reason = _cpp_kernel_fallback_reason(("solve_fullrecover_tau_path",))
        if legacy_reason is None:
            return "legacy_profiled_fullrecover", None
    return None, fallback_reason


def _with_nofe_details(path_solve: _TauPathSolve) -> _TauPathSolve:
    details = dict(path_solve.details)
    details.setdefault("nofe_algorithm", "singleton_intercept_profiled_iols")
    details.setdefault(
        "nofe_implicit_intercept_policy",
        "profiled_singleton_intercept",
    )
    return _TauPathSolve(
        beta=path_solve.beta,
        linear=path_solve.linear,
        tau_fit=path_solve.tau_fit,
        converged=path_solve.converged,
        iteration=path_solve.iteration,
        details=details,
    )


def _fullrecover_boundary_separation_taupath_solve(
    *,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    tau_target: float,
    control: IOLSKernelControl,
    support_details: Mapping[str, Any],
) -> _TauPathSolve:
    diagnostics = dict(support_details)
    diagnostics.update(
        {
            "phase1_attempts": 0,
            "phase1_accepted": 0,
            "phase1_rejected": 0,
            "rho_candidate_count": 0,
            "fe_state_policy": _fullrecover_fe_policy(groups),
            "anderson_policy": "not_started_boundary_support_failure",
            "failure_reason": TAU2_BOUNDARY_SEPARATION_CODE,
        }
    )
    iteration = IOLSIterationDiagnostics(
        converged=False,
        phase="support_check",
        iterations=0,
        restarts=0,
        final_rho=None,
        rho_path=(),
        final_criterion=None,
        anderson_m_initial=int(control.anderson_m),
        anderson_m_final=0,
        failure_reason=TAU2_BOUNDARY_SEPARATION_CODE,
        details=diagnostics,
    )
    path_record = {
        "tau": 2.0,
        "from_tau": None,
        "converged": False,
        "beta_reduced": None,
        "rho": None,
        "iterations": 0,
        "restarts": 0,
        "anderson_m_initial": int(control.anderson_m),
        "anderson_m_final": 0,
        "failure_reason": TAU2_BOUNDARY_SEPARATION_CODE,
        "beta_init_source_tau": None,
        "rho_hint": None,
        "_linear_predictor": None,
        "diagnostics": diagnostics,
    }
    details = {
        "path_status": "anchor_failed",
        "tau_anchor": 2.0,
        "tau_target": float(tau_target),
        "tau_attained": 2.0,
        "tau_failed": 2.0,
        "failure_reason": TAU2_BOUNDARY_SEPARATION_CODE,
        "n_path_evaluations": 0,
        "n_path_successes": 0,
        "n_path_failures": 0,
        "tau_grid_policy": "quadratic_endpoint_dense_with_adaptive_bisection",
        "score_validation_policy": "diagnostic",
        "path": (path_record,),
    }
    return _TauPathSolve(
        beta=np.full(exog.shape[1], np.nan, dtype=float),
        linear=None,
        tau_fit=2.0,
        converged=False,
        iteration=iteration,
        details=details,
    )


def _solve_profiled_fullrecover_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
    boundary_support_details: Mapping[str, Any] | None = None,
) -> _TauPathSolve:
    if boundary_support_details is not None and bool(
        boundary_support_details.get("separation_detected", False)
    ):
        return _fullrecover_boundary_separation_taupath_solve(
            exog=exog,
            groups=groups,
            tau_target=tau_target,
            control=control,
            support_details=boundary_support_details,
        )
    cpp_kernel = _load_cpp_kernel_module()
    if cpp_kernel is None:
        raise ImportError("pyiols._kernel_cpp is not importable")
    if _uses_joint_hdfe_state(groups):
        raw = cpp_kernel.solve_fullrecover_tau_path_manyway(
            np.ascontiguousarray(endog, dtype=np.float64),
            np.ascontiguousarray(exog, dtype=np.float64),
            float(tau_target),
            groups=tuple(np.asarray(group, dtype=np.int64) for group in groups),
            n_groups=tuple(int(count) for count in counts),
            control=_control_to_cpp_mapping(control),
        )
    else:
        group = np.asarray(groups[0], dtype=np.int64)
        raw = cpp_kernel.solve_fullrecover_tau_path(
            np.ascontiguousarray(endog, dtype=np.float64),
            np.ascontiguousarray(exog, dtype=np.float64),
            float(tau_target),
            group=group,
            n_groups=int(counts[0]),
            control=_control_to_cpp_mapping(control),
        )
    iteration = _iteration_from_cpp_mapping(raw["iteration"])
    details = dict(raw["details"])
    return _with_nofe_details(
        _TauPathSolve(
            beta=np.asarray(raw["beta"], dtype=float),
            linear=None
            if raw["linear"] is None
            else np.asarray(raw["linear"], dtype=float),
            tau_fit=float(raw["tau_fit"]),
            converged=bool(raw["converged"]),
            iteration=iteration,
            details=details,
        )
    )


def _solve_nofe_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    tau_target: float,
    control: IOLSKernelControl,
) -> _TauPathSolve:
    cpp_kernel = _load_cpp_kernel_module()
    if cpp_kernel is None:
        raise ImportError("pyiols._kernel_cpp is not importable")
    raw = cpp_kernel.solve_nofe_tau_path(
        np.ascontiguousarray(endog, dtype=np.float64),
        np.ascontiguousarray(exog, dtype=np.float64),
        float(tau_target),
        control=_control_to_cpp_mapping(control),
    )
    iteration = _iteration_from_cpp_mapping(raw["iteration"])
    details = dict(raw["details"])
    return _TauPathSolve(
        beta=np.asarray(raw["beta"], dtype=float),
        linear=None
        if raw["linear"] is None
        else np.asarray(raw["linear"], dtype=float),
        tau_fit=float(raw["tau_fit"]),
        converged=bool(raw["converged"]),
        iteration=iteration,
        details=details,
    )


def _solve_nofe_configured_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    tau_targets: tuple[float, ...],
    control: IOLSKernelControl,
) -> _TauPathSolve:
    if not tau_targets:
        raise ValueError("tau_path_targets must be non-empty when tau_target is None")
    path_solves = tuple(
        _solve_nofe_tau_path_cpp(
            endog=endog,
            exog=exog,
            tau_target=float(tau_target),
            control=control,
        )
        for tau_target in tau_targets
    )
    return _with_nofe_details(_merge_configured_tau_paths(path_solves, tau_targets))


def _solve_fullrecover_configured_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_targets: tuple[float, ...],
    control: IOLSKernelControl,
    boundary_support_details: Mapping[str, Any] | None = None,
) -> _TauPathSolve:
    if not tau_targets:
        raise ValueError("tau_path_targets must be non-empty when tau_target is None")
    path_solves = tuple(
        _solve_profiled_fullrecover_tau_path_cpp(
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
            tau_target=float(tau_target),
            control=control,
            boundary_support_details=boundary_support_details,
        )
        for tau_target in tau_targets
    )
    return _merge_configured_tau_paths(path_solves, tau_targets)


def _fullcancel_cpp_supported(
    *,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> bool:
    return _fullcancel_cpp_fallback_reason(groups=groups, counts=counts) is None


def _fullcancel_cpp_fallback_reason(
    *,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
) -> str | None:
    if not groups or len(groups) != len(counts):
        return "invalid_fe_groups"
    return _cpp_kernel_fallback_reason(("solve_fullcancel_tau_path",))


def _mixed_cpp_fallback_reason(
    *,
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
) -> str | None:
    if not cancel_groups or len(cancel_groups) != len(cancel_counts):
        return "invalid_cancel_fe_groups"
    if not recover_groups or len(recover_groups) != len(recover_counts):
        return "invalid_recover_fe_groups"
    return _cpp_kernel_fallback_reason(
        (
            "mixed_design_matrix",
            "mixed_support_conditioning_preflight",
            "solve_mixed_fullcancel_tau_path",
        )
    )


def _solve_fullcancel_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
) -> _TauPathSolve:
    cpp_kernel = _load_cpp_kernel_module()
    if cpp_kernel is None:
        raise ImportError("pyiols._kernel_cpp is not importable")
    groups_cpp = tuple(np.asarray(group, dtype=np.int64) for group in groups)
    raw = cpp_kernel.solve_fullcancel_tau_path(
        np.ascontiguousarray(endog, dtype=np.float64),
        np.ascontiguousarray(exog, dtype=np.float64),
        float(tau_target),
        groups=groups_cpp,
        n_groups=tuple(int(count) for count in counts),
        control=_control_to_cpp_mapping(control),
    )
    iteration = _iteration_from_cpp_mapping(raw["iteration"])
    details = dict(raw["details"])
    return _TauPathSolve(
        beta=np.asarray(raw["beta"], dtype=float),
        linear=None
        if raw["linear"] is None
        else np.asarray(raw["linear"], dtype=float),
        tau_fit=float(raw["tau_fit"]),
        converged=bool(raw["converged"]),
        iteration=iteration,
        details=details,
    )


def _solve_mixed_fullcancel_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    cancel_groups: tuple[np.ndarray, ...],
    cancel_counts: tuple[int, ...],
    recover_groups: tuple[np.ndarray, ...],
    recover_counts: tuple[int, ...],
    tau_target: float,
    control: IOLSKernelControl,
) -> _TauPathSolve:
    cpp_kernel = _load_cpp_kernel_module()
    if cpp_kernel is None:
        raise ImportError("pyiols._kernel_cpp is not importable")
    if not hasattr(cpp_kernel, "solve_mixed_fullcancel_tau_path"):
        mixed_design, _ = _mixed_design_matrix_cpp(
            exog=exog,
            recover_groups=recover_groups,
            recover_counts=recover_counts,
            recover_names=tuple(
                f"recover{index}" for index in range(len(recover_groups))
            ),
        )
        return _solve_fullcancel_tau_path_cpp(
            endog=endog,
            exog=mixed_design,
            groups=cancel_groups,
            counts=cancel_counts,
            tau_target=tau_target,
            control=control,
        )
    raw = cpp_kernel.solve_mixed_fullcancel_tau_path(
        np.ascontiguousarray(endog, dtype=np.float64),
        np.ascontiguousarray(exog, dtype=np.float64),
        float(tau_target),
        cancel_groups=tuple(
            np.asarray(group, dtype=np.int64) for group in cancel_groups
        ),
        cancel_counts=tuple(int(count) for count in cancel_counts),
        recover_groups=tuple(
            np.asarray(group, dtype=np.int64) for group in recover_groups
        ),
        recover_counts=tuple(int(count) for count in recover_counts),
        control=_control_to_cpp_mapping(control),
    )
    iteration = _iteration_from_cpp_mapping(raw["iteration"])
    details = dict(raw["details"])
    return _TauPathSolve(
        beta=np.asarray(raw["beta"], dtype=float),
        linear=None
        if raw["linear"] is None
        else np.asarray(raw["linear"], dtype=float),
        tau_fit=float(raw["tau_fit"]),
        converged=bool(raw["converged"]),
        iteration=iteration,
        details=details,
    )


def _solve_fullcancel_configured_tau_path(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_targets: tuple[float, ...],
    control: IOLSKernelControl,
) -> _TauPathSolve:
    if not tau_targets:
        raise ValueError("tau_path_targets must be non-empty when tau_target is None")
    path_solves = tuple(
        _solve_fullcancel_tau_path(
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
            tau_target=float(tau_target),
            control=control,
        )
        for tau_target in tau_targets
    )
    return _merge_configured_tau_paths(path_solves, tau_targets)


def _solve_fullcancel_configured_tau_path_cpp(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    tau_targets: tuple[float, ...],
    control: IOLSKernelControl,
) -> _TauPathSolve:
    if not tau_targets:
        raise ValueError("tau_path_targets must be non-empty when tau_target is None")
    path_solves = tuple(
        _solve_fullcancel_tau_path_cpp(
            endog=endog,
            exog=exog,
            groups=groups,
            counts=counts,
            tau_target=float(tau_target),
            control=control,
        )
        for tau_target in tau_targets
    )
    return _merge_configured_tau_paths(path_solves, tau_targets)


def _kernel_groups_from_spec(
    spec: IOLSKernelSpec,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[int, ...]]:
    if spec.topology == "noFE":
        n_obs = np.asarray(spec.endog).shape[0]
        return ("__intercept__",), (np.zeros(n_obs, dtype=np.int32),), (1,)
    if spec.topology == "fullrecover":
        if spec.cancel_fe_blocks:
            raise ValueError("fullrecover topology cannot receive cancel_fe_blocks")
        if spec.recover_fe_counts:
            return _encoded_groups_from_blocks(
                spec.recover_fe_blocks,
                spec.recover_fe_counts,
            )
        return _remap_groups(spec.recover_fe_blocks)
    if spec.topology == "fullcancel":
        if spec.recover_fe_blocks:
            raise ValueError("fullcancel topology cannot receive recover_fe_blocks")
        names, groups, counts = _remap_groups(spec.cancel_fe_blocks)
        if not groups:
            raise ValueError(
                "fullcancel topology requires at least one cancel FE block"
            )
        return names, groups, counts
    if spec.topology == "mixed":
        raise NotImplementedError(
            "P0-S2c mixed support must combine fullcancel and fullrecover "
            "blocks in the same package-owned kernel contract."
        )
    raise ValueError(f"Unsupported kernel topology: {spec.topology!r}")


def _mixed_groups_from_spec(
    spec: IOLSKernelSpec,
) -> tuple[
    tuple[str, ...],
    tuple[np.ndarray, ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[np.ndarray, ...],
    tuple[int, ...],
]:
    if not spec.cancel_fe_blocks:
        raise ValueError("mixed topology requires at least one cancel FE block")
    if not spec.recover_fe_blocks:
        raise ValueError("mixed topology requires at least one recover FE block")
    overlap = set(spec.cancel_fe_blocks).intersection(spec.recover_fe_blocks)
    if overlap:
        names = ", ".join(sorted(str(name) for name in overlap))
        raise ValueError(
            f"mixed FE blocks cannot be both cancelled and recovered: {names}"
        )
    cancel_names, cancel_groups, cancel_counts = _remap_groups(spec.cancel_fe_blocks)
    recover_names, recover_groups, recover_counts = _remap_groups(
        spec.recover_fe_blocks
    )
    return (
        cancel_names,
        cancel_groups,
        cancel_counts,
        recover_names,
        recover_groups,
        recover_counts,
    )


def solve_iols_kernel(spec: IOLSKernelSpec) -> IOLSKernelResult:
    """Internal iOLS tau-path kernel. Not exported to the public API.

    The production algorithm solves the tau=2 iOLS Phase 1 + Phase 2 anchor,
    then continues to any non-anchor target through warm-started Phase 2 path
    steps. GPML/Newton/IRLS comparator solvers are not valid production
    anchors for this iOLS kernel.
    """

    endog = _as_float_vector(spec.endog, name="endog")
    exog_full = _as_float_matrix(spec.exog, name="exog")
    if exog_full.shape[0] != endog.shape[0]:
        raise ValueError("endog and exog must have the same number of rows")
    tau_target = None if spec.tau_target is None else float(spec.tau_target)
    tau_values_to_validate = (
        tuple(float(value) for value in spec.tau_path_targets)
        if tau_target is None
        else (tau_target,)
    )
    for tau_value in tau_values_to_validate:
        if tau_value < 0.0 or tau_value > 3.0:
            raise ValueError("tau values must stay in the public domain [0, 3]")

    exog, has_intercept, _ = _split_intercept(exog_full, None)
    exog_canonical = exog
    if spec.topology == "mixed":
        (
            cancel_names,
            cancel_groups,
            cancel_counts,
            recover_names,
            recover_groups,
            recover_counts,
        ) = _mixed_groups_from_spec(spec)
        fe_names = tuple(f"cancel:{name}" for name in cancel_names) + tuple(
            f"recover:{name}" for name in recover_names
        )
        groups = cancel_groups
        counts = cancel_counts
    else:
        fe_names, groups, counts = _kernel_groups_from_spec(spec)

    effective_control, speed_control_details = _speed_fixed_tau_control(
        spec.control,
        tau_target=tau_target,
    )
    normalization_state = _build_transformed_normalization_state(
        topology=spec.topology,
        exog=exog_canonical,
        groups=groups,
        counts=counts,
        control=effective_control,
    )
    exog = _apply_transformed_normalization(exog_canonical, normalization_state)

    kernel_backend = KERNEL_BACKEND_PYTHON_REFERENCE
    cpp_backend_policy = _cpp_kernel_policy()
    cpp_backend_status = "fallback_python_reference"
    cpp_backend_fallback_reason: str | None = None
    boundary_speed_details: dict[str, Any] = {}

    if tau_target is None:
        if spec.topology == "noFE":
            nofe_cpp_strategy, cpp_backend_fallback_reason = (
                _nofe_cpp_backend_strategy()
            )
            if nofe_cpp_strategy == "dedicated_nofe":
                path_solve = _solve_nofe_configured_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    tau_targets=tuple(float(value) for value in spec.tau_path_targets),
                    control=effective_control,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_nofe"
            elif nofe_cpp_strategy == "legacy_profiled_fullrecover":
                path_solve = _with_nofe_details(
                    _solve_fullrecover_configured_tau_path_cpp(
                        endog=endog,
                        exog=exog,
                        groups=groups,
                        counts=counts,
                        tau_targets=tuple(
                            float(value) for value in spec.tau_path_targets
                        ),
                        control=effective_control,
                    )
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_profiled_fullrecover_legacy_nofe"
            else:
                path_solve = _with_nofe_details(
                    _solve_fullrecover_configured_tau_path(
                        endog=endog,
                        exog=exog,
                        groups=groups,
                        counts=counts,
                        tau_targets=tuple(
                            float(value) for value in spec.tau_path_targets
                        ),
                        control=effective_control,
                    )
                )
        elif spec.topology == "mixed":
            cpp_backend_fallback_reason = _mixed_cpp_fallback_reason(
                cancel_groups=cancel_groups,
                cancel_counts=cancel_counts,
                recover_groups=recover_groups,
                recover_counts=recover_counts,
            )
            if cpp_backend_fallback_reason is None:
                path_solve = _solve_mixed_configured_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    cancel_names=cancel_names,
                    cancel_groups=cancel_groups,
                    cancel_counts=cancel_counts,
                    recover_names=recover_names,
                    recover_groups=recover_groups,
                    recover_counts=recover_counts,
                    tau_targets=tuple(float(value) for value in spec.tau_path_targets),
                    control=effective_control,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_mixed"
            else:
                path_solve = _solve_mixed_configured_tau_path(
                    endog=endog,
                    exog=exog,
                    cancel_names=cancel_names,
                    cancel_groups=cancel_groups,
                    cancel_counts=cancel_counts,
                    recover_names=recover_names,
                    recover_groups=recover_groups,
                    recover_counts=recover_counts,
                    tau_targets=tuple(float(value) for value in spec.tau_path_targets),
                    control=effective_control,
                )
        elif spec.topology == "fullcancel":
            cpp_backend_fallback_reason = _fullcancel_cpp_fallback_reason(
                groups=groups,
                counts=counts,
            )
            if cpp_backend_fallback_reason is None:
                path_solve = _solve_fullcancel_configured_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_targets=tuple(float(value) for value in spec.tau_path_targets),
                    control=effective_control,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_fullcancel"
            else:
                path_solve = _solve_fullcancel_configured_tau_path(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_targets=tuple(float(value) for value in spec.tau_path_targets),
                    control=effective_control,
                )
        else:
            cpp_backend_fallback_reason = _fullrecover_cpp_fallback_reason(
                spec=spec,
                endog=endog,
                exog=exog,
                groups=groups,
                counts=counts,
            )
            boundary_support_details, boundary_speed_details = (
                _fullrecover_boundary_support_details(
                    topology=spec.topology,
                    tau_target=tau_target,
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                )
            )
            if cpp_backend_fallback_reason is None:
                path_solve = _solve_fullrecover_configured_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_targets=tuple(float(value) for value in spec.tau_path_targets),
                    control=effective_control,
                    boundary_support_details=boundary_support_details,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = (
                    "used_manyway_fullrecover"
                    if _uses_joint_hdfe_state(groups)
                    else "used_profiled_fullrecover"
                )
            else:
                path_solve = _solve_fullrecover_configured_tau_path(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_targets=tuple(float(value) for value in spec.tau_path_targets),
                    control=effective_control,
                )
    else:
        if spec.topology == "noFE":
            nofe_cpp_strategy, cpp_backend_fallback_reason = (
                _nofe_cpp_backend_strategy()
            )
            if nofe_cpp_strategy == "dedicated_nofe":
                path_solve = _solve_nofe_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    tau_target=tau_target,
                    control=effective_control,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_nofe"
            elif nofe_cpp_strategy == "legacy_profiled_fullrecover":
                path_solve = _with_nofe_details(
                    _solve_profiled_fullrecover_tau_path_cpp(
                        endog=endog,
                        exog=exog,
                        groups=groups,
                        counts=counts,
                        tau_target=tau_target,
                        control=effective_control,
                    )
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_profiled_fullrecover_legacy_nofe"
            else:
                path_solve = _with_nofe_details(
                    _solve_fullrecover_tau_path(
                        endog=endog,
                        exog=exog,
                        groups=groups,
                        counts=counts,
                        tau_target=tau_target,
                        control=effective_control,
                    )
                )
        elif spec.topology == "mixed":
            cpp_backend_fallback_reason = _mixed_cpp_fallback_reason(
                cancel_groups=cancel_groups,
                cancel_counts=cancel_counts,
                recover_groups=recover_groups,
                recover_counts=recover_counts,
            )
            if cpp_backend_fallback_reason is None:
                path_solve = _solve_mixed_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    cancel_names=cancel_names,
                    cancel_groups=cancel_groups,
                    cancel_counts=cancel_counts,
                    recover_names=recover_names,
                    recover_groups=recover_groups,
                    recover_counts=recover_counts,
                    tau_target=tau_target,
                    control=effective_control,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_mixed"
            else:
                path_solve = _solve_mixed_tau_path(
                    endog=endog,
                    exog=exog,
                    cancel_names=cancel_names,
                    cancel_groups=cancel_groups,
                    cancel_counts=cancel_counts,
                    recover_names=recover_names,
                    recover_groups=recover_groups,
                    recover_counts=recover_counts,
                    tau_target=tau_target,
                    control=effective_control,
                )
        elif spec.topology == "fullcancel":
            cpp_backend_fallback_reason = _fullcancel_cpp_fallback_reason(
                groups=groups,
                counts=counts,
            )
            if cpp_backend_fallback_reason is None:
                path_solve = _solve_fullcancel_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_target=tau_target,
                    control=effective_control,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = "used_fullcancel"
            else:
                path_solve = _solve_fullcancel_tau_path(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_target=tau_target,
                    control=effective_control,
                )
        else:
            cpp_backend_fallback_reason = _profiled_fullrecover_cpp_fallback_reason(
                spec=spec,
                endog=endog,
                exog=exog,
                groups=groups,
                counts=counts,
                tau_target=tau_target,
            )
            boundary_support_details, boundary_speed_details = (
                _fullrecover_boundary_support_details(
                    topology=spec.topology,
                    tau_target=tau_target,
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                )
            )
            if cpp_backend_fallback_reason is None:
                path_solve = _solve_profiled_fullrecover_tau_path_cpp(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_target=tau_target,
                    control=effective_control,
                    boundary_support_details=boundary_support_details,
                )
                kernel_backend = KERNEL_BACKEND_CPP
                cpp_backend_status = (
                    "used_manyway_fullrecover"
                    if _uses_joint_hdfe_state(groups)
                    else "used_profiled_fullrecover"
                )
            else:
                path_solve = _solve_fullrecover_tau_path(
                    endog=endog,
                    exog=exog,
                    groups=groups,
                    counts=counts,
                    tau_target=tau_target,
                    control=effective_control,
                )
    beta_star = np.asarray(path_solve.beta, dtype=float)
    beta = _back_transform_beta(beta_star, normalization_state)
    back_transform_residual = _back_transform_residual(
        beta,
        beta_star,
        normalization_state,
    )
    linear_back_transform_residual = _linear_back_transform_residual(
        exog_canonical=exog_canonical,
        exog_star=exog,
        beta=beta,
        beta_star=beta_star,
    )
    iteration = path_solve.iteration
    result_converged = bool(path_solve.converged)
    result_failure_reason = iteration.failure_reason
    tau_fit = path_solve.tau_fit
    linear = np.array([], dtype=float)
    fitted = np.array([], dtype=float)
    se = None
    covariance = None
    if np.all(np.isfinite(beta)):
        if spec.topology == "fullcancel":
            linear = _fullcancel_linear_state(exog_canonical, beta)
        elif spec.topology == "mixed" and path_solve.linear is not None:
            linear = np.asarray(path_solve.linear, dtype=float)
        elif path_solve.linear is not None:
            linear = np.asarray(path_solve.linear, dtype=float)
        else:
            try:
                linear = _profile_linear(
                    endog=endog,
                    exog=exog_canonical,
                    beta=beta,
                    groups=groups,
                    counts=counts,
                    tau=tau_fit,
                    maxiter=int(effective_control.fe_profile_maxiter),
                    tol=float(effective_control.fe_profile_tol),
                )
            except RuntimeError as exc:
                if not is_known_fe_runtime_failure(exc):
                    raise
                result_converged = False
                result_failure_reason = KERNEL_FAILURE_INVALID_PHI_STEP
                iteration = replace(
                    iteration,
                    converged=False,
                    failure_reason=result_failure_reason,
                    details={
                        **dict(iteration.details),
                        "post_solution_failure_reason": str(exc),
                    },
                )
                linear = np.full(endog.shape, np.nan, dtype=float)
        fitted = exp_with_float64_overflow_guard(linear)
    if (
        result_converged
        and spec.topology not in {"fullcancel", "mixed"}
        and spec.vcov_request.lower() == "iid"
        and not _uses_joint_hdfe_state(groups)
        and np.all(np.isfinite(beta))
    ):
        try:
            se, covariance = _manyway_iols_covariance(
                endog=endog,
                exog=exog_canonical,
                beta=beta,
                groups=groups,
                tau=tau_fit,
                se_type="iid",
                linear=linear,
            )
        except RuntimeError as exc:
            if not is_known_fe_runtime_failure(exc):
                raise
            se = None
            covariance = None

    beta_full, se_full, cov_full = _expand_intercept(
        beta,
        se,
        covariance,
        original_k=exog_full.shape[1],
        has_intercept=has_intercept,
    )
    if _speed_compact_fixed_tau_details_enabled(tau_target):
        path_details = _compact_fixed_tau_path_details(
            raw_details=path_solve.details,
            tau_target=tau_target,
            tau_fit=tau_fit,
            result_converged=result_converged,
            failure_reason=result_failure_reason,
        )
    else:
        canonical_path_details = _path_details_back_transformed(
            path_solve.details,
            normalization_state,
            exog_canonical=exog_canonical,
            exog_star=exog,
        )
        path_details = _path_details_with_public_coefficients(
            canonical_path_details,
            param_names=spec.param_names,
            original_k=exog_full.shape[1],
            has_intercept=has_intercept,
            endog=endog,
            exog=exog_canonical,
            groups=groups,
            vcov_request=spec.vcov_request,
            path_inference_allowed=spec.topology not in {"fullcancel", "mixed"},
        )

    if spec.topology == "noFE":
        granular_case = "nofe"
    elif spec.topology == "fullcancel":
        granular_case = "fullcancel_direct"
    elif spec.topology == "mixed":
        granular_case = "mixed_direct"
    else:
        granular_case = "fullrecover_direct"
    kernel_status = (
        "package_tau2_anchor_p0_s2b_fullcancel"
        if spec.topology == "fullcancel"
        else (
            "package_tau2_anchor_p0_s2c_mixed"
            if spec.topology == "mixed"
            else "package_tau2_anchor_p0_s2a"
        )
    )
    anchor_algorithm = (
        "iols_phase1_phase2_fullcancel"
        if spec.topology == "fullcancel"
        else (
            "iols_phase1_phase2_mixed_cancel_recover_design"
            if spec.topology == "mixed"
            else "iols_phase1_phase2"
        )
    )

    solver_details = {
        "kernel_backend": kernel_backend,
        "kernel": kernel_status,
        "anchor_algorithm": anchor_algorithm,
        "anchor_tau": 2.0,
        "tau_target": tau_target,
        "tau_path_targets": tuple(float(value) for value in spec.tau_path_targets),
        "tau_fit": tau_fit,
        "path_status": path_details.get("path_status"),
        "failure_reason": result_failure_reason,
        "tau_path": path_details,
        "external_warm_start_policy": "not_accepted",
        "fe_names": fe_names,
        "fullrecover_fe_policy": _fullrecover_fe_policy(groups)
        if spec.topology not in {"fullcancel", "mixed"}
        else "not_applicable",
        "fullcancel_fe_policy": "weighted_within_cancel_all"
        if spec.topology in {"fullcancel", "mixed"}
        else "not_applicable",
        "fullcancel_index_policy": "slope_only_raw_exog"
        if spec.topology == "fullcancel"
        else "not_applicable",
        "mixed_algorithm": path_details.get("mixed_algorithm", "not_applicable"),
        "mixed_cancel_fe_policy": path_details.get(
            "mixed_cancel_fe_policy",
            "not_applicable",
        ),
        "mixed_recover_fe_policy": path_details.get(
            "mixed_recover_fe_policy",
            "not_applicable",
        ),
        "mixed_retained_index_policy": path_details.get(
            "mixed_retained_index_policy",
            "not_applicable",
        ),
        "nofe_algorithm": path_details.get("nofe_algorithm", "not_applicable"),
        "nofe_implicit_intercept_policy": path_details.get(
            "nofe_implicit_intercept_policy",
            "not_applicable",
        ),
        "has_intercept": has_intercept,
        "vcov_request": spec.vcov_request,
        "exp_guard_policy": "float64_overflow_boundary_only",
        **_normalization_details(
            normalization_state,
            back_transform_residual=back_transform_residual,
            linear_back_transform_residual=linear_back_transform_residual,
        ),
        **_cpp_solver_metadata(
            policy=cpp_backend_policy,
            status=cpp_backend_status,
            fallback_reason=cpp_backend_fallback_reason,
        ),
        **speed_control_details,
        **boundary_speed_details,
    }

    return IOLSKernelResult(
        beta=beta_full,
        se=se_full,
        covariance=cov_full,
        converged=result_converged,
        tau_fit=tau_fit,
        linear_predictor=linear,
        fittedvalues=fitted,
        topology=spec.topology,
        granular_case=granular_case,
        iteration=iteration,
        solver_details=solver_details,
    )


# Research-only direct fixed-tau noFE route. Not wired into solve_iols_kernel;
# the C++ twin is _kernel_cpp.solve_nofe_direct_fixed_tau and both return the
# same key sets below.
# runtime_v2 (2026-10-02 preflight): the direct route keeps the last Phase 2
# iterate on native failure, counts Phase 1 map evaluations, and the C++ twin
# matches the NumPy kappa window start and adapted Anderson history. The
# result schema is unchanged; the version is reported in backend_details.
# runtime_v4 (2026-10-03, campaign v6 candidate): Phase 2 stagnation, the
# windowed kappa monitor and the Anderson memory adaptation read the max-abs
# plain-map residual at the current iterate instead of the accepted move
# (which is an Anderson extrapolation length after an accepted step); the eps
# stop, the post-check and every control are unchanged. An invalid final
# verification map now clears native convergence.
DIRECT_NOFE_RUNTIME_VERSION: Final = "direct_nofe_runtime_v4_2026_10_03"
# runtime_v3 direct-route monitor policies (both backends report the same
# kappa policy string; the q-hat guard exists only in the NumPy reference).
# runtime_v4 renames the kappa policy because its input history changed.
DIRECT_KAPPA_MONITOR_POLICY: Final = "windowed_plain_residual_ratio_10"
DIRECT_QHAT_GUARD_POLICY_NUMPY: Final = "advisory_numpy_reference_direct_route"
DIRECT_PROGRESS_MONITOR_POLICY: Final = (
    "plain_map_residual_maxabs_star_at_current_iterate"
)
DIRECT_POST_CHECK_POLICY: Final = "invalid_verification_map_clears_convergence"
DIRECT_NOFE_FIXED_TAU_SCHEMA: Final = "pyiols.direct_nofe_fixed_tau.v1"
DIRECT_NOFE_FIXED_TAU_ROUTE: Final = "direct_nofe_fixed_tau"
DIRECT_NOFE_FIXED_TAU_ALGORITHM: Final = "iols_phase1_phase2_at_requested_tau"
DIRECT_NOFE_FIXED_TAU_RESULT_KEYS: Final = (
    "schema",
    "route",
    "backend",
    "kernel_backend",
    "algorithm",
    "intercept_policy",
    "tau_requested",
    "tau_fit",
    "phase1_tau",
    "phase2_tau",
    "tau2_anchor_used",
    "tau_continuation_used",
    "converged",
    "failure_reason",
    "beta",
    "intercept",
    "linear",
    "iteration",
    "diagnostics",
    "control",
    "backend_details",
)
DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS: Final = (
    "converged",
    "phase",
    "iterations",
    "restarts",
    "final_rho",
    "rho_path",
    "final_criterion",
    "anderson_m_initial",
    "anderson_m_final",
    "failure_reason",
)
DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS: Final = (
    "phase1_attempts",
    "phase1_accepted",
    "phase1_rejected",
    "phase2_evaluations",
    "last_finite_criterion",
    "best_finite_criterion",
    "post_convergence_residual",
    "post_convergence_kappa_est",
    "post_convergence_score_residual",
    "post_convergence_fe_score_residual",
    "score_tolerance",
    "score_validation_status",
    "score_warning_codes",
)
DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS: Final = (
    "anderson_m",
    "eps",
    "stopcrit",
    "max_phase1_iterations",
    "max_phase2_evaluations",
    "delta_log_min",
    "delta_log_max",
    "delta_grid_size",
    "rho_log_min",
    "rho_log_max",
    "rho_grid_size",
)
# runtime_v3: `direct_restart_start` is deliberately NOT echoed in `control`
# (the harness direct_route._normalize casts every non-float control echo with
# int()); both backends report it in backend_details instead.
_DIRECT_NOFE_INT_DIAGNOSTICS: Final = frozenset(
    {"phase1_attempts", "phase1_accepted", "phase1_rejected", "phase2_evaluations"}
)


def _direct_nofe_validated_inputs(
    endog: np.ndarray,
    exog_slopes: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.array(endog, dtype=np.float64, copy=True)
    x = np.array(exog_slopes, dtype=np.float64, copy=True, order="C")
    if y.ndim != 1:
        raise ValueError("endog must be one-dimensional")
    if x.ndim != 2:
        raise ValueError("exog_slopes must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError("endog and exog_slopes must have the same number of rows")
    if x.shape[1] < 1:
        raise ValueError(
            "exog_slopes must have at least one slope column; the intercept is "
            "profiled"
        )
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("endog must be finite and nonnegative")
    if not np.any(y > 0.0):
        raise ValueError("endog must contain at least one positive value")
    if not np.all(np.isfinite(x)):
        raise ValueError("exog_slopes must be finite")
    for column in range(x.shape[1]):
        if np.max(x[:, column]) == np.min(x[:, column]):
            raise ValueError(
                f"exog_slopes column {column} is constant; pass slopes only, the "
                "intercept is profiled"
            )
    tau_value = float(tau)
    if not np.isfinite(tau_value) or tau_value < 0.0 or tau_value > 3.0:
        raise ValueError("tau must be finite and in [0, 3]")
    return y, x, tau_value


def _direct_nofe_validate_control(control: IOLSKernelControl) -> None:
    if not isinstance(control, IOLSKernelControl):
        raise TypeError("control must be an IOLSKernelControl")
    if _transformed_normalization_mode(control) != "none":
        raise ValueError(
            "direct fixed-tau route requires transformed_design_normalization='none'"
        )
    if int(control.delta_grid_size) < 1:
        raise ValueError(
            "direct fixed-tau route requires delta_grid_size >= 1 so Phase 1 runs"
        )
    if int(control.anderson_m) < 0:
        raise ValueError("anderson_m must be nonnegative")
    if not np.isfinite(float(control.eps)) or float(control.eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    if int(control.stopcrit) < 2 or int(control.max_phase1_iterations) < 2:
        raise ValueError("stopcrit and max_phase1_iterations must be at least 2")
    if int(control.max_phase2_evaluations) < 1:
        raise ValueError("max_phase2_evaluations must be at least 1")
    if str(control.direct_restart_start) not in DIRECT_RESTART_START_POLICIES:
        raise ValueError(
            "direct_restart_start must be one of "
            f"{DIRECT_RESTART_START_POLICIES}"
        )


# rho_grid_hook_v1: opt-in Phase-2 rho-grid policies (direct no-FE route only).
DIRECT_RHO_GRID_POLICIES: Final = ("legacy", "step_grid", "curvature_step_grid")
DIRECT_RHO_GRID_HOOK_VERSION: Final = "rho_grid_hook_v1_2026_10_05"


def _direct_step_rho_grid(
    alpha0: float, alpha_min: float
) -> tuple[tuple[float, ...], bool]:
    # rho_grid_hook_v1: halving alpha grid in plain float arithmetic.
    alpha0 = float(alpha0)
    alpha_min = float(alpha_min)
    if not (math.isfinite(alpha0) and alpha0 > 0.0):
        raise RuntimeError(f"direct_rho_grid_invalid: alpha0={alpha0!r}")
    if not (math.isfinite(alpha_min) and alpha_min > 0.0):
        raise RuntimeError(f"direct_rho_grid_invalid: alpha_min={alpha_min!r}")
    clipped = alpha0 <= alpha_min
    if clipped:
        alphas = [alpha_min]
    else:
        alphas = []
        a = alpha0
        while a > alpha_min:
            alphas.append(a)
            a = a * 0.5
        alphas.append(alpha_min)
    rho = tuple(1.0 / a - 1.0 for a in alphas)
    previous = None
    for value in rho:
        if not (math.isfinite(value) and value > 0.0):
            raise RuntimeError(f"direct_rho_grid_invalid: rho={value!r}")
        if previous is not None and not value > previous:
            raise RuntimeError("direct_rho_grid_invalid: not strictly increasing")
        previous = value
    return rho, clipped


def _direct_profiled_curvature_L0(
    *,
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: tuple[np.ndarray, ...],
    counts: tuple[int, ...],
    x_demeaned: np.ndarray,
    tau: float,
    control: IOLSKernelControl,
) -> dict[str, Any]:
    # rho_grid_hook_v1: max generalized eigenvalue of (profiled H, Z'Z/n) at beta.
    start_ns = time.perf_counter_ns()
    linear = _profile_linear(
        endog=endog,
        exog=exog,
        beta=beta,
        groups=groups,
        counts=counts,
        tau=tau,
        maxiter=int(control.fe_profile_maxiter),
        tol=float(control.fe_profile_tol),
    )
    linear = np.asarray(linear, dtype=np.float64)
    if not np.all(np.isfinite(linear)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_index")
    y = np.asarray(endog, dtype=np.float64)
    z = np.asarray(x_demeaned, dtype=np.float64)
    n = z.shape[0]
    c1 = float(tau) - 1.0
    c2 = 2.0 - float(tau)
    h = np.zeros(n, dtype=np.float64)
    if c1 != 0.0:
        positive = y > 0.0
        h[positive] += c1 * exp_with_float64_overflow_guard(
            np.log(y[positive]) + (1.0 - float(tau)) * linear[positive]
        )
    if c2 != 0.0:
        h += c2 * exp_with_float64_overflow_guard((2.0 - float(tau)) * linear)
    if not np.all(np.isfinite(h)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_row_curvature")
    total = float(np.sum(h))
    if not (math.isfinite(total) and total > 0.0):
        raise RuntimeError("direct_rho_curvature_failed: nonpositive_curvature_sum")
    g = z.T @ h
    hess = ((z.T * h) @ z - np.outer(g, g) / total) / float(n)
    hess = 0.5 * (hess + hess.T)
    if not np.all(np.isfinite(hess)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_hessian")
    gram = (z.T @ z) / float(n)
    gram = 0.5 * (gram + gram.T)
    try:
        chol = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "direct_rho_curvature_failed: gram_cholesky_failed"
        ) from exc
    left = np.linalg.solve(chol, hess)
    cmat = np.linalg.solve(chol, left.T)
    cmat = 0.5 * (cmat + cmat.T)
    if not np.all(np.isfinite(cmat)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_hessian")
    try:
        eigenvalues = np.linalg.eigvalsh(cmat)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "direct_rho_curvature_failed: eigensolver_failed"
        ) from exc
    if not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("direct_rho_curvature_failed: nonfinite_eigenvalue")
    return {
        "L0": float(np.max(eigenvalues)),
        "lambda_min": float(np.min(eigenvalues)),
        "wall_ns": int(time.perf_counter_ns() - start_ns),
    }


def _direct_policy_rho_path(
    *,
    policy: str,
    linear: np.ndarray,
    tau: float,
    control: IOLSKernelControl,
    curvature: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    # rho_grid_hook_v1: build the step-grid rho list for a non-legacy policy.
    alpha_max = 1.0 / (1.0 + math.exp(float(control.rho_log_min)))
    alpha_min = 1.0 / (1.0 + math.exp(float(control.rho_log_max)))
    if policy == "step_grid":
        alpha0 = alpha_max
    elif policy == "curvature_step_grid":
        if curvature is None:
            raise RuntimeError("direct_rho_grid_invalid: missing curvature")
        alpha0 = 1.0 / max(
            1.0 + math.exp(float(control.rho_log_min)), float(curvature["L0"])
        )
    else:
        raise ValueError(f"rho_grid_policy must be one of {DIRECT_RHO_GRID_POLICIES}")
    values, clipped = _direct_step_rho_grid(alpha0, alpha_min)
    rho_path = np.asarray(values, dtype=float)
    linear_arr = np.asarray(linear, dtype=float)
    details: dict[str, Any] = {
        "rho_policy": "direct_" + policy,
        "linear_min": float(np.nanmin(linear_arr)) if linear_arr.size else float("nan"),
        "linear_max": float(np.nanmax(linear_arr)) if linear_arr.size else float("nan"),
        "rho_candidate_count": int(rho_path.size),
        "rho_hint": None,
        "rho_log_min": float(control.rho_log_min),
        "rho_log_max": float(control.rho_log_max),
        "rho_grid_size": int(control.rho_grid_size),
        "rho_min": float(rho_path[0]),
        "rho_max": float(rho_path[-1]),
        "_rho_grid_alpha0": float(alpha0),
        "_rho_grid_alpha_min": float(alpha_min),
        "_rho_grid_clipped": bool(clipped),
    }
    return rho_path, details


def _solve_nofe_direct_fixed_tau(
    *,
    endog: np.ndarray,
    exog_slopes: np.ndarray,
    tau: float,
    control: IOLSKernelControl,
    rho_grid_policy: str = "legacy",
) -> dict[str, Any]:
    """Research-only direct fixed-tau noFE iOLS solve (NumPy reference).

    Runs the existing profiled Phase 1 (delta grid) and Phase 2 (rho grid) once
    at the requested tau: no tau=2 anchor, no tau continuation and no speed
    control environment flags. The intercept is profiled analytically through
    the singleton-intercept group, so exog_slopes must exclude it. The returned
    linear index is X @ beta plus the profiled intercept at the requested tau.
    """
    # rho_grid_hook_v1: validate the opt-in policy before any work.
    if rho_grid_policy not in DIRECT_RHO_GRID_POLICIES:
        raise ValueError(f"rho_grid_policy must be one of {DIRECT_RHO_GRID_POLICIES}")
    y, x, tau_value = _direct_nofe_validated_inputs(endog, exog_slopes, tau)
    _direct_nofe_validate_control(control)
    groups = (np.zeros(y.shape[0], dtype=np.int32),)
    counts = (1,)
    state = _fullrecover_invariant_state(
        exog=x,
        groups=groups,
        counts=counts,
        control=control,
    )
    solve = _concentrated_fullrecover(
        endog=y,
        exog=x,
        groups=groups,
        counts=counts,
        tau=tau_value,
        control=control,
        beta_init=None,
        rho_hint=None,
        invariant_state=state,
        direct_nofe_fixed_tau=True,
        direct_rho_grid_policy=rho_grid_policy,
    )
    beta = np.array(solve.beta, dtype=np.float64, copy=True)
    intercept = float("nan")
    linear = None
    if np.all(np.isfinite(beta)):
        # Same arithmetic as _profile_linear for the singleton-intercept group.
        base = x @ beta
        intercept = _profile_intercept_adjustment(endog=y, linear=base, tau=tau_value)
        candidate = base + intercept
        if np.all(np.isfinite(candidate)):
            linear = candidate
    iteration = solve.iteration
    details = dict(iteration.details)
    diagnostics: dict[str, Any] = {}
    for key in DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS:
        if key in _DIRECT_NOFE_INT_DIAGNOSTICS:
            diagnostics[key] = int(details.get(key, 0))
        elif key == "score_validation_status":
            diagnostics[key] = str(details.get(key, "not_checked"))
        elif key == "score_warning_codes":
            diagnostics[key] = tuple(str(code) for code in details.get(key, ()))
        else:
            diagnostics[key] = float(details.get(key, float("nan")))
    return {
        "schema": DIRECT_NOFE_FIXED_TAU_SCHEMA,
        "route": DIRECT_NOFE_FIXED_TAU_ROUTE,
        "backend": "numpy",
        "kernel_backend": KERNEL_BACKEND_PYTHON_REFERENCE,
        "algorithm": DIRECT_NOFE_FIXED_TAU_ALGORITHM,
        "intercept_policy": "profiled_singleton_intercept",
        "tau_requested": tau_value,
        "tau_fit": tau_value,
        "phase1_tau": tau_value,
        "phase2_tau": tau_value,
        "tau2_anchor_used": False,
        "tau_continuation_used": False,
        "converged": bool(iteration.converged),
        "failure_reason": iteration.failure_reason,
        "beta": beta,
        "intercept": float(intercept),
        "linear": linear,
        "iteration": {
            "converged": bool(iteration.converged),
            "phase": str(iteration.phase),
            "iterations": int(iteration.iterations),
            "restarts": int(iteration.restarts),
            "final_rho": iteration.final_rho,
            "rho_path": tuple(float(value) for value in iteration.rho_path),
            "final_criterion": iteration.final_criterion,
            "anderson_m_initial": int(iteration.anderson_m_initial),
            "anderson_m_final": int(iteration.anderson_m_final),
            "failure_reason": iteration.failure_reason,
        },
        "diagnostics": diagnostics,
        "control": {
            key: getattr(control, key) for key in DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS
        },
        "backend_details": {
            **{
                key: value
                for key, value in details.items()
                if key not in DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS
            },
            "runtime_version": DIRECT_NOFE_RUNTIME_VERSION,
            "anderson_history_policy": "grows_with_adapted_m",
            "kappa_window_first_entry": 4,
            "qhat_check": DIRECT_QHAT_GUARD_POLICY_NUMPY,
        },
    }
