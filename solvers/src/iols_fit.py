"""Historical two-phase fixed-tau iOLS, with local portable initialization."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from collections.abc import Mapping
import numpy as np
import direct_route
_AdapterState = SimpleNamespace
_STATE = None

BACKEND_DETAIL_KEYS = (
    "phase1_evaluations",
    "qhat_invalid_count",
    "kappa_monitor_max",
    "kappa_monitor_checks",
    "runtime_version",
    "nonconverged_iterate_policy",
    "phase2_attempts",
    "phase2_warm_restarts",
    "best_score_norm",
    "best_score_rho",
    "kappa_coordinate_row_max",
    # runtime_v4 policy labels / counter (diagnostic-only: the engine's compact whitelist
    # does not forward them; direct callers such as host_verification/v6_verify.py read them).
    "kappa_monitor_policy",
    "progress_monitor_policy",
    "post_check_policy",
    "post_convergence_invalid_map_count",
    "phase1_post_convergence_invalid_map_count",
    # Curvature rerun (2026-10-05): scalar rho_grid_hook_v1 diagnostics (absent before the
    # Phase-2 handoff; None when not applicable). rho_grid_values/alpha_min are not forwarded.
    "rho_grid_policy",
    "rho_grid_hook_version",
    "rho_grid_count",
    "rho_grid_alpha0",
    "rho_grid_clipped",
    "rho_curvature_L0",
    "rho_curvature_lambda_min",
    "rho_curvature_wall_ns",
    "rho_curvature_pilot",
)

SUPPORTED_TAUS = (1.0, 2.0)

SUPPORTED_BACKENDS = ("numpy", "cpp")

REQUIRED_CONTROL_KEYS = (
    "eps",
    "anderson_m",
    "stopcrit",
    "max_phase1_iterations",
    "max_phase2_evaluations",
)

KERNEL_CONTROL_DEFAULTS_L113_133 = {
    "anderson_m": 5,
    "eps": 1e-6,
    "stopcrit": 10000,
    "tau_path_steps": 20,
    "min_dtau": None,
    "max_path_evaluations": 10000,
    "max_phase1_iterations": 10000,
    "max_phase2_evaluations": 50000,
    "fe_profile_maxiter": 10000,
    "fe_profile_tol": 1e-6,
    "fe_demean_maxiter": 10000,
    "fe_demean_tol": 1e-10,
    "delta_log_min": -5.0,
    "delta_log_max": 5.0,
    "delta_grid_size": 5,
    "rho_log_min": -7.0,
    "rho_log_max": 14.0,
    "rho_grid_size": 150,
    "transformed_design_normalization": "none",
}

FORBIDDEN_ENV_PREFIXES = ("PYIOLS_SPEED_", "PYIOLS_CPP_SPEED_")

FORBIDDEN_ENV_EXACT = ("PYIOLS_CPP_TIERED_FE_DEMEAN",)

CAP_FAILURE_REASONS = frozenset(
    {
        "phase2_evaluation_budget_exhausted",
    }
)

NUMERICAL_FAILURE_REASONS = frozenset(
    {
        "not_converged",
        "invalid_phi_step",
        "criterion_stagnation",
        "criterion_explosion",
        "noncontractive",
        "post_convergence_residual",
        "post_convergence_noncontractive",
        "post_convergence_invalid_map",
        "q_convergence_invalid",
        "phase1_candidate_unusable",
        "empty_finite_rho_path",
        "tau2_anchor_failed",
        "tau2_anchor_missing_rho",
        "not_started",
    }
)

SUPPORT_FAILURE_REASONS = frozenset({"tau2_boundary_separation"})

_NUMERICAL_RUNTIME_MESSAGE_PREFIXES = (
    "LAPACK DGELSD least-squares solve failed",
    "LAPACK DGELSD workspace query failed",
    # Curvature rerun: rho_grid_hook_v1 curvature/grid failures are recorded numerical
    # failures of that fit (never a fallback to another grid, never a campaign stop).
    "direct_rho_curvature_failed",
    "direct_rho_grid_invalid",
)

_S_SINGULAR_PREFIX = "S_singular"

class IOLSAdapterError(Exception):
    """Base class for adapter errors. These must surface to the harness."""

class IOLSContractError(IOLSAdapterError):
    """Inputs or environment violate the adapter contract (not numerical)."""

def forbidden_env_flags(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    env = os.environ if environ is None else environ
    hits = [
        name
        for name in env
        if name.startswith(FORBIDDEN_ENV_PREFIXES) or name in FORBIDDEN_ENV_EXACT
    ]
    return tuple(sorted(hits))

def classify_failure_reason(reason: str | None) -> str:
    """Map a kernel failure reason to a coarse class. Never an existence claim."""

    if reason is None:
        return "unspecified_nonconvergence"
    if reason in CAP_FAILURE_REASONS:
        return "cap_or_refinement_limit"
    if reason in NUMERICAL_FAILURE_REASONS:
        return "numerical_nonconvergence"
    if reason in SUPPORT_FAILURE_REASONS:
        return "support_check"
    return "unrecognized_kernel_reason"

def validate_control_mapping(
    mapping: Mapping[str, Any],
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    """Validate an iOLS control mapping against IOLSKernelControl field names."""

    if not isinstance(mapping, Mapping):
        raise IOLSContractError("iols_control must be a mapping")
    values = dict(mapping)
    unknown = sorted(set(values) - set(field_names))
    if unknown:
        raise IOLSContractError(f"unknown iols_control keys: {unknown}")
    missing = [key for key in REQUIRED_CONTROL_KEYS if key not in values]
    if missing:
        raise IOLSContractError(
            f"iols_control must set these explicitly (no silent defaults): {missing}"
        )
    eps = values["eps"]
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise IOLSContractError("eps must be a real number")
    if not (math.isfinite(float(eps)) and float(eps) > 0.0):
        raise IOLSContractError("eps must be finite and > 0")
    int_keys = (
        "anderson_m",
        "stopcrit",
        "max_phase1_iterations",
        "max_phase2_evaluations",
        "tau_path_steps",
        "max_path_evaluations",
        "fe_profile_maxiter",
        "fe_demean_maxiter",
        "delta_grid_size",
        "rho_grid_size",
    )
    for key in int_keys:
        if key in values:
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise IOLSContractError(f"{key} must be an integer")
            values[key] = int(value)
    if values["anderson_m"] < 0:
        raise IOLSContractError("anderson_m must be >= 0")
    for key in ("stopcrit", "max_phase1_iterations", "max_phase2_evaluations"):
        if values[key] < 2:
            raise IOLSContractError(f"{key} must be >= 2")
    # Phase 1 (delta grid) is what makes this a two-phase solve; size 0 is
    # the fixed-tau fast control shortcut (kernel.py L5150-5156).
    if int(values.get("delta_grid_size", 5)) < 1:
        raise IOLSContractError(
            "delta_grid_size < 1 disables Phase 1; not a two-phase iOLS run"
        )
    # The C++ mapping (kernel.py L5309-5327, bind.cpp L9654-9698) does not
    # forward fe_profile_*; changing them would break NumPy/C++ parity.
    for key in ("fe_profile_maxiter", "fe_profile_tol"):
        if key in values and values[key] != KERNEL_CONTROL_DEFAULTS_L113_133[key]:
            raise IOLSContractError(
                f"{key} is not forwarded to C++; overriding it breaks parity"
            )
    mode = values.get("transformed_design_normalization", "none")
    if mode not in ("none", "scale"):
        raise IOLSContractError(
            "transformed_design_normalization must be 'none' or 'scale'"
        )
    min_dtau = values.get("min_dtau")
    if min_dtau is not None:
        if isinstance(min_dtau, bool) or not isinstance(min_dtau, (int, float)):
            raise IOLSContractError("min_dtau must be None or a real number")
        values["min_dtau"] = float(min_dtau)
    for key in ("eps", "fe_demean_tol", "delta_log_min", "delta_log_max",
                "rho_log_min", "rho_log_max", "fe_profile_tol"):
        if key in values:
            values[key] = float(values[key])
    return values

def check_inputs(X: Any, y: Any, tau: Any, backend: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Cheap O(n k) contract checks. Raises IOLSContractError on violation."""

    if backend not in SUPPORTED_BACKENDS:
        raise IOLSContractError(f"backend must be one of {SUPPORTED_BACKENDS}")
    if isinstance(tau, bool) or not isinstance(tau, (int, float, np.floating, np.integer)):
        raise IOLSContractError("tau must be a real scalar")
    tau_f = float(tau)
    if tau_f not in SUPPORTED_TAUS:
        raise IOLSContractError(f"tau must be exactly one of {SUPPORTED_TAUS}")
    x = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or yy.ndim != 1:
        raise IOLSContractError("X must be 2-D and y 1-D")
    n_obs, n_cols = x.shape
    if yy.shape[0] != n_obs:
        raise IOLSContractError("X and y row counts differ")
    if n_cols < 2:
        raise IOLSContractError("X must contain the intercept column and >= 1 slope")
    if n_obs <= n_cols:
        raise IOLSContractError("need n_obs > n_cols")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(yy))):
        raise IOLSContractError("X and y must be finite")
    if np.any(yy < 0.0):
        raise IOLSContractError("y must be nonnegative (kernel does not validate)")
    if not np.any(yy > 0.0):
        raise IOLSContractError(
            "y has no positive values; profiled intercept silently returns 0 "
            "(kernel.py L528-530)"
        )
    # The kernel detects the intercept only as an all-ones column 0
    # (kernel.py L318-326) and ALWAYS profiles a singleton intercept for noFE
    # (kernel.py L5874-5876). Any other constant column is collinear with
    # that implicit intercept; the min-norm projection would not fail loudly.
    if not np.all(x[:, 0] == 1.0):
        raise IOLSContractError(
            "X[:, 0] must be exactly 1.0 (outer column scaling must leave the "
            "intercept column untouched)"
        )
    slopes = x[:, 1:]
    constant = np.flatnonzero(np.ptp(slopes, axis=0) == 0.0)
    if constant.size:
        raise IOLSContractError(
            f"slope columns {[int(c) + 1 for c in constant]} are constant "
            "(collinear with the implicit intercept)"
        )
    return x, yy, tau_f

def _resolve_control(state: _AdapterState, control: Mapping[str, Any] | None) -> Any:
    if control is None:
        return state.control
    key = tuple(sorted((str(k), v) for k, v in dict(control).items()))
    cached = state.control_cache.get(key)
    if cached is not None:
        return cached
    field_names = tuple(f.name for f in dataclasses.fields(state.kernel.IOLSKernelControl))
    values = validate_control_mapping(control, field_names)
    built = state.kernel.IOLSKernelControl(**values)
    state.control_cache[key] = built
    return built

def _result(
    *,
    beta: np.ndarray | None,
    native_converged: bool,
    native_status: str,
    phase2_iterations: int | None,
    rho_restarts: int | None,
    backend: str,
    diagnostics: dict[str, Any],
    engine_calls: int = 1,
) -> dict[str, Any]:
    return {
        "beta": beta,
        "native_converged": bool(native_converged),
        "native_status": native_status,
        "iterations": None,
        "phase1_iterations": None,
        "phase2_iterations": phase2_iterations,
        "rho_restarts": rho_restarts,
        "actual_backend": backend,
        "engine_calls": engine_calls,
        "budget_exhausted": diagnostics.get("iteration_failure_reason") in CAP_FAILURE_REASONS,
        "diagnostics": diagnostics,
    }

def fit(
    X: Any,
    y: Any,
    tau: float,
    backend: str,
    control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Direct fixed-tau two-phase fit; no GPML prerequisite for PPML."""
    state = _STATE
    if state is None:
        raise IOLSContractError("initialize() must be called before fit()")
    if forbidden_env_flags():
        raise IOLSContractError("historical shortcut flags must remain disabled")
    if backend not in state.backends:
        raise IOLSContractError(f"backend {backend!r} was not initialized")
    x_full, endog, tau_f = check_inputs(X, y, tau, backend)
    kctl, K = _resolve_control(state, control), state.kernel
    exog, has_intercept, _ = K._split_intercept(x_full, None)
    if not has_intercept:
        raise IOLSContractError("intercept column not detected")
    _, groups, counts = K._kernel_groups_from_spec(
        K.IOLSKernelSpec(endog=endog, exog=x_full, topology="noFE", tau_target=tau_f)
    )
    diagnostics = {
        "route": backend, "tau_target": tau_f,
        "start_policy": "kernel_internal_log1p_ols_on_demeaned_x",
    }
    try:
        norm = K._build_transformed_normalization_state(
            topology="noFE", exog=exog, groups=groups, counts=counts, control=kctl,
        )
    except ValueError as exc:
        if not str(exc).startswith(_S_SINGULAR_PREFIX):
            raise
        diagnostics.update(failure_class="design_degenerate", exception=str(exc)[:200])
        return _result(
            beta=None, native_converged=False, native_status="design_screen_failed:S_singular",
            phase2_iterations=None, rho_restarts=None, backend=backend,
            diagnostics=diagnostics, engine_calls=0,
        )
    exog_star = K._apply_transformed_normalization(exog, norm)
    # Normalization is applied once outside the native/reference entrypoint.
    # The fixed-tau core requires "none" to prevent an inconsistent second scaling.
    core_control = dataclasses.replace(kctl, transformed_design_normalization="none")
    try:
        with np.errstate(all="ignore"):
            raw = state.direct.solve_direct_fixed_tau(
                kernel=K, native=state.native, backend=backend,
                exog_slopes=exog_star, endog=endog, tau=tau_f, control=core_control,
            )
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        return _numerical_exception_result(exc, backend, diagnostics)
    except RuntimeError as exc:
        if str(exc).startswith(_NUMERICAL_RUNTIME_MESSAGE_PREFIXES) or K.is_known_fe_runtime_failure(exc):
            return _numerical_exception_result(exc, backend, diagnostics)
        raise

    iteration, detail = raw["iteration"], raw["diagnostics"]
    backend_details = raw.get("backend_details") or {}
    # The direct-route wrapper validates backend, exact tau and absence of an anchor.
    reason = raw["failure_reason"] or iteration["failure_reason"]
    native_converged = bool(raw["converged"])
    beta_star = np.asarray(raw["beta"], dtype=float)
    if beta_star.shape != (x_full.shape[1] - 1,):
        raise IOLSContractError("direct route returned the wrong coefficient dimension")
    slopes = np.asarray(K._back_transform_beta(beta_star, norm), dtype=float)
    phase2 = int(detail["phase2_evaluations"])
    diagnostics.update({
        "tau_fit": raw["tau_fit"], "phase1_tau": raw["phase1_tau"],
        "phase2_tau": raw["phase2_tau"], "tau2_anchor_used": False,
        "tau_continuation_used": False, "iteration_failure_reason": reason,
        "final_rho": iteration["final_rho"],
        "final_criterion_star": iteration["final_criterion"],
        "anderson_m_final": iteration["anderson_m_final"],
        "native_iteration_counter_sum": int(iteration["iterations"]),
        "phase1_attempts": int(detail["phase1_attempts"]),
        "phase1_accepted": int(detail["phase1_accepted"]),
        "phase1_rejected": int(detail["phase1_rejected"]),
        "final_step_post_convergence_residual_star": detail["post_convergence_residual"],
        "final_step_slope_score_max_abs_per_n_star": detail["post_convergence_score_residual"],
        "final_step_intercept_score_abs_per_n": detail["post_convergence_fe_score_residual"],
        "final_step_score_validation_status": detail["score_validation_status"],
        "phase2_count_trustworthy": True,
        "normalization_condition_after": norm.condition_after,
        "back_transform_residual": K._back_transform_residual(slopes, beta_star, norm),
        "failure_class": "none" if native_converged else classify_failure_reason(reason),
    })
    for key in BACKEND_DETAIL_KEYS:
        if key in backend_details:
            value = backend_details[key]
            if value is None or isinstance(value, (bool, int, str)):
                diagnostics[key] = value
            else:
                diagnostics[key] = float(value)
    # runtime_v3 per-attempt trace (list of small dicts); non-scalar, so the
    # engine's compact record drops it. Only direct callers (host diagnostic) see it.
    trace = backend_details.get("phase2_attempt_trace")
    if trace is not None:
        diagnostics["phase2_attempt_trace"] = [dict(item) for item in trace]
    # Preserve a finite last iterate even after a native unsuccessful stop.
    # The independent exact-equation audit, not the native flag, decides accuracy.
    beta = None
    if np.all(np.isfinite(slopes)) and np.isfinite(raw["intercept"]):
        base = exog @ slopes
        intercept = float(K._profile_intercept_adjustment(endog=endog, linear=base, tau=tau_f))
        candidate = np.r_[intercept, slopes]
        if np.all(np.isfinite(candidate)) and np.all(np.isfinite(base + intercept)):
            beta = candidate
            if raw["linear"] is not None:
                diagnostics["fitted_index_consistency_max_abs"] = float(
                    np.max(np.abs(np.asarray(raw["linear"]) - (base + intercept)))
                )
    status = "converged" if native_converged else f"direct_nonconvergence:{reason}"
    if beta is None:
        status += ":nonfinite_beta_or_index"
        diagnostics["failure_class"] = "nonfinite_beta_or_index"
    return _result(
        beta=beta, native_converged=native_converged, native_status=status,
        phase2_iterations=phase2, rho_restarts=int(iteration["restarts"]),
        backend=backend, diagnostics=diagnostics,
    )

def _numerical_exception_result(
    exc: BaseException,
    backend: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    diagnostics["failure_class"] = "numerical_exception"
    diagnostics["exception_type"] = type(exc).__name__
    diagnostics["exception"] = str(exc).splitlines()[0][:200] if str(exc) else ""
    return _result(
        beta=None,
        native_converged=False,
        native_status=f"numerical_exception:{type(exc).__name__}",
        phase2_iterations=None,
        rho_restarts=None,
        backend=backend,
        diagnostics=diagnostics,
    )

def initialize(control, backends=("numpy", "cpp"), native_path=None):
    global _STATE
    from historical_iols.solvers import kernel
    native = None
    if "cpp" in backends:
        if native_path is None:
            build = Path(__file__).resolve().parents[1] / "build"
            candidates = sorted(build.glob("_kernel_cpp*.so")) + sorted(build.glob("_kernel_cpp*.pyd"))
            if len(candidates) != 1:
                raise RuntimeError("Build the C++ kernel first: python solvers/build_native.py --eigen-include PATH")
            native_path = candidates[0]
        spec = importlib.util.spec_from_file_location("historical_iols._kernel_cpp", Path(native_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the supplied C++ kernel")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        if native.backend_info().get("kernel_backend") != "cpp":
            raise RuntimeError("The native module did not identify itself as the C++ kernel")
    def native_loader():
        if native is None:
            raise RuntimeError("C++ requested without a loaded native kernel")
        return native
    kernel._load_cpp_kernel_module = native_loader
    names = tuple(f.name for f in dataclasses.fields(kernel.IOLSKernelControl))
    values = validate_control_mapping(control, names)
    _STATE = SimpleNamespace(kernel=kernel, native=native, direct=direct_route,
        backends=tuple(backends), control=kernel.IOLSKernelControl(**values),
        control_mapping=values, control_cache={})

