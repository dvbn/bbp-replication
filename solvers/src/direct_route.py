"""Research-only adapter for the direct fixed-tau noFE iOLS route.

The caller supplies the already imported, patched ``pyiols.solvers.kernel``
module and, for the C++ backend, the already imported, patched
``pyiols._kernel_cpp`` extension. This module never imports, builds or switches
backends: a missing symbol, a schema mismatch or a tau mismatch raises.

Both backends run the existing two-phase iOLS (Phase 1 delta grid, then
Phase 2 rho grid) once at the requested tau. There is no tau=2 anchor, no tau
continuation and no Newton or IRLS step. Convergence flags are the solver's own
small-update criteria; certifying the score is the caller's job.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np

ADAPTER_VERSION = "direct_route.v4-bbp-curvature-rerun"
BACKENDS = ("numpy", "cpp")
NUMPY_SYMBOL = "_solve_nofe_direct_fixed_tau"
NATIVE_SYMBOL = "solve_nofe_direct_fixed_tau"
# Curvature rerun (2026-10-05): the tested combined curvature-initialized halving-step
# grid, passed explicitly to both backends. Nothing else about the route changes.
RHO_GRID_POLICY = "curvature_step_grid"
RHO_GRID_HOOK_VERSION = "rho_grid_hook_v1_2026_10_05"
_FLOAT_CONTROL_KEYS = frozenset(
    {"eps", "delta_log_min", "delta_log_max", "rho_log_min", "rho_log_max"}
)

# Controls planned for the PPML/GPML study. Every other IOLSKernelControl field
# keeps its library default (delta grid 5 points on [-5, 5], rho grid 150
# points on [-7, 14], no transformed-design normalization).
FUTURE_STUDY_CONTROL: dict[str, Any] = {
    "eps": 1e-10,
    "max_phase1_iterations": 100_000,
    "stopcrit": 1_000_000,
    "max_phase2_evaluations": 5_000_000,
    "anderson_m": 5,
}


def study_control(kernel: ModuleType, **overrides: Any) -> Any:
    """Build the study IOLSKernelControl from the supplied kernel module."""
    values = dict(FUTURE_STUDY_CONTROL)
    values.update(overrides)
    return kernel.IOLSKernelControl(**values)


def solve_direct_fixed_tau(
    *,
    kernel: ModuleType,
    native: ModuleType | None,
    backend: str,
    exog_slopes: np.ndarray,
    endog: np.ndarray,
    tau: float,
    control: Any,
) -> dict[str, Any]:
    """Solve the noFE model at exactly ``tau`` and return a common plain dict.

    ``exog_slopes`` excludes the intercept column; the intercept is profiled
    analytically and reported as ``intercept`` together with the full fitted
    index ``linear`` (``exog_slopes @ beta + intercept``).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    control_obj = _coerce_control(kernel, control)
    # The C++ mapping cannot carry transformed_design_normalization, so the
    # NumPy control validator gates both backends before dispatch.
    kernel._direct_nofe_validate_control(control_obj)
    tau_value = float(tau)

    native_backend_info: dict[str, Any] | None = None
    if backend == "numpy":
        solver = getattr(kernel, NUMPY_SYMBOL, None)
        if solver is None:
            raise RuntimeError(
                f"kernel module has no {NUMPY_SYMBOL}; the direct_fork patch is "
                "not applied"
            )
        raw = solver(
            endog=endog,
            exog_slopes=exog_slopes,
            tau=tau_value,
            control=control_obj,
            rho_grid_policy=RHO_GRID_POLICY,
        )
    else:
        if native is None:
            raise RuntimeError(
                "backend='cpp' requires the patched native extension; refusing to "
                "substitute the NumPy backend"
            )
        solver = getattr(native, NATIVE_SYMBOL, None)
        if solver is None:
            raise RuntimeError(
                f"native extension has no {NATIVE_SYMBOL}; rebuild the isolated "
                "research extension; refusing to substitute the NumPy backend"
            )
        raw = solver(
            np.ascontiguousarray(endog, dtype=np.float64),
            np.ascontiguousarray(exog_slopes, dtype=np.float64),
            tau_value,
            control=kernel._control_to_cpp_mapping(control_obj),
            rho_grid_policy=RHO_GRID_POLICY,
        )
        info = getattr(native, "backend_info", None)
        if callable(info):
            native_backend_info = dict(info())

    n_obs = int(np.asarray(endog).shape[0])
    return _normalize(
        kernel,
        raw,
        backend=backend,
        tau=tau_value,
        n_obs=n_obs,
        native_backend_info=native_backend_info,
    )


def _coerce_control(kernel: ModuleType, control: Any) -> Any:
    if isinstance(control, kernel.IOLSKernelControl):
        return control
    if isinstance(control, Mapping):
        # Unknown field names raise TypeError from the dataclass constructor.
        return kernel.IOLSKernelControl(**dict(control))
    raise TypeError("control must be an IOLSKernelControl or a mapping of its fields")


def _require_exact_keys(
    payload: Any,
    expected: tuple[str, ...],
    *,
    where: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(payload).__name__}")
    actual = set(payload)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        raise RuntimeError(f"{where} key mismatch: missing={missing} extra={extra}")
    return payload


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _normalize(
    kernel: ModuleType,
    raw: Any,
    *,
    backend: str,
    tau: float,
    n_obs: int,
    native_backend_info: dict[str, Any] | None,
) -> dict[str, Any]:
    result = _require_exact_keys(
        raw, tuple(kernel.DIRECT_NOFE_FIXED_TAU_RESULT_KEYS), where="result"
    )
    identity = {
        "schema": kernel.DIRECT_NOFE_FIXED_TAU_SCHEMA,
        "route": kernel.DIRECT_NOFE_FIXED_TAU_ROUTE,
        "algorithm": kernel.DIRECT_NOFE_FIXED_TAU_ALGORITHM,
        "backend": backend,
        "intercept_policy": "profiled_singleton_intercept",
    }
    for key, expected in identity.items():
        if result[key] != expected:
            raise RuntimeError(f"{key}={result[key]!r}, expected {expected!r}")
    for key in ("tau_requested", "tau_fit", "phase1_tau", "phase2_tau"):
        if float(result[key]) != tau:
            raise RuntimeError(f"{key}={result[key]!r} differs from requested {tau!r}")
    if result["tau2_anchor_used"] is not False:
        raise RuntimeError("direct route reported a tau=2 anchor")
    if result["tau_continuation_used"] is not False:
        raise RuntimeError("direct route reported tau continuation")
    # Present only once the Phase-2 handoff is reached; when present it must echo the policy.
    details = result["backend_details"]
    for key, expected in (("rho_grid_policy", RHO_GRID_POLICY),
                          ("rho_grid_hook_version", RHO_GRID_HOOK_VERSION)):
        if key in details and details[key] != expected:
            raise RuntimeError(f"backend_details {key}={details[key]!r}, expected {expected!r}")

    iteration_raw = _require_exact_keys(
        result["iteration"],
        tuple(kernel.DIRECT_NOFE_FIXED_TAU_ITERATION_KEYS),
        where="iteration",
    )
    diagnostics_raw = _require_exact_keys(
        result["diagnostics"],
        tuple(kernel.DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS),
        where="diagnostics",
    )
    control_raw = _require_exact_keys(
        result["control"],
        tuple(kernel.DIRECT_NOFE_FIXED_TAU_CONTROL_KEYS),
        where="control",
    )

    converged = bool(result["converged"])
    if converged != bool(iteration_raw["converged"]):
        raise RuntimeError("top-level and iteration convergence flags disagree")
    beta = np.array(result["beta"], dtype=np.float64, copy=True)
    if beta.ndim != 1:
        raise RuntimeError("beta must be one-dimensional")
    linear = None
    if result["linear"] is not None:
        linear = np.array(result["linear"], dtype=np.float64, copy=True)
        if linear.shape != (n_obs,):
            raise RuntimeError(f"linear has shape {linear.shape}, expected ({n_obs},)")
    # A native success flag with no finite index is a numerical result to audit,
    # not a schema/engineering exception. Preserve it for the independent gate.

    iteration = {
        "converged": bool(iteration_raw["converged"]),
        "phase": str(iteration_raw["phase"]),
        "iterations": int(iteration_raw["iterations"]),
        "restarts": int(iteration_raw["restarts"]),
        "final_rho": _optional_float(iteration_raw["final_rho"]),
        "rho_path": tuple(float(value) for value in iteration_raw["rho_path"]),
        "final_criterion": _optional_float(iteration_raw["final_criterion"]),
        "anderson_m_initial": int(iteration_raw["anderson_m_initial"]),
        "anderson_m_final": int(iteration_raw["anderson_m_final"]),
        "failure_reason": _optional_str(iteration_raw["failure_reason"]),
    }
    diagnostics: dict[str, Any] = {}
    for key in kernel.DIRECT_NOFE_FIXED_TAU_DIAGNOSTIC_KEYS:
        value = diagnostics_raw[key]
        if key in {
            "phase1_attempts",
            "phase1_accepted",
            "phase1_rejected",
            "phase2_evaluations",
        }:
            diagnostics[key] = int(value)
        elif key == "score_validation_status":
            diagnostics[key] = str(value)
        elif key == "score_warning_codes":
            diagnostics[key] = tuple(str(code) for code in value)
        else:
            diagnostics[key] = float(value)
    control_echo = {
        key: (str(value) if key == "direct_restart_start" else
              float(value) if key in _FLOAT_CONTROL_KEYS else int(value))
        for key, value in control_raw.items()
    }

    return {
        "adapter": ADAPTER_VERSION,
        "schema": str(result["schema"]),
        "route": str(result["route"]),
        "backend": backend,
        "kernel_backend": str(result["kernel_backend"]),
        "algorithm": str(result["algorithm"]),
        "intercept_policy": str(result["intercept_policy"]),
        "tau_requested": float(result["tau_requested"]),
        "tau_fit": float(result["tau_fit"]),
        "phase1_tau": float(result["phase1_tau"]),
        "phase2_tau": float(result["phase2_tau"]),
        "tau2_anchor_used": False,
        "tau_continuation_used": False,
        "converged": converged,
        "failure_reason": _optional_str(result["failure_reason"]),
        "beta": beta,
        "intercept": float(result["intercept"]),
        "linear": linear,
        "iteration": iteration,
        "diagnostics": diagnostics,
        "control": control_echo,
        "backend_details": dict(result["backend_details"]),
        "native_backend_info": native_backend_info,
    }
