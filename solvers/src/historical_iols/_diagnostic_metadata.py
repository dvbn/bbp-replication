from __future__ import annotations

from typing import Any, Final

KNOWN_FE_RUNTIME_FAILURES: Final[frozenset[str]] = frozenset(
    {
        "manyway_fe_demean_not_converged",
        "manyway_fe_profile_not_converged",
        "twoway_fe_profile_not_converged",
    }
)


def runtime_failure_reason(exc: BaseException) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__


def is_known_fe_runtime_failure(exc: BaseException) -> bool:
    return runtime_failure_reason(exc) in KNOWN_FE_RUNTIME_FAILURES


def phase1_convergence_metadata(
    *,
    endpoint_kernel_converged: bool,
    public_converged: bool,
    phase1_selection_had_converged_targets: bool,
) -> dict[str, Any]:
    endpoint = bool(endpoint_kernel_converged)
    public = bool(public_converged)
    return {
        "endpoint_kernel_converged": endpoint,
        "public_converged": public,
        "phase1_status_overrode_kernel_convergence": bool(endpoint and not public),
        "phase1_selection_had_converged_targets": bool(
            phase1_selection_had_converged_targets
        ),
    }
