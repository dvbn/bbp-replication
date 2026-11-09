from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from ..results import IOLSResults
from ..specs import SolverControl

if TYPE_CHECKING:
    from ..api import NormalizedInputs


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(value)


def _copy_mapping_sequence(
    value: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in value)


_MISSING = object()
PYTHON_REFERENCE_KERNEL_BACKEND: Final = "python_reference"
_COMPACT_PUBLIC_RESULTS_FLAG: Final = "PYIOLS_SPEED_COMPACT_PUBLIC_RESULTS"


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compact_public_results_enabled() -> bool:
    return _truthy_env(os.environ.get(_COMPACT_PUBLIC_RESULTS_FLAG))


def _result_sequence(value: Sequence[float]) -> Sequence[float]:
    if _compact_public_results_enabled():
        return value
    return tuple(float(item) for item in value)


def _scalar_path_value(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return _MISSING


def _compact_path_record(record: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "tau",
        "from_tau",
        "source_tau",
        "step",
        "path_target",
        "converged",
        "rho",
        "iterations",
        "restarts",
        "anderson_m_initial",
        "anderson_m_final",
        "failure_reason",
        "beta_init_source_tau",
        "rho_hint",
        "path_coefficient_status",
        "path_inference_status",
        "path_inference_reason",
    ):
        if key not in record:
            continue
        value = _scalar_path_value(record[key])
        if value is not _MISSING:
            compact[key] = value
    diagnostics = record.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        compact_diagnostics: dict[str, Any] = {}
        for key, raw_value in diagnostics.items():
            value = _scalar_path_value(raw_value)
            if value is not _MISSING:
                compact_diagnostics[str(key)] = value
        if compact_diagnostics:
            compact["diagnostics"] = compact_diagnostics
    return compact


def _compact_tau_path_payload(path_details: Mapping[str, Any]) -> dict[str, Any]:
    records = tuple(path_details.get("path", ()) or ())
    compact: dict[str, Any] = {
        "n_tau_steps": int(len(records)),
        "tau_path_summary": tuple(
            _compact_path_record(record)
            for record in records
            if isinstance(record, Mapping)
        ),
    }
    for key in (
        "path_status",
        "path_mode",
        "tau_anchor",
        "tau_target",
        "tau_targets",
        "tau_path_targets",
        "tau_attained",
        "tau_failed",
        "failure_reason",
        "target_details",
        "n_path_evaluations",
        "n_path_successes",
        "n_path_failures",
        "tau_grid_policy",
        "score_validation_policy",
        "path_coefficient_status",
        "path_inference_status",
        "path_inference_reason",
        "path_covariance_scope",
        "path_covariance_fixed_effect_policy",
    ):
        if key in path_details:
            compact[key] = path_details[key]
    if "tau_attained" not in compact:
        converged_taus = [
            float(record["tau"])
            for record in records
            if isinstance(record, Mapping)
            and bool(record.get("converged", False))
            and record.get("tau") is not None
        ]
        if converged_taus:
            compact["tau_attained"] = converged_taus[-1]
    return compact


def _apply_return_path_policy(
    payload: Mapping[str, Any],
    *,
    return_path: bool,
) -> dict[str, Any]:
    updated = _copy_mapping(payload)
    if return_path:
        return updated
    path_details = updated.get("tau_path")
    if isinstance(path_details, Mapping):
        updated.update(_compact_tau_path_payload(path_details))
    elif isinstance(updated.get("path"), Sequence):
        updated.update(_compact_tau_path_payload(updated))
    updated.pop("tau_path", None)
    updated.pop("path", None)
    return updated


@dataclass(frozen=True)
class SolverRequest:
    backend: str
    family: str
    tau_mode: str
    tau_target: float | None
    tau_range: tuple[float, float]
    tau_grid_size: int
    tau_inference: str
    vcov_type: str
    vcov_hc_type: str | None
    cluster_dims: tuple[str, ...]
    fe_method: str
    fe_dim_case: str
    fe_topology: str
    fe_names: tuple[str, ...]
    control: SolverControl
    param_names: tuple[str, ...]
    nobs: int
    return_design: bool = False
    return_path: bool = False
    routing_diagnostics: Mapping[str, Any] | None = None

    @classmethod
    def from_normalized(
        cls,
        normalized: NormalizedInputs,
        *,
        backend: str,
        nobs: int,
        param_names: Sequence[str],
    ) -> SolverRequest:
        spec = normalized.spec
        fixef = spec.fixef
        return cls(
            backend=backend,
            family=spec.family.name,
            tau_mode=spec.tau.mode,
            tau_target=spec.tau.tau_fixed,
            tau_range=spec.tau.tau_range,
            tau_grid_size=spec.tau.tau_grid_size,
            tau_inference=spec.tau.inference,
            vcov_type=spec.vcov.type,
            vcov_hc_type=spec.vcov.hc_type,
            cluster_dims=spec.vcov.cluster_dims or (),
            fe_method=fixef.effective_method,
            fe_dim_case=fixef.fe_dim_case,
            fe_topology=fixef.fe_topology,
            fe_names=fixef.names,
            control=normalized.control,
            param_names=tuple(param_names),
            nobs=int(nobs),
            return_design=normalized.return_design,
            return_path=normalized.return_path,
            routing_diagnostics=normalized.routing_diagnostics,
        )

    def tau_info_base(self) -> dict[str, Any]:
        return {
            "mode": self.tau_mode,
            "tau_target": self.tau_target,
            "tau_range": self.tau_range,
            "tau_grid_size": self.tau_grid_size,
            "tau_inference": self.tau_inference,
        }


@dataclass(frozen=True)
class SolverResponse:
    request: SolverRequest
    params: Mapping[str, float]
    bse: Mapping[str, float]
    pvalues: Mapping[str, float] = field(default_factory=dict)
    tvalues: Mapping[str, float] = field(default_factory=dict)
    fittedvalues: Sequence[float] = field(default_factory=tuple)
    resid_response: Sequence[float] = field(default_factory=tuple)
    nobs: int = 0
    df_model: int = 0
    df_resid: int = 0
    family: str = "tweedie"
    tau: float | str | None = None
    vcov_type: str = "iid"
    converged: bool = False
    fe_method: str | None = None
    fe_partition: str | None = None
    partition_warning: str | None = None
    dropped_columns: Sequence[str] = field(default_factory=tuple)
    dropped_observations: Sequence[int] = field(default_factory=tuple)
    singleton_info: Mapping[str, Any] = field(default_factory=dict)
    separation_info: Mapping[str, Any] = field(default_factory=dict)
    collinearity_info: Mapping[str, Any] = field(default_factory=dict)
    preprocess_log: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    solver_details: Mapping[str, Any] = field(default_factory=dict)
    runtime_info: Mapping[str, Any] = field(default_factory=dict)
    tau_details: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_details: Mapping[str, Any] = field(default_factory=dict)
    covariance: Sequence[Sequence[float]] | None = None

    def _kernel_backend(self) -> str:
        raw = self.solver_details.get(
            "kernel_backend",
            self.diagnostic_details.get(
                "kernel_backend",
                PYTHON_REFERENCE_KERNEL_BACKEND,
            ),
        )
        return str(raw)

    def solver_info_mapping(self) -> dict[str, Any]:
        payload = {
            "backend": self.request.backend,
            "kernel_backend": self._kernel_backend(),
        }
        payload.update(
            _apply_return_path_policy(
                self.solver_details,
                return_path=self.request.return_path,
            )
        )
        payload["auto_routing"] = self.request.routing_diagnostics
        return payload

    def tau_info_mapping(self) -> dict[str, Any]:
        payload = self.request.tau_info_base()
        payload.update(
            _apply_return_path_policy(
                self.tau_details,
                return_path=self.request.return_path,
            )
        )
        return payload

    def diagnostics_mapping(self) -> dict[str, Any]:
        payload = {
            "backend": self.request.backend,
            "kernel_backend": self._kernel_backend(),
        }
        payload.update(
            _apply_return_path_policy(
                self.diagnostic_details,
                return_path=self.request.return_path,
            )
        )
        payload["auto_routing"] = self.request.routing_diagnostics
        return payload

    def to_results(self) -> IOLSResults:
        return IOLSResults(
            params=dict(self.params),
            bse=dict(self.bse),
            pvalues=dict(self.pvalues),
            tvalues=dict(self.tvalues),
            fittedvalues=_result_sequence(self.fittedvalues),
            resid_response=_result_sequence(self.resid_response),
            nobs=int(self.nobs),
            df_model=int(self.df_model),
            df_resid=int(self.df_resid),
            family=self.family,
            tau=self.tau,
            vcov_type=self.vcov_type,
            converged=bool(self.converged),
            fe_method=self.fe_method,
            fe_partition=self.fe_partition,
            partition_warning=self.partition_warning,
            dropped_columns=tuple(self.dropped_columns),
            dropped_observations=tuple(self.dropped_observations),
            singleton_info=_copy_mapping(self.singleton_info),
            separation_info=_copy_mapping(self.separation_info),
            collinearity_info=_copy_mapping(self.collinearity_info),
            preprocess_log=_copy_mapping_sequence(self.preprocess_log),
            solver_info=self.solver_info_mapping(),
            runtime_info=_copy_mapping(self.runtime_info),
            tau_info=self.tau_info_mapping(),
            diagnostics=self.diagnostics_mapping(),
            _covariance=self.covariance,
        )
