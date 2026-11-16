from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal

FamilyName = Literal["tweedie", "poisson", "gamma", "invgauss"]
VcovType = Literal["iid", "robust", "cluster"]
AccelerationMode = Literal["anderson", "none", "auto"]
TauInferenceMode = Literal["conditional", "propagate", "auto"]
MethodFamily = Literal["canonical_iols", "phase1_tau_guide"]
TAU_BOUNDS = (0.0, 3.0)


@dataclass(frozen=True)
class FamilySpec:
    name: FamilyName
    implied_tau: float | None


@dataclass(frozen=True)
class IVSpec:
    endogenous: tuple[str, ...]
    instruments: tuple[str, ...]
    exposure: Literal["public"] = "public"


@dataclass(frozen=True)
class RouteSpec:
    method_family: MethodFamily = "canonical_iols"
    target_taus: tuple[float, ...] = ()
    phase1_anchor_tau: float = 2.0
    target_taus_input_order: tuple[float, ...] = ()


@dataclass(frozen=True)
class PartitionSpec:
    cancel: tuple[str, ...]
    recover: tuple[str, ...]
    tau_select: str | None
    mode: Literal["auto", "manual"] = "manual"

    def __post_init__(self) -> None:
        overlap = set(self.cancel).intersection(self.recover)
        if overlap:
            raise ValueError(
                "PartitionSpec cancel/recover must be disjoint, "
                f"got overlap: {sorted(overlap)!r}"
            )
        if self.tau_select is not None and self.tau_select not in self.cancel:
            raise ValueError(
                "tau_select must be None or one of the cancelled FE dimensions"
            )


@dataclass(frozen=True)
class VcovSpec:
    type: VcovType
    hc_type: Literal["HC0", "HC1", "HC2", "HC3"] | None = None
    cluster_dims: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.type == "iid" and (
            self.hc_type is not None or self.cluster_dims is not None
        ):
            raise ValueError("iid vcov does not take hc_type or cluster_dims")
        if self.type == "robust" and self.hc_type is None:
            raise ValueError("robust vcov requires an hc_type")
        if self.type == "cluster" and not self.cluster_dims:
            raise ValueError("cluster vcov requires at least one cluster dimension")


@dataclass(frozen=True)
class TauSpec:
    mode: Literal["fixed", "select", "path"]
    tau_fixed: float | None
    tau_range: tuple[float, float]
    tau_grid: tuple[float, ...] | None
    tau_grid_size: int
    inference: TauInferenceMode = "conditional"

    def __post_init__(self) -> None:
        if self.inference not in {"conditional", "propagate", "auto"}:
            raise ValueError(
                "tau_inference must be one of conditional, propagate, auto"
            )
        single_explicit_grid = (
            self.tau_grid is not None
            and len(self.tau_grid) == 1
            and self.tau_grid_size == 1
        )
        if self.tau_grid_size < 2 and not single_explicit_grid:
            raise ValueError("tau_grid_size must be at least 2")
        start, stop = self.tau_range
        if start >= stop:
            raise ValueError("tau_range must be strictly increasing")
        if start < TAU_BOUNDS[0] or stop > TAU_BOUNDS[1]:
            raise ValueError("tau_range must stay within [0.0, 3.0]")
        if self.mode == "fixed" and self.tau_fixed is None:
            raise ValueError("fixed tau mode requires tau_fixed")
        if self.mode == "fixed" and not (
            TAU_BOUNDS[0] <= self.tau_fixed <= TAU_BOUNDS[1]
        ):
            raise ValueError("tau_fixed must stay within [0.0, 3.0]")
        if self.mode == "select" and self.tau_fixed is not None:
            raise ValueError("select tau mode cannot define tau_fixed")
        if self.mode == "path" and self.tau_fixed is not None:
            raise ValueError("path tau mode cannot define tau_fixed")
        if self.mode != "select" and self.inference != "conditional":
            raise ValueError("tau_inference is only meaningful with tau='select'")


@dataclass(frozen=True)
class FixedEffectSpec:
    names: tuple[str, ...]
    method: Literal["fd", "hdfe", "auto"]
    effective_method: Literal["fd", "hdfe"]
    fe_dim_case: Literal["noFE", "oneFE", "multiFE"]
    fe_topology: Literal["noFE", "fullcancel", "mixed", "fullrecover"]
    partition: PartitionSpec | None
    tau_select_dim: str | None
    auto_rule: str | None

    def __post_init__(self) -> None:
        if self.fe_dim_case == "noFE" and self.names:
            raise ValueError("noFE dimension case cannot carry FE names")
        if self.fe_dim_case != "noFE" and not self.names:
            raise ValueError("FE specs with dimensions must carry FE names")
        if self.partition is not None and self.effective_method != "fd":
            raise ValueError("partition is only valid for fd effective_method")
        if self.fe_topology == "mixed" and self.partition is None:
            raise ValueError("mixed FE topology requires a partition")
        if self.partition is not None:
            covered = set(self.partition.cancel).union(self.partition.recover)
            expected = set(self.names)
            if covered != expected:
                missing = sorted(expected - covered)
                extra = sorted(covered - expected)
                details = []
                if missing:
                    details.append(f"missing={missing!r}")
                if extra:
                    details.append(f"extra={extra!r}")
                raise ValueError(
                    "partition must cover each FE name exactly once"
                    + (f" ({', '.join(details)})" if details else "")
                )


@dataclass(frozen=True)
class ModelSpec:
    formula: str | None
    family: FamilySpec
    fixef: FixedEffectSpec | None
    iv: IVSpec | None
    tau: TauSpec
    vcov: VcovSpec
    collinearity: Literal["drop", "error"]
    separation: Literal["auto", "none", "fe", "simplex", "relu"]
    drop_singletons: bool
    route: RouteSpec = field(default_factory=RouteSpec)


@dataclass
class ModelData:
    endog: Any
    exog: Any
    exog_names: tuple[str, ...]
    fe_blocks: Mapping[str, Any] | None
    instruments: Any
    instrument_names: tuple[str, ...] | None
    weights: Any
    offset: Any
    cluster: Mapping[str, Any] | None
    subset_mask: Any


@dataclass
class SolverControl:
    max_inner: int = 300
    tol: float = 1e-6
    accel: AccelerationMode = "auto"
    anderson_m: int = 5
    delta_grid: tuple[float, ...] | None = None
    rho_grid: tuple[float, ...] | None = None
    strict: bool = False
    verbose: int = 0
    collect_runtime: bool = False
    transformed_design_normalization: Literal["none", "scale"] = "scale"
    eps: float | None = None
    stopcrit: int | None = None
    tau_path_steps: int | None = None
    min_dtau: float | None = None
    max_path_evaluations: int | None = None
    max_phase1_iterations: int | None = None
    max_phase2_evaluations: int | None = None
    fe_profile_maxiter: int | None = None
    fe_profile_tol: float | None = None
    fe_demean_maxiter: int | None = None
    fe_demean_tol: float | None = None
    delta_log_min: float | None = None
    delta_log_max: float | None = None
    delta_grid_size: int | None = None
    rho_log_min: float | None = None
    rho_log_max: float | None = None
    rho_grid_size: int | None = None
    first_stage_f_warn: float = 10.0
    first_stage_f_auto_guardrail: float = 10.0
    first_stage_f_hard: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate public control values, including post-construction mutation."""

        def _require_integral(field_name: str, value: object) -> int:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{field_name} must be an integer")
            return int(value)

        def _require_finite_real(field_name: str, value: object) -> float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{field_name} must be finite")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{field_name} must be finite")
            return result

        if self.accel not in {"anderson", "none", "auto"}:
            raise ValueError("accel must be one of 'anderson', 'none', or 'auto'")
        if self.delta_grid is not None:
            raise ValueError(
                "delta_grid is not supported; use delta_log_min, "
                "delta_log_max, and delta_grid_size"
            )
        if self.rho_grid is not None:
            raise ValueError(
                "rho_grid is not supported; use rho_log_min, "
                "rho_log_max, and rho_grid_size"
            )

        max_inner = _require_integral("max_inner", self.max_inner)
        if max_inner <= 0:
            raise ValueError("max_inner must be positive")
        tol = _require_finite_real("tol", self.tol)
        if tol <= 0:
            raise ValueError("tol must be positive")
        anderson_m = _require_integral("anderson_m", self.anderson_m)
        if anderson_m < 0:
            raise ValueError("anderson_m must be non-negative")

        positive_float_fields = (
            "eps",
            "min_dtau",
            "fe_profile_tol",
            "fe_demean_tol",
            "first_stage_f_warn",
            "first_stage_f_auto_guardrail",
            "first_stage_f_hard",
        )
        for field_name in positive_float_fields:
            value = getattr(self, field_name)
            if value is not None:
                real_value = _require_finite_real(field_name, value)
                if real_value <= 0:
                    raise ValueError(f"{field_name} must be positive")
        positive_int_fields = (
            "stopcrit",
            "tau_path_steps",
            "max_path_evaluations",
            "max_phase1_iterations",
            "max_phase2_evaluations",
            "fe_profile_maxiter",
            "fe_demean_maxiter",
        )
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if value is not None:
                int_value = _require_integral(field_name, value)
                if int_value <= 0:
                    raise ValueError(f"{field_name} must be positive")
        if self.delta_grid_size is not None:
            delta_grid_size = _require_integral(
                "delta_grid_size",
                self.delta_grid_size,
            )
            if delta_grid_size < 0:
                raise ValueError("delta_grid_size must be non-negative")
        if self.rho_grid_size is not None:
            rho_grid_size = _require_integral("rho_grid_size", self.rho_grid_size)
            if rho_grid_size < 2:
                raise ValueError("rho_grid_size must be at least 2")
        log_range_fields = (
            "delta_log_min",
            "delta_log_max",
            "rho_log_min",
            "rho_log_max",
        )
        for field_name in log_range_fields:
            value = getattr(self, field_name)
            if value is not None:
                _require_finite_real(field_name, value)
        if (
            self.delta_log_min is not None
            and self.delta_log_max is not None
            and float(self.delta_log_max) <= float(self.delta_log_min)
        ):
            raise ValueError("delta_log_max must be greater than delta_log_min")
        if (
            self.rho_log_min is not None
            and self.rho_log_max is not None
            and float(self.rho_log_max) <= float(self.rho_log_min)
        ):
            raise ValueError("rho_log_max must be greater than rho_log_min")
        if self.transformed_design_normalization not in {"none", "scale"}:
            raise ValueError(
                "transformed_design_normalization must be one of 'none' or 'scale'"
            )

    def kernel_control_kwargs(self, *, anderson_m: int | None = None) -> dict[str, Any]:
        """Return public solver controls in package kernel keyword form."""

        self.validate()
        effective_anderson_m = self.anderson_m if anderson_m is None else anderson_m
        kwargs: dict[str, Any] = {
            "anderson_m": int(effective_anderson_m),
            "eps": float(self.tol if self.eps is None else self.eps),
            "transformed_design_normalization": self.transformed_design_normalization,
        }
        if self.stopcrit is not None:
            kwargs["stopcrit"] = int(self.stopcrit)
        elif self.max_inner != 300:
            kwargs["stopcrit"] = int(self.max_inner)
        optional_fields = (
            "tau_path_steps",
            "min_dtau",
            "max_path_evaluations",
            "max_phase1_iterations",
            "max_phase2_evaluations",
            "fe_profile_maxiter",
            "fe_profile_tol",
            "fe_demean_maxiter",
            "fe_demean_tol",
            "delta_log_min",
            "delta_log_max",
            "delta_grid_size",
            "rho_log_min",
            "rho_log_max",
            "rho_grid_size",
        )
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                kwargs[field_name] = value
        return kwargs
