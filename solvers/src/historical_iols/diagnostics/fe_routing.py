from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..specs import FixedEffectSpec, VcovSpec


def _counts_for_levels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse)
    return inverse.astype(int, copy=False), counts.astype(int, copy=False)


def _ordered_pairs(
    names: Sequence[str],
    mapping: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    return tuple((str(name), int(mapping.get(str(name), 0))) for name in names)


@dataclass(frozen=True)
class FESupportDiagnostics:
    fe_names: tuple[str, ...]
    fe_levels_pairs: tuple[tuple[str, int], ...]
    singleton_levels_pairs: tuple[tuple[str, int], ...]
    zero_positive_levels_pairs: tuple[tuple[str, int], ...]
    zero_sum_levels_pairs: tuple[tuple[str, int], ...]

    @property
    def has_singletons(self) -> bool:
        return any(count > 0 for _, count in self.singleton_levels_pairs)

    @property
    def has_zero_positive_levels(self) -> bool:
        return any(count > 0 for _, count in self.zero_positive_levels_pairs)

    @property
    def has_zero_sum_levels(self) -> bool:
        return any(count > 0 for _, count in self.zero_sum_levels_pairs)

    def as_mapping(self) -> dict[str, Any]:
        fe_levels = dict(self.fe_levels_pairs)
        singleton_levels = dict(self.singleton_levels_pairs)
        zero_positive_levels = dict(self.zero_positive_levels_pairs)
        zero_sum_levels = dict(self.zero_sum_levels_pairs)
        return {
            "fe_names": self.fe_names,
            "fe_levels": fe_levels,
            "singleton_levels": singleton_levels,
            "zero_positive_levels": zero_positive_levels,
            "zero_sum_levels": zero_sum_levels,
            "has_singletons": any(singleton_levels.values()),
            "has_zero_positive_levels": any(zero_positive_levels.values()),
            "has_zero_sum_levels": any(zero_sum_levels.values()),
        }


@dataclass(frozen=True)
class AutoFERoutingDecision:
    effective_method: str
    effective_topology: str
    auto_rule: str
    status: str
    reason_codes: tuple[str, ...]
    message: str | None
    fe_support: FESupportDiagnostics

    @property
    def promoted(self) -> bool:
        return self.effective_method == "hdfe"

    def log_step(self) -> dict[str, Any]:
        step = {
            "step": "fe_auto_routing",
            "status": self.status,
            "auto_rule": self.auto_rule,
            "effective_method": self.effective_method,
            "effective_topology": self.effective_topology,
            "promoted": self.promoted,
            "reason_codes": self.reason_codes,
        }
        if self.message is not None:
            step["message"] = self.message
        return step

    def as_mapping(self) -> dict[str, Any]:
        return {
            "requested_method": "auto",
            "effective_method": self.effective_method,
            "effective_topology": self.effective_topology,
            "auto_rule": self.auto_rule,
            "status": self.status,
            "promoted": self.promoted,
            "reason_codes": self.reason_codes,
            "message": self.message,
            "fe_support": self.fe_support.as_mapping(),
            "log_step": self.log_step(),
        }


def diagnose_fe_support(
    endog: Any,
    fe_blocks: Mapping[str, Sequence[Any]] | None,
) -> FESupportDiagnostics:
    if not fe_blocks:
        return FESupportDiagnostics(
            fe_names=(),
            fe_levels_pairs=(),
            singleton_levels_pairs=(),
            zero_positive_levels_pairs=(),
            zero_sum_levels_pairs=(),
        )

    y = np.asarray(endog, dtype=float)
    if y.ndim != 1:
        raise ValueError("endog must be one-dimensional for FE support diagnostics")

    fe_names = tuple(str(name) for name in fe_blocks.keys())
    fe_levels: dict[str, int] = {}
    singleton_levels: dict[str, int] = {}
    zero_positive_levels: dict[str, int] = {}
    zero_sum_levels: dict[str, int] = {}
    positive = np.asarray(y > 0.0, dtype=float)

    for name, values in fe_blocks.items():
        array = np.asarray(values)
        if array.shape[0] != y.shape[0]:
            raise ValueError(f"{name!r} has a different length than endog")
        inverse, counts = _counts_for_levels(array)
        fe_levels[str(name)] = int(counts.size)
        singleton_levels[str(name)] = int(np.sum(counts == 1))
        pos_counts = np.bincount(
            inverse,
            weights=positive,
            minlength=int(counts.size),
        )
        sum_counts = np.bincount(
            inverse,
            weights=y,
            minlength=int(counts.size),
        )
        zero_positive_levels[str(name)] = int(np.sum(pos_counts <= 0))
        zero_sum_levels[str(name)] = int(np.sum(sum_counts <= 0))

    return FESupportDiagnostics(
        fe_names=fe_names,
        fe_levels_pairs=_ordered_pairs(fe_names, fe_levels),
        singleton_levels_pairs=_ordered_pairs(fe_names, singleton_levels),
        zero_positive_levels_pairs=_ordered_pairs(fe_names, zero_positive_levels),
        zero_sum_levels_pairs=_ordered_pairs(fe_names, zero_sum_levels),
    )


_PUBLIC_FULLRECOVER_FIXED_TAU_VALUES = (0.0, 1.0, 1.5, 2.0, 3.0)


def _is_public_fullrecover_tau(
    *,
    tau_mode: str,
    tau_fixed: float | None,
) -> bool:
    if tau_mode in {"select", "path"}:
        return True
    if tau_mode != "fixed" or tau_fixed is None:
        return False
    return any(
        abs(float(tau_fixed) - value) <= 1e-12
        for value in _PUBLIC_FULLRECOVER_FIXED_TAU_VALUES
    )


def _is_public_fullrecover_vcov(
    *,
    vcov_spec: VcovSpec,
    fe_count: int,
) -> bool:
    if vcov_spec.type == "iid":
        return True
    if fe_count <= 2:
        if vcov_spec.type == "cluster":
            return len(vcov_spec.cluster_dims or ()) == 1
        return (
            vcov_spec.type == "robust"
            and vcov_spec.hc_type in {"HC0", "HC1", "HC2", "HC3"}
        )
    if fe_count <= 4:
        return vcov_spec.type == "robust" and vcov_spec.hc_type == "HC0"
    return False


def resolve_auto_fe_routing(
    *,
    endog: Any,
    fixef_spec: FixedEffectSpec,
    fe_blocks: Mapping[str, Sequence[Any]] | None,
    tau_mode: str,
    tau_fixed: float | None,
    vcov_spec: VcovSpec,
) -> AutoFERoutingDecision | None:
    if fixef_spec.method != "auto" or not fixef_spec.names:
        return None
    if fixef_spec.partition is None:
        return None
    if fe_blocks is None:
        raise NotImplementedError(
            "Auto FE routing is only wired for array-mode FE blocks "
            "with explicit values."
        )

    fe_support = diagnose_fe_support(endog=endog, fe_blocks=fe_blocks)
    if fixef_spec.partition.mode != "auto":
        return AutoFERoutingDecision(
            effective_method="fd",
            effective_topology=fixef_spec.fe_topology,
            auto_rule="p0_s28_auto_manual_partition_preserved_fd",
            status="info",
            reason_codes=("manual_partition_present",),
            message=(
                "fe_method='auto' preserved the user-supplied FD FE partition "
                "and did not apply automatic fullrecover promotion."
            ),
            fe_support=fe_support,
        )

    reason_codes: list[str] = []
    fe_count = len(fixef_spec.names)
    if not _is_public_fullrecover_tau(
        tau_mode=tau_mode,
        tau_fixed=tau_fixed,
    ):
        reason_codes.append("fixed_tau_outside_fullrecover_scope")
    if fe_count > 4:
        reason_codes.append("more_than_four_fe_dimensions")
    if not _is_public_fullrecover_vcov(vcov_spec=vcov_spec, fe_count=fe_count):
        reason_codes.append("vcov_outside_fullrecover_scope")
    if fe_support.has_singletons:
        reason_codes.append("singleton_fe_levels_present")
    if fe_support.has_zero_positive_levels:
        reason_codes.append("zero_positive_support_detected")
    if fe_support.has_zero_sum_levels:
        reason_codes.append("zero_sum_support_detected")

    if not reason_codes:
        return AutoFERoutingDecision(
            effective_method="hdfe",
            effective_topology="fullrecover",
            auto_rule="p0_s28_auto_fullrecover_public_scope_whitelist",
            status="pass",
            reason_codes=(
                "public_fullrecover_scope",
                "strong_fe_support",
            ),
            message=(
                "fe_method='auto' promoted to public hdfe/fullrecover because "
                "the tau mode, FE count, covariance request, and FE support "
                "are inside the audited public fullrecover_iols scope."
            ),
            fe_support=fe_support,
        )

    return AutoFERoutingDecision(
        effective_method="fd",
        effective_topology="fullcancel",
        auto_rule="p0_s28_auto_fd_fullcancel_fallback_guardrail",
        status="warn",
        reason_codes=tuple(reason_codes),
        message=(
            "fe_method='auto' stayed on fd/fullcancel because the requested "
            "case is outside the automatic public fullrecover promotion "
            f"scope: {', '.join(reason_codes)}."
        ),
        fe_support=fe_support,
    )
