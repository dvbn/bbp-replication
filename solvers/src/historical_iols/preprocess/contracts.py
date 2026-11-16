from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _freeze_pairs(mapping: Mapping[str, int] | None) -> tuple[tuple[str, int], ...]:
    if not mapping:
        return ()
    return tuple((str(key), int(value)) for key, value in mapping.items())


def _mapping_from_pairs(
    pairs: tuple[tuple[str, int], ...],
) -> dict[str, int]:
    return {key: value for key, value in pairs}


def _freeze_details(details: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(key), _freeze_value(value)) for key, value in details.items())


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(_freeze_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _freeze_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def fe_level_counts(fe_blocks: Mapping[str, Sequence[Any]] | None) -> dict[str, int]:
    if not fe_blocks:
        return {}
    counts: dict[str, int] = {}
    for name, values in fe_blocks.items():
        array = np.asarray(values)
        counts[str(name)] = int(np.unique(array).size)
    return counts


def dropped_observations_from_mask(
    kept_obs_mask: Sequence[bool] | np.ndarray,
) -> tuple[int, ...]:
    mask = np.asarray(kept_obs_mask, dtype=bool)
    return tuple(int(index) for index in np.flatnonzero(~mask))


@dataclass(frozen=True)
class PreprocessStepRecord:
    step: str
    status: str
    details: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        *,
        step: str,
        status: str,
        details: Mapping[str, Any] | None = None,
    ) -> PreprocessStepRecord:
        return cls(
            step=step,
            status=status,
            details=() if details is None else _freeze_details(details),
        )

    def as_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {"step": self.step, "status": self.status}
        mapping.update(dict(self.details))
        return mapping


@dataclass(frozen=True)
class PreprocessSampleAccounting:
    n_obs_in: int
    n_obs_out: int
    dropped_observations: tuple[int, ...] = ()
    dropped_columns: tuple[str, ...] = ()
    kept_obs_mask: tuple[bool, ...] = ()
    kept_x_idx: tuple[int, ...] = ()
    dropped_x_idx: tuple[int, ...] = ()
    intercept_idx: int | None = None

    @property
    def dropped_observation_count(self) -> int:
        return len(self.dropped_observations)

    @property
    def dropped_column_count(self) -> int:
        return len(self.dropped_columns)


@dataclass(frozen=True)
class FESupportSummary:
    fe_levels_before_pairs: tuple[tuple[str, int], ...] = ()
    fe_levels_after_pairs: tuple[tuple[str, int], ...] = ()
    zero_positive_levels_before_pairs: tuple[tuple[str, int], ...] = ()
    zero_sum_levels_before_pairs: tuple[tuple[str, int], ...] = ()
    zero_positive_levels_after_pairs: tuple[tuple[str, int], ...] = ()
    zero_sum_levels_after_pairs: tuple[tuple[str, int], ...] = ()

    @property
    def fe_levels_before(self) -> dict[str, int]:
        return _mapping_from_pairs(self.fe_levels_before_pairs)

    @property
    def fe_levels_after(self) -> dict[str, int]:
        return _mapping_from_pairs(self.fe_levels_after_pairs)

    @property
    def zero_positive_levels_before(self) -> dict[str, int]:
        return _mapping_from_pairs(self.zero_positive_levels_before_pairs)

    @property
    def zero_sum_levels_before(self) -> dict[str, int]:
        return _mapping_from_pairs(self.zero_sum_levels_before_pairs)

    @property
    def zero_positive_levels_after(self) -> dict[str, int]:
        return _mapping_from_pairs(self.zero_positive_levels_after_pairs)

    @property
    def zero_sum_levels_after(self) -> dict[str, int]:
        return _mapping_from_pairs(self.zero_sum_levels_after_pairs)


@dataclass(frozen=True)
class SingletonDiagnosticsContract:
    checked: bool
    has_singletons: bool
    singleton_rounds: int = 0
    singleton_dropped: int = 0
    count_screen_rounds: int = 0
    count_screen_dropped: int = 0
    fe_name: str | None = None
    fe_names: tuple[str, ...] = ()
    policy: str | None = None

    def as_result_mapping(
        self,
        *,
        include_rounds: bool = False,
        include_fe_names: bool = False,
        include_policy: bool = False,
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "checked": self.checked,
            "has_singletons": self.has_singletons,
        }
        if include_rounds:
            mapping["singleton_rounds"] = self.singleton_rounds
            mapping["singleton_dropped"] = self.singleton_dropped
        if self.fe_name is not None:
            mapping["fe_name"] = self.fe_name
        if include_fe_names and self.fe_names:
            mapping["fe_names"] = self.fe_names
        if include_policy and self.policy is not None:
            mapping["policy"] = self.policy
        return mapping


@dataclass(frozen=True)
class CollinearityDiagnosticsContract:
    mode: str
    status: str
    dropped_columns: tuple[str, ...] = ()
    duplicate_constant_dropped: bool = False
    regressor_collinearity_dropped: bool = False
    transformed_collinearity_dropped: bool = False

    def as_result_mapping(
        self,
        *,
        include_drop_flags: bool = False,
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "mode": self.mode,
            "status": self.status,
        }
        if self.dropped_columns:
            mapping["dropped_columns"] = self.dropped_columns
        if include_drop_flags:
            mapping["transformed_collinearity_dropped"] = (
                self.transformed_collinearity_dropped
            )
            mapping["duplicate_constant_dropped"] = (
                self.duplicate_constant_dropped
            )
            mapping["regressor_collinearity_dropped"] = (
                self.regressor_collinearity_dropped
            )
        return mapping


@dataclass(frozen=True)
class PreprocessContract:
    contract_version: str = "stage3_packet_a_v1"
    sample: PreprocessSampleAccounting = field(
        default_factory=lambda: PreprocessSampleAccounting(
            n_obs_in=0,
            n_obs_out=0,
        )
    )
    fe_support: FESupportSummary = field(default_factory=FESupportSummary)
    singleton: SingletonDiagnosticsContract = field(
        default_factory=lambda: SingletonDiagnosticsContract(
            checked=False,
            has_singletons=False,
        )
    )
    collinearity: CollinearityDiagnosticsContract = field(
        default_factory=lambda: CollinearityDiagnosticsContract(
            mode="drop",
            status="not_applied",
        )
    )
    steps: tuple[PreprocessStepRecord, ...] = ()

    def preprocess_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(step.as_mapping() for step in self.steps)


def scope_only_contract(
    *,
    step_name: str,
    n_obs: int,
    status: str,
    fe_blocks: Mapping[str, Sequence[Any]] | None = None,
    step_details: Mapping[str, Any] | None = None,
    singleton_checked: bool,
    has_singletons: bool,
    fe_name: str | None = None,
    fe_names: Sequence[str] = (),
    policy: str | None = None,
    collinearity_mode: str,
    collinearity_status: str,
) -> PreprocessContract:
    levels = fe_level_counts(fe_blocks)
    return PreprocessContract(
        sample=PreprocessSampleAccounting(
            n_obs_in=int(n_obs),
            n_obs_out=int(n_obs),
        ),
        fe_support=FESupportSummary(
            fe_levels_before_pairs=_freeze_pairs(levels),
            fe_levels_after_pairs=_freeze_pairs(levels),
        ),
        singleton=SingletonDiagnosticsContract(
            checked=singleton_checked,
            has_singletons=has_singletons,
            fe_name=fe_name,
            fe_names=tuple(str(name) for name in fe_names),
            policy=policy,
        ),
        collinearity=CollinearityDiagnosticsContract(
            mode=collinearity_mode,
            status=collinearity_status,
        ),
        steps=(
            PreprocessStepRecord.from_mapping(
                step=step_name,
                status=status,
                details=step_details,
            ),
        ),
    )


def partitioned_research_contract(
    *,
    prep_meta: Any,
    diagnostics: Any,
    partition_case: str,
    research_backend: str,
    dropped_columns: Sequence[str],
) -> PreprocessContract:
    return PreprocessContract(
        sample=PreprocessSampleAccounting(
            n_obs_in=int(prep_meta.n_obs_in),
            n_obs_out=int(prep_meta.n_obs_out),
            dropped_observations=dropped_observations_from_mask(prep_meta.kept_obs_mask),
            dropped_columns=tuple(str(name) for name in dropped_columns),
            kept_obs_mask=tuple(
                bool(value) for value in np.asarray(prep_meta.kept_obs_mask)
            ),
            kept_x_idx=tuple(int(index) for index in np.asarray(prep_meta.kept_x_idx)),
            dropped_x_idx=tuple(
                int(index) for index in np.asarray(prep_meta.dropped_x_idx)
            ),
            intercept_idx=(
                None
                if prep_meta.intercept_idx is None
                else int(prep_meta.intercept_idx)
            ),
        ),
        fe_support=FESupportSummary(
            fe_levels_before_pairs=_freeze_pairs(prep_meta.fe_levels_before),
            fe_levels_after_pairs=_freeze_pairs(prep_meta.fe_levels_after),
            zero_positive_levels_before_pairs=_freeze_pairs(
                getattr(prep_meta, "zero_positive_levels_before", None)
            ),
            zero_sum_levels_before_pairs=_freeze_pairs(
                getattr(prep_meta, "zero_sum_levels_before", None)
            ),
            zero_positive_levels_after_pairs=_freeze_pairs(
                getattr(prep_meta, "zero_positive_levels", None)
            ),
            zero_sum_levels_after_pairs=_freeze_pairs(
                getattr(prep_meta, "zero_sum_levels", None)
            ),
        ),
        singleton=SingletonDiagnosticsContract(
            checked=True,
            has_singletons=int(prep_meta.singleton_dropped) > 0,
            singleton_rounds=int(prep_meta.singleton_rounds),
            singleton_dropped=int(prep_meta.singleton_dropped),
            count_screen_rounds=int(prep_meta.count_screen_rounds),
            count_screen_dropped=int(prep_meta.count_screen_dropped),
        ),
        collinearity=CollinearityDiagnosticsContract(
            mode="drop",
            status="preprocess_drop",
            dropped_columns=tuple(str(name) for name in dropped_columns),
            duplicate_constant_dropped=bool(
                getattr(diagnostics, "duplicate_constant_dropped", False)
            ),
            regressor_collinearity_dropped=bool(
                getattr(diagnostics, "regressor_collinearity_dropped", False)
            ),
            transformed_collinearity_dropped=bool(
                getattr(diagnostics, "transformed_collinearity_dropped", False)
            ),
        ),
        steps=(
            PreprocessStepRecord.from_mapping(
                step="fd_partitioned_scope_check",
                status="pass",
                details={
                    "partition_case": partition_case,
                    "research_backend": research_backend,
                },
            ),
            PreprocessStepRecord.from_mapping(
                step="partitioned_preprocess",
                status=str(diagnostics.status),
                details={
                    "n_obs_in": int(prep_meta.n_obs_in),
                    "n_obs_out": int(prep_meta.n_obs_out),
                    "singleton_dropped": int(prep_meta.singleton_dropped),
                    "count_screen_dropped": int(prep_meta.count_screen_dropped),
                    "dropped_x_idx": tuple(
                        int(index) for index in np.asarray(prep_meta.dropped_x_idx)
                    ),
                },
            ),
        ),
    )


def fullrecover_research_contract(
    *,
    prep_meta: Any,
    fe_names: Sequence[str],
    research_backend: str,
) -> PreprocessContract:
    ordered_names = tuple(str(name) for name in fe_names)

    def _ordered_pairs(
        mapping: Mapping[str, int] | None,
    ) -> tuple[tuple[str, int], ...]:
        mapping = {} if mapping is None else mapping
        return tuple((name, int(mapping.get(name, 0))) for name in ordered_names)

    dropped_observations = dropped_observations_from_mask(prep_meta.kept_obs_mask)
    dropped_obs_count = len(dropped_observations)
    singleton_dropped = int(prep_meta.singleton_dropped)
    count_screen_dropped = int(prep_meta.count_screen_dropped)
    preprocess_status = (
        "drop"
        if dropped_obs_count > 0 or singleton_dropped > 0 or count_screen_dropped > 0
        else "pass"
    )

    return PreprocessContract(
        sample=PreprocessSampleAccounting(
            n_obs_in=int(prep_meta.n_obs_in),
            n_obs_out=int(prep_meta.n_obs_out),
            dropped_observations=dropped_observations,
            kept_obs_mask=tuple(
                bool(value) for value in np.asarray(prep_meta.kept_obs_mask)
            ),
            kept_x_idx=tuple(int(index) for index in np.asarray(prep_meta.kept_x_idx)),
            dropped_x_idx=tuple(
                int(index) for index in np.asarray(prep_meta.dropped_x_idx)
            ),
            intercept_idx=(
                None
                if prep_meta.intercept_idx is None
                else int(prep_meta.intercept_idx)
            ),
        ),
        fe_support=FESupportSummary(
            fe_levels_before_pairs=_ordered_pairs(prep_meta.fe_levels_before),
            fe_levels_after_pairs=_ordered_pairs(prep_meta.fe_levels_after),
            zero_positive_levels_before_pairs=_ordered_pairs(
                getattr(prep_meta, "zero_positive_levels_before", None)
            ),
            zero_sum_levels_before_pairs=_ordered_pairs(
                getattr(prep_meta, "zero_sum_levels_before", None)
            ),
            zero_positive_levels_after_pairs=_ordered_pairs(
                getattr(prep_meta, "zero_positive_levels", None)
            ),
            zero_sum_levels_after_pairs=_ordered_pairs(
                getattr(prep_meta, "zero_sum_levels", None)
            ),
        ),
        singleton=SingletonDiagnosticsContract(
            checked=True,
            has_singletons=singleton_dropped > 0,
            singleton_rounds=int(prep_meta.singleton_rounds),
            singleton_dropped=singleton_dropped,
            count_screen_rounds=int(prep_meta.count_screen_rounds),
            count_screen_dropped=count_screen_dropped,
            fe_names=ordered_names,
            policy="drop_if_present",
        ),
        collinearity=CollinearityDiagnosticsContract(
            mode="drop",
            status="not_applied",
        ),
        steps=(
            PreprocessStepRecord.from_mapping(
                step="fullrecover_iols_scope_check",
                status="pass",
                details={
                    "research_backend": str(research_backend),
                },
            ),
            PreprocessStepRecord.from_mapping(
                step="fullrecover_preprocess",
                status=preprocess_status,
                details={
                    "n_obs_in": int(prep_meta.n_obs_in),
                    "n_obs_out": int(prep_meta.n_obs_out),
                    "dropped_observation_count": dropped_obs_count,
                    "singleton_rounds": int(prep_meta.singleton_rounds),
                    "singleton_dropped": singleton_dropped,
                    "count_screen_rounds": int(prep_meta.count_screen_rounds),
                    "count_screen_dropped": count_screen_dropped,
                },
            ),
        ),
    )
