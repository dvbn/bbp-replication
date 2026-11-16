from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any


def _ordered_names(mapping: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(mapping.keys())


@dataclass
class IOLSResults:
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
    solver_info: Mapping[str, Any] = field(default_factory=dict)
    runtime_info: Mapping[str, Any] = field(default_factory=dict)
    tau_info: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    _covariance: Sequence[Sequence[float]] | None = None
    design: Any | None = None

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        """Canonical nonconvergence reasons, deduped in source priority order."""
        if self.converged:
            return ()
        reasons: list[str] = []

        def add_reason(value: Any) -> None:
            if value is None:
                return
            text = str(value)
            if text and text not in reasons:
                reasons.append(text)

        def add_many(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                add_reason(value)
                return
            try:
                iterator = iter(value)
            except TypeError:
                add_reason(value)
                return
            for item in iterator:
                add_reason(item)

        sources = tuple(
            source
            for source in (self.solver_info, self.diagnostics, self.runtime_info)
            if isinstance(source, Mapping)
        )
        for source in sources:
            add_reason(source.get("failure_reason"))

        if isinstance(self.solver_info, Mapping):
            tau_path = self.solver_info.get("tau_path")
            if isinstance(tau_path, Mapping):
                add_reason(tau_path.get("failure_reason"))

        for source in sources:
            add_reason(source.get("inference_reason"))
            add_many(source.get("inference_reasons"))

        return tuple(reasons)

    @property
    def failure_reason(self) -> str | None:
        """Primary nonconvergence reason, or ``None`` for converged results."""
        reasons = self.failure_reasons
        return reasons[0] if reasons else None

    def _metadata_value(self, key: str) -> Any:
        """Return public metadata using the same solver-last precedence as IV views."""

        value = None
        for source in (self.diagnostics, self.solver_info):
            if isinstance(source, Mapping) and key in source:
                value = source[key]
        return value

    def _metadata_value_any(self, keys: tuple[str, ...]) -> Any:
        """Return public metadata for aliases using solver-last precedence."""

        value = None
        for source in (self.diagnostics, self.solver_info):
            if not isinstance(source, Mapping):
                continue
            for key in keys:
                if key in source:
                    value = source[key]
                    break
        return value

    @property
    def inference_status(self) -> str | None:
        """Public inference status when covariance/inference metadata is present."""

        value = self._metadata_value("inference_status")
        return None if value is None else str(value)

    @property
    def inference_available(self) -> bool:
        """Whether the result exposes usable coefficient-level inference."""

        status = self.inference_status
        if status not in {"available", "available_provisional"}:
            return False
        explicit = self._metadata_value("inference_available")
        if explicit is not None:
            return bool(explicit)
        return True

    @property
    def inference_reason(self) -> str | None:
        """Reason associated with the public inference status, when present."""

        value = self._metadata_value("inference_reason")
        return None if value is None else str(value)

    @property
    def inference_withheld_reason(self) -> str | None:
        """Reason inference was withheld, or ``None`` when not withheld."""

        value = self._metadata_value("inference_withheld_reason")
        return None if value is None else str(value)

    @property
    def cov_source(self) -> str | None:
        """Canonical covariance source tag, when the result carries one."""

        value = self._metadata_value_any(("cov_source", "covariance_source"))
        return None if value is None else str(value)

    @property
    def details(self) -> dict[str, Any]:
        """Compatibility metadata view used by public evidence/reporting code.

        The solver and diagnostic mappings are already public. This property
        gives read-only consumers one filtered namespace without exposing all
        solver bookkeeping; later sources keep the same solver-last precedence
        used by the IV accessors.
        """

        explicit = {
            "blocked_surface",
            "cov_source",
            "covariance_source",
            "failure_reason",
            "fe_dim_case",
            "guardrail_refusal",
            "inference_available",
            "inference_reason",
            "inference_status",
            "inference_withheld_reason",
            "iv_status",
            "method_family",
            "mgd_state_signature",
            "n_fe_dims",
            "package_route",
            "phase1_endpoint_covariance_copy_failure_reason",
            "phase1_endpoint_covariance_copy_status",
            "phase1_endpoint_covariance_status",
            "route",
            "tau_mode_blocked",
            "topology",
        }
        prefixes = (
            "first_stage_f_auto_guardrail_",
            "rank_preselection_",
            "support_failure_",
            "tau_path_",
            "tau_selection_",
            "weak_iv_hard_block_",
        )
        payload: dict[str, Any] = {}
        for source in (self.diagnostics, self.solver_info):
            if isinstance(source, Mapping):
                for key, value in source.items():
                    key_text = str(key)
                    if key_text in explicit or key_text.startswith(prefixes):
                        payload[key_text] = value
        return payload

    @property
    def iv_diagnostics(self) -> dict[str, Any]:
        """Return public i2SLS diagnostic statistics when present."""

        if not any(
            isinstance(source, Mapping) and source.get("iv_status") == "i2sls"
            for source in (self.solver_info, self.diagnostics)
        ):
            return {}
        prefixes = (
            "first_stage_",
            "cragg_",
            "kleibergen_",
            "sargan_",
            "weak_iv_",
            "algorithm11_",
            "overid_",
            "overidentification_",
            "identification_rank_",
        )
        explicit = {
            "iv_status",
            "method_family",
            "endogenous_names",
            "instrument_names",
            "inference_status",
            "inference_reason",
            "inference_withheld_reason",
            "cov_source",
            "failure_reason",
        }
        payload: dict[str, Any] = {}
        for source in (self.diagnostics, self.solver_info):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():
                key_text = str(key)
                if key_text in explicit or key_text.startswith(prefixes):
                    payload[key_text] = value
        return payload

    @property
    def iv_cov_diagnostics(self) -> dict[str, Any]:
        """Return public i2SLS covariance diagnostics when present."""

        if not any(
            isinstance(source, Mapping) and source.get("iv_status") == "i2sls"
            for source in (self.solver_info, self.diagnostics)
        ):
            return {}
        prefixes = (
            "algorithm10_",
            "covariance_",
            "dof_",
            "df_resid_",
        )
        explicit = {
            "cov_source",
            "covariance_source",
            "inference_status",
            "inference_reason",
            "inference_withheld_reason",
            "inference_warning",
        }
        payload: dict[str, Any] = {}
        for source in (self.diagnostics, self.solver_info):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():
                key_text = str(key)
                if key_text in explicit or key_text.startswith(prefixes):
                    payload[key_text] = value
        return payload

    @property
    def path_diagnostics(self) -> dict[float, dict[str, Any]]:
        """Return stored per-tau path diagnostics keyed by tau when available."""

        payload: dict[float, dict[str, Any]] = {}
        for record in _result_tau_path_records(self):
            tau = _coerce_float(record.get("tau"))
            if not math.isfinite(tau):
                continue
            diagnostics = record.get("diagnostics")
            item: dict[str, Any] = (
                dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
            )
            for key in (
                "converged",
                "failure_reason",
                "iterations",
                "rho",
                "path_target",
                "path_inference_status",
                "path_inference_reason",
            ):
                if key in record:
                    item[key] = record[key]
            payload[float(tau)] = item
        return payload

    def _nonconvergence_warning_message(self) -> str | None:
        if self.converged:
            return None
        reasons = self.failure_reasons
        if not reasons:
            return None
        joined = ", ".join(reasons)
        return (
            "Result did not converge; estimates and inference should be treated "
            f"as diagnostic only (reason: {joined})."
        )

    def _warning_messages(self) -> tuple[str, ...]:
        messages: list[str] = []
        if isinstance(self.diagnostics, Mapping):
            single = self.diagnostics.get("inference_warning")
            if single is not None:
                messages.append(str(single))
            multiple = self.diagnostics.get("warning_messages")
            if multiple is not None:
                if isinstance(multiple, str):
                    messages.append(multiple)
                else:
                    for value in multiple:
                        messages.append(str(value))
        nonconvergence = self._nonconvergence_warning_message()
        if nonconvergence is not None:
            messages.append(nonconvergence)
        deduped: list[str] = []
        seen: set[str] = set()
        for message in messages:
            if message not in seen:
                deduped.append(message)
                seen.add(message)
        return tuple(deduped)

    def _warn_inference_messages(self) -> None:
        for message in self._warning_messages():
            warnings.warn(message, stacklevel=2)

    def _warn_nonconvergence_message(self) -> None:
        message = self._nonconvergence_warning_message()
        if message is not None:
            warnings.warn(message, stacklevel=2)

    def summary(self) -> str:
        lines = [
            "pyiols scaffold result",
            f"family={self.family} tau={self.tau} vcov={self.vcov_type}",
            f"nobs={self.nobs} df_model={self.df_model} df_resid={self.df_resid}",
            f"converged={self.converged}",
        ]
        if isinstance(self.diagnostics, Mapping):
            inference_status = self.diagnostics.get("inference_status")
            if inference_status is not None:
                lines.append(f"inference_status={inference_status}")
            path_inference_status = self.diagnostics.get("path_inference_status")
            if path_inference_status is not None:
                lines.append(f"path_inference_status={path_inference_status}")
                if str(path_inference_status) in {"not_computed", "unavailable"}:
                    lines.append(
                        "path_note: final-tau inference only; path covariance "
                        "and path-level standard errors are not computed"
                    )
            path_inference_reason = self.diagnostics.get("path_inference_reason")
            if path_inference_reason is not None:
                lines.append(f"path_inference_reason={path_inference_reason}")
            target_source = (
                self.tau_info.get("tau_path_targets_source")
                if isinstance(self.tau_info, Mapping)
                else None
            )
            if target_source is None and isinstance(self.solver_info, Mapping):
                target_source = self.solver_info.get("tau_path_targets_source")
            if target_source is not None:
                lines.append(f"tau_path_targets_source={target_source}")
            prediction_warning = self.diagnostics.get("prediction_warning")
            if prediction_warning is not None:
                lines.append(f"prediction_note: {prediction_warning}")
        for message in self._warning_messages():
            lines.append(f"warning: {message}")
        for name, value in self.params.items():
            se = self.bse.get(name)
            if se is not None:
                lines.append(f"{name}: coef={value:.6g} se={se:.6g}")
            else:
                lines.append(f"{name}: coef={value:.6g}")
        return "\n".join(lines)

    def cov_params(self) -> tuple[tuple[float, ...], ...]:
        self._warn_inference_messages()
        if self._covariance is not None:
            return tuple(tuple(row) for row in self._covariance)
        warnings.warn(
            "cov_params() is returning a diagonal fallback because the scaffold "
            "result does not carry a full covariance matrix.",
            stacklevel=2,
        )
        names = _ordered_names(self.params)
        matrix = []
        for row_name in names:
            row = []
            try:
                row_se = float(self.bse[row_name])
            except (TypeError, ValueError):
                row_se = float("nan")
            for col_name in names:
                try:
                    col_se = float(self.bse[col_name])
                except (TypeError, ValueError):
                    col_se = float("nan")
                if not (math.isfinite(row_se) and math.isfinite(col_se)):
                    row.append(float("nan"))
                elif row_name == col_name:
                    row.append(row_se**2)
                else:
                    row.append(float("nan"))
            matrix.append(tuple(row))
        return tuple(matrix)

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        self._warn_inference_messages()
        z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        intervals: dict[str, tuple[float, float]] = {}
        for name, estimate in self.params.items():
            se = self.bse[name]
            intervals[name] = (estimate - z_value * se, estimate + z_value * se)
        return intervals

    def tidy(self):
        import pandas as pd

        intervals = self.conf_int()
        rows = []
        for name, estimate in self.params.items():
            rows.append(
                {
                    "term": name,
                    "estimate": estimate,
                    "std_error": self.bse.get(name),
                    "statistic": self.tvalues.get(name),
                    "pvalue": self.pvalues.get(name),
                    "conf_low": intervals[name][0],
                    "conf_high": intervals[name][1],
                }
            )
        return pd.DataFrame(rows)

    def tau_path_frame(
        self,
        *,
        terms: str | Sequence[str] | None = None,
        alpha: float = 0.05,
    ):
        """Return stored coefficient-path estimates in long tabular form.

        The method is intentionally read-only: it consumes coefficient snapshots
        carried on the result and never re-runs a tau solve.
        """

        frame = _tau_path_frame(self, terms=terms, alpha=alpha)
        self._warn_nonconvergence_message()
        return frame

    def plot_tau_path(
        self,
        *,
        terms: str | Sequence[str] | None = None,
        alpha: float = 0.05,
        tau_min: float = 0.0,
        tau_max: float = 3.0,
        key_taus: Sequence[float] = (1.0, 2.0),
        selected_tau: float | None = None,
        show_ci: bool = True,
        ax: Any = None,
        colors: Sequence[str] | None = None,
    ):
        """Plot stored tau-path estimates for selected coefficients.

        Confidence ribbons are drawn only for path rows that carry per-tau
        standard errors. Missing path-level inference is not imputed from the
        final-tau standard error.
        """

        return _plot_tau_path(
            self,
            terms=terms,
            alpha=alpha,
            tau_min=tau_min,
            tau_max=tau_max,
            key_taus=key_taus,
            selected_tau=selected_tau,
            show_ci=show_ci,
            ax=ax,
            colors=colors,
        )

    def regression_table(
        self,
        *,
        model_names: Sequence[str] | None = None,
        terms: Sequence[str] | None = None,
        digits: int = 3,
        statistic: str = "std_error",
        stars: bool = True,
        include_stats: bool = True,
        stats: Sequence[str] | None = None,
    ):
        return regression_table(
            self,
            model_names=model_names,
            terms=terms,
            digits=digits,
            statistic=statistic,
            stars=stars,
            include_stats=include_stats,
            stats=stats,
        )

    def to_latex(
        self,
        *,
        path: str | Path | None = None,
        model_names: Sequence[str] | None = None,
        terms: Sequence[str] | None = None,
        digits: int = 3,
        statistic: str = "std_error",
        stars: bool = True,
        include_stats: bool = True,
        stats: Sequence[str] | None = None,
        caption: str | None = None,
        label: str | None = None,
        notes: Sequence[str] | None = None,
    ) -> str:
        return esttab(
            self,
            path=path,
            model_names=model_names,
            terms=terms,
            digits=digits,
            statistic=statistic,
            stars=stars,
            include_stats=include_stats,
            stats=stats,
            caption=caption,
            label=label,
            notes=notes,
        )

    def predict(self, data: Any = None, *, which: str = "mean"):
        if data is not None:
            raise NotImplementedError(
                "Out-of-sample prediction is not wired in the scaffold"
            )
        if which in {"mean", "response"}:
            predicted = tuple(self.fittedvalues)
            self._warn_nonconvergence_message()
            return predicted
        if which in {"linear", "link"}:
            linear = self.diagnostics.get("linear_predictor")
            if linear is None:
                raise NotImplementedError(
                    "Linear predictor output is unavailable for this result"
                )
            predicted = tuple(linear)
            self._warn_nonconvergence_message()
            return predicted
        raise ValueError(f"Unsupported prediction type: {which!r}")

    def get_prediction(
        self,
        data: Any = None,
        *,
        which: str = "mean",
    ) -> Mapping[str, Any]:
        predicted = self.predict(data=data, which=which)
        return {
            "predicted_mean": predicted,
            "which": which,
            "nobs": len(predicted),
        }


def _normalize_terms(terms: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if terms is None:
        return None
    if isinstance(terms, str):
        return (terms,)
    return tuple(str(term) for term in terms)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _result_tau_path_records(result: IOLSResults) -> tuple[Mapping[str, Any], ...]:
    candidates = (
        result.tau_info.get("path") if isinstance(result.tau_info, Mapping) else None,
        result.tau_info.get("tau_path_summary")
        if isinstance(result.tau_info, Mapping)
        else None,
        result.solver_info.get("path")
        if isinstance(result.solver_info, Mapping)
        else None,
        result.solver_info.get("tau_path_summary")
        if isinstance(result.solver_info, Mapping)
        else None,
        result.diagnostics.get("tau_path_summary")
        if isinstance(result.diagnostics, Mapping)
        else None,
        result.solver_info.get("tau_path", {}).get("path")
        if isinstance(result.solver_info, Mapping)
        and isinstance(result.solver_info.get("tau_path"), Mapping)
        else None,
        result.tau_info.get("tau_path", {}).get("path")
        if isinstance(result.tau_info, Mapping)
        and isinstance(result.tau_info.get("tau_path"), Mapping)
        else None,
    )
    for candidate in candidates:
        if candidate:
            return tuple(candidate)
    return ()


def _path_inference_metadata(result: IOLSResults) -> tuple[str | None, str | None]:
    sources: list[Mapping[str, Any]] = []
    if isinstance(result.diagnostics, Mapping):
        sources.append(result.diagnostics)
    if isinstance(result.solver_info, Mapping):
        sources.append(result.solver_info)
        tau_path = result.solver_info.get("tau_path")
        if isinstance(tau_path, Mapping):
            sources.append(tau_path)
    if isinstance(result.tau_info, Mapping):
        sources.append(result.tau_info)
        tau_path = result.tau_info.get("tau_path")
        if isinstance(tau_path, Mapping):
            sources.append(tau_path)

    status: str | None = None
    reason: str | None = None
    for source in sources:
        if status is None and source.get("path_inference_status") is not None:
            status = str(source["path_inference_status"])
        if reason is None and source.get("path_inference_reason") is not None:
            reason = str(source["path_inference_reason"])
        if status is not None and reason is not None:
            break
    return status, reason


def _path_inference_unavailable_message(result: IOLSResults) -> str | None:
    status, reason = _path_inference_metadata(result)
    if status not in {"not_computed", "unavailable"}:
        return None
    reason_text = f", reason={reason}" if reason else ""
    return (
        "Path-level covariance/inference is not available for this tau path "
        f"(status={status}{reason_text}); coefficient path estimates remain "
        "available, but path std_error/confidence columns are NaN and no CI "
        "ribbon is drawn."
    )


def _path_record_params(
    record: Mapping[str, Any],
    names: tuple[str, ...],
) -> dict[str, float] | None:
    params = record.get("params")
    if isinstance(params, Mapping):
        return {str(name): _coerce_float(value) for name, value in params.items()}
    beta = record.get("beta")
    if beta is None:
        beta = record.get("beta_reduced")
    if beta is None:
        return None
    values = tuple(_coerce_float(value) for value in beta)
    if len(values) != len(names):
        return None
    return {name: value for name, value in zip(names, values, strict=True)}


def _path_record_bse(
    record: Mapping[str, Any],
    names: tuple[str, ...],
) -> dict[str, float]:
    bse = record.get("bse")
    if isinstance(bse, Mapping):
        return {str(name): _coerce_float(value) for name, value in bse.items()}
    std_errors = record.get("std_errors")
    if std_errors is None:
        std_errors = record.get("se")
    if std_errors is None:
        return {}
    values = tuple(_coerce_float(value) for value in std_errors)
    if len(values) != len(names):
        return {}
    return {name: value for name, value in zip(names, values, strict=True)}


def _selected_tau(result: IOLSResults) -> float | None:
    if not isinstance(result.tau_info, Mapping):
        return None
    for key in ("tau_selected", "selected_tau"):
        value = _coerce_float(result.tau_info.get(key))
        if math.isfinite(value):
            return value
    if result.tau_info.get("mode") == "select":
        value = _coerce_float(result.tau_info.get("tau_fit"))
        if math.isfinite(value):
            return value
    return None


def _tau_path_frame(
    result: IOLSResults,
    *,
    terms: str | Sequence[str] | None,
    alpha: float,
):
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    import pandas as pd

    term_filter = _normalize_terms(terms)
    names = _ordered_names(result.params)
    records = _result_tau_path_records(result)
    z_value = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    selected = _selected_tau(result)
    path_inference_status, path_inference_reason = _path_inference_metadata(result)
    rows: list[dict[str, Any]] = []
    filtered_nonconverged_count = 0
    filtered_invalid_tau_count = 0
    filtered_missing_params_count = 0
    returned_record_count = 0
    filtered_nonconverged_taus: list[float] = []
    filtered_nonconverged_reasons: list[str] = []

    def add_filtered_reason(value: Any) -> None:
        if value is None:
            return
        text = str(value)
        if text and text not in filtered_nonconverged_reasons:
            filtered_nonconverged_reasons.append(text)

    for record in records:
        tau = _coerce_float(record.get("tau"))
        if not bool(record.get("converged", True)):
            filtered_nonconverged_count += 1
            if math.isfinite(tau):
                filtered_nonconverged_taus.append(float(tau))
            add_filtered_reason(record.get("failure_reason"))
            continue
        if not math.isfinite(tau):
            filtered_invalid_tau_count += 1
            continue
        params = _path_record_params(record, names)
        if not params:
            filtered_missing_params_count += 1
            continue
        bse = _path_record_bse(record, names)
        row_count_before_record = len(rows)
        for term, estimate in params.items():
            if term_filter is not None and term not in term_filter:
                continue
            se = bse.get(term, float("nan"))
            record_status = path_inference_status
            record_reason = path_inference_reason
            if record.get("path_inference_status") is not None:
                record_status = str(record["path_inference_status"])
            if record.get("path_inference_reason") is not None:
                record_reason = str(record["path_inference_reason"])
            if record_status in {"not_computed", "unavailable"}:
                se = float("nan")
            conf_low = float("nan")
            conf_high = float("nan")
            if math.isfinite(se):
                conf_low = estimate - z_value * se
                conf_high = estimate + z_value * se
            rows.append(
                {
                    "tau": tau,
                    "term": term,
                    "estimate": estimate,
                    "std_error": se,
                    "conf_low": conf_low,
                    "conf_high": conf_high,
                    "path_inference_available": math.isfinite(se),
                    "path_inference_status": record_status,
                    "path_inference_reason": record_reason,
                    "is_tau_1": abs(tau - 1.0) <= 1e-10,
                    "is_tau_2": abs(tau - 2.0) <= 1e-10,
                    "is_selected_tau": selected is not None
                    and abs(tau - selected) <= 1e-10,
                }
            )
        if len(rows) > row_count_before_record:
            returned_record_count += 1
    if not rows:
        raise NotImplementedError(
            "This result does not carry coefficient-level tau-path snapshots. "
            "The solver must store converged path params/beta records before "
            "tau_path_frame() or plot_tau_path() can be used without re-solving."
        )
    frame = pd.DataFrame(rows)
    frame.attrs["path_inference_status"] = path_inference_status
    frame.attrs["path_inference_reason"] = path_inference_reason
    frame.attrs["path_record_count"] = len(records)
    frame.attrs["path_returned_record_count"] = int(returned_record_count)
    frame.attrs["path_returned_row_count"] = int(len(rows))
    frame.attrs["path_filtered_nonconverged_record_count"] = int(
        filtered_nonconverged_count
    )
    frame.attrs["path_filtered_nonconverged_taus"] = tuple(
        filtered_nonconverged_taus
    )
    frame.attrs["path_filtered_nonconverged_failure_reasons"] = tuple(
        filtered_nonconverged_reasons
    )
    frame.attrs["path_filtered_invalid_tau_record_count"] = int(
        filtered_invalid_tau_count
    )
    frame.attrs["path_filtered_missing_params_record_count"] = int(
        filtered_missing_params_count
    )
    return frame


def _plot_tau_path(
    result: IOLSResults,
    *,
    terms: str | Sequence[str] | None,
    alpha: float,
    tau_min: float,
    tau_max: float,
    key_taus: Sequence[float],
    selected_tau: float | None,
    show_ci: bool,
    ax: Any,
    colors: Sequence[str] | None,
):
    if float(tau_min) >= float(tau_max):
        raise ValueError("tau_min must be less than tau_max")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_tau_path() requires matplotlib. Install matplotlib or use "
            "tau_path_frame() for a tabular path artifact."
        ) from exc

    frame = result.tau_path_frame(terms=terms, alpha=alpha)
    frame = frame[(frame["tau"] >= tau_min) & (frame["tau"] <= tau_max)]
    if frame.empty:
        raise ValueError("No stored tau-path rows fall inside the requested tau range")
    if show_ci:
        has_ci = (
            frame["conf_low"].map(math.isfinite)
            & frame["conf_high"].map(math.isfinite)
        ).any()
        if not bool(has_ci):
            message = _path_inference_unavailable_message(result)
            if message is not None:
                warnings.warn(message, stacklevel=2)
    selected_terms = _normalize_terms(terms)
    if selected_terms is None:
        selected_terms = tuple(dict.fromkeys(str(term) for term in frame["term"]))
    palette = tuple(
        colors
        or (
            "#8ecae6",
            "#ffb3a7",
            "#b8de6f",
            "#cdb4db",
            "#ffd166",
            "#a8dadc",
        )
    )
    if ax is None:
        fig, axes_grid = plt.subplots(
            len(selected_terms),
            1,
            sharex=True,
            figsize=(7.0, max(2.8, 2.4 * len(selected_terms))),
            squeeze=False,
        )
        axes = list(axes_grid.ravel())
    elif hasattr(ax, "plot"):
        if len(selected_terms) != 1:
            raise ValueError("Pass one axis per term when plotting multiple terms")
        fig = ax.figure
        axes = [ax]
    else:
        axes = list(ax)
        if len(axes) < len(selected_terms):
            raise ValueError("Not enough axes for the requested terms")
        fig = axes[0].figure

    marker_values = [float(value) for value in key_taus]
    effective_selected_tau = (
        selected_tau if selected_tau is not None else _selected_tau(result)
    )
    if effective_selected_tau is not None:
        marker_values.append(float(effective_selected_tau))

    for index, term in enumerate(selected_terms):
        axis = axes[index]
        term_frame = frame[frame["term"] == term].sort_values("tau")
        color = palette[index % len(palette)]
        x_values = [float(value) for value in term_frame["tau"]]
        y_values = [float(value) for value in term_frame["estimate"]]
        axis.plot(x_values, y_values, color=color, linewidth=2.0, label=str(term))
        if show_ci:
            ci_frame = term_frame[
                term_frame["conf_low"].map(math.isfinite)
                & term_frame["conf_high"].map(math.isfinite)
            ]
            if not ci_frame.empty:
                axis.fill_between(
                    [float(value) for value in ci_frame["tau"]],
                    [float(value) for value in ci_frame["conf_low"]],
                    [float(value) for value in ci_frame["conf_high"]],
                    color=color,
                    alpha=0.25,
                    linewidth=0.0,
                )
        axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        for marker in marker_values:
            if tau_min <= marker <= tau_max:
                linestyle = ":" if marker in {1.0, 2.0} else "-."
                axis.axvline(
                    marker,
                    color="#666666",
                    linewidth=1.0,
                    linestyle=linestyle,
                )
        axis.set_ylabel(str(term))
        axis.legend(loc="best", frameon=False)
    axes[-1].set_xlabel("tau")
    for axis in axes:
        axis.set_xlim(float(tau_min), float(tau_max))
    return fig, axes[0] if len(axes) == 1 else tuple(axes)


def _coerce_results(
    results: IOLSResults | Sequence[IOLSResults],
) -> tuple[IOLSResults, ...]:
    if isinstance(results, IOLSResults):
        return (results,)
    return tuple(results)


def _format_float(value: Any, digits: int) -> str:
    numeric = _coerce_float(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{int(digits)}f}"


def _star_suffix(pvalue: Any) -> str:
    pvalue_float = _coerce_float(pvalue)
    if not math.isfinite(pvalue_float):
        return ""
    for cutoff, suffix in ((0.01, "***"), (0.05, "**"), (0.10, "*")):
        if pvalue_float <= cutoff:
            return suffix
    return ""


def _statistic_value(result: IOLSResults, term: str, statistic: str) -> Any:
    if statistic == "std_error":
        return result.bse.get(term)
    if statistic in {"t", "tvalue", "statistic"}:
        return result.tvalues.get(term)
    if statistic == "pvalue":
        return result.pvalues.get(term)
    raise ValueError(
        "statistic must be one of 'std_error', 'tvalue', 'statistic', or 'pvalue'"
    )


def _statistic_cell(result: IOLSResults, term: str, statistic: str, digits: int) -> str:
    formatted = _format_float(_statistic_value(result, term, statistic), digits)
    if not formatted:
        return ""
    if statistic == "pvalue":
        return f"[{formatted}]"
    return f"({formatted})"


def _default_terms(results: tuple[IOLSResults, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for result in results:
        for term in result.params:
            if term not in seen:
                terms.append(term)
                seen.add(term)
    return tuple(terms)


def _stat_label(stat: str) -> str:
    labels = {
        "family": "Family",
        "tau": "Tau",
        "vcov_type": "Vcov",
        "fe_method": "FE method",
        "fe_partition": "FE partition",
        "nobs": "N",
        "df_model": "df model",
        "df_resid": "df resid",
        "converged": "Converged",
    }
    return labels.get(stat, stat)


def _result_stat(result: IOLSResults, stat: str) -> Any:
    if stat == "family":
        return result.family
    if stat == "tau":
        return result.tau
    if stat == "vcov_type":
        return result.vcov_type
    if stat == "fe_method":
        return result.fe_method
    if stat == "fe_partition":
        return result.fe_partition
    if stat == "nobs":
        return result.nobs
    if stat == "df_model":
        return result.df_model
    if stat == "df_resid":
        return result.df_resid
    if stat == "converged":
        return "Yes" if result.converged else "No"
    raise ValueError(f"Unsupported regression table statistic: {stat!r}")


def _format_stat_value(value: Any, digits: int) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value, digits)
    if value is None:
        return ""
    return str(value)


def regression_table(
    results: IOLSResults | Sequence[IOLSResults],
    *,
    model_names: Sequence[str] | None = None,
    terms: Sequence[str] | None = None,
    digits: int = 3,
    statistic: str = "std_error",
    stars: bool = True,
    include_stats: bool = True,
    stats: Sequence[str] | None = None,
):
    import pandas as pd

    result_list = _coerce_results(results)
    if not result_list:
        raise ValueError("At least one result is required")
    names = tuple(
        model_names or (f"Model {index + 1}" for index in range(len(result_list)))
    )
    if len(names) != len(result_list):
        raise ValueError("model_names length must match the number of results")
    term_list = tuple(terms or _default_terms(result_list))
    rows: list[dict[str, str]] = []
    for term in term_list:
        coef_row = {"term": str(term)}
        stat_row = {"term": ""}
        for model_name, result in zip(names, result_list, strict=True):
            estimate = _format_float(result.params.get(term), digits)
            if estimate and stars:
                estimate = f"{estimate}{_star_suffix(result.pvalues.get(term))}"
            coef_row[model_name] = estimate
            stat_row[model_name] = _statistic_cell(result, term, statistic, digits)
        rows.append(coef_row)
        rows.append(stat_row)

    if include_stats:
        stat_list = tuple(
            stats
            or (
                "family",
                "tau",
                "vcov_type",
                "fe_method",
                "fe_partition",
                "nobs",
                "df_model",
                "df_resid",
                "converged",
            )
        )
        for stat in stat_list:
            row = {"term": _stat_label(stat)}
            for model_name, result in zip(names, result_list, strict=True):
                row[model_name] = _format_stat_value(
                    _result_stat(result, stat),
                    digits,
                )
            rows.append(row)
    table = pd.DataFrame(rows, columns=("term",) + names)
    for result in result_list:
        result._warn_nonconvergence_message()
    return table


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def esttab(
    results: IOLSResults | Sequence[IOLSResults],
    *,
    fmt: str = "latex",
    path: str | Path | None = None,
    model_names: Sequence[str] | None = None,
    terms: Sequence[str] | None = None,
    digits: int = 3,
    statistic: str = "std_error",
    stars: bool = True,
    include_stats: bool = True,
    stats: Sequence[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
    notes: Sequence[str] | None = None,
) -> str:
    if fmt not in {"latex", "tex"}:
        raise ValueError("esttab only supports fmt='latex' or fmt='tex'")
    table = regression_table(
        results,
        model_names=model_names,
        terms=terms,
        digits=digits,
        statistic=statistic,
        stars=stars,
        include_stats=include_stats,
        stats=stats,
    )
    column_format = "l" + "c" * (len(table.columns) - 1)
    lines: list[str] = []
    if caption is not None or label is not None:
        lines.append(r"\begin{table}[!htbp]")
        lines.append(r"\centering")
        if caption is not None:
            lines.append(rf"\caption{{{_latex_escape(caption)}}}")
        if label is not None:
            lines.append(rf"\label{{{_latex_escape(label)}}}")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\hline")
    lines.append(
        " & ".join(_latex_escape(column) for column in table.columns) + r" \\"
    )
    lines.append(r"\hline")
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    if notes:
        lines.append(r"\hline")
        note_text = " ".join(_latex_escape(note) for note in notes)
        colspan = len(table.columns)
        lines.append(
            rf"\multicolumn{{{colspan}}}{{l}}{{\footnotesize {note_text}}} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if caption is not None or label is not None:
        lines.append(r"\end{table}")
    latex = "\n".join(lines)
    if path is not None:
        Path(path).write_text(latex, encoding="utf-8")
    return latex
