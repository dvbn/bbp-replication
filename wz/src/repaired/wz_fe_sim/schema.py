"""Result-row schema, admissible coefficient/SE pairings and status vocabulary.

Standard library only: the schema is imported by the driver, the summaries and
the stdlib-only checks. One row per
(packet, archive_version, dgp, N, T, rep, estimator_version, tau, coef_version, se_version).
"""

from __future__ import annotations

import math

KEY_FIELDS = (
    "packet",
    "archive_version",
    "dgp",
    "N",
    "T",
    "rep",
    "estimator_version",
    "tau",
    "coef_version",
    "se_version",
)

# Order is the CSV column order. Everything not produced is NaN / "" (never 0).
# draw_hash = content hash of the master innovation arrays; draw_file_sha256 =
# sha256 of the stored immutable draw file (contract Sec. 3), or of the
# historical input file for packet H.  beta_last_iterate / theta_max_abs_last
# describe the terminal solver iterate; for a failed root they are diagnostics
# only and never an estimate (``beta`` stays NaN unless status_root == "ok").
# resid_metric / newton_corr_metric are the mean-scale fixed-metric norms
# (metric_normalization: raw_sum_over_n for H_Gamma, already_mean for B'B/n;
# n = n_obs); metric_scope says whether the metric gates applied.
# attempts / attempts_ok count calls of the requested target equation (direct
# and continuation endpoint calls); attempts_intermediate(_ok) count
# continuation calls at other tau; newton_calls = every executed Newton call of
# the root.  Like the elapsed_* fields, newton_calls / attempts_intermediate*
# are carried once (on the C or J row with its own sandwich), so summing rows
# never multiplies work.
ROW_FIELDS = KEY_FIELDS + (
    "offdiag",
    "draw_hash",
    "draw_file_sha256",
    "config_hash",
    "code_hash",
    "beta",
    "se",
    "bias_hat",
    "bias_term1",
    "bias_term2",
    "wz_bias",
    "wz_bias_no_factor",
    "solver_version",
    "acceptance_rule_version",
    "start_policy_version",
    "start_used",
    "attempts",
    "attempts_ok",
    "attempts_intermediate",
    "attempts_intermediate_ok",
    "newton_calls",
    "continuation_status",
    "continuation_seed",
    "continuation_last_tau",
    "iterations",
    "backtracks",
    "certified_full_step",
    "status_root",
    "status_inference",
    "status_correction",
    "resid_scaled",
    "resid_merit",
    "beta_last_iterate",
    "theta_max_abs_last",
    "resid_target_over_se",
    "newton_corr_norm",
    "rcond",
    "resid_metric",
    "newton_corr_metric",
    "metric_scope",
    "metric_normalization",
    "schur_beta",
    "schur_beta_over_xx",
    "w_min",
    "w_max",
    "root_ambiguous",
    "ambiguity_max_diff",
    "jac_asym",
    "merit_source",
    "n_obs",
    "n_pairs",
    "d",
    "elapsed_setup_s",
    "elapsed_root_s",
    "elapsed_retry_s",
    "elapsed_cov_s",
    "elapsed_corr_s",
    "elapsed_extra_s",
    "note",
)

# Columns that legitimately differ between two otherwise identical runs.
TIMING_FIELDS = tuple(f for f in ROW_FIELDS if f.startswith("elapsed_"))

FLOAT_FIELDS = frozenset(
    {
        "tau",
        "beta",
        "se",
        "bias_hat",
        "bias_term1",
        "bias_term2",
        "wz_bias",
        "wz_bias_no_factor",
        "resid_scaled",
        "resid_merit",
        "beta_last_iterate",
        "theta_max_abs_last",
        "resid_target_over_se",
        "newton_corr_norm",
        "rcond",
        "resid_metric",
        "newton_corr_metric",
        "continuation_last_tau",
        "schur_beta",
        "schur_beta_over_xx",
        "w_min",
        "w_max",
        "ambiguity_max_diff",
        "jac_asym",
    }
    | set(TIMING_FIELDS)
)
INT_FIELDS = frozenset(
    {"dgp", "N", "T", "rep", "attempts", "attempts_ok", "attempts_intermediate", "attempts_intermediate_ok",
     "newton_calls", "iterations", "backtracks", "n_obs", "n_pairs", "d"}
)
BOOL_FIELDS = frozenset({"offdiag", "root_ambiguous", "certified_full_step"})

PACKETS = ("M", "H", "H_R", "P")  # P = preflight/pilot packet, never pooled with M
ESTIMATOR_VERSIONS = ("C", "CB", "J", "WZ-bc", "WZ-spj")
COEF_VERSIONS = ("raw", "manuscript_bc", "wz_bc", "spj")
SE_VERSIONS = ("pair_sandwich", "joint_naive", "wz_unadjusted", "wz_adjusted", "none")

ROOT_STATUSES = (
    "ok",
    "singular_jacobian",
    "line_search_failed",
    "max_iter",
    "root_check_failed",
    "numerical_failure",
    "not_attempted",
)
STAGE_STATUSES = ROOT_STATUSES + ("failed", "inference_check_failed", "skipped_root_failed")
CONTINUATION_STATUSES = ("not_needed", "disabled", "no_certified_seed", "endpoint_certified",
                         "min_step_reached", "budget_exhausted", "no_numerical_progress",
                         "not_applicable", "no_admissible_start", "")
METRIC_SCOPES = ("fixed_metric", "objective_coordinate_only", "gamma_pilot_preliminary_coordinate_only",
                 "gamma_pilot_fixed_metric_admission", "objective_only_ppml_subfit", "")

# (estimator_version, coef_version) -> allowed se_versions.  PPML-only versions
# require tau == 1.  WZ conventions (wz_unadjusted = P/(P-1) x pair sandwich,
# wz_adjusted = WZ one-sided leverage covariance) are PPML-only labels.
ALLOWED_PAIRINGS = {
    ("C", "raw"): ("pair_sandwich", "wz_unadjusted", "wz_adjusted"),
    ("CB", "manuscript_bc"): ("pair_sandwich", "wz_unadjusted", "wz_adjusted"),
    ("J", "raw"): ("joint_naive",),
    ("WZ-bc", "wz_bc"): ("pair_sandwich", "wz_unadjusted", "wz_adjusted"),
    ("WZ-spj", "spj"): ("pair_sandwich", "wz_unadjusted", "wz_adjusted"),
}
PPML_ONLY_SE = frozenset({"wz_unadjusted", "wz_adjusted"})
PPML_ONLY_ESTIMATORS = frozenset({"WZ-bc", "WZ-spj"})


def expected_keyset(tau_grid):
    """Configured estimator/SE key set {(estimator_version, tau, coef_version, se_version)}
    per draw-cell, derived from ALLOWED_PAIRINGS: PPML-only SE versions and
    estimators only at tau = 1, and J(1) (== C(1)) excluded.  Its size is the
    number of rows per draw-cell (30 for the seven-power grid).  Standard library
    only, so the CLI plan uses the same key set as the preflight checks."""
    out = set()
    for tau in tau_grid:
        tau = round(float(tau), 12)
        ppml = abs(tau - 1.0) < 1e-12
        for (est, coef), ses in ALLOWED_PAIRINGS.items():
            if est in PPML_ONLY_ESTIMATORS and not ppml:
                continue
            if est == "J" and ppml:
                continue
            for se in ses:
                if se in PPML_ONLY_SE and not ppml:
                    continue
                out.add((est, tau, coef, se))
    return out


def core_roots_per_draw_cell(tau_grid):
    """C(tau) for every tau plus J(tau) for tau != 1 (J(1) = C(1) is not refitted)."""
    taus = {round(float(t), 12) for t in tau_grid}
    return 2 * len(taus) - (1 if 1.0 in taus else 0)


def nan_row(**values):
    """A full row with NaN/empty defaults; unknown keys are rejected."""
    unknown = set(values) - set(ROW_FIELDS)
    if unknown:
        raise KeyError(f"unknown row fields: {sorted(unknown)}")
    row = {}
    for field in ROW_FIELDS:
        if field in FLOAT_FIELDS:
            row[field] = math.nan
        elif field in INT_FIELDS:
            row[field] = -1
        elif field in BOOL_FIELDS:
            row[field] = False
        else:
            row[field] = ""
    row.update(values)
    return row


def check_pairing(estimator_version, coef_version, se_version, tau):
    """Raise if a coefficient is paired with an inadmissible SE version."""
    allowed = ALLOWED_PAIRINGS.get((estimator_version, coef_version))
    if allowed is None:
        raise ValueError(f"unknown estimator/coef pair {estimator_version}/{coef_version}")
    if se_version not in allowed:
        raise ValueError(
            f"{estimator_version}/{coef_version} cannot use se_version={se_version}"
        )
    is_ppml = abs(float(tau) - 1.0) < 1e-12
    if (se_version in PPML_ONLY_SE or estimator_version in PPML_ONLY_ESTIMATORS) and not is_ppml:
        raise ValueError(f"{estimator_version}/{se_version} is PPML-only (tau=1)")
    if estimator_version == "J" and is_ppml:
        raise ValueError("J(1) equals C(1); report C(1), not a third PPML estimate")


def validate_row(row):
    """Schema validation of one row dict (types, vocabulary, pairing)."""
    missing = [f for f in ROW_FIELDS if f not in row]
    if missing:
        raise KeyError(f"row missing fields {missing}")
    if row["packet"] not in PACKETS:
        raise ValueError(f"bad packet {row['packet']!r}")
    if row["status_root"] not in ROOT_STATUSES:
        raise ValueError(f"bad status_root {row['status_root']!r}")
    for stage in ("status_inference", "status_correction"):
        if row[stage] not in STAGE_STATUSES + ("",):
            raise ValueError(f"bad {stage} {row[stage]!r}")
    if row["continuation_status"] not in CONTINUATION_STATUSES:
        raise ValueError(f"bad continuation_status {row['continuation_status']!r}")
    if row["metric_scope"] not in METRIC_SCOPES:
        raise ValueError(f"bad metric_scope {row['metric_scope']!r}")
    check_pairing(row["estimator_version"], row["coef_version"], row["se_version"], row["tau"])
    return True


def format_value(field, value):
    """Deterministic CSV text: floats with repr (round-trip exact), NaN as 'nan'."""
    if field in FLOAT_FIELDS:
        value = float(value)
        return "nan" if math.isnan(value) else repr(value)
    if field in BOOL_FIELDS:
        return "1" if value else "0"
    return str(value)


def parse_value(field, text):
    """Inverse of format_value for reading row CSVs."""
    if field in FLOAT_FIELDS:
        return float(text)
    if field in INT_FIELDS:
        return int(text)
    if field in BOOL_FIELDS:
        return text in ("1", "True", "true")
    return text
