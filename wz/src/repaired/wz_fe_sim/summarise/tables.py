"""Per-cell summary statistics with replication-bundle bootstrap MCSEs.

Conventions
-----------
* A *group* is (packet, archive_version, dgp, N, T, estimator_version, tau,
  coef_version, se_version); a *cell* is (packet, archive_version, dgp, N, T).
  Different archive_version values (and packets) are NEVER pooled: each gets its
  own cell and every output row carries ``archive_version``.
* Stage-aware eligibility (defined once, in ``accepted_mask`` / ``delivered_mask``,
  and reused by paired.py and figures.py):
    accepted   status_root == "ok" AND finite beta AND, for corrected coefficient
               versions (manuscript_bc, wz_bc, spj), status_correction == "ok".
               A finite trial beta from a failed root is never accepted.
    delivered  accepted AND status_inference == "ok" AND finite se > 0.  A finite
               SE with failed / inference_check_failed status is NOT delivered.
  A failed correction removes only the corrected row; the raw C row is separate.
* R = distinct rep ids in the group (failures included).
* n_beta = accepted coefficients; D = delivered intervals (also reported as
  n_se / n_cov); C = delivered rows whose 1.96-normal interval contains beta0 = 1.
  Per-stage failure counts: n_root_failed, n_inference_failed (accepted but not
  delivered), n_correction_failed, n_excluded_trial_beta (finite beta, root not ok).
* bias, sd (ddof=1), rmse use the n_beta draws; mean/median SE and mean CI width
  use the D delivered draws.  bias_over_mean_se is the WZ convention
  (bias on n_beta draws divided by mean SE on D draws).
* Coverage C/D = coverage among delivered; C/R = all-draw coverage (an
  undelivered interval counts as non-coverage); delivery = D/R.
* MCSE: bias = sd/sqrt(n_beta); coverage C/D = binomial sqrt(p(1-p)/D); every
  other statistic = SD over B bootstrap replicates that resample WHOLE rep ids
  with replacement.  The resampled rep multiset is a deterministic function of
  (seed, packet, dgp, N, T) and is shared by every estimator in the cell, so
  paired structure is preserved.  No normal-theory MCSE for SD is used.
"""

from __future__ import annotations

import csv
import math
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..schema import PACKETS, ROOT_STATUSES, STAGE_STATUSES

Z95 = 1.96
BETA0 = 1.0
DEFAULT_N_BOOT = 999
EST_ORDER = {"J": 0, "C": 1, "CB": 2, "WZ-bc": 3, "WZ-spj": 4}
BOOT_KEYS = (
    "sd", "rmse", "median_se", "mean_se_over_sd", "bias_over_sd",
    "bias_over_mean_se", "coverage_cr", "delivery",
)
GROUP_FIELDS = ("packet", "archive_version", "dgp", "N", "T", "estimator_version", "tau",
                "coef_version", "se_version")
CORRECTED_COEF = frozenset({"manuscript_bc", "wz_bc", "spj"})


def accepted_mask(beta, status_root, coef_version, status_correction):
    """Accepted-coefficient mask (stage-aware; the single definition used everywhere)."""
    beta = np.asarray(beta, dtype=float)
    ok = (np.asarray(status_root, dtype=object) == "ok") & np.isfinite(beta)
    corrected = np.isin(np.asarray(coef_version, dtype=object), list(CORRECTED_COEF))
    return ok & (~corrected | (np.asarray(status_correction, dtype=object) == "ok"))


def delivered_mask(accepted, status_inference, se):
    """Delivered-interval mask: accepted AND status_inference == "ok" AND finite se > 0."""
    se = np.asarray(se, dtype=float)
    with np.errstate(invalid="ignore"):
        return (np.asarray(accepted, dtype=bool) & (np.asarray(status_inference, dtype=object) == "ok")
                & np.isfinite(se) & (se > 0))


def row_masks(data):
    """(accepted, delivered) boolean arrays over all rows of ``data``."""
    acc = accepted_mask(data["beta"], data["status_root"], data["coef_version"], data["status_correction"])
    return acc, delivered_mask(acc, data["status_inference"], data["se"])


def check_single_packet(data, allow_multi_packet=False):
    """Refuse inputs holding several packets unless explicitly allowed."""
    packets = sorted(set(data["packet"].tolist()))
    if len(packets) > 1 and not allow_multi_packet:
        raise ValueError(f"input mixes packets {packets}; pass allow_multi_packet=True explicitly")
    return packets


def _tau_key(tau):
    return round(float(tau), 12)


@dataclass
class GroupData:
    """Per-group arrays aligned to the cell's sorted rep ids (NaN / False if absent)."""
    key: tuple
    idx: np.ndarray        # row indices into the input columns
    beta: np.ndarray       # NaN unless the coefficient is accepted
    se: np.ndarray         # NaN unless the interval is delivered
    present: np.ndarray    # rep has a row in this group (attempted)
    n_ambiguous: int
    draw_hash: np.ndarray = None    # object array aligned to rep_ids ("" if absent)
    draw_sha: np.ndarray = None


@dataclass
class Cell:
    key: tuple
    rep_ids: np.ndarray
    groups: dict


def _group_sort_key(gk):
    return (EST_ORDER.get(gk[5], 99), gk[6], gk[7], gk[8])


def build_cells(data):
    """Group rows into cells and groups; raises on duplicate (group, rep) rows."""
    cols = [data[k].tolist() for k in GROUP_FIELDS]
    acc, dlv = row_masks(data)
    members = {}
    for i, key in enumerate(zip(*cols)):
        gk = key[:6] + (_tau_key(key[6]),) + key[7:]
        members.setdefault(gk, []).append(i)
    by_cell = {}
    for gk, idx in members.items():
        by_cell.setdefault(gk[:5], []).append((gk, np.asarray(idx, dtype=np.intp)))
    rep = data["rep"]
    cells = {}
    for ck in sorted(by_cell):
        items = by_cell[ck]
        rep_ids = np.unique(rep[np.concatenate([ix for _, ix in items])])
        groups = {}
        for gk, ix in sorted(items, key=lambda it: _group_sort_key(it[0])):
            reps = rep[ix]
            if np.unique(reps).size != reps.size:
                raise ValueError(f"duplicate rep rows within group {gk}")
            pos = np.searchsorted(rep_ids, reps)
            beta = np.full(rep_ids.size, np.nan)
            se = np.full(rep_ids.size, np.nan)
            present = np.zeros(rep_ids.size, dtype=bool)
            hsh = np.full(rep_ids.size, "", dtype=object)
            sha = np.full(rep_ids.size, "", dtype=object)
            beta[pos] = np.where(acc[ix], data["beta"][ix], np.nan)
            se[pos] = np.where(dlv[ix], data["se"][ix], np.nan)
            present[pos] = True
            hsh[pos] = data["draw_hash"][ix]
            sha[pos] = data["draw_file_sha256"][ix]
            groups[gk] = GroupData(gk, ix, beta, se, present, int(np.sum(data["root_ambiguous"][ix])),
                                   hsh, sha)
        cells[ck] = Cell(ck, rep_ids, groups)
    return cells


def boot_indices(seed, cell_key, n_reps, n_boot):
    """(n_boot, n_reps) rep-position resamples; deterministic in (seed, cell).

    Uses PCG64 seeded through a SeedSequence of (seed, packet index, dgp, N, T),
    so results do not depend on which other cells are present (archive_version is
    not part of the seed: archives are separate cells with the same resampling rule).  The same array is
    applied to every estimator in the cell (paired resampling).
    """
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    packet, dgp, n, t = cell_key
    pidx = PACKETS.index(packet) if packet in PACKETS else len(PACKETS)
    ss = np.random.SeedSequence([int(seed), pidx, int(dgp), int(n), int(t)])
    rng = np.random.Generator(np.random.PCG64(ss))
    return rng.integers(0, n_reps, size=(n_boot, n_reps))


def _stats_nd(beta, se, present):
    """Statistics reducing over the last axis; inputs share shape (..., R)."""
    with np.errstate(all="ignore"):
        vb = present & np.isfinite(beta)
        dl = vb & np.isfinite(se) & (se > 0)
        R = present.sum(-1).astype(float)
        n = vb.sum(-1).astype(float)
        D = dl.sum(-1).astype(float)
        d = np.where(vb, beta - BETA0, 0.0)
        bias = d.sum(-1) / n
        dev = np.where(vb, d - bias[..., None], 0.0)
        sd = np.where(n >= 2, np.sqrt((dev ** 2).sum(-1) / (n - 1)), np.nan)
        rmse = np.sqrt((d ** 2).sum(-1) / n)
        se_d = np.where(dl, se, np.nan)
        mean_se = np.where(dl, se, 0.0).sum(-1) / D
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_se = np.nanmedian(se_d, axis=-1)
        covered = dl & (np.abs(d) <= Z95 * np.where(dl, se, 0.0))
        C = covered.sum(-1).astype(float)
        return {
            "R": R, "n": n, "D": D, "C": C, "bias": bias, "sd": sd, "rmse": rmse,
            "mean_se": mean_se, "median_se": median_se,
            "mean_se_over_sd": mean_se / sd, "bias_over_sd": bias / sd,
            "bias_over_mean_se": bias / mean_se,
            "coverage_cd": C / D, "coverage_cr": C / R, "delivery": D / R,
        }


def _boot_stats(g, idx, chunk=64):
    out = {k: [] for k in BOOT_KEYS}
    for s in range(0, idx.shape[0], chunk):
        ii = idx[s:s + chunk]
        st = _stats_nd(g.beta[ii], g.se[ii], g.present[ii])
        for k in BOOT_KEYS:
            out[k].append(st[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def bootstrap_sd(values):
    """SD (ddof=1) of the finite bootstrap replicates; NaN if fewer than 2."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if v.size >= 2 else math.nan


def _status_counts(prefix, values, vocab):
    cnt = Counter(values)
    out = {}
    for s in vocab:
        out[f"{prefix}_{s}"] = int(cnt.pop(s, 0))
    out[f"{prefix}_blank"] = int(cnt.pop("", 0)) if "" not in vocab else 0
    out[f"{prefix}_other"] = int(sum(cnt.values()))
    return out


def _summarise_group(data, gk, g, idx, seed, n_boot):
    st = {k: float(v) for k, v in _stats_nd(g.beta, g.se, g.present).items()}
    boot = _boot_stats(g, idx)
    n, D, C, R = st["n"], st["D"], st["C"], st["R"]
    p = st["coverage_cd"]
    s_root = np.asarray(data["status_root"][g.idx], dtype=object)
    s_corr = np.asarray(data["status_correction"][g.idx], dtype=object)
    corrected = gk[7] in CORRECTED_COEF
    root_bad = s_root != "ok"
    finite_beta = np.isfinite(data["beta"][g.idx])
    row = {
        "packet": gk[0], "archive_version": gk[1], "dgp": gk[2], "N": gk[3], "T": gk[4],
        "estimator_version": gk[5], "tau": gk[6], "coef_version": gk[7], "se_version": gk[8],
        "R": int(R), "n_beta": int(n), "D": int(D), "C": int(C),
        "n_se": int(D), "n_cov": int(D), "n_covering": int(C),
        "n_root_failed": int(root_bad.sum()),
        "n_inference_failed": int(n - D),
        "n_correction_failed": int(((~root_bad) & (s_corr != "ok")).sum()) if corrected else 0,
        "n_excluded_trial_beta": int((root_bad & finite_beta).sum()),
        "n_bias": int(n), "n_sd": int(n), "n_rmse": int(n),
        "n_mean_se_over_sd": int(D), "n_bias_over_sd": int(n), "n_bias_over_mean_se": int(D),
        "bias": st["bias"], "bias_x100": st["bias"] * 100.0,
        "mcse_bias": st["sd"] / math.sqrt(n) if n >= 2 else math.nan,
        "sd": st["sd"], "mcse_sd": bootstrap_sd(boot["sd"]),
        "rmse": st["rmse"], "mcse_rmse": bootstrap_sd(boot["rmse"]),
        "mean_se": st["mean_se"], "median_se": st["median_se"],
        "mcse_median_se": bootstrap_sd(boot["median_se"]),
        "mean_se_over_sd": st["mean_se_over_sd"],
        "mcse_mean_se_over_sd": bootstrap_sd(boot["mean_se_over_sd"]),
        "coverage_cd": p,
        "mcse_coverage_cd": math.sqrt(p * (1 - p) / D) if D >= 1 else math.nan,
        "coverage_cr": st["coverage_cr"], "mcse_coverage_cr": bootstrap_sd(boot["coverage_cr"]),
        "delivery": st["delivery"], "mcse_delivery": bootstrap_sd(boot["delivery"]),
        "mean_ci_width": 2.0 * Z95 * st["mean_se"],
        "bias_over_sd": st["bias_over_sd"], "mcse_bias_over_sd": bootstrap_sd(boot["bias_over_sd"]),
        "bias_over_mean_se_wz": st["bias_over_mean_se"],
        "mcse_bias_over_mean_se_wz": bootstrap_sd(boot["bias_over_mean_se"]),
        "root_ambiguous": g.n_ambiguous,
        "n_boot": int(n_boot), "seed": int(seed),
    }
    row.update(_status_counts("root", data["status_root"][g.idx].tolist(), ROOT_STATUSES))
    row.update(_status_counts("inf", data["status_inference"][g.idx].tolist(), STAGE_STATUSES))
    row.update(_status_counts("corr", data["status_correction"][g.idx].tolist(), STAGE_STATUSES))
    return row


def summarise_cells(data, seed, n_boot=DEFAULT_N_BOOT, allow_multi_packet=False):
    """List of per-group summary dicts (see module docstring), sorted deterministically."""
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    check_single_packet(data, allow_multi_packet)
    rows = []
    for ck, cell in build_cells(data).items():
        idx = boot_indices(seed, ck[:1] + ck[2:], cell.rep_ids.size, n_boot)
        for gk, g in cell.groups.items():
            rows.append(_summarise_group(data, gk, g, idx, seed, n_boot))
    return rows


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "nan" if math.isnan(v) else repr(v)
    return str(v)


def write_summary_csv(rows, path):
    """Write dict rows to CSV (union of keys, first-seen order); refuses to overwrite."""
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "x", newline="") as fh:   # 'x' raises FileExistsError
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([_fmt(r[k]) if k in r else "" for k in fields])
    return path
