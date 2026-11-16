"""Estimation sample, within-country slope design and balanced country cohorts.

Sample rule: complete cases on the outcome, the 12 controls and both keys, then
drop countries with a single observation (country singletons).  Zero outcomes
are kept.  All-zero country groups are reported in the audit, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTCOME = "mean_lights0708"
CONTROLS = ("centr_tribe", "lnpdmean00s", "capdistance1", "seadist1", "borderdist1", "lnwaterkm",
            "lnkm2split", "mean_elev", "mean_suit", "malariasuit", "petroleum", "diamondd")
TARGET = "centr_tribe"
COUNTRY, FAMILY = "wbcode", "cluster"


@dataclass(frozen=True)
class Sample:
    y: np.ndarray  # (n,) raw outcome
    X: np.ndarray  # (n, p) controls in CONTROLS order, original units
    country: np.ndarray  # (n,) integer country code 0..Nc-1
    family: np.ndarray  # (n,) integer family code 0..Nf-1
    inter: np.ndarray  # (n,) integer country x family code 0..Ni-1
    country_labels: tuple
    family_labels: tuple
    audit: dict

    @property
    def n(self):
        return len(self.y)


def _codes(values):
    labels, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return codes.astype(np.int64), tuple(labels.tolist())


def load_sample(path):
    raw = pd.read_csv(path, dtype={COUNTRY: str, FAMILY: str}, keep_default_na=True)
    num = raw[[OUTCOME, *CONTROLS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keys_ok = np.ones(len(raw), bool)
    for k in (COUNTRY, FAMILY):
        keys_ok &= raw[k].notna().to_numpy() & (raw[k].fillna("").str.strip() != "").to_numpy()
    complete = np.isfinite(num).all(axis=1) & keys_ok
    cc = raw.loc[complete].copy()
    size = cc.groupby(COUNTRY)[OUTCOME].transform("size").to_numpy()
    singles = cc.loc[size == 1, COUNTRY].tolist()
    df = cc.loc[size > 1].reset_index(drop=True)

    y = df[OUTCOME].to_numpy(float)
    if np.any(y < 0):
        raise ValueError("The exponential-mean application requires nonnegative outcomes.")
    X = df[list(CONTROLS)].to_numpy(float)
    country, clab = _codes(df[COUNTRY])
    family, flab = _codes(df[FAMILY])
    inter = np.unique(country * len(flab) + family, return_inverse=True)[1].astype(np.int64)

    sizes = np.bincount(country)
    totals = np.bincount(country, weights=y)
    all_zero = [clab[g] for g in np.flatnonzero(totals == 0)]
    audit = {
        "rows_raw": int(len(raw)),
        "rows_complete_case": int(complete.sum()),
        "rows_dropped_incomplete": int((~complete).sum()),
        "country_singletons_dropped": [str(s) for s in singles],
        "n_obs": int(len(df)),
        "n_countries": len(clab),
        "n_families": len(flab),
        "n_country_family_intersections": int(inter.max() + 1),
        "n_zero_outcomes": int((y == 0).sum()),
        "n_negative_outcomes": int((y < 0).sum()),
        "country_size_min": int(sizes.min()),
        "country_size_max": int(sizes.max()),
        "all_zero_country_groups": all_zero,
        "n_all_zero_country_groups": len(all_zero),
        "families_spanning_multiple_countries": int(
            sum(len(np.unique(country[family == f])) > 1 for f in range(len(flab)))),
    }
    return Sample(y, X, country, family, inter, clab, flab, audit)


def within_design(sample):
    """B = (X - country mean) / s with fixed column scales s (RMS of the centred
    column).  Returns (B, s); the slope in original units is theta / s."""
    nc = int(sample.country.max() + 1)
    cnt = np.bincount(sample.country, minlength=nc).astype(float)
    means = np.stack([np.bincount(sample.country, weights=col, minlength=nc) for col in sample.X.T], axis=1)
    Xc = sample.X - (means / cnt[:, None])[sample.country]
    s = np.sqrt(np.mean(Xc * Xc, axis=0))
    if np.any(s == 0):
        bad = [CONTROLS[k] for k in np.flatnonzero(s == 0)]
        raise ValueError(f"no within-country variation in {bad}")
    return Xc / s, s


def design_rank(sample, B):
    """Rank of the within design and of the explicit [X, country dummies] design."""
    D = np.zeros((sample.n, int(sample.country.max() + 1)))
    D[np.arange(sample.n), sample.country] = 1.0
    sv = np.linalg.svd(B, compute_uv=False)
    full = np.hstack([sample.X, D])
    return {
        "within_design_rank": int(np.linalg.matrix_rank(B)),
        "within_design_columns": int(B.shape[1]),
        "within_design_singular_value_ratio": float(sv.min() / sv.max()),
        "dummy_design_rank": int(np.linalg.matrix_rank(full)),
        "dummy_design_columns": int(full.shape[1]),
    }


def cohort_rows(country):
    """Balanced cohorts: one (G, T) array of row indices per distinct country size T.

    Countries are ordered by code and rows keep sample order; every row appears
    exactly once and no cohort is padded."""
    out = []
    sizes = np.bincount(country)
    for T in np.unique(sizes[sizes > 0]):
        members = np.flatnonzero(sizes == T)
        out.append(np.stack([np.flatnonzero(country == g) for g in members]))
    return out
