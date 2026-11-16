"""Exact complete-case, tail-deletion and country-singleton sample rules."""

from __future__ import annotations
import io
import pathlib
import numpy as np
import pandas as pd
from .data import COUNTRY,FAMILY,OUTCOME,CONTROLS,load_sample

def _rule_rows(raw, excluded=()):
    """Source-row mapping of the existing parser's finite-complete-case/singleton rule."""
    available = raw.drop(index=list(excluded))
    numeric = available[[OUTCOME, *CONTROLS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    complete = np.isfinite(numeric).all(axis=1)
    for key in (COUNTRY, FAMILY):
        complete &= available[key].notna().to_numpy() & (available[key].fillna("").str.strip() != "").to_numpy()
    eligible = available.loc[complete]
    sizes = eligible.groupby(COUNTRY)[OUTCOME].transform("size")
    return eligible.index[sizes > 1].tolist(), eligible.index[sizes == 1].tolist()

def deletion_samples(data_path):
    """Deterministic plans and unchanged-parser samples, preserving original CSV bytes."""
    payload = pathlib.Path(data_path).read_bytes()
    raw = pd.read_csv(io.BytesIO(payload), dtype={COUNTRY: str, FAMILY: str}, keep_default_na=True)
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(raw) + 1:
        raise ValueError("Tail runner requires one physical data line per CSV record; no rows were rewritten.")
    header, records = lines[0], lines[1:]
    baseline_rows, baseline_singletons = _rule_rows(raw)
    order = sorted(baseline_rows, key=lambda row: (-float(raw.at[row, OUTCOME]), row))
    n = len(order)
    counts = (("full", 0), ("bright_top1row", 1), ("bright_top1pct", (n + 99) // 100),
              ("bright_top2pct", (2 * n + 99) // 100), ("bright_top5pct", (5 * n + 99) // 100))
    samples = []
    for sample_id, k in counts:
        if k >= n:
            raise ValueError(f"{sample_id}: deletion leaves no baseline observations")
        requested = order[:k]
        requested_set = set(requested)
        retained, singleton_rows = _rule_rows(raw, requested)
        stream = header + b"".join(line if line.endswith((b"\n", b"\r")) else line + b"\n"
                                  for i, line in enumerate(records) if i not in requested_set)
        sample = load_sample(io.BytesIO(stream))
        expected = raw.loc[retained]
        np.testing.assert_array_equal(sample.y, expected[OUTCOME].to_numpy(float))
        np.testing.assert_array_equal(sample.X, expected[list(CONTROLS)].to_numpy(float))
        np.testing.assert_array_equal(np.asarray(sample.country_labels)[sample.country], expected[COUNTRY].to_numpy(str))
        np.testing.assert_array_equal(np.asarray(sample.family_labels)[sample.family], expected[FAMILY].to_numpy(str))
        cascade = [i for i in singleton_rows if i not in set(baseline_singletons)]
        removed = sorted(set(baseline_rows) - set(retained))
        cutoff = float(raw.at[order[k - 1], OUTCOME]) if k else np.nan
        next_y = float(raw.at[order[k], OUTCOME]) if k else np.nan
        meta = {
            "sample_id": sample_id, "n_requested_deletions": k, "n_additional_singletons": len(cascade),
            "n_removed_total": len(removed), "n_obs": sample.n,
            "n_zero_outcomes": sample.audit["n_zero_outcomes"], "n_countries": sample.audit["n_countries"],
            "n_families": sample.audit["n_families"], "n_intersections": sample.audit["n_country_family_intersections"],
            "additional_singleton_countries": "|".join(raw.loc[cascade, COUNTRY].astype(str)),
            "cutoff_raw_y": cutoff, "next_rank_raw_y": next_y,
            "tie_at_cutoff": bool(k and cutoff == next_y),
            "removed_raw_outcome_share": float(raw.loc[removed, OUTCOME].sum() / raw.loc[baseline_rows, OUTCOME].sum()),
        }
        samples.append({"sample": sample, "metadata": meta, "requested_source_rows": requested,
                        "singleton_cascade_source_rows": cascade, "retained_source_rows": retained})
    return raw, baseline_rows, baseline_singletons, order, samples

def all_samples(data_path):
    raw, baseline, baseline_singletons, order, plans = deletion_samples(data_path)
    excluded = [i for i in baseline if float(raw.at[i, OUTCOME]) == 0.0]
    retained, singleton_rows = _rule_rows(raw, excluded)
    payload = pathlib.Path(data_path).read_bytes()
    lines = payload.splitlines(keepends=True)
    exclude_set = set(excluded)
    stream = lines[0] + b"".join(line if line.endswith((b"\n", b"\r")) else line+b"\n"
                               for i,line in enumerate(lines[1:]) if i not in exclude_set)
    sample = load_sample(io.BytesIO(stream))
    expected = raw.loc[retained]
    np.testing.assert_array_equal(sample.y,expected[OUTCOME].to_numpy(float))
    np.testing.assert_array_equal(sample.X,expected[list(CONTROLS)].to_numpy(float))
    cascade = [i for i in singleton_rows if i not in set(baseline_singletons)]
    removed = sorted(set(baseline)-set(retained))
    meta = dict(sample_id="positive_full", n_requested_deletions=len(excluded),
        n_additional_singletons=len(cascade), n_removed_total=len(removed), n_obs=sample.n,
        n_zero_outcomes=sample.audit["n_zero_outcomes"], n_countries=sample.audit["n_countries"],
        n_families=sample.audit["n_families"], n_intersections=sample.audit["n_country_family_intersections"])
    plans.append(dict(sample=sample, metadata=meta, requested_source_rows=excluded,
        singleton_cascade_source_rows=cascade, retained_source_rows=retained))
    return plans
