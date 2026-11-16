"""Validated scientific configurations for the WZ replication study.

Configurations are loaded once and deep-frozen. Numerical settings and
random-stream dimensions are retained from the paper's study.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from types import MappingProxyType

from .versions import ACCEPTANCE_RULE_VERSION

REQUIRED_KEYS = (
    "config_name",
    "packet",
    "base_seed",
    "master_N",
    "master_T",
    "dgps",
    "cells",
    "reps",
    "tau_grid",
    "cancel_solve_order",
    "continuation_neighbor",
    "run_all_starts",
    "contrast_basis",
    "offdiag",
    "dgp_params",
    "solver",
    "tolerances",
    "tolerances_frozen",
    "joint_continuation",
    "wz",
    "summary",
    "main_run",
)
TAU_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
# TAU_GRID is the SUPPORTED grid (historical seven-power configurations stay
# readable).  A configuration's ``tau_grid`` is its scientific grid: a sorted,
# duplicate-free subset of TAU_GRID that must contain PPML (1) and GPML (2).
# Supporting a subset is not a choice of the study grid.
SUPPORTED_TAU_GRID = TAU_GRID
REQUIRED_TAUS = (1.0, 2.0)
# Every root-acceptance threshold (all applied on every acceptance route).
ACCEPTANCE_TOLERANCES = ("tol_scaled", "tol_step", "rcond_min", "tol_metric_residual", "tol_metric_step")

# Bounded joint-root continuation controls.
JOINT_CONTINUATION_KEYS = ("enabled", "step_initial", "step_max", "step_min", "contraction", "growth",
                           "growth_after_successes", "max_calls_per_target", "seed_policy")
JOINT_CONTINUATION_SEED_POLICY = "nearest_certified_direct_J_or_C1_abs_tau_tie_lower"


class ConfigError(ValueError):
    pass


def canonical_json(obj):
    return json.dumps(_thaw(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(cfg):
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()


def _freeze(obj):
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _thaw(obj):
    if isinstance(obj, (dict, MappingProxyType)):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_thaw(v) for v in obj]
    return obj


def _is_int(v):
    """True integers only: bool, floats (even 2.0), NaN/inf and strings are not."""
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _seq(v, what):
    if not isinstance(v, (list, tuple)):
        raise ConfigError(f"{what} must be a list, got {type(v).__name__}")
    return v


def check_dgps(dgps, allowed=None):
    """Non-empty, duplicate-free tuple of ints in {1,2,3,4}; subset of ``allowed``."""
    vals = _seq(dgps, "dgps")
    if not vals:
        raise ConfigError("dgps must be non-empty")
    if not all(_is_int(d) for d in vals):
        raise ConfigError(f"dgps must be integers, got {list(vals)!r}")
    out = tuple(int(d) for d in vals)
    if len(set(out)) != len(out):
        raise ConfigError(f"duplicate dgps in {list(out)}")
    if not set(out) <= {1, 2, 3, 4}:
        raise ConfigError(f"dgps must lie in {{1,2,3,4}}, got {list(out)}")
    if allowed is not None and not set(out) <= {int(a) for a in allowed}:
        raise ConfigError(f"dgps {list(out)} are not configured ({list(allowed)})")
    return out


def check_cells(cells, master_N, master_T, allowed=None):
    """Non-empty, duplicate-free tuple of integer (N, T) inside the master array;
    for execution overrides ``allowed`` is the configured cell list."""
    vals = _seq(cells, "cells")
    if not vals:
        raise ConfigError("cells must be non-empty")
    out = []
    for c in vals:
        if not isinstance(c, (list, tuple)) or len(c) != 2 or not all(_is_int(v) for v in c):
            raise ConfigError(f"cell {c!r} must be an integer (N, T) pair")
        n, t = int(c[0]), int(c[1])
        if not (2 <= n <= int(master_N) and 2 <= t <= int(master_T)):
            raise ConfigError(f"cell ({n},{t}) outside the master array")
        out.append((n, t))
    if len(set(out)) != len(out):
        raise ConfigError(f"duplicate cells in {out}")
    if allowed is not None:
        ok = {(int(a), int(b)) for a, b in allowed}
        if not set(out) <= ok:
            raise ConfigError(f"cells {out} are not all configured cells")
    return tuple(out)


def check_reps(reps, allowed=None):
    """Closed integer interval (lo, hi), 1 <= lo <= hi; inside ``allowed`` if given.
    A rep list such as [1, 2, 2] is malformed, not an ID list."""
    vals = _seq(reps, "reps")
    if len(vals) != 2 or not all(_is_int(v) for v in vals):
        raise ConfigError(f"reps must be an integer [first, last] interval, got {list(vals)!r}")
    lo, hi = int(vals[0]), int(vals[1])
    if not 1 <= lo <= hi:
        raise ConfigError("reps must be [first, last] with 1 <= first <= last")
    if allowed is not None and not (int(allowed[0]) <= lo and hi <= int(allowed[1])):
        raise ConfigError(f"reps [{lo},{hi}] outside the configured interval {list(allowed)}")
    return lo, hi


def _tau_value(v, what):
    """One supported power: a real number (not bool/str), finite, in TAU_GRID."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ConfigError(f"{what} must be a number, got {v!r}")
    t = float(v)
    if not math.isfinite(t):
        raise ConfigError(f"{what} must be finite, got {v!r}")
    if t not in TAU_GRID:
        raise ConfigError(f"{what}={v!r} is not in the supported tau grid {TAU_GRID}")
    return t


def check_tau_grid(values):
    """Scientific tau grid: sorted, unique subset of TAU_GRID containing 1 and 2."""
    if not isinstance(values, (list, tuple)):
        raise ConfigError(f"tau grid (tau_grid) must be a list, got {type(values).__name__}")
    grid = tuple(_tau_value(v, "tau grid entry") for v in values)
    if len(set(grid)) != len(grid):
        raise ConfigError(f"tau grid has duplicate powers {list(grid)}")
    if list(grid) != sorted(grid):
        raise ConfigError(f"tau grid must be sorted ascending, got {list(grid)}")
    missing = [t for t in REQUIRED_TAUS if t not in grid]
    if missing:
        raise ConfigError(f"tau grid must contain PPML (1) and GPML (2); missing {missing}")
    return grid


def configured_taus(cfg):
    """The configuration's scientific grid as floats (already validated)."""
    return tuple(float(t) for t in cfg["tau_grid"])


def _check_solve_plan(raw, grid):
    """cancel_solve_order: a permutation of the grid starting at Gamma (2).
    continuation_neighbor: exactly one entry per grid power (no numeric alias
    keys such as "1" and "1.0"); each value null or a grid power solved earlier."""
    order = tuple(_tau_value(v, "cancel_solve_order entry") for v in _seq(raw["cancel_solve_order"],
                                                                        "cancel_solve_order"))
    if sorted(order) != list(grid) or order[0] != 2.0:
        raise ConfigError("cancel_solve_order must permute the tau grid and start at the Gamma root (2)")
    nb_raw = raw["continuation_neighbor"]
    if not isinstance(nb_raw, dict) and not hasattr(nb_raw, "keys"):
        raise ConfigError("continuation_neighbor must be an object")
    nb = {}
    for k, v in nb_raw.items():
        try:
            key = float(k) if isinstance(k, str) else k
        except ValueError:
            raise ConfigError(f"continuation_neighbor key {k!r} is not a number") from None
        t = _tau_value(key, f"continuation_neighbor key {k!r}")
        if t in nb:
            raise ConfigError(f"continuation_neighbor has duplicate keys for tau={t:g}")
        nb[t] = v
    if set(nb) != set(grid):
        raise ConfigError("continuation_neighbor needs one entry per tau")
    pos = {t: i for i, t in enumerate(order)}
    for t, v in nb.items():
        if v is None:
            continue
        ref = _tau_value(v, f"continuation_neighbor[{t:g}]")
        if ref not in pos:
            raise ConfigError(f"neighbour {ref:g} of tau={t:g} is not in the tau grid")
        if pos[ref] >= pos[t]:
            raise ConfigError(f"neighbour of tau={t} must be solved earlier")
    return order, nb


def _dyadic(x):
    """Exact power of two (so every path point of the 0.5-spaced grid is exact)."""
    m, _ = math.frexp(x)
    return x > 0 and m == 0.5


def validate_joint_continuation(jc):
    """Complete, internally consistent continuation controls, else ConfigError.

    Steps must be exact powers of two so each path tau is exactly representable;
    the per-target call budget is a finite positive integer."""
    if not isinstance(jc, dict) and not hasattr(jc, "keys"):
        raise ConfigError("joint_continuation must be an object")
    missing = [k for k in JOINT_CONTINUATION_KEYS if k not in jc]
    if missing:
        raise ConfigError(f"joint_continuation missing keys {missing}")
    extra = sorted(set(jc) - set(JOINT_CONTINUATION_KEYS))
    if extra:
        raise ConfigError(f"joint_continuation has unknown keys {extra}")
    if not isinstance(jc["enabled"], bool):
        raise ConfigError("joint_continuation.enabled must be true or false")
    for k in ("step_initial", "step_max", "step_min", "contraction", "growth"):
        v = jc[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ConfigError(f"joint_continuation.{k} must be a finite number")
    for k in ("growth_after_successes", "max_calls_per_target"):
        if not _is_int(jc[k]) or jc[k] < 1:
            raise ConfigError(f"joint_continuation.{k} must be an integer >= 1")
    s0, smax, smin = float(jc["step_initial"]), float(jc["step_max"]), float(jc["step_min"])
    if not (0 < smin <= s0 <= smax <= TAU_GRID[-1] - TAU_GRID[0]):
        raise ConfigError("joint_continuation needs 0 < step_min <= step_initial <= step_max <= 3")
    for k, v in (("step_initial", s0), ("step_max", smax), ("step_min", smin)):
        if not _dyadic(v):
            raise ConfigError(f"joint_continuation.{k} must be an exact power of two (exact tau arithmetic)")
    if not (0 < float(jc["contraction"]) < 1 and _dyadic(float(jc["contraction"]))):
        raise ConfigError("joint_continuation.contraction must be a power of two in (0, 1)")
    if not (float(jc["growth"]) > 1 and _dyadic(float(jc["growth"]))):
        raise ConfigError("joint_continuation.growth must be a power of two > 1")
    if jc["seed_policy"] != JOINT_CONTINUATION_SEED_POLICY:
        raise ConfigError(f"joint_continuation.seed_policy must be {JOINT_CONTINUATION_SEED_POLICY!r}")
    return jc


def validate(raw):
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"config missing keys {missing}")
    if raw["packet"] not in ("M", "H", "H_R", "P"):
        raise ConfigError("packet must be M, H, H_R or P")
    grid = check_tau_grid(raw["tau_grid"])
    _check_solve_plan(raw, grid)
    for key in ("master_N", "master_T"):
        if not _is_int(raw[key]) or raw[key] < 2:
            raise ConfigError(f"{key} must be an integer >= 2")
    check_dgps(raw["dgps"])
    check_cells(raw["cells"], raw["master_N"], raw["master_T"])
    check_reps(raw["reps"])
    if raw["main_run"] and raw["packet"] != "M":
        raise ConfigError("only the packet-M configuration may be a main run")
    if not raw["main_run"] and raw["packet"] == "M":
        raise ConfigError("packet M is reserved for the main configuration")
    tol = raw["tolerances"]
    for key in ACCEPTANCE_TOLERANCES + ("tol_root", "inference_rel"):
        if key not in tol:
            raise ConfigError(f"tolerance {key} is missing")
        v = tol[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v > 0):
            raise ConfigError(f"tolerance {key} must be a finite positive number")
    for key in ("tol_metric_residual_normalisation", "tol_metric_step_normalisation"):
        if not isinstance(tol.get(key), str) or not tol[key].strip():
            raise ConfigError(f"tolerances.{key} must state the metric normalisation")
    if tol.get("acceptance_rule_version") != ACCEPTANCE_RULE_VERSION:
        raise ConfigError(f"tolerances.acceptance_rule_version must be {ACCEPTANCE_RULE_VERSION!r} "
                          f"(the acceptance rule implemented by this code), got {tol.get('acceptance_rule_version')!r}")
    validate_joint_continuation(raw["joint_continuation"])
    return raw


def load_config(path):
    with open(path) as fh:
        raw = json.load(fh)
    return _freeze(validate(raw))


def from_dict(raw):
    return _freeze(validate(json.loads(json.dumps(raw))))


def with_overrides(cfg, **changes):
    """New frozen config (new hash) with top-level overrides; used by tests and
    by the preflight to derive bounded sub-runs.  Never mutates ``cfg``."""
    raw = _thaw(cfg)
    raw.update(changes)
    return from_dict(raw)
