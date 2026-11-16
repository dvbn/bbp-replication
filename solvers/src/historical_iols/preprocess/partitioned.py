from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr


@dataclass(frozen=True)
class PartitionedPreprocessMeta:
    n_obs_in: int
    n_obs_out: int
    kept_obs_mask: np.ndarray
    kept_x_idx: np.ndarray
    dropped_x_idx: np.ndarray
    intercept_idx: int | None
    singleton_rounds: int
    singleton_dropped: int
    count_screen_rounds: int
    count_screen_dropped: int
    fe_levels_before: dict[str, int]
    fe_levels_after: dict[str, int]
    zero_positive_levels_before: dict[str, int]
    zero_sum_levels_before: dict[str, int]
    zero_positive_levels: dict[str, int]
    zero_sum_levels: dict[str, int]
    constant_dropped_x_idx: np.ndarray
    rank_dropped_x_idx: np.ndarray
    transformed_rank: int


@dataclass(frozen=True)
class RegressorRankScreen:
    exog: np.ndarray
    kept_x_idx: np.ndarray
    dropped_x_idx: np.ndarray
    intercept_idx: int | None
    constant_dropped_x_idx: np.ndarray
    rank_dropped_x_idx: np.ndarray
    transformed_rank: int
    rank_basis: str

    @property
    def dropped_column_count(self) -> int:
        return int(self.dropped_x_idx.size)

    @property
    def has_dropped_columns(self) -> bool:
        return bool(self.dropped_column_count)


def _drop_rows(arr: Sequence[object] | np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(arr)[mask]


def _nonzero_constant_columns(X: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if X.shape[0] == 0 or X.shape[1] == 0:
        return np.array([], dtype=int)
    ptp = np.ptp(X, axis=0)
    nonzero = np.max(np.abs(X), axis=0) > tol
    return np.where((ptp <= tol) & nonzero)[0]


def _handle_constant_columns(
    X: np.ndarray,
    *,
    has_constant: str = "skip",
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    X = np.asarray(X, dtype=float)
    const_idx = _nonzero_constant_columns(X, tol=tol)
    if len(const_idx) == 0:
        return X, np.arange(X.shape[1]), np.array([], dtype=int), None

    if has_constant == "raise":
        raise ValueError(f"constant columns present at indices {const_idx.tolist()}")
    if has_constant == "add":
        return X, np.arange(X.shape[1]), np.array([], dtype=int), None
    if has_constant != "skip":
        raise ValueError("has_constant must be one of {'skip','raise','add'}")

    keep_const = int(const_idx[0])
    drop_const = np.array([index for index in const_idx[1:]], dtype=int)
    keep_mask = np.ones(X.shape[1], dtype=bool)
    keep_mask[drop_const] = False
    kept_idx = np.where(keep_mask)[0]
    return X[:, kept_idx], kept_idx, drop_const, keep_const


def _normalize_fe_levels(
    fe_blocks: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    out: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, values in fe_blocks.items():
        _, inv = np.unique(np.asarray(values), return_inverse=True)
        out[str(name)] = inv
        counts[str(name)] = int(inv.max()) + 1 if len(inv) else 0
    return out, counts


def _iterative_support_mask(
    Y: np.ndarray,
    fe_blocks: Mapping[str, np.ndarray],
    *,
    drop_singletons: bool,
    count_model: bool,
    positive_eps: float = 0.0,
) -> tuple[np.ndarray, int, int, int, int]:
    if not fe_blocks:
        n_obs = len(Y)
        return np.ones(n_obs, dtype=bool), 0, 0, 0, 0

    n_obs = len(next(iter(fe_blocks.values())))
    keep = np.ones(n_obs, dtype=bool)
    singleton_rounds = 0
    count_rounds = 0
    singleton_total = 0
    count_total = 0

    while True:
        changed = False

        if count_model:
            to_drop = np.zeros(n_obs, dtype=bool)
            positive = np.asarray(Y > positive_eps, dtype=float)
            for values in fe_blocks.values():
                values_kept = values[keep]
                if len(values_kept) == 0:
                    continue
                pos_counts = np.bincount(
                    values_kept,
                    weights=positive[keep],
                    minlength=int(values_kept.max()) + 1,
                )
                bad_levels = np.where(pos_counts == 0)[0]
                if len(bad_levels) == 0:
                    continue
                cur = np.zeros(n_obs, dtype=bool)
                cur[keep] = np.isin(values_kept, bad_levels)
                to_drop |= cur
            new_drop = to_drop & keep
            if np.any(new_drop):
                keep &= ~new_drop
                count_rounds += 1
                count_total += int(np.sum(new_drop))
                changed = True

        if drop_singletons and np.any(keep):
            to_drop = np.zeros(n_obs, dtype=bool)
            for values in fe_blocks.values():
                values_kept = values[keep]
                if len(values_kept) == 0:
                    continue
                counts = np.bincount(values_kept)
                singleton_levels = np.where(counts == 1)[0]
                if len(singleton_levels) == 0:
                    continue
                cur = np.zeros(n_obs, dtype=bool)
                cur[keep] = np.isin(values_kept, singleton_levels)
                to_drop |= cur
            new_drop = to_drop & keep
            if np.any(new_drop):
                keep &= ~new_drop
                singleton_rounds += 1
                singleton_total += int(np.sum(new_drop))
                changed = True

        if not changed:
            break

    return keep, singleton_rounds, singleton_total, count_rounds, count_total


def _positive_support_diagnostics(
    Y: np.ndarray,
    fe_blocks: Mapping[str, np.ndarray],
) -> tuple[dict[str, int], dict[str, int]]:
    Y = np.asarray(Y, dtype=float)
    positive = Y > 0
    zero_positive_levels: dict[str, int] = {}
    zero_sum_levels: dict[str, int] = {}
    for name, values in fe_blocks.items():
        arr = np.asarray(values)
        if len(arr) == 0:
            zero_positive_levels[str(name)] = 0
            zero_sum_levels[str(name)] = 0
            continue
        n_levels = int(arr.max()) + 1
        pos_counts = np.bincount(
            arr,
            weights=positive.astype(float),
            minlength=n_levels,
        )
        y_sums = np.bincount(arr, weights=Y, minlength=n_levels)
        zero_positive_levels[str(name)] = int(np.sum(pos_counts <= 0))
        zero_sum_levels[str(name)] = int(np.sum(y_sums <= 0))
    return zero_positive_levels, zero_sum_levels


def rank_reveal_qr(
    X: np.ndarray,
    *,
    tol: float = 1e-10,
    protected: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, int]:
    X = np.asarray(X, dtype=float)
    K = X.shape[1]
    protected = tuple(
        sorted(set(int(index) for index in protected if 0 <= int(index) < K))
    )

    if K == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    _, R, piv = qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if len(diag) == 0:
        rank = 0
    else:
        cutoff = tol * max(diag[0], 1.0)
        rank = int(np.sum(diag > cutoff))
    kept = list(sorted(piv[:rank].tolist()))

    for index in protected:
        if index not in kept:
            kept.append(index)
    kept_arr = np.array(sorted(set(kept)), dtype=int)
    dropped_arr = np.array(
        [index for index in range(K) if index not in set(kept_arr.tolist())],
        dtype=int,
    )
    return kept_arr, dropped_arr, rank


def solve_ols_qr_dropcollinear(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-10,
    protected: Sequence[int] = (),
) -> tuple[np.ndarray, dict[str, object]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    kept, dropped, rank = rank_reveal_qr(X, tol=tol, protected=protected)
    beta = np.zeros(X.shape[1], dtype=float)
    if len(kept) > 0:
        beta_kept, *_ = np.linalg.lstsq(X[:, kept], y, rcond=None)
        beta[kept] = beta_kept
    meta = {"kept": kept, "dropped": dropped, "rank": rank}
    return beta, meta


def screen_regressor_matrix(
    X: np.ndarray,
    *,
    rank_design: np.ndarray | None = None,
    rank_basis: str = "raw_exog",
    has_constant: str = "skip",
    tol: float = 1e-10,
) -> RegressorRankScreen:
    """Drop duplicate constants and QR-detected non-estimable regressors."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if rank_design is None:
        rank_design_arr = X
    else:
        rank_design_arr = np.asarray(rank_design, dtype=float)
        if rank_design_arr.shape != X.shape:
            raise ValueError("rank_design must have the same shape as X")

    X_const, kept_const_idx, dropped_const_idx, intercept_idx = (
        _handle_constant_columns(
            X,
            has_constant=has_constant,
        )
    )
    rank_const = rank_design_arr[:, kept_const_idx]
    protected: tuple[int, ...] = ()
    if intercept_idx is not None and rank_basis in {"raw_exog", "raw_retained_exog"}:
        intercept_position = np.flatnonzero(kept_const_idx == int(intercept_idx))
        if intercept_position.size:
            protected = (int(intercept_position[0]),)

    kept_rank_pos, dropped_rank_pos, transformed_rank = rank_reveal_qr(
        rank_const,
        tol=tol,
        protected=protected,
    )
    kept_x_idx = kept_const_idx[kept_rank_pos]
    rank_dropped_x_idx = kept_const_idx[dropped_rank_pos]
    dropped_x_idx = np.array(
        sorted(
            set(int(index) for index in dropped_const_idx.tolist())
            | set(int(index) for index in rank_dropped_x_idx.tolist())
        ),
        dtype=int,
    )
    return RegressorRankScreen(
        exog=X_const[:, kept_rank_pos],
        kept_x_idx=kept_x_idx,
        dropped_x_idx=dropped_x_idx,
        intercept_idx=intercept_idx,
        constant_dropped_x_idx=dropped_const_idx,
        rank_dropped_x_idx=rank_dropped_x_idx,
        transformed_rank=int(transformed_rank),
        rank_basis=str(rank_basis),
    )


def preprocess_partitioned_inputs(
    Y: np.ndarray,
    X: np.ndarray,
    fe_blocks: Mapping[str, np.ndarray],
    *,
    has_constant: str = "skip",
    drop_singletons: bool = True,
    count_model: bool = True,
    positive_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], PartitionedPreprocessMeta]:
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.ndim != 1:
        raise ValueError("Y must be 1-dimensional")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if len(Y) != X.shape[0]:
        raise ValueError("Y and X must have the same number of observations")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains non-finite values")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    raw_blocks = {str(name): np.asarray(values) for name, values in fe_blocks.items()}
    if any(len(values) != len(Y) for values in raw_blocks.values()):
        raise ValueError("all FE blocks must have length equal to len(Y)")

    fe_norm, fe_before = _normalize_fe_levels(raw_blocks)

    kept_obs = np.ones(len(Y), dtype=bool)
    singleton_rounds = 0
    singleton_dropped = 0
    count_screen_rounds = 0
    count_screen_dropped = 0
    if fe_norm:
        (
            kept_obs,
            singleton_rounds,
            singleton_dropped,
            count_screen_rounds,
            count_screen_dropped,
        ) = _iterative_support_mask(
            Y,
            fe_norm,
            drop_singletons=drop_singletons,
            count_model=count_model,
            positive_eps=positive_eps,
        )

    Y1 = _drop_rows(Y, kept_obs)
    X1 = _drop_rows(X, kept_obs)
    fe1_raw = {
        name: _drop_rows(values, kept_obs)
        for name, values in raw_blocks.items()
    }
    fe1, fe_after = _normalize_fe_levels(fe1_raw)

    zero_positive_before, zero_sum_before = _positive_support_diagnostics(Y, fe_norm)
    zero_positive_levels, zero_sum_levels = _positive_support_diagnostics(Y1, fe1)

    screen = screen_regressor_matrix(
        X1,
        has_constant=has_constant,
        rank_basis="raw_retained_exog",
    )

    meta = PartitionedPreprocessMeta(
        n_obs_in=len(Y),
        n_obs_out=len(Y1),
        kept_obs_mask=kept_obs,
        kept_x_idx=screen.kept_x_idx,
        dropped_x_idx=screen.dropped_x_idx,
        intercept_idx=screen.intercept_idx,
        singleton_rounds=singleton_rounds,
        singleton_dropped=singleton_dropped,
        count_screen_rounds=count_screen_rounds,
        count_screen_dropped=count_screen_dropped,
        fe_levels_before=fe_before,
        fe_levels_after=fe_after,
        zero_positive_levels_before=zero_positive_before,
        zero_sum_levels_before=zero_sum_before,
        zero_positive_levels=zero_positive_levels,
        zero_sum_levels=zero_sum_levels,
        constant_dropped_x_idx=screen.constant_dropped_x_idx,
        rank_dropped_x_idx=screen.rank_dropped_x_idx,
        transformed_rank=screen.transformed_rank,
    )
    return Y1, screen.exog, fe1, meta
