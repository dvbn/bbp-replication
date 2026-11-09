from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _as_group_array(group: np.ndarray) -> np.ndarray:
    raw = np.asarray(group)
    if raw.ndim != 1:
        raise ValueError("FE group labels must be one-dimensional")
    if np.issubdtype(raw.dtype, np.integer):
        if raw.size and np.min(raw) < 0:
            raise ValueError("FE group labels must be non-negative integers")
        if raw.size and int(np.max(raw)) <= np.iinfo(np.int32).max:
            return raw.astype(np.int32, copy=False)
        return raw.astype(np.int64, copy=False)

    array = np.asarray(raw, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("FE group labels must be one-dimensional")
    if array.size and np.min(array) < 0:
        raise ValueError("FE group labels must be non-negative integers")
    if array.size and int(np.max(array)) <= np.iinfo(np.int32).max:
        return array.astype(np.int32, copy=False)
    return array


def _resolve_n_groups(group: np.ndarray, n_groups: int | None = None) -> int:
    if n_groups is not None:
        return int(n_groups)
    return int(group.max()) + 1 if group.size else 0


def demean_oneway(
    values: np.ndarray,
    group: np.ndarray,
    n_groups: int | None = None,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Within-group demean a vector or matrix, optionally with weights."""

    array = _as_float_array(values)
    group_arr = _as_group_array(group)
    if array.shape[0] != group_arr.shape[0]:
        raise ValueError("values and group must have the same number of rows")
    if array.ndim == 2 and array.shape[1] == 0:
        return array.copy()

    n_group = _resolve_n_groups(group_arr, n_groups)
    return _demean_oneway_prepared(
        array,
        group_arr,
        n_group,
        weights=weights,
    )


def _demean_oneway_prepared(
    array: np.ndarray,
    group_arr: np.ndarray,
    n_group: int,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Within-group demean for already validated arrays."""

    if weights is None:
        counts = np.bincount(group_arr, minlength=n_group).astype(float)
        denom = np.maximum(counts, 1.0)
        if array.ndim == 1:
            sums = np.bincount(group_arr, weights=array, minlength=n_group)
            return array - (sums / denom)[group_arr]
        out = np.empty_like(array, dtype=float)
        for column in range(array.shape[1]):
            sums = np.bincount(
                group_arr,
                weights=array[:, column],
                minlength=n_group,
            )
            out[:, column] = array[:, column] - (sums / denom)[group_arr]
        return out

    weight_arr = _as_float_array(weights)
    if weight_arr.shape != group_arr.shape:
        raise ValueError("weights and group must have the same shape")
    weight_sum = np.bincount(group_arr, weights=weight_arr, minlength=n_group)
    valid = weight_sum > 1e-15

    if array.ndim == 1:
        sums = np.bincount(
            group_arr,
            weights=weight_arr * array,
            minlength=n_group,
        )
        means = np.zeros_like(weight_sum, dtype=float)
        means[valid] = sums[valid] / weight_sum[valid]
        return array - means[group_arr]

    out = np.empty_like(array, dtype=float)
    for column in range(array.shape[1]):
        sums = np.bincount(
            group_arr,
            weights=weight_arr * array[:, column],
            minlength=n_group,
        )
        means = np.zeros_like(weight_sum, dtype=float)
        means[valid] = sums[valid] / weight_sum[valid]
        out[:, column] = array[:, column] - means[group_arr]
    return out


def _demean_oneway_prepared_into(
    array: np.ndarray,
    group_arr: np.ndarray,
    n_group: int,
    *,
    out: np.ndarray,
    weights: np.ndarray | None = None,
    denom: np.ndarray | None = None,
    weight_sum: np.ndarray | None = None,
    valid_weight_group: np.ndarray | None = None,
    weighted_values: np.ndarray | None = None,
) -> np.ndarray:
    """Within-group demean into a caller-owned output array."""

    if weights is None:
        if denom is None:
            counts = np.bincount(group_arr, minlength=n_group).astype(float)
            denom = np.maximum(counts, 1.0)
        if array.ndim == 1:
            sums = np.bincount(group_arr, weights=array, minlength=n_group)
            means = sums / denom
            np.subtract(array, means[group_arr], out=out)
            return out
        for column in range(array.shape[1]):
            sums = np.bincount(
                group_arr,
                weights=array[:, column],
                minlength=n_group,
            )
            means = sums / denom
            np.subtract(array[:, column], means[group_arr], out=out[:, column])
        return out

    weight_arr = _as_float_array(weights)
    if weight_arr.shape != group_arr.shape:
        raise ValueError("weights and group must have the same shape")
    if weight_sum is None:
        weight_sum = np.bincount(group_arr, weights=weight_arr, minlength=n_group)
    if valid_weight_group is None:
        valid_weight_group = weight_sum > 1e-15

    if array.ndim == 1:
        if weighted_values is None:
            weighted = weight_arr * array
        else:
            np.multiply(weight_arr, array, out=weighted_values)
            weighted = weighted_values
        sums = np.bincount(group_arr, weights=weighted, minlength=n_group)
        means = np.zeros_like(weight_sum, dtype=float)
        means[valid_weight_group] = (
            sums[valid_weight_group] / weight_sum[valid_weight_group]
        )
        np.subtract(array, means[group_arr], out=out)
        return out

    if weighted_values is None:
        weighted_values = np.empty(array.shape[0], dtype=float)
    for column in range(array.shape[1]):
        np.multiply(weight_arr, array[:, column], out=weighted_values)
        sums = np.bincount(group_arr, weights=weighted_values, minlength=n_group)
        means = np.zeros_like(weight_sum, dtype=float)
        means[valid_weight_group] = (
            sums[valid_weight_group] / weight_sum[valid_weight_group]
        )
        np.subtract(array[:, column], means[group_arr], out=out[:, column])
    return out


def demean(
    values: np.ndarray,
    group_i: np.ndarray | None = None,
    group_t: np.ndarray | None = None,
    n_i: int | None = None,
    n_t: int | None = None,
    *,
    weights: np.ndarray | None = None,
    tol: float = 1e-12,
    maxiter: int = 10000,
) -> np.ndarray:
    """Dispatch one- or two-way alternating-projection demeaning."""

    groups: list[np.ndarray] = []
    n_groups: list[int | None] = []
    if group_i is not None:
        groups.append(group_i)
        n_groups.append(n_i)
    if group_t is not None:
        groups.append(group_t)
        n_groups.append(n_t)
    if not groups:
        return _as_float_array(values).copy()
    if len(groups) == 1:
        return demean_oneway(values, groups[0], n_groups[0], weights=weights)
    return demean_manyway(
        values,
        tuple(groups),
        n_groups=tuple(n_groups),
        weights=weights,
        tol=tol,
        maxiter=maxiter,
    )


def demean_manyway(
    values: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    n_groups: Sequence[int | None] | None = None,
    weights: np.ndarray | None = None,
    tol: float = 1e-12,
    maxiter: int = 10000,
) -> np.ndarray:
    """Many-way alternating-projection demeaning over arbitrary FE blocks."""
    out, converged = _demean_manyway_checked(
        values,
        groups,
        n_groups=n_groups,
        weights=weights,
        tol=tol,
        maxiter=maxiter,
    )
    if not converged:
        raise RuntimeError("manyway_fe_demean_not_converged")
    return out


def _demean_manyway_checked(
    values: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    n_groups: Sequence[int | None] | None = None,
    weights: np.ndarray | None = None,
    tol: float = 1e-12,
    maxiter: int = 10000,
) -> tuple[np.ndarray, bool]:
    """Demean and report whether alternating projection met `tol`."""

    array = _as_float_array(values)
    if len(groups) == 0:
        return array.copy(), True
    if array.ndim == 2 and array.shape[1] == 0:
        return array.copy(), True

    group_arrays = tuple(_as_group_array(group) for group in groups)
    for group_arr in group_arrays:
        if array.shape[0] != group_arr.shape[0]:
            raise ValueError("values and every FE group must have matching rows")
    if n_groups is None:
        n_group_tuple = tuple(_resolve_n_groups(group) for group in group_arrays)
    else:
        n_group_input = tuple(n_groups)
        if len(n_group_input) != len(group_arrays):
            raise ValueError("n_groups must match the number of FE groups")
        n_group_tuple = tuple(
            _resolve_n_groups(group, n_group)
            for group, n_group in zip(group_arrays, n_group_input, strict=True)
        )

    weight_arr = None if weights is None else _as_float_array(weights)
    if len(group_arrays) == 1:
        return (
            _demean_oneway_prepared(
                array,
                group_arrays[0],
                n_group_tuple[0],
                weights=weight_arr,
            ),
            True,
        )

    if array.ndim == 2:
        out = np.array(array, dtype=float, order="F", copy=True)
        scratch = np.empty(array.shape, dtype=float, order="F")
        previous = np.empty(array.shape, dtype=float, order="F")
    else:
        out = array.copy()
        scratch = np.empty_like(out, dtype=float)
        previous = np.empty_like(out, dtype=float)
    weighted_values = (
        None if weight_arr is None else np.empty(array.shape[0], dtype=float)
    )
    group_stats: list[tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]
    group_stats = []
    for group_arr, n_group in zip(group_arrays, n_group_tuple, strict=True):
        if weight_arr is None:
            counts = np.bincount(group_arr, minlength=n_group).astype(float)
            group_stats.append((np.maximum(counts, 1.0), None, None))
        else:
            weight_sum = np.bincount(
                group_arr,
                weights=weight_arr,
                minlength=n_group,
            )
            group_stats.append((None, weight_sum, weight_sum > 1e-15))

    for _ in range(maxiter):
        np.copyto(previous, out)
        current = out
        target = scratch
        for group_index, (group_arr, n_group) in enumerate(
            zip(group_arrays, n_group_tuple, strict=True)
        ):
            denom, weight_sum, valid_weight_group = group_stats[group_index]
            _demean_oneway_prepared_into(
                current,
                group_arr,
                n_group,
                weights=weight_arr,
                out=target,
                denom=denom,
                weight_sum=weight_sum,
                valid_weight_group=valid_weight_group,
                weighted_values=weighted_values,
            )
            current, target = target, current
        if current is not out:
            np.copyto(out, current)
        if out.size and (not np.isfinite(np.max(out)) or not np.isfinite(np.min(out))):
            return out, False
        np.subtract(out, previous, out=scratch)
        np.abs(scratch, out=scratch)
        if np.max(scratch) < tol:
            return out, True
    return out, False


def group_logsumexp(
    values: np.ndarray,
    group: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Groupwise log-sum-exp with empty-group protection."""

    values_arr = _as_float_array(values)
    group_arr = _as_group_array(group)
    return _group_logsumexp_prepared(values_arr, group_arr, int(n_groups))


def _group_logsumexp_prepared(
    values_arr: np.ndarray,
    group_arr: np.ndarray,
    n_group: int,
) -> np.ndarray:
    """Groupwise log-sum-exp for already validated arrays."""

    out = np.full(n_group, -np.inf, dtype=float)
    if values_arr.size == 0 or n_group == 0:
        return out

    group_max = np.full(n_group, -np.inf, dtype=float)
    np.maximum.at(group_max, group_arr, values_arr)
    posinf = np.isposinf(group_max)
    finite_groups = np.isfinite(group_max)
    valid_obs = finite_groups[group_arr] & np.isfinite(values_arr)
    if np.any(valid_obs):
        shifted = np.zeros_like(values_arr, dtype=float)
        shifted[valid_obs] = np.exp(
            values_arr[valid_obs] - group_max[group_arr][valid_obs]
        )
        sums = np.bincount(group_arr, weights=shifted, minlength=n_group)
        valid_sums = finite_groups & (sums > 0.0)
        out[valid_sums] = group_max[valid_sums] + np.log(sums[valid_sums])
    out[posinf] = np.inf
    return out


def _profile_one_effect(
    endog: np.ndarray,
    partial_linear: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    tau: float,
) -> np.ndarray:
    positive = endog > 0.0
    log_numerator_terms = np.full(partial_linear.shape[0], -np.inf, dtype=float)
    log_numerator_terms[positive] = (
        np.log(endog[positive]) + (1.0 - tau) * partial_linear[positive]
    )
    log_denominator_terms = (2.0 - tau) * partial_linear
    log_numerator = _group_logsumexp_prepared(
        log_numerator_terms,
        group,
        int(n_groups),
    )
    log_denominator = _group_logsumexp_prepared(
        log_denominator_terms,
        group,
        int(n_groups),
    )

    effect = np.zeros(int(n_groups), dtype=float)
    valid = np.isfinite(log_numerator) & np.isfinite(log_denominator)
    effect[valid] = log_numerator[valid] - log_denominator[valid]
    return effect


def profile_fe(
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    group_i: np.ndarray | None,
    group_t: np.ndarray | None,
    n_i: int | None,
    n_t: int | None,
    *,
    tau: float = 2.0,
    maxiter: int = 10000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Profile one- or two-way FE by stable log-sum-exp first-order updates."""

    y = _as_float_array(endog)
    x = _as_float_array(exog)
    b = _as_float_array(beta)
    if y.ndim != 1:
        raise ValueError("endog must be one-dimensional")
    if x.ndim != 2:
        raise ValueError("exog must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError("endog and exog must have the same row count")

    base_linear = x @ b
    if group_i is None and group_t is None:
        return np.zeros(0, dtype=float), None
    if group_i is None:
        group_i = group_t
        n_i = n_t
        group_t = None
        n_t = None

    i_arr = _as_group_array(group_i)
    i_count = _resolve_n_groups(i_arr, n_i)
    alpha = np.zeros(i_count, dtype=float)
    if group_t is None:
        return _profile_one_effect(y, base_linear, i_arr, i_count, tau), None

    t_arr = _as_group_array(group_t)
    t_count = _resolve_n_groups(t_arr, n_t)
    gamma = np.zeros(t_count, dtype=float)

    for _ in range(maxiter):
        alpha_new = _profile_one_effect(
            y,
            base_linear + gamma[t_arr],
            i_arr,
            i_count,
            tau,
        )
        gamma_new = _profile_one_effect(
            y,
            base_linear + alpha_new[i_arr],
            t_arr,
            t_count,
            tau,
        )
        if alpha_new.size:
            shift = float(alpha_new[0])
            alpha_new = alpha_new - shift
            gamma_new = gamma_new + shift
        if not (np.all(np.isfinite(alpha_new)) and np.all(np.isfinite(gamma_new))):
            raise RuntimeError("twoway_fe_profile_not_converged")
        diff = max(
            float(np.max(np.abs(alpha_new - alpha))) if alpha.size else 0.0,
            float(np.max(np.abs(gamma_new - gamma))) if gamma.size else 0.0,
        )
        alpha = alpha_new
        gamma = gamma_new
        if diff < tol:
            return alpha, gamma
    raise RuntimeError("twoway_fe_profile_not_converged")


def profile_fe_manyway(
    endog: np.ndarray,
    exog: np.ndarray,
    beta: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    tau: float = 2.0,
    maxiter: int = 10000,
    tol: float = 1e-10,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Profile arbitrary many-way FE and return effects plus full linear index."""

    xb = _as_float_array(exog) @ _as_float_array(beta)
    if len(groups) == 0:
        return (), xb

    y = _as_float_array(endog)
    group_arrays = tuple(_as_group_array(group) for group in groups)
    offset = 0.0
    effects = [
        np.zeros(_resolve_n_groups(group), dtype=float) for group in group_arrays
    ]

    for _ in range(maxiter):
        previous_linear = xb + offset
        for effect, group in zip(effects, group_arrays, strict=True):
            if effect.size:
                previous_linear += effect[group]

        for index, group in enumerate(group_arrays):
            partial = xb + offset
            for other_index, other_group in enumerate(group_arrays):
                if other_index == index:
                    continue
                other_effect = effects[other_index]
                if other_effect.size:
                    partial += other_effect[other_group]

            updated = _profile_one_effect(
                y,
                partial,
                group,
                effects[index].size,
                tau,
            )
            if updated.size:
                shift = float(updated[0])
                updated = updated - shift
                offset += shift
            effects[index] = updated

        linear = xb + offset
        for effect, group in zip(effects, group_arrays, strict=True):
            if effect.size:
                linear += effect[group]
        if not np.all(np.isfinite(linear)):
            raise RuntimeError("manyway_fe_profile_not_converged")
        if np.max(np.abs(linear - previous_linear)) < tol:
            return tuple(effects), linear

    raise RuntimeError("manyway_fe_profile_not_converged")
