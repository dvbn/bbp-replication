"""Cell data: one (sub)panel of origins x destinations x periods.

Arrays are indexed by origin position, destination position and period.  The
country labels identify diagonal pairs (origin label == destination label);
they are excluded only under the explicit secondary ``offdiag`` specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retained import RetainedDesign


@dataclass(frozen=True)
class CellData:
    x: np.ndarray  # (No, Nd, T)
    y: np.ndarray  # (No, Nd, T)
    origin_labels: tuple
    dest_labels: tuple
    offdiag: bool = False

    def __post_init__(self):
        x = np.array(self.x, dtype=float)
        y = np.array(self.y, dtype=float)
        if x.ndim != 3 or x.shape != y.shape:
            raise ValueError("x and y must be aligned (No, Nd, T) arrays")
        if len(self.origin_labels) != x.shape[0] or len(self.dest_labels) != x.shape[1]:
            raise ValueError("labels must match the panel shape")
        if not (np.isfinite(x).all() and np.isfinite(y).all()) or np.any(y < 0):
            raise ValueError("x must be finite and y finite nonnegative")
        x.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "origin_labels", tuple(self.origin_labels))
        object.__setattr__(self, "dest_labels", tuple(self.dest_labels))

    @property
    def shape(self):
        return self.x.shape

    def pairs(self):
        """(pair_o, pair_d) positions in origin-major order, diagonal per spec."""
        No, Nd, _ = self.x.shape
        po, pd = np.meshgrid(np.arange(No), np.arange(Nd), indexing="ij")
        po, pd = po.ravel(), pd.ravel()
        if self.offdiag:
            o_lab = np.array(self.origin_labels, dtype=object)[po]
            d_lab = np.array(self.dest_labels, dtype=object)[pd]
            keep = o_lab != d_lab
            po, pd = po[keep], pd[keep]
        return po, pd

    def pair_arrays(self):
        po, pd = self.pairs()
        return po, pd, self.x[po, pd, :], self.y[po, pd, :]

    def design(self, F="helmert", dropped_dest=None):
        po, pd, xg, _ = self.pair_arrays()
        No, Nd, _ = self.x.shape
        return RetainedDesign(xg, po, pd, No, Nd, F, dropped_dest)

    def outcome_pairs(self):
        return self.pair_arrays()[3]

    def subset(self, origin_pos, dest_pos, T=None):
        """Rectangular sub-panel (origin positions x destination positions)."""
        origin_pos = np.asarray(origin_pos, dtype=np.int64)
        dest_pos = np.asarray(dest_pos, dtype=np.int64)
        T = self.x.shape[2] if T is None else int(T)
        sl = np.ix_(origin_pos, dest_pos, np.arange(T))
        return CellData(
            self.x[sl],
            self.y[sl],
            tuple(self.origin_labels[i] for i in origin_pos),
            tuple(self.dest_labels[j] for j in dest_pos),
            self.offdiag,
        )

    def content_hash(self):
        """sha256 of the exact cell content (x, y, labels, offdiag)."""
        import hashlib

        h = hashlib.sha256()
        h.update(repr((self.x.shape, self.origin_labels, self.dest_labels, self.offdiag)).encode())
        h.update(np.ascontiguousarray(self.x, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.y, dtype="<f8").tobytes())
        return h.hexdigest()
