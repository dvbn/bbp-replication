"""Counter-based master innovations keyed only by (base_seed, packet, rep).

Key:   Philox-4x64 key = first 16 bytes of
       sha256(b"wz_fe_sim/innovations/v1|<base_seed>|<packet>|<rep>"), counter 0.
       No dependence on DGP, N, T, shard, estimator or processing order.
Draws: standard normals by Box-Muller from the raw 64-bit Philox words
       (u = ((word >> 11) + 0.5) * 2^-53 in (0, 1)), so the stream depends on
       the Philox bit generator, not on NumPy's Generator.normal algorithm.
Order: alpha (N,T), gamma (N,T), eta (N,N), nu (N,N,T), eps (N,N,T), each block
       starting on a fresh Box-Muller pair.
Stata seed labels (1234..1243) are provenance strings only: no bit-equivalence
with the public Stata draws is claimed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

INNOVATION_SCHEME = "philox4x64-sha256key-boxmuller/v1"
BLOCKS = ("alpha", "gamma", "eta", "nu", "eps")


def philox_key(base_seed, packet, rep):
    msg = f"wz_fe_sim/innovations/v1|{int(base_seed)}|{packet}|{int(rep)}".encode()
    digest = hashlib.sha256(msg).digest()
    return np.frombuffer(digest[:16], dtype="<u8").astype(np.uint64)


def _normals(bitgen, n):
    m = (n + 1) // 2
    words = bitgen.random_raw(2 * m)
    u = ((words >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    u1, u2 = u[0::2], u[1::2]
    rad = np.sqrt(-2.0 * np.log(u1))
    ang = 2.0 * np.pi * u2
    z = np.empty(2 * m)
    z[0::2] = rad * np.cos(ang)
    z[1::2] = rad * np.sin(ang)
    return z[:n]


@dataclass(frozen=True)
class MasterInnovations:
    base_seed: int
    packet: str
    rep: int
    N: int
    T: int
    alpha: np.ndarray  # (N, T) origin-time, standard normal
    gamma: np.ndarray  # (N, T) destination-time
    eta: np.ndarray  # (N, N) pair
    nu: np.ndarray  # (N, N, T) regressor innovation
    eps: np.ndarray  # (N, N, T) disturbance innovation (before AR filter)
    draw_hash: str


def master_innovations(base_seed, packet, rep, N=100, T=10):
    bg = np.random.Philox(key=philox_key(base_seed, packet, rep))
    shapes = {"alpha": (N, T), "gamma": (N, T), "eta": (N, N), "nu": (N, N, T), "eps": (N, N, T)}
    arrays = {}
    h = hashlib.sha256(f"{INNOVATION_SCHEME}|{N}|{T}".encode())
    for name in BLOCKS:
        shape = shapes[name]
        arr = _normals(bg, int(np.prod(shape))).reshape(shape)
        arr.setflags(write=False)
        arrays[name] = arr
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return MasterInnovations(int(base_seed), str(packet), int(rep), N, T, draw_hash=h.hexdigest(), **arrays)
