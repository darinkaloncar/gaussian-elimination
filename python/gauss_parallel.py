from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

from utils import SolveConfig, save_state, save_metadata, back_substitution, residual_norm, now_ms


@dataclass
class ShmArray:
    shm: SharedMemory
    shape: Tuple[int, ...]
    dtype: np.dtype

    def as_ndarray(self) -> np.ndarray:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


def _attach_shm(name: str, shape: Tuple[int, ...], dtype_str: str) -> ShmArray:
    shm = SharedMemory(name=name)
    dtype = np.dtype(dtype_str)
    return ShmArray(shm=shm, shape=shape, dtype=dtype)


def _worker_eliminate(args):
    (shmA_name, shmB_name, shapeA, dtypeA_str, k, start, end, pivot_row, pivot_b) = args
    A_sh = _attach_shm(shmA_name, tuple(shapeA), dtypeA_str)
    B_sh = _attach_shm(shmB_name, (shapeA[0],), dtypeA_str)

    A = A_sh.as_ndarray()
    b = B_sh.as_ndarray()

    piv = pivot_row[k]
    if piv == 0:
        A_sh.close()
        B_sh.close()
        return

    for i in range(start, end):
        factor = A[i, k] / piv
        if factor != 0:
            A[i, k:] -= factor * pivot_row[k:]
            b[i] -= factor * pivot_b
            A[i, k] = 0.0

    A_sh.close()
    B_sh.close()


def gaussian_elimination_mp(A: np.ndarray, b: np.ndarray, cfg: SolveConfig, nprocs: int) -> Tuple[np.ndarray, float]:
    n = A.shape[0]
    dtype = A.dtype

    # Shared memory for A
    shmA = SharedMemory(create=True, size=A.nbytes)
    A_sh = np.ndarray(A.shape, dtype=dtype, buffer=shmA.buf)
    A_sh[:] = A

    # Shared memory for b
    shmB = SharedMemory(create=True, size=b.nbytes)
    b_sh = np.ndarray(b.shape, dtype=dtype, buffer=shmB.buf)
    b_sh[:] = b

    t0 = time.perf_counter()

    try:
        if cfg.save_states:
            save_state(cfg.out_dir, 0, A_sh.copy(), b_sh.copy())

        # Važno: na macOS-u default je spawn; to je ok, ali mora __main__ guard (ima ga u main()).
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=nprocs)

        try:
            for k in range(n - 1):
                # pivoting
                pivot_row_idx = k + int(np.argmax(np.abs(A_sh[k:, k])))
                if A_sh[pivot_row_idx, k] == 0:
                    raise ValueError(f"Singular or near-singular matrix: zero pivot at column {k}")

                if pivot_row_idx != k:
                    A_sh[[k, pivot_row_idx], :] = A_sh[[pivot_row_idx, k], :]
                    b_sh[[k, pivot_row_idx]] = b_sh[[pivot_row_idx, k]]

                # pivot copied, not to be dependant
                pivot_row = A_sh[k, :].copy()
                pivot_b = float(b_sh[k])

                # rows: k+1..n-1
                start_row = k + 1
                total = n - start_row
                if total <= 0:
                    break

                # chunking rows between processes
                chunks: List[Tuple[int, int]] = []
                chunk = (total + nprocs - 1) // nprocs
                for p in range(nprocs):
                    s = start_row + p * chunk
                    e = min(start_row + (p + 1) * chunk, n)
                    if s < e:
                        chunks.append((s, e))

                tasks = [
                    (shmA.name, shmB.name, A_sh.shape, str(dtype), k, s, e, pivot_row, pivot_b)
                    for (s, e) in chunks
                ]
                pool.map(_worker_eliminate, tasks)

                if cfg.save_states and ((k + 1) % cfg.save_every == 0):
                    save_state(cfg.out_dir, k + 1, A_sh.copy(), b_sh.copy())

        finally:
            pool.close()
            pool.join()

        x = back_substitution(A_sh, b_sh)
        t1 = time.perf_counter()

        meta = {
            "impl": "python_multiprocessing",
            "n": int(n),
            "dtype": str(dtype),
            "nprocs": int(nprocs),
            "save_states": bool(cfg.save_states),
            "save_every": int(cfg.save_every),
            "out_dir": cfg.out_dir,
            "elapsed_s": float(t1 - t0),
            "timestamp_ms": now_ms(),
        }
        save_metadata(cfg.out_dir, meta)
        return x, (t1 - t0)

    finally:
        shmA.close()
        shmA.unlink()
        shmB.close()
        shmB.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", type=str, required=True, help="NPZ fajl sa A i b")
    ap.add_argument("--out", type=str, default="outputs/mp_run")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--save-states", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    args = ap.parse_args()

    data = np.load(args.system)
    A, b = data["A"], data["b"]

    cfg = SolveConfig(save_states=args.save_states, save_every=args.save_every, out_dir=args.out)
    x, elapsed = gaussian_elimination_mp(A, b, cfg, nprocs=args.procs)

    # rezidual nad ORIGINALNIM sistemom
    res = residual_norm(A, x, b)
    print(f"[MP] procs={args.procs} elapsed={elapsed:.6f}s residual_norm={res:.3e} x[0..3]={x[:4]}")

if __name__ == "__main__":
    main()
