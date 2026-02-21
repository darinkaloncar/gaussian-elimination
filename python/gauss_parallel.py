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
    (shmA_name, shmB_name, shapeA, dtypeA_str, dtypeB_str, k, start, end, eps) = args

    A_sh = _attach_shm(shmA_name, tuple(shapeA), dtypeA_str)
    B_sh = _attach_shm(shmB_name, (shapeA[0],), dtypeB_str)

    A = A_sh.as_ndarray()
    b = B_sh.as_ndarray()

    piv = A[k, k]
    if abs(float(piv)) <= eps:
        A_sh.close()
        B_sh.close()
        return

    rowk = A[k, k:]          # view u shared memory (read-only usage)
    bk = float(b[k])

    for i in range(start, end):
        factor = A[i, k] / piv
        if factor != 0:
            A[i, k:] -= factor * rowk
            b[i] -= factor * bk
            A[i, k] = 0.0

    A_sh.close()
    B_sh.close()


def gaussian_elimination_mp(A: np.ndarray, b: np.ndarray, cfg: SolveConfig, nprocs: int, eps: float = 0.0,
                           tail_seq_threshold_mult: int = 4) -> Tuple[np.ndarray, float]:
    n = A.shape[0]
    dtypeA = A.dtype
    dtypeB = b.dtype

    # Shared memory for A
    shmA = SharedMemory(create=True, size=A.nbytes)
    A_sh = np.ndarray(A.shape, dtype=dtypeA, buffer=shmA.buf)
    A_sh[:] = A

    # Shared memory for b
    shmB = SharedMemory(create=True, size=b.nbytes)
    b_sh = np.ndarray(b.shape, dtype=dtypeB, buffer=shmB.buf)
    b_sh[:] = b

    t0 = time.perf_counter()

    try:
        if cfg.save_states:
            save_state(cfg.out_dir, 0, A_sh.copy(), b_sh.copy())

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=nprocs)

        try:
            for k in range(n - 1):
                # pivoting (sekvencijalno)
                pivot_row_idx = k + int(np.argmax(np.abs(A_sh[k:, k])))
                piv = A_sh[pivot_row_idx, k]
                if abs(float(piv)) <= eps:
                    raise ValueError(f"Singular/near-singular matrix: zero pivot at column {k}")

                if pivot_row_idx != k:
                    A_sh[[k, pivot_row_idx], :] = A_sh[[pivot_row_idx, k], :]
                    b_sh[[k, pivot_row_idx]] = b_sh[[pivot_row_idx, k]]

                start_row = k + 1
                total = n - start_row
                if total <= 0:
                    break

                # if rep is small -> seq (smaller overhead)
                if total < tail_seq_threshold_mult * nprocs:
                    piv = A_sh[k, k]
                    rowk = A_sh[k, k:].copy()  # can copy (small-ish) or view
                    bk = float(b_sh[k])
                    for i in range(start_row, n):
                        factor = A_sh[i, k] / piv
                        if factor != 0:
                            A_sh[i, k:] -= factor * rowk
                            b_sh[i] -= factor * bk
                            A_sh[i, k] = 0.0
                else:
                    # chunking of rows
                    chunk = (total + nprocs - 1) // nprocs
                    chunks: List[Tuple[int, int]] = []
                    for p in range(nprocs):
                        s = start_row + p * chunk
                        e = min(start_row + (p + 1) * chunk, n)
                        if s < e:
                            chunks.append((s, e))

                    tasks = [
                        (shmA.name, shmB.name, A_sh.shape, str(dtypeA), str(dtypeB), k, s, e, eps)
                        for (s, e) in chunks
                    ]
                    pool.map(_worker_eliminate, tasks, chunksize=1)

                if cfg.save_states and ((k + 1) % cfg.save_every == 0):
                    save_state(cfg.out_dir, k + 1, A_sh.copy(), b_sh.copy())

        finally:
            pool.close()
            pool.join()

        x = back_substitution(A_sh, b_sh, eps=eps)
        t1 = time.perf_counter()

        meta = {
            "impl": "python_multiprocessing",
            "n": int(n),
            "dtypeA": str(dtypeA),
            "dtypeB": str(dtypeB),
            "nprocs": int(nprocs),
            "save_states": bool(cfg.save_states),
            "save_every": int(cfg.save_every),
            "out_dir": cfg.out_dir,
            "elapsed_s": float(t1 - t0),
            "timestamp_ms": now_ms(),
            "residual_norm": residual_norm(A, x, b),
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
    ap.add_argument("--out", type=str, default="../outputs/mp_run")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--save-states", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.0, help="Prag za near-zero pivot (0.0 = striktno)")
    args = ap.parse_args()

    data = np.load(args.system)
    A, b = data["A"], data["b"]

    cfg = SolveConfig(save_states=args.save_states, save_every=args.save_every, out_dir=args.out)
    x, elapsed = gaussian_elimination_mp(A, b, cfg, nprocs=args.procs, eps=args.eps)

    print(f"[MP] procs={args.procs} elapsed={elapsed:.6f}s residual={residual_norm(A,x,b):.3e} x[0..3]={x[:4]}")


if __name__ == "__main__":
    main()