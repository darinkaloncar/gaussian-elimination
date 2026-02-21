import argparse
import time
import numpy as np

from utils import SolveConfig, save_state, save_metadata, back_substitution, residual_norm, now_ms


def gaussian_elimination_seq(A: np.ndarray, b: np.ndarray, cfg: SolveConfig, eps: float = 0.0):
    n = A.shape[0]
    A_work = A.copy()
    b_work = b.copy()

    t0 = time.perf_counter()

    if cfg.save_states:
        save_state(cfg.out_dir, 0, A_work, b_work)

    for k in range(n - 1):
        # partial pivoting
        pivot_row = k + int(np.argmax(np.abs(A_work[k:, k])))
        piv = A_work[pivot_row, k]
        if abs(float(piv)) <= eps:
            raise ValueError(f"Singular/near-singular matrix: zero pivot at column {k}")

        if pivot_row != k:
            A_work[[k, pivot_row], :] = A_work[[pivot_row, k], :]
            b_work[[k, pivot_row]] = b_work[[pivot_row, k]]

        piv = A_work[k, k]

        # elimination
        for i in range(k + 1, n):
            factor = A_work[i, k] / piv
            if factor != 0:
                A_work[i, k:] -= factor * A_work[k, k:]
                b_work[i] -= factor * b_work[k]
                A_work[i, k] = 0.0

        if cfg.save_states and ((k + 1) % cfg.save_every == 0):
            save_state(cfg.out_dir, k + 1, A_work, b_work)

    x = back_substitution(A_work, b_work, eps=eps)
    t1 = time.perf_counter()

    meta = {
        "impl": "python_seq",
        "n": int(n),
        "dtype": str(A.dtype),
        "save_states": bool(cfg.save_states),
        "save_every": int(cfg.save_every),
        "out_dir": cfg.out_dir,
        "elapsed_s": float(t1 - t0),
        "timestamp_ms": now_ms(),
        "residual_norm": residual_norm(A, x, b),
    }
    save_metadata(cfg.out_dir, meta)

    return x, (t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", type=str, required=True, help="NPZ fajl sa A i b")
    ap.add_argument("--out", type=str, default="../outputs/seq_run")
    ap.add_argument("--save-states", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.0, help="Prag za near-zero pivot (0.0 = striktno)")
    args = ap.parse_args()

    data = np.load(args.system)
    A, b = data["A"], data["b"]

    cfg = SolveConfig(save_states=args.save_states, save_every=args.save_every, out_dir=args.out)
    x, elapsed = gaussian_elimination_seq(A, b, cfg, eps=args.eps)

    print(f"[SEQ] elapsed={elapsed:.6f}s residual={residual_norm(A,x,b):.3e} x[0..3]={x[:4]}")


if __name__ == "__main__":
    main()