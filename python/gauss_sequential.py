import argparse
import time
import numpy as np
from utils import SolveConfig, save_state, save_metadata, back_substitution, residual_norm, now_ms

def gaussian_elimination_seq(A: np.ndarray, b: np.ndarray, cfg: SolveConfig):
    n = A.shape[0]
    A = A.copy()
    b = b.copy()

    t0 = time.perf_counter()

    if cfg.save_states:
        save_state(cfg.out_dir, 0, A, b)

    for k in range(n - 1):
        # partial pivoting: argmax |A[i,k]| for i>=k
        pivot_row = k + int(np.argmax(np.abs(A[k:, k])))
        if A[pivot_row, k] == 0:
            raise ValueError(f"Singular or near-singular matrix: zero pivot at column {k}")

        if pivot_row != k:
            A[[k, pivot_row], :] = A[[pivot_row, k], :]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        piv = A[k, k]
        # elimination
        for i in range(k + 1, n):
            factor = A[i, k] / piv
            if factor != 0:
                A[i, k:] -= factor * A[k, k:]
                b[i] -= factor * b[k]
                A[i, k] = 0.0

        if cfg.save_states and ((k + 1) % cfg.save_every == 0):
            save_state(cfg.out_dir, k + 1, A, b)

    x = back_substitution(A, b)
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
        "residual_norm": residual_norm(A=np.load(cfg.system_path)["A"], x=x, b=np.load(cfg.system_path)["b"]) if hasattr(cfg, "system_path") else None,
    }
    save_metadata(cfg.out_dir, meta)
    return x, (t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", type=str, required=True, help="NPZ file containing A and b")
    ap.add_argument("--out", type=str, default="outputs/seq_run")
    ap.add_argument("--save-states", action="store_true")
    ap.add_argument("--save-every", type=int, default=1)
    args = ap.parse_args()

    data = np.load(args.system)
    A, b = data["A"], data["b"]

    cfg = SolveConfig(save_states=args.save_states, save_every=args.save_every, out_dir=args.out)
    cfg.system_path = args.system  # type: ignore

    x, elapsed = gaussian_elimination_seq(A, b, cfg)
    print(f"[SEQ] elapsed={elapsed:.6f}s, x[0..3]={x[:4]}")

if __name__ == "__main__":
    main()
