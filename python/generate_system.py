import argparse
import numpy as np
from utils import make_diagonally_dominant_system, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="../outputs/system.npz")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    args = ap.parse_args()

    dtype = np.float64 if args.dtype == "float64" else np.float32
    A, b = make_diagonally_dominant_system(args.n, seed=args.seed, dtype=dtype)

    ensure_dir("/".join(args.out.split("/")[:-1]) or ".")
    np.savez_compressed(args.out, A=A, b=b)
    print(f"Saved system to {args.out} (n={args.n}, seed={args.seed}, dtype={args.dtype})")

if __name__ == "__main__":
    main()
