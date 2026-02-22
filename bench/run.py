import argparse
import json
import math
import os
import shutil
import statistics as stats
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def iqr_outliers(xs: List[float]) -> List[float]:
    """Return values considered outliers by Tukey rule (1.5*IQR)."""
    if len(xs) < 4:
        return []
    x = sorted(xs)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [v for v in xs if v < lo or v > hi]


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.stdev(xs)


def amdahl_speedup(p: int, f: float) -> float:
    return 1.0 / (f + (1.0 - f) / p)


def gustafson_speedup(p: int, f: float) -> float:
    return p - f * (p - 1)


def estimate_f_from_amdahl(p: int, speedup: float) -> float:
    """Estimate serial fraction f from one measured speedup point."""
    if p <= 1 or speedup <= 0:
        return 1.0
    # f = (1/S - 1/p) / (1 - 1/p)
    f = (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)
    return float(max(0.0, min(1.0, f)))


# ---------------- Command runners ----------------

def run_cmd(cmd: List[str], cwd: str | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("Command failed:", " ".join(cmd))
        print("STDOUT:\n", r.stdout)
        print("STDERR:\n", r.stderr)
        raise RuntimeError("Command failed")


def read_meta_elapsed(meta_path: Path) -> float:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    # Python writes elapsed_s, Rust writes elapsed_s
    return float(meta["elapsed_s"])


# ---------------- Benchmark config ----------------

@dataclass(frozen=True)
class Point:
    lang: str        # "python" or "rust"
    mode: str        # "seq" or "par"
    p: int           # processes/threads
    n: int
    kind: str        # "strong" or "weak"


def n_weak(n0: int, p: int) -> int:
    return int(round(n0 * (p ** (1.0 / 3.0))))


# ---------------- Benchmark execution ----------------

def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def gen_system(python_dir: Path, out_npz: Path, n: int, seed: int = 0) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        sys.executable, str(python_dir / "generate_system.py"),
        "--n", str(n),
        "--seed", str(seed),
        "--out", str(out_npz),
        "--dtype", "float64",
    ])


def run_python_seq(python_dir: Path, system_npz: Path, out_dir: Path) -> float:
    ensure_clean_dir(out_dir)
    run_cmd([
        sys.executable, str(python_dir / "gauss_sequential.py"),
        "--system", str(system_npz),
        "--out", str(out_dir),
        # no --save-states in benchmark
    ])
    return read_meta_elapsed(out_dir / "meta.json")


def run_python_mp(python_dir: Path, system_npz: Path, out_dir: Path, procs: int) -> float:
    ensure_clean_dir(out_dir)
    run_cmd([
        sys.executable, str(python_dir / "gauss_parallel.py"),
        "--system", str(system_npz),
        "--out", str(out_dir),
        "--procs", str(procs),
    ])
    return read_meta_elapsed(out_dir / "meta.json")


def run_rust_seq(rust_bin: Path, system_npz: Path, out_dir: Path) -> float:
    ensure_clean_dir(out_dir)
    run_cmd([
        str(rust_bin),
        "--system", str(system_npz),
        "--out", str(out_dir),
        "--mode", "seq",
    ])
    return read_meta_elapsed(out_dir / "meta.json")


def run_rust_thr(rust_bin: Path, system_npz: Path, out_dir: Path, threads: int) -> float:
    ensure_clean_dir(out_dir)
    run_cmd([
        str(rust_bin),
        "--system", str(system_npz),
        "--out", str(out_dir),
        "--mode", "thr",
        "--threads", str(threads),
    ])
    return read_meta_elapsed(out_dir / "meta.json")


def bench_points(
    python_dir: Path,
    rust_bin: Path,
    base_out: Path,
    reps: int,
    points: List[Point],
    seed: int = 0
) -> Dict[Point, List[float]]:
    results: Dict[Point, List[float]] = {}
    systems_dir = base_out / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    # cache systems by N so we don't regenerate each time
    system_cache: Dict[int, Path] = {}

    for pt in points:
        if pt.n not in system_cache:
            system_npz = systems_dir / f"system_n{pt.n}.npz"
            gen_system(python_dir, system_npz, pt.n, seed=seed)
            system_cache[pt.n] = system_npz

        system_npz = system_cache[pt.n]
        times: List[float] = []

        for r in range(reps):
            out_dir = base_out / "runs" / pt.kind / pt.lang / f"n{pt.n}" / f"p{pt.p}" / pt.mode / f"rep_{r:02d}"
            if pt.lang == "python":
                if pt.mode == "seq":
                    t = run_python_seq(python_dir, system_npz, out_dir)
                else:
                    t = run_python_mp(python_dir, system_npz, out_dir, procs=pt.p)
            else:
                if pt.mode == "seq":
                    t = run_rust_seq(rust_bin, system_npz, out_dir)
                else:
                    t = run_rust_thr(rust_bin, system_npz, out_dir, threads=pt.p)

            times.append(t)

        results[pt] = times
        m, s = mean_std(times)
        outs = iqr_outliers(times)
        print(f"[{pt.kind}] {pt.lang} {pt.mode} n={pt.n} p={pt.p}: mean={m:.6f}s std={s:.6f}s outliers={len(outs)}")

    return results


# ---------------- CSV + Plots ----------------

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def plot_speedup(
    out_png: Path,
    title: str,
    ps: List[int],
    speedups: List[float],
    f_hat: float,
    law: str,  # "amdahl" or "gustafson"
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.title(title)
    plt.xlabel("Broj jezgara (p)")
    plt.ylabel("Ubrzanje S(p)")

    # Measured
    plt.plot(ps, speedups, marker="o", linestyle="-", label="Ostvareno ubrzanje")

    # Ideal
    plt.plot(ps, ps, linestyle="--", label="Idealno (S=p)")

    # Theoretical
    if law == "amdahl":
        theo = [amdahl_speedup(p, f_hat) for p in ps]
        plt.plot(ps, theo, linestyle=":", label=f"Amdahl (f={f_hat:.3f})")
    else:
        theo = [gustafson_speedup(p, f_hat) for p in ps]
        plt.plot(ps, theo, linestyle=":", label=f"Gustafson (f={f_hat:.3f})")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def summarize(results: Dict[Point, List[float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pt, times in results.items():
        m, s = mean_std(times)
        outs = iqr_outliers(times)
        rows.append({
            "kind": pt.kind,
            "lang": pt.lang,
            "mode": pt.mode,
            "n": pt.n,
            "p": pt.p,
            "reps": len(times),
            "mean_s": round(m, 6),
            "std_s": round(s, 6),
            "outliers": len(outs),
        })
    # stable ordering
    rows.sort(key=lambda r: (r["kind"], r["lang"], r["n"], r["p"], r["mode"]))
    return rows


def compute_speedups(table_rows: List[Dict[str, object]], kind: str, lang: str) -> Tuple[List[int], List[float], float]:
    # pick rows for this kind+lang
    rows = [r for r in table_rows if r["kind"] == kind and r["lang"] == lang]
    # need mean time for seq p=1 and par for each p
    # for strong: seq uses n fixed; for weak: seq uses n(p=1) = n0 and par uses n(p)
    # We'll compute speedup per p as mean(seq@p=1)/mean(par@p)
    seq_rows = [r for r in rows if r["mode"] == "seq" and int(r["p"]) == 1]
    if not seq_rows:
        raise RuntimeError(f"Missing seq p=1 for {kind} {lang}")
    t1 = float(seq_rows[0]["mean_s"])

    par_rows = [r for r in rows if r["mode"] == "par"]
    par_rows.sort(key=lambda r: int(r["p"]))
    ps = [int(r["p"]) for r in par_rows]
    sp = [t1 / float(r["mean_s"]) for r in par_rows]

    # estimate f from the last point
    f_hat = estimate_f_from_amdahl(ps[-1], sp[-1]) if ps else 1.0
    return ps, sp, f_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--strong-n", type=int, default=1024)
    ap.add_argument("--weak-n0", type=int, default=600)
    ap.add_argument("--ps", type=str, default="1,2,4,8")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--python-dir", type=str, default="../python")
    ap.add_argument("--rust-bin", type=str, default="../rust/target/release/gauss_rust")
    ap.add_argument("--out", type=str, default="../outputs/bench")
    args = ap.parse_args()

    python_dir = Path(args.python_dir).resolve()
    rust_bin = Path(args.rust_bin).resolve()
    base_out = Path(args.out).resolve()

    ps = [int(x.strip()) for x in args.ps.split(",") if x.strip()]
    if 1 not in ps:
        ps = [1] + ps

    if not rust_bin.exists():
        raise SystemExit(f"Rust binary not found: {rust_bin} (build with: cargo build --release in rust/)")

    # Define all points
    points: List[Point] = []

    # STRONG: fixed N
    Nstrong = args.strong_n
    for p in ps:
        points += [
            Point("python", "seq", 1, Nstrong, "strong"),
            Point("python", "par", p, Nstrong, "strong"),
            Point("rust", "seq", 1, Nstrong, "strong"),
            Point("rust", "par", p, Nstrong, "strong"),
        ]

    # WEAK: N depends on p (seq only at p=1 with N0)
    N0 = args.weak_n0
    for p in ps:
        Np = n_weak(N0, p)
        points += [
            Point("python", "seq", 1, N0, "weak") if p == 1 else None,
            Point("python", "par", p, Np, "weak"),
            Point("rust", "seq", 1, N0, "weak") if p == 1 else None,
            Point("rust", "par", p, Np, "weak"),
        ]
    points = [pt for pt in points if pt is not None]

    # Run benchmark
    results = bench_points(python_dir, rust_bin, base_out, args.reps, points, seed=args.seed)

    # Summarize to table
    table_rows = summarize(results)
    write_csv(base_out / "tables" / "all_results.csv", table_rows)

    # Strong plots
    ps_py, sp_py, f_py = compute_speedups(table_rows, "strong", "python")
    plot_speedup(
        base_out / "plots" / "strong_python.png",
        f"Jako skaliranje (Python) N={Nstrong}",
        ps_py, sp_py, f_py, law="amdahl"
    )

    ps_rs, sp_rs, f_rs = compute_speedups(table_rows, "strong", "rust")
    plot_speedup(
        base_out / "plots" / "strong_rust.png",
        f"Jako skaliranje (Rust) N={Nstrong}",
        ps_rs, sp_rs, f_rs, law="amdahl"
    )

    # Weak plots
    ps_py_w, sp_py_w, f_py_w = compute_speedups(table_rows, "weak", "python")
    plot_speedup(
        base_out / "plots" / "weak_python.png",
        f"Slabo skaliranje (Python) N0={N0}, N(p)=N0*p^(1/3)",
        ps_py_w, sp_py_w, f_py_w, law="gustafson"
    )

    ps_rs_w, sp_rs_w, f_rs_w = compute_speedups(table_rows, "weak", "rust")
    plot_speedup(
        base_out / "plots" / "weak_rust.png",
        f"Slabo skaliranje (Rust) N0={N0}, N(p)=N0*p^(1/3)",
        ps_rs_w, sp_rs_w, f_rs_w, law="gustafson"
    )

    # Also produce per-plot support tables
    # (Filter to strong/weak + lang)
    def write_support(kind: str, lang: str, name: str):
        rows = [r for r in table_rows if r["kind"] == kind and r["lang"] == lang]
        write_csv(base_out / "tables" / name, rows)

    write_support("strong", "python", "strong_python_table.csv")
    write_support("strong", "rust", "strong_rust_table.csv")
    write_support("weak", "python", "weak_python_table.csv")
    write_support("weak", "rust", "weak_rust_table.csv")

    print("\nDONE.")
    print("Tables:", base_out / "tables")
    print("Plots :", base_out / "plots")


if __name__ == "__main__":
    main()
