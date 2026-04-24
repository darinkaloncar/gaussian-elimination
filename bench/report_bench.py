import argparse
import json
import shutil
import statistics as stats
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
RUST_BIN = ROOT / "rust" / "target" / "release" / "gauss_rust"

DEFAULT_OUT_DIR = ROOT / "outputs" / "report_bench"
DEFAULT_PS = [1, 2, 4, 8]
DEFAULT_REPS = 30
DEFAULT_SEED = 0

STRONG_PY_N = 1536
WEAK_PY_N0 = 768
STRONG_RS_N = 1536
WEAK_RS_N0 = 768

VALID_STAGES = {
    "strong_python_seq",
    "strong_python_par",
    "weak_python_par",
    "strong_rust_seq",
    "strong_rust_par",
    "weak_rust_par",
}


def n_weak(n0: int, p: int) -> int:
    return int(round(n0 * (p ** (1.0 / 3.0))))


def mean_std(xs: list[float]) -> tuple[float, float]:
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.stdev(xs)


def iqr_outliers(xs: list[float]) -> list[float]:
    if len(xs) < 4:
        return []
    q1 = np.percentile(xs, 25)
    q3 = np.percentile(xs, 75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [x for x in xs if x < lo or x > hi]


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("FAILED:", " ".join(cmd))
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("command failed")


def read_elapsed(meta_path: Path) -> float:
    with meta_path.open("r", encoding="utf-8") as f:
        return float(json.load(f)["elapsed_s"])


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_system(out_dir: Path, n: int, seed: int) -> Path:
    systems_dir = out_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    npz_path = systems_dir / f"system_n{n}.npz"
    if not npz_path.exists():
        run_cmd(
            [
                sys.executable,
                str(PYTHON_DIR / "generate_system.py"),
                "--n",
                str(n),
                "--seed",
                str(seed),
                "--out",
                str(npz_path),
                "--dtype",
                "float64",
            ]
        )
    return npz_path


def run_python_seq(system_npz: Path, out_dir: Path) -> float:
    ensure_clean_dir(out_dir)
    run_cmd(
        [
            sys.executable,
            str(PYTHON_DIR / "gauss_sequential.py"),
            "--system",
            str(system_npz),
            "--out",
            str(out_dir),
        ]
    )
    return read_elapsed(out_dir / "meta.json")


def run_python_par(system_npz: Path, out_dir: Path, procs: int) -> float:
    ensure_clean_dir(out_dir)
    run_cmd(
        [
            sys.executable,
            str(PYTHON_DIR / "gauss_parallel.py"),
            "--system",
            str(system_npz),
            "--out",
            str(out_dir),
            "--procs",
            str(procs),
        ]
    )
    return read_elapsed(out_dir / "meta.json")


def run_rust_seq(system_npz: Path, out_dir: Path) -> float:
    ensure_clean_dir(out_dir)
    run_cmd(
        [
            str(RUST_BIN),
            "--system",
            str(system_npz),
            "--out",
            str(out_dir),
            "--mode",
            "seq",
        ]
    )
    return read_elapsed(out_dir / "meta.json")


def run_rust_par(system_npz: Path, out_dir: Path, threads: int) -> float:
    ensure_clean_dir(out_dir)
    run_cmd(
        [
            str(RUST_BIN),
            "--system",
            str(system_npz),
            "--out",
            str(out_dir),
            "--mode",
            "thr",
            "--threads",
            str(threads),
        ]
    )
    return read_elapsed(out_dir / "meta.json")


@dataclass(frozen=True)
class Series:
    stage: str
    kind: str
    lang: str
    p: int
    n: int
    baseline: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ps", type=str, default="1,2,4,8")
    ap.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated subset of stages, or 'all'.",
    )
    ap.add_argument("--resume", action="store_true", help="Skip repetitions that already have meta.json.")
    ap.add_argument("--aggregate-only", action="store_true", help="Do not run benchmarks; only rebuild tables/plots from existing results.")
    ap.add_argument("--skip-aggregate", action="store_true", help="Run selected stages only; do not rebuild summary yet.")
    ap.add_argument("--sleep-between-reps", type=float, default=3.0)
    ap.add_argument("--sleep-between-series", type=float, default=20.0)
    ap.add_argument("--sleep-between-stages", type=float, default=60.0)
    return ap.parse_args()


def parse_ps(raw: str) -> list[int]:
    ps = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if 1 not in ps:
        ps = [1] + ps
    return sorted(set(ps))


def parse_stages(raw: str) -> list[str]:
    if raw == "all":
        return sorted(VALID_STAGES)
    stages = [x.strip() for x in raw.split(",") if x.strip()]
    invalid = [x for x in stages if x not in VALID_STAGES]
    if invalid:
        raise SystemExit(f"Unknown stage(s): {', '.join(invalid)}")
    return stages


def build_series(ps: list[int]) -> list[Series]:
    out: list[Series] = [Series("strong_python_seq", "strong", "python", 1, STRONG_PY_N, "seq")]

    out.extend(Series("strong_python_par", "strong", "python", p, STRONG_PY_N, "par") for p in ps)
    out.extend(Series("weak_python_par", "weak", "python", p, n_weak(WEAK_PY_N0, p), "par") for p in ps)

    out.append(Series("strong_rust_seq", "strong", "rust", 1, STRONG_RS_N, "seq"))
    out.extend(Series("strong_rust_par", "strong", "rust", p, STRONG_RS_N, "par") for p in ps)
    out.extend(Series("weak_rust_par", "weak", "rust", p, n_weak(WEAK_RS_N0, p), "par") for p in ps)
    return out


def rep_dir(base_out: Path, series: Series, rep: int) -> Path:
    return base_out / "runs" / series.stage / series.lang / f"n{series.n}" / f"p{series.p}" / f"rep_{rep:02d}"


def collect_times(
    base_out: Path,
    series: Series,
    reps: int,
    seed: int,
    resume: bool,
    sleep_between_reps: float,
) -> list[float]:
    system_npz = ensure_system(base_out, series.n, seed)
    times: list[float] = []

    for rep in range(reps):
        out_dir = rep_dir(base_out, series, rep)
        meta_path = out_dir / "meta.json"
        if resume and meta_path.exists():
            elapsed = read_elapsed(meta_path)
            times.append(elapsed)
            print(f"[resume] {series.stage} p={series.p} rep={rep:02d} elapsed={elapsed:.6f}s")
            continue

        if series.lang == "python" and series.baseline == "seq":
            elapsed = run_python_seq(system_npz, out_dir)
        elif series.lang == "python":
            elapsed = run_python_par(system_npz, out_dir, series.p)
        elif series.lang == "rust" and series.baseline == "seq":
            elapsed = run_rust_seq(system_npz, out_dir)
        else:
            elapsed = run_rust_par(system_npz, out_dir, series.p)

        times.append(elapsed)
        print(f"[run] {series.stage} p={series.p} rep={rep:02d} elapsed={elapsed:.6f}s")
        if sleep_between_reps > 0 and rep != reps - 1:
            time.sleep(sleep_between_reps)

    return times


def load_times(base_out: Path, series: Series, reps: int) -> list[float]:
    times: list[float] = []
    for rep in range(reps):
        meta_path = rep_dir(base_out, series, rep) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing result: {meta_path}")
        times.append(read_elapsed(meta_path))
    return times


def amdahl_speedup(p: int, f: float) -> float:
    return 1.0 / (f + (1.0 - f) / p)


def gustafson_speedup(p: int, alpha: float) -> float:
    return p - alpha * (p - 1)


def fit_amdahl(ps: list[int], speeds: list[float]) -> float:
    best_f = 1.0
    best_err = float("inf")
    for f in np.linspace(0.0, 1.0, 2001):
        err = sum((amdahl_speedup(p, float(f)) - s) ** 2 for p, s in zip(ps, speeds))
        if err < best_err:
            best_err = err
            best_f = float(f)
    return best_f


def amdahl_from_point(p: int, speedup: float) -> float:
    if p <= 1 or speedup <= 0:
        return 1.0
    denom = 1.0 - 1.0 / p
    if denom == 0.0:
        return 1.0
    f = (1.0 / speedup - 1.0 / p) / denom
    return float(max(0.0, min(1.0, f)))


def amdahl_from_best_point(ps: list[int], speeds: list[float]) -> float:
    candidates = [(p, s) for p, s in zip(ps, speeds) if p > 1]
    if not candidates:
        return 1.0
    best_p, best_s = max(candidates, key=lambda item: item[1])
    return amdahl_from_point(best_p, best_s)


def gustafson_from_point(p: int, speedup: float) -> float:
    if p <= 1:
        return 1.0
    alpha = (p - speedup) / (p - 1)
    return float(max(0.0, min(1.0, alpha)))


def gustafson_from_best_point(ps: list[int], speeds: list[float]) -> float:
    candidates = [(p, s) for p, s in zip(ps, speeds) if p > 1]
    if not candidates:
        return 1.0
    best_p, best_s = max(candidates, key=lambda item: item[1])
    return gustafson_from_point(best_p, best_s)


def fit_gustafson(ps: list[int], speeds: list[float]) -> float:
    best_a = 1.0
    best_err = float("inf")
    for alpha in np.linspace(0.0, 1.0, 2001):
        err = sum((gustafson_speedup(p, float(alpha)) - s) ** 2 for p, s in zip(ps, speeds))
        if err < best_err:
            best_err = err
            best_a = float(alpha)
    return best_a


def plot_series(out_png: Path, title: str, ps: list[int], measured: list[float], theoretical: list[float], theory_label: str) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(ps, measured, marker="o", linewidth=2, label="Ostvareno ubrzanje")
    plt.plot(ps, ps, linestyle="--", linewidth=1.5, label="Idealno skaliranje")
    plt.plot(ps, theoretical, linestyle=":", linewidth=2, label=theory_label)
    plt.title(title)
    plt.xlabel("Broj procesorskih jezgara")
    plt.ylabel("Ubrzanje")
    plt.xticks(ps)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in cols) + "\n")


def write_svg_plot(out_svg: Path, title: str, ps: list[int], measured: list[float], theoretical: list[float], theory_label: str) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    w, h = 760, 480
    ml, mr, mt, mb = 70, 30, 40, 55
    ymax = max(max(measured), max(theoretical), max(ps)) * 1.1

    def sx(p: int) -> float:
        return ml + (p - min(ps)) / (max(ps) - min(ps)) * (w - ml - mr)

    def sy(y: float) -> float:
        return h - mb - y / ymax * (h - mt - mb)

    def poly(xs: list[int], ys: list[float]) -> str:
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="480" viewBox="0 0 760 480">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="380" y="24" text-anchor="middle" font-size="18" font-family="Helvetica">{title}</text>',
        f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="black"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="black"/>',
    ]

    for p in ps:
        x = sx(p)
        svg.append(f'<line x1="{x:.1f}" y1="{h-mb}" x2="{x:.1f}" y2="{h-mb+6}" stroke="black"/>')
        svg.append(f'<text x="{x:.1f}" y="{h-mb+22}" text-anchor="middle" font-size="12" font-family="Helvetica">{p}</text>')

    y_step = 1 if ymax <= 10 else 2
    for y in range(0, int(ymax) + 1, y_step):
        yy = sy(float(y))
        svg.append(f'<line x1="{ml-6}" y1="{yy:.1f}" x2="{ml}" y2="{yy:.1f}" stroke="black"/>')
        svg.append(f'<text x="{ml-10}" y="{yy+4:.1f}" text-anchor="end" font-size="12" font-family="Helvetica">{y}</text>')

    svg.append(f'<text x="380" y="470" text-anchor="middle" font-size="14" font-family="Helvetica">Broj procesorskih jezgara</text>')
    svg.append(f'<text x="20" y="240" text-anchor="middle" font-size="14" font-family="Helvetica" transform="rotate(-90 20 240)">Ubrzanje</text>')
    svg.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{poly(ps, measured)}"/>')
    svg.append(f'<polyline fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="8 6" points="{poly(ps, [float(p) for p in ps])}"/>')
    svg.append(f'<polyline fill="none" stroke="#16a34a" stroke-width="2" stroke-dasharray="4 4" points="{poly(ps, theoretical)}"/>')

    for x, y in zip(ps, measured):
        svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#2563eb"/>')

    legend_x, legend_y = 480, 60
    items = [
        ("#2563eb", "", "Ostvareno ubrzanje"),
        ("#ef4444", "8 6", "Idealno skaliranje"),
        ("#16a34a", "4 4", theory_label),
    ]
    for i, (color, dash, label) in enumerate(items):
        y = legend_y + i * 22
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+30}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        svg.append(f'<text x="{legend_x+38}" y="{y+4}" font-size="12" font-family="Helvetica">{label}</text>')

    svg.append("</svg>")
    out_svg.write_text("\n".join(svg), encoding="utf-8")


def aggregate_results(base_out: Path, ps: list[int], reps: int) -> dict[str, dict[str, object]]:
    strong_python_seq = load_times(base_out, Series("strong_python_seq", "strong", "python", 1, STRONG_PY_N, "seq"), reps)
    strong_rust_seq = load_times(base_out, Series("strong_rust_seq", "strong", "rust", 1, STRONG_RS_N, "seq"), reps)

    strong_python_par = {
        p: load_times(base_out, Series("strong_python_par", "strong", "python", p, STRONG_PY_N, "par"), reps)
        for p in ps
    }
    weak_python_par = {
        p: load_times(base_out, Series("weak_python_par", "weak", "python", p, n_weak(WEAK_PY_N0, p), "par"), reps)
        for p in ps
    }
    strong_rust_par = {
        p: load_times(base_out, Series("strong_rust_par", "strong", "rust", p, STRONG_RS_N, "par"), reps)
        for p in ps
    }
    weak_rust_par = {
        p: load_times(base_out, Series("weak_rust_par", "weak", "rust", p, n_weak(WEAK_RS_N0, p), "par"), reps)
        for p in ps
    }

    py_seq_mean, _ = mean_std(strong_python_seq)
    rs_seq_mean, _ = mean_std(strong_rust_seq)
    py_weak_base_mean, _ = mean_std(weak_python_par[1])
    rs_weak_base_mean, _ = mean_std(weak_rust_par[1])

    tables: dict[str, list[dict[str, object]]] = {
        "strong_python": [],
        "strong_rust": [],
        "weak_python": [],
        "weak_rust": [],
    }

    strong_py_speedups: list[float] = []
    strong_rs_speedups: list[float] = []
    weak_py_speedups: list[float] = []
    weak_rs_speedups: list[float] = []

    for p in ps:
        py_times = strong_python_par[p]
        rs_times = strong_rust_par[p]
        wpy_times = weak_python_par[p]
        wrs_times = weak_rust_par[p]

        py_mean, py_std = mean_std(py_times)
        rs_mean, rs_std = mean_std(rs_times)
        wpy_mean, wpy_std = mean_std(wpy_times)
        wrs_mean, wrs_std = mean_std(wrs_times)

        strong_py_speedups.append(py_seq_mean / py_mean)
        strong_rs_speedups.append(rs_seq_mean / rs_mean)
        weak_py_speedups.append((p * py_weak_base_mean) / wpy_mean)
        weak_rs_speedups.append((p * rs_weak_base_mean) / wrs_mean)

        tables["strong_python"].append(
            {
                "p": p,
                "n": STRONG_PY_N,
                "mean_s": round(py_mean, 6),
                "std_s": round(py_std, 6),
                "outliers": len(iqr_outliers(py_times)),
                "speedup": round(strong_py_speedups[-1], 6),
            }
        )
        tables["strong_rust"].append(
            {
                "p": p,
                "n": STRONG_RS_N,
                "mean_s": round(rs_mean, 6),
                "std_s": round(rs_std, 6),
                "outliers": len(iqr_outliers(rs_times)),
                "speedup": round(strong_rs_speedups[-1], 6),
            }
        )
        tables["weak_python"].append(
            {
                "p": p,
                "n": n_weak(WEAK_PY_N0, p),
                "mean_s": round(wpy_mean, 6),
                "std_s": round(wpy_std, 6),
                "outliers": len(iqr_outliers(wpy_times)),
                "speedup": round(weak_py_speedups[-1], 6),
            }
        )
        tables["weak_rust"].append(
            {
                "p": p,
                "n": n_weak(WEAK_RS_N0, p),
                "mean_s": round(wrs_mean, 6),
                "std_s": round(wrs_std, 6),
                "outliers": len(iqr_outliers(wrs_times)),
                "speedup": round(weak_rs_speedups[-1], 6),
            }
        )

    f_py = amdahl_from_best_point(ps, strong_py_speedups)
    f_rs = amdahl_from_best_point(ps, strong_rs_speedups)
    alpha_py = gustafson_from_best_point(ps, weak_py_speedups)
    alpha_rs = gustafson_from_best_point(ps, weak_rs_speedups)

    plot_series(
        base_out / "plots" / "strong_python.png",
        f"Jako skaliranje Python implementacije (N={STRONG_PY_N})",
        ps,
        strong_py_speedups,
        [amdahl_speedup(p, f_py) for p in ps],
        f"Amdahl (f={f_py:.3f})",
    )
    plot_series(
        base_out / "plots" / "strong_rust.png",
        f"Jako skaliranje Rust implementacije (N={STRONG_RS_N})",
        ps,
        strong_rs_speedups,
        [amdahl_speedup(p, f_rs) for p in ps],
        f"Amdahl (f={f_rs:.3f})",
    )
    plot_series(
        base_out / "plots" / "weak_python.png",
        f"Slabo skaliranje Python implementacije (N0={WEAK_PY_N0})",
        ps,
        weak_py_speedups,
        [gustafson_speedup(p, alpha_py) for p in ps],
        f"Gustafson (alpha={alpha_py:.3f})",
    )
    plot_series(
        base_out / "plots" / "weak_rust.png",
        f"Slabo skaliranje Rust implementacije (N0={WEAK_RS_N0})",
        ps,
        weak_rs_speedups,
        [gustafson_speedup(p, alpha_rs) for p in ps],
        f"Gustafson (alpha={alpha_rs:.3f})",
    )

    write_svg_plot(
        base_out / "plots" / "weak_python.svg",
        f"Slabo skaliranje Python implementacije (N0={WEAK_PY_N0})",
        ps,
        weak_py_speedups,
        [gustafson_speedup(p, alpha_py) for p in ps],
        f"Gustafson (alpha={alpha_py:.3f})",
    )
    write_svg_plot(
        base_out / "plots" / "weak_rust.svg",
        f"Slabo skaliranje Rust implementacije (N0={WEAK_RS_N0})",
        ps,
        weak_rs_speedups,
        [gustafson_speedup(p, alpha_rs) for p in ps],
        f"Gustafson (alpha={alpha_rs:.3f})",
    )

    all_rows: list[dict[str, object]] = []
    for row in tables["strong_python"]:
        all_rows.append({"kind": "strong", "lang": "python", **row})
    all_rows.append({"kind": "strong", "lang": "python", "p": 1, "n": STRONG_PY_N, "mean_s": round(py_seq_mean, 6), "std_s": round(mean_std(strong_python_seq)[1], 6), "outliers": len(iqr_outliers(strong_python_seq)), "speedup": 1.0, "mode": "seq"})
    for row in tables["strong_rust"]:
        all_rows.append({"kind": "strong", "lang": "rust", **row})
    all_rows.append({"kind": "strong", "lang": "rust", "p": 1, "n": STRONG_RS_N, "mean_s": round(rs_seq_mean, 6), "std_s": round(mean_std(strong_rust_seq)[1], 6), "outliers": len(iqr_outliers(strong_rust_seq)), "speedup": 1.0, "mode": "seq"})
    for row in tables["weak_python"]:
        all_rows.append({"kind": "weak", "lang": "python", **row})
    for row in tables["weak_rust"]:
        all_rows.append({"kind": "weak", "lang": "rust", **row})

    for name, rows in tables.items():
        write_csv(base_out / "tables" / f"{name}.csv", rows)
    write_csv(base_out / "tables" / "all_results.csv", all_rows)

    summary = {
        "strong_python": {
            "seq_mean_s": py_seq_mean,
            "serial_fraction": f_py,
            "parallel_fraction": 1.0 - f_py,
            "amdahl_limit": 1.0 / f_py if f_py > 0 else float("inf"),
            "speedups": dict(zip(ps, strong_py_speedups)),
        },
        "strong_rust": {
            "seq_mean_s": rs_seq_mean,
            "serial_fraction": f_rs,
            "parallel_fraction": 1.0 - f_rs,
            "amdahl_limit": 1.0 / f_rs if f_rs > 0 else float("inf"),
            "speedups": dict(zip(ps, strong_rs_speedups)),
        },
        "weak_python": {
            "par1_mean_s": py_weak_base_mean,
            "serial_fraction": alpha_py,
            "parallel_fraction": 1.0 - alpha_py,
            "speedups": dict(zip(ps, weak_py_speedups)),
        },
        "weak_rust": {
            "par1_mean_s": rs_weak_base_mean,
            "serial_fraction": alpha_rs,
            "parallel_fraction": 1.0 - alpha_rs,
            "speedups": dict(zip(ps, weak_rs_speedups)),
        },
    }
    with (base_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_selected_stages(args: argparse.Namespace, ps: list[int], stages: list[str]) -> None:
    series_by_stage: dict[str, list[Series]] = {}
    for series in build_series(ps):
        series_by_stage.setdefault(series.stage, []).append(series)

    for idx, stage in enumerate(stages):
        print(f"=== Stage: {stage} ===")
        for s_idx, series in enumerate(series_by_stage[stage]):
            collect_times(
                args.out,
                series,
                args.reps,
                args.seed,
                args.resume,
                args.sleep_between_reps,
            )
            if args.sleep_between_series > 0 and s_idx != len(series_by_stage[stage]) - 1:
                print(f"[cooldown] sleeping {args.sleep_between_series:.1f}s between series")
                time.sleep(args.sleep_between_series)
        if args.sleep_between_stages > 0 and idx != len(stages) - 1:
            print(f"[cooldown] sleeping {args.sleep_between_stages:.1f}s between stages")
            time.sleep(args.sleep_between_stages)


def main() -> None:
    args = parse_args()
    ps = parse_ps(args.ps)
    stages = parse_stages(args.stages)

    if not args.aggregate_only:
        run_selected_stages(args, ps, stages)

    if args.skip_aggregate:
        print("Skipped aggregation by request.")
        return

    summary = aggregate_results(args.out, ps, args.reps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
