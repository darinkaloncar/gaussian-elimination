import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np


def list_states(states_dir: Path) -> list[Path]:
    files = sorted(states_dir.glob("state_*.npz"))
    if not files:
        raise FileNotFoundError(f"No state_*.npz files found in {states_dir}")
    return files


def below_diag_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.tril(a, k=-1)))


def render_frame(state_path: Path, out_png: Path, vlim: float | None) -> None:
    data = np.load(state_path)
    a = data["A"]
    b = data["b"]

    iter_no = int(state_path.stem.split("_")[1])
    norm_val = below_diag_norm(a)
    vmax = vlim if vlim is not None else float(np.max(np.abs(a)))
    vmax = max(vmax, 1e-9)

    fig = plt.figure(figsize=(10, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.4])

    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.imshow(a, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax_a.set_title(f"Matrica A - iteracija {iter_no}")
    ax_a.set_xticks(range(a.shape[1]))
    ax_a.set_yticks(range(a.shape[0]))
    ax_a.set_xticklabels(range(a.shape[1]))
    ax_a.set_yticklabels(range(a.shape[0]))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax_a.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Vrednost", rotation=90)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    info = [
        f"Stanje: {state_path.name}",
        "",
        f"||A_donji|| = {norm_val:.4f}",
        "",
        "Vektor b:",
        *[f"b[{i}] = {val:.3f}" for i, val in enumerate(b)],
    ]
    ax_b.text(0.0, 1.0, "\n".join(info), va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("Gaussian Elimination - tok iteracija", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def build_video(frames_dir: Path, out_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    states = list_states(args.states_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out.parent / f"{args.out.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    vmax = max(float(np.max(np.abs(np.load(path)["A"]))) for path in states)

    for idx, state_path in enumerate(states):
        out_png = frames_dir / f"frame_{idx:04d}.png"
        render_frame(state_path, out_png, vmax)

    build_video(frames_dir, args.out, args.fps)

    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
