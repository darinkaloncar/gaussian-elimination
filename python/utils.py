import os
import json
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class SolveConfig:
    save_states: bool = True
    save_every: int = 1
    out_dir: str = "outputs/run"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def make_diagonally_dominant_system(n: int, seed: int = 0, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, size=(n, n)).astype(dtype)
    b = rng.normal(0, 1, size=(n,)).astype(dtype)

    row_sums = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
    A[np.diag_indices(n)] = row_sums + 1.0
    return A, b


def save_state(out_dir: str, k: int, A: np.ndarray, b: np.ndarray) -> None:
    ensure_dir(out_dir)
    np.savez_compressed(os.path.join(out_dir, f"state_{k:04d}.npz"), A=A, b=b)


def save_metadata(out_dir: str, meta: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def back_substitution(U: np.ndarray, y: np.ndarray, eps: float = 0.0) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros(n, dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        piv = U[i, i]
        if abs(float(piv)) <= eps:
            raise ZeroDivisionError(f"Zero/near-zero pivot at row {i} during back-substitution.")
        s = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - s) / piv
    return x


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    r = A @ x - b
    return float(np.linalg.norm(r, ord=2))