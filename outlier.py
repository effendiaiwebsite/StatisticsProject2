"""Hybrid outlier-detection ensemble.

Four detectors reconciled by a majority vote:
  1. Modified Z-Score (MAD)
  2. Tukey IQR fences
  3. Isolation Forest (scikit-learn)
  4. Iterative mean/std trimming

Vote: 0 = Normal, 1 = Possible, >= 2 = High-confidence.

Detector logic is ported verbatim from the standalone outlier project
(sp2/main.py). The Isolation Forest import is guarded so the dashboard
still runs (as a three-detector vote) when scikit-learn is unavailable.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when sklearn is missing
    IsolationForest = None
    PCA = None
    SKLEARN_AVAILABLE = False

STRICTNESS_PRESETS = {
    "lenient":  {"tau": 4.0, "k": 2.0},
    "balanced": {"tau": 3.5, "k": 1.5},
    "strict":   {"tau": 3.0, "k": 1.0},
}


def detect_mad(values: np.ndarray, tau: float) -> dict[str, Any]:
    """Modified Z-Score using MAD; fall back to mean abs deviation when MAD=0."""
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    abs_dev = np.abs(values - median)
    mad = float(np.median(abs_dev))
    fallback_used = False
    if mad == 0.0:
        mean_abs_dev = float(np.mean(abs_dev))
        fallback_used = True
        if mean_abs_dev == 0.0:
            scores = np.zeros_like(values)
        else:
            scores = 0.6745 * (values - median) / (1.2533 * mean_abs_dev)
    else:
        scores = 0.6745 * (values - median) / mad
    flags = np.abs(scores) > tau
    return {
        "scores": scores.tolist(),
        "flags": flags.tolist(),
        "stats": {
            "median": median,
            "mad": mad,
            "tau": tau,
            "fallback_used": fallback_used,
            "flagged": int(flags.sum()),
        },
    }


def detect_iqr(values: np.ndarray, k: float) -> dict[str, Any]:
    """Standard Tukey IQR fence test."""
    values = np.asarray(values, dtype=float)
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    flags = (values < lo) | (values > hi)
    scores = np.where(values < lo, (lo - values) / max(iqr, 1e-12),
                      np.where(values > hi, (values - hi) / max(iqr, 1e-12), 0.0))
    return {
        "scores": scores.tolist(),
        "flags": flags.tolist(),
        "stats": {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "k": k,
            "lower_fence": lo,
            "upper_fence": hi,
            "flagged": int(flags.sum()),
        },
    }


def detect_isoforest(matrix: np.ndarray, contamination: float, seed: int = 42) -> dict[str, Any]:
    """Isolation Forest on the numeric matrix (n_samples x n_features)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    n = matrix.shape[0]
    contamination = float(np.clip(contamination, 0.01, 0.20))
    if not SKLEARN_AVAILABLE:
        return {
            "scores": [0.0] * n,
            "flags": [False] * n,
            "stats": {"contamination": contamination, "available": False, "flagged": 0},
        }
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(matrix)
    raw = model.decision_function(matrix)
    preds = model.predict(matrix)
    flags = (preds == -1)
    anomaly_score = -raw
    threshold = float(-model.offset_) if hasattr(model, "offset_") else 0.0
    return {
        "scores": anomaly_score.tolist(),
        "flags": flags.tolist(),
        "stats": {
            "contamination": contamination,
            "n_estimators": int(model.n_estimators),
            "score_threshold": threshold,
            "available": True,
            "flagged": int(flags.sum()),
        },
    }


def detect_iterative(values: np.ndarray, iterations: int) -> dict[str, Any]:
    """Iterative mean/std trimming.

    For i = 1..I, with f_i = (I - i) / 10:
        S_i = { d in S_{i-1} : |d - mu_{i-1}| <= f_i * sigma_{i-1} }

    A point is removed if dropped in any iteration; it is flagged for the
    vote only if removed while f_i >= 1.5.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    iterations = int(np.clip(iterations, 1, 100))

    alive = np.ones(n, dtype=bool)
    removed_iter = np.full(n, -1, dtype=int)
    removed_threshold = np.full(n, np.nan)

    mu_hist: list[float] = []
    sigma_hist: list[float] = []
    f_hist: list[float] = []
    alive_count_hist: list[int] = []

    surv = values.copy()
    mu = float(np.mean(surv)) if len(surv) else 0.0
    sigma = float(np.std(surv, ddof=0)) if len(surv) else 0.0
    mu_hist.append(mu)
    sigma_hist.append(sigma)
    f_hist.append(float("nan"))
    alive_count_hist.append(int(alive.sum()))

    for i in range(1, iterations + 1):
        f_i = (iterations - i) / 10.0
        if sigma == 0.0 or not np.isfinite(sigma):
            mu_hist.append(mu)
            sigma_hist.append(sigma)
            f_hist.append(f_i)
            alive_count_hist.append(int(alive.sum()))
            continue
        cutoff = f_i * sigma
        alive_idx = np.where(alive)[0]
        if len(alive_idx) == 0:
            mu_hist.append(mu)
            sigma_hist.append(sigma)
            f_hist.append(f_i)
            alive_count_hist.append(0)
            continue
        diffs = np.abs(values[alive_idx] - mu)
        to_remove = alive_idx[diffs > cutoff]
        if len(to_remove):
            alive[to_remove] = False
            removed_iter[to_remove] = i
            removed_threshold[to_remove] = f_i
        if alive.any():
            mu = float(np.mean(values[alive]))
            sigma = float(np.std(values[alive], ddof=0))
        mu_hist.append(mu)
        sigma_hist.append(sigma)
        f_hist.append(f_i)
        alive_count_hist.append(int(alive.sum()))

    total_removed_mask = ~alive
    vote_mask = np.zeros(n, dtype=bool)
    rt = np.nan_to_num(removed_threshold, nan=-1.0)
    vote_mask[(removed_iter > 0) & (rt >= 1.5)] = True

    score = np.where(removed_iter > 0, np.nan_to_num(removed_threshold, nan=0.0), 0.0)

    K = None
    for idx in range(1, len(sigma_hist)):
        prev_s = sigma_hist[idx - 1]
        cur_s = sigma_hist[idx]
        if prev_s <= 0 or not math.isfinite(prev_s):
            continue
        rel_drop = (prev_s - cur_s) / prev_s
        if rel_drop < 0.01:
            K = idx
            break

    return {
        "scores": score.tolist(),
        "flags": vote_mask.tolist(),
        "removed": total_removed_mask.tolist(),
        "removed_iter": removed_iter.tolist(),
        "stats": {
            "iterations": iterations,
            "total_removed": int(total_removed_mask.sum()),
            "flagged_for_vote": int(vote_mask.sum()),
            "final_mu": mu_hist[-1],
            "final_sigma": sigma_hist[-1],
            "initial_mu": mu_hist[0],
            "initial_sigma": sigma_hist[0],
            "K_flat": K,
        },
        "history": {
            "mu": mu_hist,
            "sigma": sigma_hist,
            "f": f_hist,
            "alive_count": alive_count_hist,
        },
    }


def run_ensemble(values: np.ndarray, strictness: str, contamination: float,
                 iterations: int) -> dict[str, Any]:
    """Run all four detectors on a 1-D series and reconcile by majority vote."""
    values = np.asarray(values, dtype=float)
    preset = STRICTNESS_PRESETS.get(strictness, STRICTNESS_PRESETS["balanced"])
    tau, k = preset["tau"], preset["k"]

    mad = detect_mad(values, tau)
    iqr = detect_iqr(values, k)
    iso = detect_isoforest(values.reshape(-1, 1), contamination)
    itr = detect_iterative(values, iterations)

    flags_matrix = np.array([mad["flags"], iqr["flags"], iso["flags"], itr["flags"]], dtype=bool)
    vote_count = flags_matrix.sum(axis=0)
    verdict = np.where(vote_count >= 2, "high",
                       np.where(vote_count == 1, "possible", "normal"))

    return {
        "n": int(len(values)),
        "values": values.tolist(),
        "params": {"strictness": strictness, "tau": tau, "k": k,
                   "contamination": contamination, "iterations": iterations},
        "mad": mad,
        "iqr": iqr,
        "iso": iso,
        "itr": itr,
        "vote": {
            "count": vote_count.tolist(),
            "verdict": verdict.tolist(),
            "tallies": {
                "high": int((verdict == "high").sum()),
                "possible": int((verdict == "possible").sum()),
                "normal": int((verdict == "normal").sum()),
            },
            "per_method": {
                "mad": int(flags_matrix[0].sum()),
                "iqr": int(flags_matrix[1].sum()),
                "iso": int(flags_matrix[2].sum()),
                "itr": int(flags_matrix[3].sum()),
            },
        },
    }


def run_multivariate(matrix: np.ndarray, columns: list[str], strictness: str,
                     contamination: float, iterations: int) -> dict[str, Any]:
    """Run the ensemble across multiple numeric features.

    MAD, IQR, and the iterative method run per column and a row is flagged
    if any column flags it (OR aggregation). Isolation Forest runs once on
    the full matrix so it can catch combination outliers. A row's vote is
    the number of the four methods that flag it.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    n, p = matrix.shape
    preset = STRICTNESS_PRESETS.get(strictness, STRICTNESS_PRESETS["balanced"])
    tau, k = preset["tau"], preset["k"]

    mad_row = np.zeros(n, dtype=bool)
    iqr_row = np.zeros(n, dtype=bool)
    itr_row = np.zeros(n, dtype=bool)
    per_column: dict[str, dict[str, Any]] = {}

    for j, col in enumerate(columns):
        v = matrix[:, j]
        m = detect_mad(v, tau)
        q = detect_iqr(v, k)
        it = detect_iterative(v, iterations)
        per_column[col] = {
            "mad": m["stats"]["flagged"],
            "iqr": q["stats"]["flagged"],
            "itr": it["stats"]["flagged_for_vote"],
        }
        mad_row |= np.asarray(m["flags"], dtype=bool)
        iqr_row |= np.asarray(q["flags"], dtype=bool)
        itr_row |= np.asarray(it["flags"], dtype=bool)

    iso = detect_isoforest(matrix, contamination)
    iso_row = np.asarray(iso["flags"], dtype=bool)

    flags_matrix = np.array([mad_row, iqr_row, iso_row, itr_row], dtype=bool)
    vote_count = flags_matrix.sum(axis=0)
    verdict = np.where(vote_count >= 2, "high",
                       np.where(vote_count == 1, "possible", "normal"))

    # 2-D projection for plotting
    pca_proj = None
    if p >= 2 and SKLEARN_AVAILABLE:
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        try:
            pca_proj = PCA(n_components=2).fit_transform((matrix - means) / stds).tolist()
        except Exception:
            pca_proj = None
    if pca_proj is None:
        if p >= 2:
            pca_proj = matrix[:, :2].tolist()
        else:
            pca_proj = [[float(i), float(matrix[i, 0])] for i in range(n)]

    return {
        "n": int(n),
        "columns": columns,
        "params": {"strictness": strictness, "tau": tau, "k": k,
                   "contamination": contamination, "iterations": iterations},
        "row_flags": {
            "mad": mad_row.tolist(),
            "iqr": iqr_row.tolist(),
            "iso": iso_row.tolist(),
            "itr": itr_row.tolist(),
        },
        "per_column": per_column,
        "pca": pca_proj,
        "vote": {
            "count": vote_count.tolist(),
            "verdict": verdict.tolist(),
            "tallies": {
                "high": int((verdict == "high").sum()),
                "possible": int((verdict == "possible").sum()),
                "normal": int((verdict == "normal").sum()),
            },
            "per_method": {
                "mad": int(mad_row.sum()),
                "iqr": int(iqr_row.sum()),
                "iso": int(iso_row.sum()),
                "itr": int(itr_row.sum()),
            },
        },
    }
