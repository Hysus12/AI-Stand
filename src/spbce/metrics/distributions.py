from __future__ import annotations

import math

import numpy as np

EPSILON = 1e-8


def normalize_distribution(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        return np.full_like(array, 1.0 / max(len(array), 1), dtype=float)
    return array / total


def safe_kl_divergence(p: list[float] | np.ndarray, q: list[float] | np.ndarray) -> float:
    p_array = normalize_distribution(p)
    q_array = normalize_distribution(q)
    return float(np.sum(p_array * np.log((p_array + EPSILON) / (q_array + EPSILON))))


def js_divergence(p: list[float] | np.ndarray, q: list[float] | np.ndarray) -> float:
    p_array = normalize_distribution(p)
    q_array = normalize_distribution(q)
    midpoint = 0.5 * (p_array + q_array)
    return 0.5 * safe_kl_divergence(p_array, midpoint) + 0.5 * safe_kl_divergence(q_array, midpoint)


def probability_mae(p: list[float] | np.ndarray, q: list[float] | np.ndarray) -> float:
    p_array = normalize_distribution(p)
    q_array = normalize_distribution(q)
    return float(np.mean(np.abs(p_array - q_array)))


def probability_rmse(p: list[float] | np.ndarray, q: list[float] | np.ndarray) -> float:
    p_array = normalize_distribution(p)
    q_array = normalize_distribution(q)
    return float(math.sqrt(np.mean((p_array - q_array) ** 2)))


def top_option_accuracy(p: list[float] | np.ndarray, q: list[float] | np.ndarray) -> float:
    p_array = normalize_distribution(p)
    q_array = normalize_distribution(q)
    return float(int(np.argmax(p_array) == np.argmax(q_array)))
