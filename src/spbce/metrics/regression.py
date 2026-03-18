from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true_array - pred_array)))


def root_mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((true_array - pred_array) ** 2)))


def r2(y_true: list[float], y_pred: list[float]) -> float:
    return float(r2_score(y_true, y_pred))


def spearman_correlation(y_true: list[float], y_pred: list[float]) -> float:
    correlation = spearmanr(y_true, y_pred).statistic
    return float(correlation if correlation is not None else 0.0)
