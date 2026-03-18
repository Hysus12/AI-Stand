from __future__ import annotations

from spbce.metrics.distributions import (
    js_divergence,
    normalize_distribution,
    probability_mae,
    safe_kl_divergence,
    top_option_accuracy,
)


def test_distribution_metrics_are_stable() -> None:
    predicted = normalize_distribution([0.6, 0.4])
    observed = normalize_distribution([0.7, 0.3])
    assert 0.0 <= js_divergence(predicted, observed) < 1.0
    assert safe_kl_divergence(predicted, observed) >= 0.0
    assert probability_mae(predicted, observed) == 0.1
    assert top_option_accuracy(predicted, observed) == 1.0
