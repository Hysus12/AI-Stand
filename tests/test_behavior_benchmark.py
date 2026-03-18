from __future__ import annotations

from typing import Any, cast

from spbce.behavior_model.benchmark import evaluate_behavior_models
from spbce.behavior_model.models import BehaviorOutcomeModel


def test_behavior_models_emit_all_required_modes(behavior_fixture: dict[str, object]) -> None:
    models = cast(dict[str, BehaviorOutcomeModel], behavior_fixture["models"])
    augmented_records = cast(list[Any], behavior_fixture["augmented_records"])
    results = evaluate_behavior_models(models, augmented_records[1:])
    model_names = {row["model"] for row in results}
    assert model_names == {"human_only", "ai_only", "hybrid"}
    assert all(
        "mae" in row and "rmse" in row and "r2" in row and "spearman" in row for row in results
    )
