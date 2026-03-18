from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from spbce.schema.project import (
    ProjectInput,
    ProjectOutput,
    SurveyQuestion,
    SyntheticRespondentRecord,
)


def _normalized_weights(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    total = float(sum(values.values()))
    if total <= 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: float(value / total) for key, value in values.items()}


def _rounded_allocations(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {key: weight * total for key, weight in weights.items()}
    floors = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = total - sum(floors.values())
    if remainder <= 0:
        return floors
    ranked = sorted(
        weights,
        key=lambda key: (raw[key] - floors[key], weights[key]),
        reverse=True,
    )
    for key in ranked[:remainder]:
        floors[key] += 1
    return floors


def _question_utilities(question: SurveyQuestion) -> np.ndarray | None:
    option_count = len(question.options)
    if question.scoring_direction == "neutral":
        return None
    if option_count == 1:
        return np.asarray([0.0], dtype=float)
    base = np.linspace(1.0, 0.0, num=option_count, dtype=float)
    if question.scoring_direction == "positive_low":
        base = base[::-1]
    return base


def _tilted_distribution(
    probabilities: Iterable[float],
    question: SurveyQuestion,
    bias: float,
    latent_profile_strength: float,
    uncertainty: float,
) -> np.ndarray:
    base = np.asarray(list(probabilities), dtype=float)
    base = np.clip(base, 1e-6, None)
    base = base / base.sum()
    utilities = _question_utilities(question)
    if utilities is None:
        return base
    centered = utilities - float(np.mean(utilities))
    strength = latent_profile_strength * (0.5 + max(0.0, 1.0 - uncertainty))
    logits = np.log(base) + (bias * strength * centered)
    logits = logits - float(np.max(logits))
    tilted = np.exp(logits)
    return tilted / tilted.sum()


def _profile_weights(weighted_score: float | None) -> dict[str, float]:
    score = 0.5 if weighted_score is None else float(min(1.0, max(0.0, weighted_score)))
    weights = {
        "promoter": 0.2 + (0.45 * score),
        "mainstream": 0.25,
        "skeptical": 0.2 + (0.45 * (1.0 - score)),
    }
    return _normalized_weights(weights)


def _combo_weight_lookup(project: ProjectInput, output: ProjectOutput) -> dict[str, float]:
    segment_defaults = {
        segment.segment_id: segment.estimated_weight or 1.0 for segment in project.target_segments
    }
    segment_weights = _normalized_weights(
        project.generation_settings.segment_weights or segment_defaults
    )
    if not segment_weights:
        segment_weights = _normalized_weights(segment_defaults)

    if project.variants:
        variant_defaults = {variant.variant_id: 1.0 for variant in project.variants}
        variant_weights = _normalized_weights(
            project.generation_settings.variant_weights or variant_defaults
        )
    else:
        variant_weights = {"__base__": 1.0}

    combo_weights: dict[str, float] = {}
    for segment_result in output.per_segment_results:
        for variant_result in segment_result.variant_results:
            variant_key = variant_result.variant_id or "__base__"
            combo_key = f"{segment_result.segment_id}::{variant_key}"
            combo_weights[combo_key] = (
                segment_weights.get(segment_result.segment_id, 0.0)
                * variant_weights.get(variant_key, 0.0)
            )
    return _normalized_weights(combo_weights)


def generate_synthetic_respondents(
    project: ProjectInput,
    output: ProjectOutput,
    respondent_count: int | None = None,
) -> tuple[list[SyntheticRespondentRecord], list[str]]:
    target_count = respondent_count or project.generation_settings.synthetic_respondent_count
    rng = np.random.default_rng(project.generation_settings.random_seed)
    question_lookup = {question.question_id: question for question in project.survey_questions}
    combo_weights = _combo_weight_lookup(project, output)
    allocations = _rounded_allocations(combo_weights, target_count)
    respondents: list[SyntheticRespondentRecord] = []

    for segment_result in output.per_segment_results:
        for variant_result in segment_result.variant_results:
            variant_key = variant_result.variant_id or "__base__"
            combo_key = f"{segment_result.segment_id}::{variant_key}"
            combo_count = allocations.get(combo_key, 0)
            if combo_count <= 0:
                continue
            profile_weights = _profile_weights(variant_result.aggregate_signals.weighted_score)
            profiles = list(profile_weights)
            profile_probabilities = np.asarray(list(profile_weights.values()), dtype=float)
            for local_index in range(combo_count):
                profile = str(rng.choice(profiles, p=profile_probabilities))
                profile_bias_lookup = {
                    "promoter": 0.9,
                    "mainstream": 0.0,
                    "skeptical": -0.9,
                }
                bias = float(
                    np.clip(
                        rng.normal(loc=profile_bias_lookup[profile], scale=0.25),
                        -1.5,
                        1.5,
                    )
                )
                answers: dict[str, str] = {}
                for question_result in variant_result.question_results:
                    question = question_lookup[question_result.question_id]
                    ordered_probabilities = [
                        float(question_result.distribution[option]) for option in question.options
                    ]
                    respondent_distribution = _tilted_distribution(
                        probabilities=ordered_probabilities,
                        question=question,
                        bias=bias,
                        latent_profile_strength=project.generation_settings.latent_profile_strength,
                        uncertainty=question_result.uncertainty,
                    )
                    answer = str(rng.choice(question.options, p=respondent_distribution))
                    answers[question.question_id] = answer
                respondents.append(
                    SyntheticRespondentRecord(
                        respondent_id=(
                            f"{project.project_id}_{segment_result.segment_id}_"
                            f"{variant_key}_{local_index:04d}"
                        ),
                        segment_id=segment_result.segment_id,
                        segment_name=segment_result.segment_name,
                        variant_id=variant_result.variant_id,
                        variant_name=variant_result.variant_name,
                        latent_profile=profile,
                        answers=answers,
                    )
                )

    notes = [
        "Synthetic respondents are sampled from per-question segment distributions.",
        (
            "Rows include a lightweight latent profile bias to create within-respondent "
            "consistency across the questionnaire."
        ),
        (
            "This is a first-pass conditional sampler, not a learned joint respondent model; "
            "use diagnostics when distributions are close or uncertainty is high."
        ),
    ]
    return respondents, notes
