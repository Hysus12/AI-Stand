from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from spbce.metrics.distributions import js_divergence
from spbce.schema.project import (
    ProjectOutput,
    QuestionInsight,
    QuestionResult,
    RecommendationItem,
)


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _variant_label(variant_id: str | None, variant_name: str | None) -> str:
    if variant_id is None:
        return variant_name or "base"
    return f"{variant_name or variant_id} ({variant_id})"


def compute_question_insights(output: ProjectOutput) -> list[QuestionInsight]:
    buckets: dict[tuple[str | None, str], list[tuple[str, QuestionResult, str]]] = defaultdict(list)
    variant_names: dict[tuple[str | None, str], str | None] = {}
    for segment_result in output.per_segment_results:
        for variant_result in segment_result.variant_results:
            for question_result in variant_result.question_results:
                key = (variant_result.variant_id, question_result.question_id)
                buckets[key].append(
                    (segment_result.segment_id, question_result, variant_result.variant_name)
                )
                variant_names[key] = variant_result.variant_name

    insights: list[QuestionInsight] = []
    for (variant_id, question_id), items in buckets.items():
        pairwise_js_values: list[float] = []
        score_values = [
            question_result.normalized_score
            for _, question_result, _ in items
            if question_result.normalized_score is not None
        ]
        for left_index in range(len(items)):
            left_distribution = [
                float(items[left_index][1].distribution[option])
                for option in items[left_index][1].options
            ]
            for right_index in range(left_index + 1, len(items)):
                right_distribution = [
                    float(items[right_index][1].distribution[option])
                    for option in items[right_index][1].options
                ]
                pairwise_js_values.append(js_divergence(left_distribution, right_distribution))
        average_pairwise_js = _mean(pairwise_js_values)
        score_range = (
            float(max(score_values) - min(score_values)) if len(score_values) >= 2 else None
        )
        mean_confidence = _mean([question_result.confidence for _, question_result, _ in items])
        top_options = Counter(question_result.top_option for _, question_result, _ in items)
        dominant_option = top_options.most_common(1)[0][0] if top_options else None
        if average_pairwise_js >= 0.12 or mean_confidence < 0.45:
            stability_label = "unstable"
        elif average_pairwise_js >= 0.06 or mean_confidence < 0.6:
            stability_label = "watch"
        else:
            stability_label = "stable"
        insights.append(
            QuestionInsight(
                question_id=question_id,
                question_text=items[0][1].question_text,
                variant_id=variant_id,
                variant_name=variant_names[(variant_id, question_id)],
                average_pairwise_js=average_pairwise_js,
                score_range=score_range,
                dominant_option=dominant_option,
                mean_confidence=mean_confidence,
                stability_label=stability_label,
            )
        )
    insights.sort(
        key=lambda item: (item.average_pairwise_js, -(item.mean_confidence)),
        reverse=True,
    )
    return insights


def _best_segment_line(output: ProjectOutput) -> tuple[str, str]:
    ranked = sorted(
        output.per_segment_results,
        key=lambda item: item.aggregate_signals.weighted_score
        if item.aggregate_signals.weighted_score is not None
        else -1.0,
        reverse=True,
    )
    best = ranked[0]
    score = best.aggregate_signals.weighted_score
    rationale = (
        f"{best.segment_name} has the strongest aggregate score "
        f"({score:.3f}) with mean confidence {best.aggregate_signals.mean_confidence:.3f}."
        if score is not None
        else (
            f"{best.segment_name} has the strongest confidence profile "
            f"({best.aggregate_signals.mean_confidence:.3f}) among the evaluated segments."
        )
    )
    return best.segment_name, rationale


def _best_variant_line(output: ProjectOutput) -> tuple[str, str] | None:
    scores: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    for segment_result in output.per_segment_results:
        for variant_result in segment_result.variant_results:
            if variant_result.variant_id is None and len(segment_result.variant_results) == 1:
                continue
            key = variant_result.variant_id or "__base__"
            if variant_result.aggregate_signals.weighted_score is not None:
                scores[key].append(variant_result.aggregate_signals.weighted_score)
            labels[key] = _variant_label(variant_result.variant_id, variant_result.variant_name)
    if not scores:
        return None
    ranked = sorted(scores, key=lambda key: _mean(scores[key]), reverse=True)
    best_key = ranked[0]
    return labels[best_key], (
        f"{labels[best_key]} leads on average weighted score "
        f"({_mean(scores[best_key]):.3f}) across segments."
    )


def build_recommendations(
    output: ProjectOutput,
    question_insights: list[QuestionInsight],
) -> list[RecommendationItem]:
    recommendations: list[RecommendationItem] = []
    best_segment_name, best_segment_rationale = _best_segment_line(output)
    recommendations.append(
        RecommendationItem(
            title=f"Prioritize {best_segment_name}",
            rationale=best_segment_rationale,
            priority="high",
        )
    )

    best_variant = _best_variant_line(output)
    if best_variant is not None:
        variant_label, variant_rationale = best_variant
        recommendations.append(
            RecommendationItem(
                title=f"Lead with {variant_label}",
                rationale=variant_rationale,
                priority="high",
            )
        )

    if question_insights:
        most_divergent = question_insights[0]
        recommendations.append(
            RecommendationItem(
                title=f"Use {most_divergent.question_id} as a segmentation lever",
                rationale=(
                    f"{most_divergent.question_text} shows the largest cross-segment divergence "
                    f"(avg JS {most_divergent.average_pairwise_js:.3f})."
                ),
                priority="medium",
            )
        )

    if output.fallback_summary.fallback_rate > 0:
        recommendations.append(
            RecommendationItem(
                title="Track route reliability in pilot delivery",
                rationale=(
                    f"Fallbacks were triggered in {output.fallback_summary.fallback_rate:.1%} "
                    "of question predictions, so project logs and retry monitoring should stay on."
                ),
                priority="medium",
            )
        )

    unstable_count = sum(
        1 for insight in question_insights if insight.stability_label == "unstable"
    )
    if unstable_count > 0 or output.diagnostics.mean_uncertainty > 0.4:
        recommendations.append(
            RecommendationItem(
                title="Treat close calls as validation candidates",
                rationale=(
                    "Higher-uncertainty or unstable questions should be converted into "
                    "focused follow-up tests before committing large spend."
                ),
                priority="medium",
            )
        )
    return recommendations[:5]


def render_executive_summary(
    output: ProjectOutput,
    question_insights: list[QuestionInsight],
    recommendations: list[RecommendationItem],
) -> str:
    best_segment_name, best_segment_rationale = _best_segment_line(output)
    best_variant = _best_variant_line(output)
    stable = [insight for insight in question_insights if insight.stability_label == "stable"][:3]
    unstable = [
        insight for insight in question_insights if insight.stability_label == "unstable"
    ][:3]

    lines = [
        f"# Executive Summary: {output.product_name}",
        "",
        "## Snapshot",
        (
            f"- Primary route: `{output.model_route_used}` with "
            f"{output.fallback_summary.fallback_rate:.1%} fallback usage."
        ),
        (
            f"- Most promising segment: **{best_segment_name}**. "
            f"{best_segment_rationale}"
        ),
    ]
    if best_variant is not None:
        lines.append(f"- Best variant to lead with: **{best_variant[0]}**. {best_variant[1]}")
    lines.append(
        f"- Mean confidence: {output.diagnostics.mean_confidence:.3f}; "
        f"mean uncertainty: {output.diagnostics.mean_uncertainty:.3f}."
    )

    if question_insights:
        lines.extend(
            [
                "",
                "## Segment And Question Signals",
                (
                    f"- Largest divergence: **{question_insights[0].question_id}** "
                    f"(avg JS {question_insights[0].average_pairwise_js:.3f})."
                ),
            ]
        )
        if stable:
            stable_labels = ", ".join(
                f"{insight.question_id} ({insight.mean_confidence:.2f})" for insight in stable
            )
            lines.append(f"- Most stable conclusions: {stable_labels}.")
        if unstable:
            unstable_labels = ", ".join(
                f"{insight.question_id} ({insight.mean_confidence:.2f})"
                for insight in unstable
            )
            lines.append(f"- Least stable conclusions: {unstable_labels}.")

    lines.extend(["", "## Recommended Next Actions"])
    for recommendation in recommendations[:4]:
        lines.append(f"- {recommendation.title}: {recommendation.rationale}")

    if output.diagnostics.limitations:
        lines.extend(["", "## Risks And Limits"])
        for limitation in output.diagnostics.limitations[:4]:
            lines.append(f"- {limitation}")

    return "\n".join(lines).strip() + "\n"
