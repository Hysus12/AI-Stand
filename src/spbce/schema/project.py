from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from spbce.schema.canonical import PopulationStruct

QuestionType = Literal["single_choice", "likert", "ordinal"]
ScoringDirection = Literal["positive_high", "positive_low", "neutral"]


class SegmentSpec(BaseModel):
    segment_id: str
    segment_name: str
    demographic_description: str
    psychographic_description: str | None = None
    behavioral_description: str | None = None
    population_struct: PopulationStruct = Field(default_factory=PopulationStruct)
    estimated_weight: float | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("estimated_weight")
    @classmethod
    def validate_estimated_weight(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("estimated_weight must be positive when provided")
        return value

    def combined_description(self) -> str:
        parts = [self.demographic_description.strip()]
        if self.psychographic_description:
            parts.append(self.psychographic_description.strip())
        if self.behavioral_description:
            parts.append(self.behavioral_description.strip())
        return " ".join(part for part in parts if part)


class SurveyQuestion(BaseModel):
    question_id: str
    question_text: str
    question_type: QuestionType
    options: list[str]
    tags: list[str] = Field(default_factory=list)
    theme: str | None = None
    scoring_direction: ScoringDirection = "positive_high"
    decision_weight: float = 1.0

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("survey questions require at least two options")
        return value

    @field_validator("decision_weight")
    @classmethod
    def validate_decision_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("decision_weight must be positive")
        return value


class VariantSpec(BaseModel):
    variant_id: str
    variant_name: str
    variant_description: str | None = None
    price: float | None = None
    message: str | None = None
    feature: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


class GenerationSettings(BaseModel):
    synthetic_respondent_count: int = 1000
    random_seed: int = 42
    segment_weights: dict[str, float] = Field(default_factory=dict)
    variant_weights: dict[str, float] = Field(default_factory=dict)
    latent_profile_strength: float = 0.85

    @field_validator("synthetic_respondent_count")
    @classmethod
    def validate_synthetic_respondent_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("synthetic_respondent_count must be positive")
        if value > 50_000:
            raise ValueError("synthetic_respondent_count cannot exceed 50000")
        return value

    @field_validator("segment_weights", "variant_weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if any(weight <= 0 for weight in value.values()):
            raise ValueError("all generation weights must be positive")
        return value

    @field_validator("latent_profile_strength")
    @classmethod
    def validate_latent_profile_strength(cls, value: float) -> float:
        if value <= 0 or value > 3:
            raise ValueError("latent_profile_strength must be in (0, 3]")
        return value


class InferenceSettings(BaseModel):
    primary_strategy: Literal["deepseek_direct"] = "deepseek_direct"
    fallback_strategy: Literal["heuristic"] = "heuristic"
    fallback_min_confidence: float | None = None
    max_retry_attempts: int = 1

    @field_validator("fallback_min_confidence")
    @classmethod
    def validate_fallback_min_confidence(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("fallback_min_confidence must be between 0 and 1")
        return value

    @field_validator("max_retry_attempts")
    @classmethod
    def validate_max_retry_attempts(cls, value: int) -> int:
        if value < 0 or value > 5:
            raise ValueError("max_retry_attempts must be between 0 and 5")
        return value


class ProjectInput(BaseModel):
    project_id: str
    product_name: str
    product_brief: str
    category: str | None = None
    category_hints: list[str] = Field(default_factory=list)
    target_segments: list[SegmentSpec]
    survey_questions: list[SurveyQuestion]
    variants: list[VariantSpec] = Field(default_factory=list)
    generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)
    inference_settings: InferenceSettings = Field(default_factory=InferenceSettings)

    @field_validator("target_segments")
    @classmethod
    def validate_target_segments(cls, value: list[SegmentSpec]) -> list[SegmentSpec]:
        if not 1 <= len(value) <= 3:
            raise ValueError("Pilot MVP supports 1 to 3 target segments")
        return value

    @field_validator("survey_questions")
    @classmethod
    def validate_survey_questions(cls, value: list[SurveyQuestion]) -> list[SurveyQuestion]:
        if not 7 <= len(value) <= 15:
            raise ValueError("Pilot MVP supports 7 to 15 survey questions")
        return value

    @field_validator("variants")
    @classmethod
    def validate_variants(cls, value: list[VariantSpec]) -> list[VariantSpec]:
        if len(value) > 3:
            raise ValueError("Pilot MVP supports up to 3 variants")
        return value

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ProjectInput:
        segment_ids = [segment.segment_id for segment in self.target_segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment_id values must be unique")
        question_ids = [question.question_id for question in self.survey_questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question_id values must be unique")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("variant_id values must be unique")
        return self


class QuestionResult(BaseModel):
    question_id: str
    question_text: str
    question_type: QuestionType
    options: list[str]
    distribution: dict[str, float]
    top_option: str
    confidence: float
    uncertainty: float
    normalized_score: float | None = None
    requested_strategy: str
    actual_strategy_used: str
    fallback_happened: bool
    fallback_reason: str | None = None
    tags: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AggregateSignals(BaseModel):
    weighted_score: float | None = None
    mean_confidence: float
    mean_uncertainty: float
    fallback_rate: float
    invalid_rate: float
    mean_top_option_margin: float
    question_count: int


class VariantResult(BaseModel):
    variant_id: str | None = None
    variant_name: str
    variant_description: str | None = None
    question_results: list[QuestionResult]
    aggregate_signals: AggregateSignals
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SegmentResult(BaseModel):
    segment_id: str
    segment_name: str
    demographic_description: str
    psychographic_description: str | None = None
    behavioral_description: str | None = None
    variant_results: list[VariantResult]
    aggregate_signals: AggregateSignals
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class QuestionInsight(BaseModel):
    question_id: str
    question_text: str
    variant_id: str | None = None
    variant_name: str | None = None
    average_pairwise_js: float
    score_range: float | None = None
    dominant_option: str | None = None
    mean_confidence: float
    stability_label: Literal["stable", "watch", "unstable"]


class RecommendationItem(BaseModel):
    title: str
    rationale: str
    priority: Literal["high", "medium", "low"]


class FallbackSummary(BaseModel):
    total_predictions: int
    fallback_count: int
    fallback_rate: float
    reason_breakdown: dict[str, int] = Field(default_factory=dict)
    actual_strategy_breakdown: dict[str, int] = Field(default_factory=dict)


class ApiUsageDiagnostics(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_api_cost_usd: float = 0.0
    estimated_cost_per_valid_question_usd: float = 0.0
    usage_notes: list[str] = Field(default_factory=list)


class LatencyDiagnostics(BaseModel):
    total_runtime_seconds: float = 0.0
    total_request_count: int = 0
    average_latency_ms_per_request: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    slowest_requests: list[dict[str, Any]] = Field(default_factory=list)


class OutputQualityDiagnostics(BaseModel):
    total_predictions: int = 0
    direct_success_count: int = 0
    direct_success_rate: float = 0.0
    final_text_present_rate: float = 0.0
    invalid_output_rate: float = 0.0
    parse_failure_rate: float = 0.0
    response_schema_compliance: float = 0.0
    fallback_rate: float = 0.0
    fallback_events: list[dict[str, Any]] = Field(default_factory=list)


class ProjectDiagnostics(BaseModel):
    mean_confidence: float
    mean_uncertainty: float
    invalid_rate: float
    api_usage: ApiUsageDiagnostics = Field(default_factory=ApiUsageDiagnostics)
    latency: LatencyDiagnostics = Field(default_factory=LatencyDiagnostics)
    output_quality: OutputQualityDiagnostics = Field(default_factory=OutputQualityDiagnostics)
    generator_notes: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    question_level_results: list[QuestionInsight] = Field(default_factory=list)


class ProjectOutput(BaseModel):
    project_id: str
    product_name: str
    model_route_used: str
    fallback_summary: FallbackSummary
    per_segment_results: list[SegmentResult]
    synthetic_respondents_path: str | None = None
    executive_summary: str
    recommendations: list[RecommendationItem]
    diagnostics: ProjectDiagnostics


class SyntheticRespondentRecord(BaseModel):
    respondent_id: str
    segment_id: str
    segment_name: str
    variant_id: str | None = None
    variant_name: str | None = None
    latent_profile: str
    answers: dict[str, str]


class ProjectRunResult(BaseModel):
    project_output: ProjectOutput
    synthetic_respondents: list[SyntheticRespondentRecord]
    export_manifest: dict[str, str] = Field(default_factory=dict)
