from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from spbce.schema.canonical import PopulationStruct


class SurveyContext(BaseModel):
    product_category: str | None = None
    price: float | None = None
    channel: str | None = None
    campaign_type: str | None = None
    time_window: str | None = None
    market_metadata: dict[str, Any] = Field(default_factory=dict)


class PredictSurveyRequest(BaseModel):
    question_text: str
    options: list[str]
    population_text: str
    population_struct: PopulationStruct = Field(default_factory=PopulationStruct)
    context: SurveyContext = Field(default_factory=SurveyContext)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("predict requests require at least two options")
        return value


class PredictSurveyResponse(BaseModel):
    distribution: dict[str, float]
    uncertainty: float
    ood_flag: bool
    support_notes: list[str]
    calibration_notes: list[str]


class MvpCostLatencySummary(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_api_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    average_latency_ms_per_request: float = 0.0
    request_count: int = 0
    retry_count: int = 0


class MvpPredictSurveyRequest(PredictSurveyRequest):
    strategy: str | None = None


class MvpPredictSurveyResponse(BaseModel):
    distribution: dict[str, float]
    top_option: str
    confidence: float
    uncertainty: float
    invalid_flag: bool
    requested_strategy: str
    actual_strategy_used: str
    fallback_happened: bool
    fallback_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    cost_latency_summary: MvpCostLatencySummary = Field(
        default_factory=MvpCostLatencySummary
    )


class PredictBehaviorRequest(BaseModel):
    survey_payload: PredictSurveyRequest
    context: SurveyContext = Field(default_factory=SurveyContext)


class PredictBehaviorResponse(BaseModel):
    predicted_outcome: float | None
    uncertainty: float
    ood_flag: bool
    support_notes: list[str]


class SampleRespondentsRequest(BaseModel):
    survey_payload: PredictSurveyRequest
    n: int = 100

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n must be positive")
        if value > 10_000:
            raise ValueError("n cannot exceed 10000")
        return value


class SampleRespondentsResponse(BaseModel):
    respondents: list[dict[str, Any]]
    sampling_notes: list[str]
