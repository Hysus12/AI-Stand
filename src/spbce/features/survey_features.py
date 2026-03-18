from __future__ import annotations

import pandas as pd

from spbce.schema.api import PredictSurveyRequest
from spbce.schema.canonical import SurveyRecord

OPTION_ROW_COLUMNS = [
    "record_id",
    "option_text",
    "option_index",
    "combined_text",
    "population_signature",
    "domain",
    "country",
    "question_topic",
    "question_type",
    "target_probability",
]


def build_option_rows(records: list[SurveyRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        for option_index, (option_text, probability) in enumerate(
            zip(record.options, record.observed_distribution, strict=True)
        ):
            rows.append(
                {
                    "record_id": record.record_id,
                    "option_text": option_text,
                    "option_index": option_index,
                    "combined_text": " ".join(
                        [
                            record.question_text,
                            f"Population: {record.population_text}",
                            f"Option: {option_text}",
                        ]
                    ),
                    "population_signature": record.population_signature(),
                    "domain": record.domain or "unknown",
                    "country": record.country or "unknown",
                    "question_topic": record.question_topic or "unknown",
                    "question_type": record.question_type,
                    "target_probability": probability,
                }
            )
    return pd.DataFrame(rows, columns=OPTION_ROW_COLUMNS)


def build_request_frame(request: PredictSurveyRequest) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    population_signature = request.population_struct.signature() or request.population_text
    for option_index, option_text in enumerate(request.options):
        rows.append(
            {
                "record_id": "inference_request",
                "option_text": option_text,
                "option_index": option_index,
                "combined_text": " ".join(
                    [
                        request.question_text,
                        f"Population: {request.population_text}",
                        f"Option: {option_text}",
                    ]
                ),
                "population_signature": population_signature,
                "domain": request.context.product_category or "unknown",
                "country": request.population_struct.region or "unknown",
                "question_topic": request.context.campaign_type or "unknown",
                "question_type": "single_choice",
            }
        )
    columns = [column for column in OPTION_ROW_COLUMNS if column != "target_probability"]
    return pd.DataFrame(rows, columns=columns)
