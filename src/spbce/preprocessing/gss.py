from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from spbce.schema.canonical import (
    BehaviorOutcome,
    BehaviorSurveyQuestion,
    ContextFeatures,
    PairedSurveyBehaviorRecord,
    PopulationStruct,
    SurveyRecord,
    stable_hash,
)
from spbce.utils.text import infer_question_topic

DEFAULT_GSS_SURVEY_VARS = [
    "trust",
    "helpful",
    "fair",
    "partyid",
    "polviews",
    "natheal",
    "natcrime",
    "natenvir",
]

DEFAULT_GSS_OUTCOME_VARS = [
    "boycott",
    "signpet",
    "conoffcl",
    "polfunds",
]

DEFAULT_GSS_GROUP_KEYS = ["year", "sex", "degree", "age_band"]

INVALID_LABEL_SNIPPETS = {
    "dk",
    "don't know",
    "dont know",
    "no answer",
    "not applicable",
    "iap",
    "refused",
    "missing",
    "uncodable",
    "not imputable",
    "other",
    "na",
}

OUTCOME_POSITIVE_LABELS = {
    "boycott": {"yes"},
    "signpet": {"yes"},
    "conoffcl": {"yes"},
    "polfunds": {
        "have done it in the past yr",
        "have done it in the more distant past",
    },
}


@dataclass(slots=True)
class GssMetadata:
    variable_labels: dict[str, str]


def load_gss_frame(
    dta_path: str | Path,
    survey_vars: list[str] | None = None,
    outcome_vars: list[str] | None = None,
) -> tuple[pd.DataFrame, GssMetadata]:
    survey_columns = survey_vars or DEFAULT_GSS_SURVEY_VARS
    outcome_columns = outcome_vars or DEFAULT_GSS_OUTCOME_VARS
    columns = ["year", "sex", "race", "degree", "age", "wtssall", *survey_columns, *outcome_columns]
    reader = pd.io.stata.StataReader(str(dta_path), convert_categoricals=True)
    variable_labels = reader.variable_labels()
    frame = reader.read(columns=columns)
    frame["age_band"] = frame["age"].map(age_to_band)
    frame["wtssall"] = pd.to_numeric(frame["wtssall"], errors="coerce").fillna(1.0)
    return frame, GssMetadata(variable_labels=variable_labels)


def age_to_band(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        if value.strip().lower() == "89 or older":
            age = 89.0
        else:
            try:
                age = float(value)
            except ValueError:
                return None
    elif isinstance(value, (int, float)):
        age = float(value)
    else:
        try:
            age = float(str(value))
        except ValueError:
            return None
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def is_valid_label(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    if not normalized:
        return False
    return all(snippet not in normalized for snippet in INVALID_LABEL_SNIPPETS)


def weighted_distribution(
    series: pd.Series, weights: pd.Series
) -> tuple[list[str], list[float], int]:
    valid_mask = series.map(is_valid_label)
    valid_values = series.loc[valid_mask]
    valid_weights = weights.loc[valid_mask].fillna(1.0)
    if valid_values.empty:
        return [], [], 0

    option_order = []
    if hasattr(valid_values.dtype, "categories"):
        categories = [
            category for category in valid_values.dtype.categories if is_valid_label(category)
        ]
        option_order = [str(category) for category in categories]
    if not option_order:
        option_order = sorted({str(value) for value in valid_values})

    weighted_counts = (
        pd.DataFrame({"value": valid_values.astype(str), "weight": valid_weights.astype(float)})
        .groupby("value", observed=True)["weight"]
        .sum()
    )
    total = float(weighted_counts.sum())
    if total <= 0:
        return [], [], 0
    distribution = [float(weighted_counts.get(option, 0.0) / total) for option in option_order]
    sample_size = int(valid_values.shape[0])
    return option_order, distribution, sample_size


def population_struct_from_row(row: dict[str, Any]) -> PopulationStruct:
    return PopulationStruct(
        age_band=row.get("age_band"),
        gender=str(row.get("sex"))
        if row.get("sex") is not None and not pd.isna(row.get("sex"))
        else None,
        education=str(row.get("degree"))
        if row.get("degree") is not None and not pd.isna(row.get("degree"))
        else None,
        other={"race": str(row.get("race"))}
        if row.get("race") is not None and not pd.isna(row.get("race"))
        else {},
    )


def population_text_from_struct(population_struct: PopulationStruct) -> str:
    parts = [
        "Adults",
        population_struct.gender,
        population_struct.age_band,
        population_struct.education,
        population_struct.other.get("race"),
        "in the United States",
    ]
    return " ".join(part for part in parts if part)


def build_gss_survey_records(
    frame: pd.DataFrame,
    metadata: GssMetadata,
    survey_vars: list[str] | None = None,
    group_keys: list[str] | None = None,
    min_sample_size: int = 40,
) -> list[SurveyRecord]:
    question_vars = survey_vars or DEFAULT_GSS_SURVEY_VARS
    keys = group_keys or DEFAULT_GSS_GROUP_KEYS
    records: list[SurveyRecord] = []

    for group_values, group_frame in frame.groupby(keys, observed=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_dict = dict(zip(keys, group_values, strict=True))
        population_struct = population_struct_from_row(group_dict)
        population_text = population_text_from_struct(population_struct)
        year = int(group_dict["year"])
        time_start = f"{year}-01-01"
        time_end = f"{year}-12-31"
        behavior_group_id = stable_hash(
            "gss_behavior_group",
            str(year),
            population_struct.signature(),
        )

        for question_var in question_vars:
            options, distribution, sample_size = weighted_distribution(
                group_frame[question_var], group_frame["wtssall"]
            )
            if sample_size < min_sample_size or len(options) < 2:
                continue
            question_text = metadata.variable_labels.get(question_var, question_var)
            records.append(
                SurveyRecord(
                    record_id=stable_hash(
                        "gss_7224_r3",
                        question_var,
                        question_text,
                        population_text,
                        str(year),
                    ),
                    dataset_id="gss_7224_r3",
                    study_id="gss_public_use",
                    group_id=stable_hash("gss_survey_group", question_var, str(year)),
                    wave_id=str(year),
                    time_start=time_start,
                    time_end=time_end,
                    domain="general_social_survey",
                    country="United States",
                    language="en",
                    population_text=population_text,
                    population_struct=population_struct,
                    question_id=question_var,
                    question_text=question_text,
                    question_topic=infer_question_topic(question_text),
                    question_type="single_choice",
                    options=options,
                    option_order=list(range(len(options))),
                    observed_distribution=distribution,
                    sample_size=sample_size,
                    weights_available=True,
                    metadata={
                        "source_variable": question_var,
                        "year": year,
                        "behavior_group_id": behavior_group_id,
                    },
                )
            )
    return records


def outcome_rate(
    series: pd.Series, weights: pd.Series, positive_labels: set[str]
) -> tuple[float | None, int]:
    valid_mask = series.map(is_valid_label)
    valid_values = series.loc[valid_mask].astype(str).str.strip().str.lower()
    valid_weights = weights.loc[valid_mask].fillna(1.0).astype(float)
    if valid_values.empty:
        return None, 0
    positive_mask = valid_values.isin({label.lower() for label in positive_labels})
    weighted_positive = float(valid_weights.loc[positive_mask].sum())
    total = float(valid_weights.sum())
    if total <= 0:
        return None, 0
    return weighted_positive / total, int(valid_values.shape[0])


def build_gss_behavior_records(
    frame: pd.DataFrame,
    metadata: GssMetadata,
    survey_vars: list[str] | None = None,
    outcome_vars: list[str] | None = None,
    group_keys: list[str] | None = None,
    min_sample_size: int = 40,
    min_question_count: int = 3,
) -> list[PairedSurveyBehaviorRecord]:
    question_vars = survey_vars or DEFAULT_GSS_SURVEY_VARS
    behavior_vars = outcome_vars or DEFAULT_GSS_OUTCOME_VARS
    keys = group_keys or DEFAULT_GSS_GROUP_KEYS
    records: list[PairedSurveyBehaviorRecord] = []

    for group_values, group_frame in frame.groupby(keys, observed=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_dict = dict(zip(keys, group_values, strict=True))
        population_struct = population_struct_from_row(group_dict)
        population_text = population_text_from_struct(population_struct)
        year = int(group_dict["year"])
        time_start = f"{year}-01-01"
        time_end = f"{year}-12-31"
        behavior_group_id = stable_hash(
            "gss_behavior_group",
            str(year),
            population_struct.signature(),
        )

        survey_questions: list[BehaviorSurveyQuestion] = []
        distribution_features: dict[str, float] = {}
        for question_var in question_vars:
            options, distribution, sample_size = weighted_distribution(
                group_frame[question_var], group_frame["wtssall"]
            )
            if sample_size < min_sample_size or len(options) < 2:
                continue
            question_text = metadata.variable_labels.get(question_var, question_var)
            survey_questions.append(
                BehaviorSurveyQuestion(
                    question_id=question_var,
                    question_text=question_text,
                    question_topic=infer_question_topic(question_text),
                    question_type="single_choice",
                    options=options,
                    option_order=list(range(len(options))),
                    human_distribution=distribution,
                    sample_size=sample_size,
                    weights_available=True,
                    metadata={"source_variable": question_var},
                )
            )
            for option, probability in zip(options, distribution, strict=True):
                feature_key = stable_hash(question_var, option)[:10]
                distribution_features[f"human__{question_var}__{feature_key}"] = probability

        if len(survey_questions) < min_question_count:
            continue

        for outcome_var in behavior_vars:
            positive_labels = OUTCOME_POSITIVE_LABELS[outcome_var]
            outcome_value, outcome_sample_size = outcome_rate(
                group_frame[outcome_var],
                group_frame["wtssall"],
                positive_labels=positive_labels,
            )
            if outcome_value is None or outcome_sample_size < min_sample_size:
                continue
            outcome_name = metadata.variable_labels.get(outcome_var, outcome_var)
            records.append(
                PairedSurveyBehaviorRecord(
                    record_id=stable_hash("gss_behavior_record", behavior_group_id, outcome_var),
                    dataset_id="gss_behavior_proxy_7224_r3",
                    study_id="gss_public_use",
                    group_id=behavior_group_id,
                    time_start=time_start,
                    time_end=time_end,
                    domain="general_social_survey_behavior_proxy",
                    population_text=population_text,
                    population_struct=population_struct,
                    stimulus_text=(
                        "General Social Survey proxy behavior benchmark using "
                        "grouped attitude distributions "
                        "to predict grouped self-reported behavior rates."
                    ),
                    questionnaire_id=stable_hash("gss_questionnaire", *question_vars),
                    survey_questions=survey_questions,
                    survey_distribution_features=distribution_features,
                    actual_outcome=BehaviorOutcome(
                        outcome_id=outcome_var,
                        outcome_type="rate",
                        outcome_name=outcome_name,
                        outcome_value=float(outcome_value),
                        positive_label=", ".join(sorted(positive_labels)),
                        unit="weighted_rate",
                        metadata={
                            "source_variable": outcome_var,
                            "sample_size": outcome_sample_size,
                        },
                    ),
                    context_features=ContextFeatures(
                        seasonality=str(year),
                        campaign_type="survey_behavior_proxy",
                        channel="gss_public_use",
                        other={"country": "United States"},
                    ),
                    metadata={
                        "year": year,
                        "survey_variables": question_vars,
                        "outcome_variable": outcome_var,
                    },
                )
            )
    return records
