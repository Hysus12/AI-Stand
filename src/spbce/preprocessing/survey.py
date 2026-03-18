from __future__ import annotations

import ast
from typing import Any

from spbce.schema.canonical import (
    PopulationStruct,
    SurveyRecord,
    make_survey_record_id,
    stable_hash,
)
from spbce.utils.text import infer_question_topic


def percentages_to_distribution(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("distribution total must be positive")
    return [float(value) / total for value in values]


def canonicalize_llm_global_opinions_row(row: dict[str, Any]) -> list[SurveyRecord]:
    question = str(row["question"]).strip()
    raw_options = row["options"]
    if isinstance(raw_options, str):
        options = [str(option).strip() for option in ast.literal_eval(raw_options)]
    else:
        options = [str(option).strip() for option in raw_options]
    source = str(row.get("source", "unknown")).strip()
    raw_selections = row["selections"]
    if isinstance(raw_selections, str):
        start = raw_selections.find("{")
        end = raw_selections.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Could not parse selections field from llm_global_opinions row")
        selection_map = ast.literal_eval(raw_selections[start : end + 1])
    else:
        selection_map = raw_selections
    records: list[SurveyRecord] = []
    question_id = stable_hash(source, question)
    group_id = stable_hash(source, question)

    for country, values in selection_map.items():
        distribution = percentages_to_distribution([float(value) for value in values])
        population_text = f"Adults in {country}"
        records.append(
            SurveyRecord(
                record_id=make_survey_record_id("llm_global_opinions", question, population_text),
                dataset_id="llm_global_opinions",
                study_id=source,
                group_id=group_id,
                domain=source,
                country=country,
                population_text=population_text,
                population_struct=PopulationStruct(region=country),
                question_id=question_id,
                question_text=question,
                question_topic=infer_question_topic(question),
                question_type="single_choice",
                options=options,
                option_order=list(range(len(options))),
                observed_distribution=distribution,
                sample_size=None,
                weights_available=False,
                metadata={"source_dataset": source},
            )
        )
    return records
