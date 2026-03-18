from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests

from spbce.preprocessing.gss import (
    DEFAULT_GSS_OUTCOME_VARS,
    DEFAULT_GSS_SURVEY_VARS,
    build_gss_behavior_records,
    build_gss_survey_records,
    load_gss_frame,
)
from spbce.utils.io import write_jsonl

GSS_ZIP_URL = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip"


def ensure_gss_dta(zip_path: Path, dta_path: Path) -> None:
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(GSS_ZIP_URL, stream=True, timeout=120)
        response.raise_for_status()
        with zip_path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if not dta_path.exists():
        dta_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract("gss7224_r3.dta", path=dta_path.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="data/raw/gss/GSS_stata.zip")
    parser.add_argument("--dta-path", default="data/raw/gss/extracted/gss7224_r3.dta")
    parser.add_argument("--survey-output", default="data/interim/gss_survey_records.jsonl")
    parser.add_argument(
        "--behavior-output", default="data/interim/gss_behavior_proxy_records.jsonl"
    )
    parser.add_argument("--min-sample-size", type=int, default=40)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    dta_path = Path(args.dta_path)
    ensure_gss_dta(zip_path=zip_path, dta_path=dta_path)

    frame, metadata = load_gss_frame(
        dta_path=dta_path,
        survey_vars=DEFAULT_GSS_SURVEY_VARS,
        outcome_vars=DEFAULT_GSS_OUTCOME_VARS,
    )
    survey_records = build_gss_survey_records(
        frame=frame,
        metadata=metadata,
        min_sample_size=args.min_sample_size,
    )
    behavior_records = build_gss_behavior_records(
        frame=frame,
        metadata=metadata,
        min_sample_size=args.min_sample_size,
    )

    write_jsonl(args.survey_output, [record.model_dump(mode="json") for record in survey_records])
    write_jsonl(
        args.behavior_output, [record.model_dump(mode="json") for record in behavior_records]
    )
    print(f"Wrote {len(survey_records)} GSS survey records to {args.survey_output}")
    print(f"Wrote {len(behavior_records)} GSS paired behavior records to {args.behavior_output}")


if __name__ == "__main__":
    main()
