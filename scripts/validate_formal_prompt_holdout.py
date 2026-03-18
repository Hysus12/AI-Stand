from __future__ import annotations

import argparse
import sys

from spbce.data.datasets import load_survey_records
from spbce.utils.io import read_json, write_json
from spbce.utils.prompt_benchmark import (
    derive_contaminated_question_blacklist,
    validate_formal_holdout,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--extra-blacklist-manifest", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = load_survey_records(args.input)
    formal_manifest = read_json(args.formal_manifest)
    blacklist = set(derive_contaminated_question_blacklist(args.audit_report))
    for manifest_path in args.extra_blacklist_manifest:
        extra_manifest = read_json(manifest_path)
        blacklist.update(
            str(row["question_id"])
            for row in extra_manifest.get("records", [])
            if row.get("question_id")
        )
    result = validate_formal_holdout(
        records=records,
        formal_manifest=formal_manifest,
        contaminated_question_blacklist=sorted(blacklist),
    )
    write_json(args.output, result)
    print(f"Wrote formal holdout validation report to {args.output}")
    if result["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
