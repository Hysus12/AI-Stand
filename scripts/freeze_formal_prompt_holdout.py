from __future__ import annotations

import argparse

from spbce.data.datasets import load_survey_records
from spbce.utils.io import write_json
from spbce.utils.prompt_benchmark import (
    build_formal_holdout_manifest,
    derive_contaminated_question_blacklist,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()

    records = load_survey_records(args.input)
    blacklist = derive_contaminated_question_blacklist(args.audit_report)
    manifest = build_formal_holdout_manifest(
        records=records,
        source_manifest_path=args.source_split,
        blacklist_question_ids=blacklist,
        output_path=args.output,
        sample_size=args.sample_size,
    )
    write_json(args.output, manifest)
    print(f"Wrote frozen formal holdout manifest to {args.output}")


if __name__ == "__main__":
    main()
