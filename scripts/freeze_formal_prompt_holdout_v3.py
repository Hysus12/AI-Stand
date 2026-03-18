from __future__ import annotations

import argparse
from collections import Counter

from spbce.data.datasets import load_survey_records
from spbce.utils.io import write_json
from spbce.utils.prompt_benchmark import (
    build_question_id_holdout_manifest,
    derive_contaminated_question_blacklist,
    question_ids_from_formal_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--formal50-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=250)
    args = parser.parse_args()

    records = load_survey_records(args.input)
    debug_blacklist = derive_contaminated_question_blacklist(args.audit_report)
    formal50_question_ids = question_ids_from_formal_manifest(args.formal50_manifest)
    excluded_question_ids = sorted(set(debug_blacklist) | set(formal50_question_ids))
    candidate_question_counts = Counter(
        record.question_id for record in records if record.question_id not in excluded_question_ids
    )
    holdout_question_ids = [
        question_id
        for question_id, _ in sorted(
            candidate_question_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    manifest = build_question_id_holdout_manifest(
        records=records,
        holdout_question_ids=holdout_question_ids,
        sample_size=args.sample_size,
        source_label=f"held_out_question_principle::{args.source_split}",
        excluded_question_ids=excluded_question_ids,
    )
    manifest["strategy"] = "formal_prompt_holdout_v3_frozen"
    manifest["selection_policy"]["candidate_question_counts"] = dict(
        sorted(
            candidate_question_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    manifest["selection_policy"]["heldout_question_ids"] = holdout_question_ids
    manifest["selection_policy"]["excluded_question_ids"] = excluded_question_ids
    write_json(args.output, manifest)
    print(f"Wrote v3 formal holdout manifest to {args.output}")


if __name__ == "__main__":
    main()
