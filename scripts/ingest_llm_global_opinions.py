from __future__ import annotations

import argparse

from datasets import load_dataset

from spbce.preprocessing.survey import canonicalize_llm_global_opinions_row
from spbce.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/interim/llm_global_opinions.jsonl")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    dataset = load_dataset("Anthropic/llm_global_opinions", split="train")
    records: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        if args.max_records is not None and index >= args.max_records:
            break
        normalized = canonicalize_llm_global_opinions_row(dict(row))
        records.extend(record.model_dump(mode="json") for record in normalized)

    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} normalized survey records to {args.output}")


if __name__ == "__main__":
    main()
