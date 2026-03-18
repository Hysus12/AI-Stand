from __future__ import annotations

import argparse

from spbce.data.datasets import load_survey_records
from spbce.data.splits import write_split_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/processed/splits")
    args = parser.parse_args()

    records = load_survey_records(args.input)
    manifests = write_split_manifests(records, args.output_dir)
    print(f"Wrote {len(manifests)} split manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
