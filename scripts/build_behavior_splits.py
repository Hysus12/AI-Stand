from __future__ import annotations

import argparse

from spbce.data.behavior_splits import write_behavior_split_manifests
from spbce.data.datasets import load_behavior_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/processed/behavior_splits")
    args = parser.parse_args()

    records = load_behavior_records(args.input)
    manifests = write_behavior_split_manifests(records, args.output_dir)
    print(f"Wrote {len(manifests)} behavior split manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
