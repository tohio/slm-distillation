from __future__ import annotations

import argparse

from distill.preference.build_preference_dataset import build_preference_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO preference dataset.")
    parser.add_argument(
        "--config",
        default="configs/preference.yaml",
        help="Path to preference dataset config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_preference_dataset(args.config)

    print(f"Accepted: {result.accepted}")
    print(f"Rejected: {result.rejected}")
    print(f"Output: {result.output_path}")
    print(f"Rejected output: {result.rejected_path}")


if __name__ == "__main__":
    main()
