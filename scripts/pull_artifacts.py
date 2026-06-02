from __future__ import annotations

import argparse

from distill.artifacts.handoff import pull_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull generated JSONL artifacts from HF.")
    parser.add_argument(
        "--config",
        default="configs/artifacts.yaml",
        help="Path to artifact handoff config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = pull_artifacts(args.config)
    print(f"Run: {result.run_name}")
    print(f"Files copied: {result.file_count}")
    print(f"Bytes copied: {result.total_bytes}")
    print(f"Pulled artifacts from: {result.repo_id}")


if __name__ == "__main__":
    main()
