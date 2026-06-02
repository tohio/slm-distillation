from __future__ import annotations

import argparse

from distill.artifacts.handoff import push_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push generated JSONL artifacts to HF.")
    parser.add_argument(
        "--config",
        default="configs/artifacts.yaml",
        help="Path to artifact handoff config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = push_artifacts(args.config)
    print(f"Run: {result.run_name}")
    print(f"Files: {result.file_count}")
    print(f"Bytes: {result.total_bytes}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Pushed artifacts to: {result.repo_id}")


if __name__ == "__main__":
    main()
