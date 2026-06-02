from __future__ import annotations

import argparse

from distill.artifacts.handoff import unpack_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unpack generated JSONL artifacts.")
    parser.add_argument("bundle", help="Path to artifact .tar.gz bundle.")
    parser.add_argument(
        "--target-dir",
        default=".",
        help="Directory where artifacts should be unpacked.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = unpack_artifacts(args.bundle, target_dir=args.target_dir)
    print(f"Unpacked artifacts into: {target_dir}")


if __name__ == "__main__":
    main()
