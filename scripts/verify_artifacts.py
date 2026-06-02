from __future__ import annotations

import argparse

from distill.artifacts.handoff import verify_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify artifact manifest checksums.")
    parser.add_argument(
        "--manifest",
        default="manifest.json",
        help="Path to manifest.json.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory for manifest paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_manifest(args.manifest, root=args.root)
    print(f"Run: {result.run_name}")
    print(f"Files verified: {result.file_count}")
    print(f"Bytes verified: {result.total_bytes}")


if __name__ == "__main__":
    main()
