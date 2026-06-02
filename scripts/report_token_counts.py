from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from distill.datasets.token_report import report_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report estimated token counts for JSONL datasets.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "data/distill/response_distill.jsonl",
            "data/preference/dpo_pairs.jsonl",
        ],
        help="JSONL files to report.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Character/token estimate used for reporting.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = report_paths(args.paths, chars_per_token=args.chars_per_token)

    if args.json:
        print(json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True))
        return

    total_records = sum(report.records for report in reports)
    total_tokens = sum(report.estimated_tokens for report in reports)

    for report in reports:
        print(f"{report.path}: records={report.records} estimated_tokens={report.estimated_tokens}")

    print(f"TOTAL: records={total_records} estimated_tokens={total_tokens}")


if __name__ == "__main__":
    main()
