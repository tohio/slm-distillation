import json
import subprocess
import sys
from pathlib import Path


def test_summarize_cost_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "raw_teacher.jsonl"
    output_path = tmp_path / "cost.json"

    input_path.write_text(
        (
            '{"prompt_id":"p1","category":"instruction","prompt":"Say hello.",'
            '"teacher":"deepseek/deepseek-v4-flash","provider":"openrouter",'
            '"response":"hello","input_tokens":1000,"output_tokens":2000,'
            '"metadata":{}}\n'
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_cost.py",
            "--teachers",
            "configs/teachers.yaml",
            "--teacher",
            "deepseek_v4_flash",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout["records"] == 1
    assert saved["input_tokens"] == 1000
    assert saved["output_tokens"] == 2000
    assert saved["total_cost_usd"] > 0
