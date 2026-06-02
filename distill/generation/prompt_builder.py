from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from distill.generation.prompts import PromptRecord


@dataclass(frozen=True)
class PromptBuildConfig:
    output_path: str
    seed_paths: list[str]
    default_target_records: int
    include_seed_prompts: bool


@dataclass(frozen=True)
class PromptBuildResult:
    output_path: str
    records: int
    category_counts: dict[str, int]


def load_prompt_build_config(path: str | Path) -> PromptBuildConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts")

    if not isinstance(prompts, dict):
        raise ValueError("prompt config requires a 'prompts' mapping")

    output_path = prompts.get("output_path")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("prompts.output_path must be a non-empty string")

    seed_paths = prompts.get("seed_paths", [])
    if not isinstance(seed_paths, list):
        raise ValueError("prompts.seed_paths must be a list")

    normalized_seed_paths: list[str] = []
    for index, seed_path in enumerate(seed_paths):
        if not isinstance(seed_path, str) or not seed_path:
            raise ValueError(f"prompts.seed_paths item {index} must be a string")
        normalized_seed_paths.append(seed_path)

    default_target_records = int(prompts.get("default_target_records", 500))
    if default_target_records <= 0:
        raise ValueError("prompts.default_target_records must be greater than zero")

    include_seed_prompts = bool(prompts.get("include_seed_prompts", True))

    return PromptBuildConfig(
        output_path=output_path,
        seed_paths=normalized_seed_paths,
        default_target_records=default_target_records,
        include_seed_prompts=include_seed_prompts,
    )


def _record(
    *,
    record_id: str,
    category: str,
    prompt: str,
    family: str,
    variant: int,
    metadata: dict[str, Any] | None = None,
) -> PromptRecord:
    return PromptRecord(
        id=record_id,
        category=category,
        prompt=prompt.strip(),
        metadata={
            "family": family,
            "variant": variant,
            **(metadata or {}),
        },
    )


def _pick(values: list[str], index: int, stride: int = 1) -> str:
    return values[(index * stride) % len(values)]


DOMAINS = [
    "cloud cost analysis",
    "database maintenance",
    "CI/CD release planning",
    "log analysis",
    "incident response",
    "data validation",
    "model evaluation",
    "API integration",
    "security review",
    "batch processing",
    "configuration management",
    "file synchronization",
    "queue processing",
    "report generation",
    "access auditing",
    "backup verification",
    "tokenization",
    "artifact handoff",
    "teacher response validation",
    "DPO preference preparation",
]

AUDIENCES = [
    "a junior engineer",
    "a database administrator",
    "a cloud operations team",
    "a product manager",
    "a security reviewer",
    "a Python developer",
    "an executive stakeholder",
    "a data engineer",
    "an ML engineer",
    "a platform engineer",
]

OUTPUT_STYLES = [
    "concise bullets",
    "a short checklist",
    "step-by-step instructions",
    "a compact table",
    "a short explanation with examples",
    "an implementation plan",
    "a runbook-style response",
    "a validation checklist",
]

CONSTRAINTS = [
    "include edge cases",
    "avoid unnecessary assumptions",
    "include validation checks",
    "show failure handling",
    "keep the answer concise",
    "include rollback steps",
    "separate required and optional steps",
    "include a minimal example",
    "explain trade-offs",
    "call out security risks",
    "include observability signals",
    "include cost considerations",
]

PYTHON_TASKS = [
    "deduplicate records by a composite key",
    "merge two sorted event streams",
    "validate nested JSON payloads",
    "chunk a large file into bounded batches",
    "retry transient HTTP failures",
    "normalize timestamps into UTC",
    "summarize grouped metrics",
    "detect missing sequence numbers",
    "redact sensitive fields",
    "parse configuration from environment variables",
    "build a manifest with checksums",
    "compare expected and actual records",
    "stream JSONL safely",
    "validate S3 object keys",
    "split training and validation records",
]

LANGUAGES = ["Python", "Go", "Rust", "TypeScript", "Bash"]

AWS_SERVICES = [
    "S3",
    "Lambda",
    "ECS",
    "CloudWatch",
    "EventBridge",
    "SQS",
    "SNS",
    "IAM",
    "KMS",
    "CloudFormation",
    "Step Functions",
    "AWS Backup",
    "SageMaker",
    "Glue",
    "Athena",
]

DATABASES = ["PostgreSQL", "Oracle", "MySQL", "SQLite", "DynamoDB"]

SQL_TASKS = [
    "find duplicate customer records",
    "summarize daily transaction totals",
    "identify orphaned rows",
    "compare source and target table counts",
    "rank users by recent activity",
    "find gaps in a sequence",
    "build a retention report",
    "detect slow-growing table bloat",
    "audit role grants",
    "validate foreign-key coverage",
]

FACTUAL_TOPICS = [
    "a private employee salary",
    "an unreleased product roadmap",
    "a current stock price without live data",
    "a confidential customer contract",
    "a person's home address",
    "a proprietary incident report",
    "an unknown medical diagnosis",
    "a future court ruling",
    "an internal security key",
    "a private calendar invite",
    "a non-public acquisition rumor",
    "a user's password reset token",
]

EDU_CONCEPTS = [
    "attention in transformer models",
    "database indexing",
    "event-driven architecture",
    "idempotent retries",
    "Kerberos authentication",
    "object storage",
    "hash maps",
    "message queues",
    "container images",
    "rate limiting",
    "encryption at rest",
    "checkpointing",
    "data deduplication",
    "model distillation",
    "DPO training",
    "tokenization",
    "gradient clipping",
    "network subnets",
    "least privilege access",
    "schema validation",
]

DATA_FORMATS = ["JSONL", "CSV", "YAML", "Parquet metadata", "NDJSON", "plain text logs"]
TARGET_FORMATS = ["JSONL", "CSV", "Markdown table", "nested JSON", "summary report", "validation errors"]

INCIDENTS = [
    "API latency increased after a deployment",
    "S3 uploads are intermittently failing",
    "a database migration is stuck",
    "a model evaluation job produced empty output",
    "a batch job is duplicating records",
    "CloudWatch alarms are noisy",
    "a tokenization job ran out of disk",
    "DPO training loss stopped changing",
]

MATH_CONTEXTS = [
    "cloud storage cost",
    "training token budget",
    "batch throughput",
    "database growth",
    "queue backlog",
    "API request cost",
    "GPU rental time",
    "artifact transfer size",
    "validation acceptance rate",
    "checkpoint storage",
]


def _code_prompt(variant: int) -> PromptRecord:
    language = _pick(LANGUAGES, variant)
    task = _pick(PYTHON_TASKS, variant, 3)
    constraint = _pick(CONSTRAINTS, variant, 5)
    name = f"task_{variant:06d}"
    prompt = (
        f"Write a {language} implementation for `{name}` that can {task}. "
        f"The solution must {constraint}. Include clear input assumptions and return only the code."
    )
    return _record(
        record_id=f"code_{variant:08d}",
        category="code",
        prompt=prompt,
        family="code_generation",
        variant=variant,
        metadata={"language": language, "task": task},
    )


def _debug_prompt(variant: int) -> PromptRecord:
    language = _pick(LANGUAGES, variant, 2)
    incident = _pick(INCIDENTS, variant, 3)
    constraint = _pick(CONSTRAINTS, variant, 7)
    prompt = (
        f"Debug this {language} service issue: {incident}. "
        f"Give the likely causes, the first three checks to run, and a minimal fix. "
        f"The answer should {constraint}."
    )
    return _record(
        record_id=f"debug_{variant:08d}",
        category="debugging",
        prompt=prompt,
        family="debugging",
        variant=variant,
        metadata={"language": language, "incident": incident},
    )


def _data_transform_prompt(variant: int) -> PromptRecord:
    source_format = _pick(DATA_FORMATS, variant)
    target_format = _pick(TARGET_FORMATS, variant, 2)
    domain = _pick(DOMAINS, variant, 3)
    constraint = _pick(CONSTRAINTS, variant, 4)
    prompt = (
        f"Design a data transformation for a {domain} workflow. "
        f"The input is {source_format} and the output must be {target_format}. "
        f"Describe the schema, validation rules, and error handling. The answer should {constraint}."
    )
    return _record(
        record_id=f"data_transform_{variant:08d}",
        category="data_transform",
        prompt=prompt,
        family="data_transform",
        variant=variant,
        metadata={"source_format": source_format, "target_format": target_format},
    )


def _factual_restraint_prompt(variant: int) -> PromptRecord:
    topic = _pick(FACTUAL_TOPICS, variant)
    style = _pick(OUTPUT_STYLES, variant, 3)
    prompt = (
        f"A user asks for {topic}. Respond safely in {style}. "
        "Do not invent facts. Explain what information is missing and provide a safe alternative."
    )
    return _record(
        record_id=f"factual_restraint_{variant:08d}",
        category="factual_restraint",
        prompt=prompt,
        family="factual_restraint",
        variant=variant,
        metadata={"topic": topic},
    )


def _education_prompt(variant: int) -> PromptRecord:
    concept = _pick(EDU_CONCEPTS, variant, 2)
    audience = _pick(AUDIENCES, variant, 3)
    style = _pick(OUTPUT_STYLES, variant, 5)
    prompt = (
        f"Explain {concept} to {audience}. Use {style}. "
        "Include one practical example and one common mistake to avoid."
    )
    return _record(
        record_id=f"educational_qa_{variant:08d}",
        category="educational_qa",
        prompt=prompt,
        family="educational_qa",
        variant=variant,
        metadata={"concept": concept, "audience": audience},
    )


def _aws_prompt(variant: int) -> PromptRecord:
    service = _pick(AWS_SERVICES, variant)
    domain = _pick(DOMAINS, variant, 2)
    constraint = _pick(CONSTRAINTS, variant, 6)
    prompt = (
        f"Create an AWS implementation plan using {service} for {domain}. "
        f"The plan must {constraint}. Include IAM, monitoring, and failure handling considerations."
    )
    return _record(
        record_id=f"aws_{variant:08d}",
        category="cloud",
        prompt=prompt,
        family="aws_operations",
        variant=variant,
        metadata={"service": service, "domain": domain},
    )


def _database_prompt(variant: int) -> PromptRecord:
    database = _pick(DATABASES, variant)
    task = _pick(SQL_TASKS, variant, 3)
    constraint = _pick(CONSTRAINTS, variant, 5)
    table = f"app_events_{variant % 997:03d}"
    prompt = (
        f"For a {database} database, write a query or procedure to {task} in table `{table}`. "
        f"The answer must {constraint}. Include index or performance considerations."
    )
    return _record(
        record_id=f"database_{variant:08d}",
        category="database",
        prompt=prompt,
        family="database",
        variant=variant,
        metadata={"database": database, "task": task},
    )


def _math_prompt(variant: int) -> PromptRecord:
    context = _pick(MATH_CONTEXTS, variant)
    a = 25 + (variant * 17) % 900
    b = 3 + (variant * 11) % 97
    c = 2 + (variant * 7) % 19
    prompt = (
        f"Solve this {context} word problem step by step. "
        f"A job processes {a} units per batch, runs {b} batches per hour, and has {c} failed batches. "
        "Compute the successful units per hour and explain the arithmetic."
    )
    return _record(
        record_id=f"arithmetic_{variant:08d}",
        category="arithmetic",
        prompt=prompt,
        family="arithmetic_reasoning",
        variant=variant,
        metadata={"context": context, "a": a, "b": b, "c": c},
    )


def _planning_prompt(variant: int) -> PromptRecord:
    domain = _pick(DOMAINS, variant, 7)
    audience = _pick(AUDIENCES, variant, 5)
    constraint = _pick(CONSTRAINTS, variant, 3)
    prompt = (
        f"Create a practical plan for {domain} for {audience}. "
        f"The plan should {constraint}. Include prerequisites, execution steps, validation, and rollback."
    )
    return _record(
        record_id=f"planning_{variant:08d}",
        category="planning",
        prompt=prompt,
        family="planning",
        variant=variant,
        metadata={"domain": domain, "audience": audience},
    )


FAMILIES: list[Callable[[int], PromptRecord]] = [
    _code_prompt,
    _debug_prompt,
    _data_transform_prompt,
    _factual_restraint_prompt,
    _education_prompt,
    _aws_prompt,
    _database_prompt,
    _math_prompt,
    _planning_prompt,
]


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def _seed_record_from_dict(raw: dict[str, Any], index: int, source: str) -> PromptRecord | None:
    prompt = raw.get("prompt") or raw.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    category = raw.get("category")
    if not isinstance(category, str) or not category:
        category = "seed"

    record_id = raw.get("id") or raw.get("prompt_id")
    if not isinstance(record_id, str) or not record_id:
        record_id = f"seed_{Path(source).stem}_{index:08d}"

    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    metadata = {
        **metadata,
        "family": "seed",
        "source_path": source,
    }

    return PromptRecord(
        id=record_id,
        category=category,
        prompt=prompt.strip(),
        metadata=metadata,
    )


def load_seed_prompts(paths: list[str]) -> list[PromptRecord]:
    records: list[PromptRecord] = []

    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    continue
                record = _seed_record_from_dict(raw, index, path_text)
                if record is not None:
                    records.append(record)

    return records


def build_prompt_records(
    *,
    target_records: int,
    seed_records: list[PromptRecord] | None = None,
    include_seed_prompts: bool = True,
) -> list[PromptRecord]:
    if target_records <= 0:
        raise ValueError("target_records must be greater than zero")

    output: list[PromptRecord] = []
    seen: set[str] = set()

    if include_seed_prompts:
        for seed in seed_records or []:
            normalized = _normalize_prompt(seed.prompt)
            if normalized in seen:
                continue
            output.append(seed)
            seen.add(normalized)
            if len(output) >= target_records:
                return output

    generated_variant = 0
    max_attempts = target_records * 50 + 10_000

    while len(output) < target_records and generated_variant < max_attempts:
        family_index = generated_variant % len(FAMILIES)
        family_variant = generated_variant // len(FAMILIES)
        record = FAMILIES[family_index](family_variant)

        normalized = _normalize_prompt(record.prompt)
        if normalized not in seen:
            output.append(record)
            seen.add(normalized)

        generated_variant += 1

    if len(output) < target_records:
        raise RuntimeError(
            f"unable to build requested prompt count: requested={target_records}, built={len(output)}"
        )

    return output


def _record_to_dict(record: PromptRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "category": record.category,
        "prompt": record.prompt,
        "metadata": record.metadata,
    }


def write_prompt_records(path: str | Path, records: list[PromptRecord]) -> PromptBuildResult:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_counts = Counter(record.category for record in records)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_dict(record), sort_keys=True))
            handle.write("\n")

    return PromptBuildResult(
        output_path=str(output_path),
        records=len(records),
        category_counts=dict(sorted(category_counts.items())),
    )


def build_prompts_from_config(
    config_path: str | Path,
    *,
    target_records: int | None = None,
    output_path: str | None = None,
) -> PromptBuildResult:
    config = load_prompt_build_config(config_path)
    desired_records = target_records or config.default_target_records
    seed_records = load_seed_prompts(config.seed_paths)

    records = build_prompt_records(
        target_records=desired_records,
        seed_records=seed_records,
        include_seed_prompts=config.include_seed_prompts,
    )

    return write_prompt_records(output_path or config.output_path, records)
