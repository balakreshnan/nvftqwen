#!/usr/bin/env python3
"""Validate instruction-tuning JSONL in input/output or messages format."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_input_output(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not nonempty_text(record.get("input")):
        errors.append("'input' must be a non-empty string")
    if not nonempty_text(record.get("output")):
        errors.append("'output' must be a non-empty string")
    return errors


def validate_messages(record: dict[str, Any]) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["'messages' must be a non-empty list"]

    errors: list[str] = []
    roles: list[str] = []
    for index, message in enumerate(messages):
        prefix = f"messages[{index}]"
        if not isinstance(message, dict):
            errors.append(f"{prefix} must be an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            errors.append(
                f"{prefix}.role must be one of {sorted(ALLOWED_ROLES)}, got {role!r}"
            )
        else:
            roles.append(role)
        if not nonempty_text(content):
            errors.append(f"{prefix}.content must be a non-empty string")

    if "user" not in roles:
        errors.append("conversation must contain at least one user message")
    if "assistant" not in roles:
        errors.append("conversation must contain at least one assistant message")
    return errors


def validate_record(record: Any) -> tuple[str | None, list[str]]:
    if not isinstance(record, dict):
        return None, ["top-level JSON value must be an object"]

    has_io = "input" in record or "output" in record
    has_messages = "messages" in record
    if has_io and has_messages:
        return None, ["record must use one schema, not both input/output and messages"]
    if has_io:
        return "input/output", validate_input_output(record)
    if has_messages:
        return "messages", validate_messages(record)
    return None, ["record must contain either input/output keys or a messages key"]


def validate_file(path: Path) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    errors: list[str] = []

    if not path.is_file():
        return counts, [f"{path}: file does not exist or is not a regular file"]

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    counts["blank"] += 1
                    continue
                counts["records"] += 1
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    counts["invalid"] += 1
                    errors.append(
                        f"{path}:{line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
                    )
                    continue

                schema, record_errors = validate_record(record)
                if record_errors:
                    counts["invalid"] += 1
                    errors.extend(
                        f"{path}:{line_number}: {message}" for message in record_errors
                    )
                else:
                    counts["valid"] += 1
                    if schema:
                        counts[schema] += 1
    except (OSError, UnicodeError) as exc:
        errors.append(f"{path}: could not read UTF-8 JSONL: {exc}")

    if counts["records"] == 0 and not errors:
        errors.append(f"{path}: contains no JSON records")
    return counts, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate JSONL records in input/output or chat messages format."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL file(s) to validate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total: Counter[str] = Counter()
    all_errors: list[str] = []

    for path in args.paths:
        counts, errors = validate_file(path)
        total.update(counts)
        all_errors.extend(errors)
        print(
            f"{path}: records={counts['records']}, valid={counts['valid']}, "
            f"invalid={counts['invalid']}, input/output={counts['input/output']}, "
            f"messages={counts['messages']}, blank={counts['blank']}"
        )

    if all_errors:
        print("\nErrors:")
        for error in all_errors:
            print(f"- {error}")
        print(f"\nValidation failed with {len(all_errors)} error(s).")
        return 1

    print(
        f"\nValidation passed: {total['valid']} valid record(s) "
        f"across {len(args.paths)} file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
