#!/usr/bin/env python3
"""Convert input/output JSONL records to chat messages JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def convert_record(record: Any, system_prompt: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(record, dict):
        return None, "top-level JSON value must be an object"

    if "messages" in record:
        if "input" in record or "output" in record:
            return None, "record mixes messages and input/output schemas"
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            return None, "'messages' must be a non-empty list"
        return record, None

    if not nonempty_text(record.get("input")):
        return None, "'input' must be a non-empty string"
    if not nonempty_text(record.get("output")):
        return None, "'output' must be a non-empty string"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        [
            {"role": "user", "content": record["input"].strip()},
            {"role": "assistant", "content": record["output"].strip()},
        ]
    )
    converted = {
        key: value for key, value in record.items() if key not in {"input", "output"}
    }
    converted["messages"] = messages
    return converted, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert input/output JSONL to messages JSONL; existing messages records "
            "are preserved."
        )
    )
    parser.add_argument("source", type=Path, help="Input JSONL path")
    parser.add_argument("destination", type=Path, help="Output JSONL path")
    parser.add_argument(
        "--system-prompt", help="Optional system message added to converted records"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite the destination if it exists"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()

    if not source.is_file():
        raise SystemExit(f"Source file not found: {source}")
    if source == destination:
        raise SystemExit("Source and destination must be different files.")
    if destination.exists() and not args.force:
        raise SystemExit(f"Destination exists; pass --force to overwrite: {destination}")

    output_records: list[dict[str, Any]] = []
    errors: list[str] = []
    converted_count = 0
    preserved_count = 0

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON at column {exc.colno}: {exc.msg}")
                continue

            was_messages = isinstance(record, dict) and "messages" in record
            result, error = convert_record(record, args.system_prompt)
            if error:
                errors.append(f"line {line_number}: {error}")
                continue
            output_records.append(result or {})
            if was_messages:
                preserved_count += 1
            else:
                converted_count += 1

    if errors:
        print("Conversion aborted; no output was written.")
        for error in errors:
            print(f"- {source}:{error}")
        return 1
    if not output_records:
        print("Conversion aborted: source contains no JSON records.")
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in output_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"Wrote {len(output_records)} record(s) to {destination} "
        f"({converted_count} converted, {preserved_count} preserved)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
