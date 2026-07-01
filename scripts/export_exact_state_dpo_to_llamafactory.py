#!/usr/bin/env python3
"""Convert Exact-State DeciSearch preference records to LLaMA-Factory DPO data.

Input accepts either direct chosen/rejected records:

    {"prompt": "...", "system": "...", "chosen": "...", "rejected": "..."}

or branch-scoring records produced by an offline rollout job:

    {
      "prompt": "...",
      "system": "...",
      "branches": [
        {"response": "...", "score": 0.82, "valid": true},
        {"response": "...", "score": 0.31, "valid": true}
      ]
    }

For branch records, the highest-scoring valid response becomes `chosen` and the
lowest-scoring valid response becomes `rejected`, subject to `--min-score-gap`.
The output is ShareGPT preference format expected by LLaMA-Factory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to parse {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def response_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("value"), str):
            return value["value"].strip()
        if isinstance(value.get("content"), str):
            return value["content"].strip()
        if value.get("tool_calls"):
            return json.dumps(value["tool_calls"], ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def normalize_tools(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return json.dumps(text, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def choose_pair(row: dict[str, Any], min_score_gap: float) -> tuple[str, str, float | None, str | None]:
    if "chosen" in row and "rejected" in row:
        chosen = response_to_text(row.get("chosen"))
        rejected = response_to_text(row.get("rejected"))
        gap = row.get("score_gap")
        try:
            gap_value = float(gap) if gap is not None else None
        except (TypeError, ValueError):
            gap_value = None
        if not chosen or not rejected:
            return "", "", gap_value, "empty_direct_pair"
        if gap_value is not None and gap_value < min_score_gap:
            return "", "", gap_value, "small_score_gap"
        return chosen, rejected, gap_value, None

    branches = row.get("branches") or row.get("responses") or []
    if not isinstance(branches, list):
        return "", "", None, "bad_branches"

    valid = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        if branch.get("valid") is False:
            continue
        response = response_to_text(branch.get("response") or branch.get("text") or branch.get("content"))
        if not response:
            continue
        try:
            score = float(branch.get("score"))
        except (TypeError, ValueError):
            continue
        valid.append((score, response))

    if len(valid) < 2:
        return "", "", None, "not_enough_valid_branches"

    valid.sort(key=lambda item: item[0])
    low_score, rejected = valid[0]
    high_score, chosen = valid[-1]
    gap = high_score - low_score
    if gap < min_score_gap:
        return "", "", gap, "small_score_gap"
    if chosen == rejected:
        return "", "", gap, "duplicate_responses"
    return chosen, rejected, gap, None


def build_record(row: dict[str, Any], chosen: str, rejected: str, gap: float | None) -> dict[str, Any]:
    prompt = row.get("prompt") or row.get("state_prompt") or row.get("input") or row.get("instruction") or ""
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, ensure_ascii=False)

    record: dict[str, Any] = {
        "id": row.get("id") or row.get("snapshot_id") or row.get("instance_id") or "",
        "conversations": [{"from": "human", "value": prompt}],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
        "metadata": {
            "source_id": row.get("id") or row.get("snapshot_id"),
            "instance_id": row.get("instance_id"),
            "component": row.get("component"),
            "score_gap": gap,
            "score": row.get("score"),
        },
    }

    system = row.get("system") or row.get("system_prompt")
    if isinstance(system, str) and system.strip():
        record["system"] = system

    tools = normalize_tools(row.get("tools"))
    if tools:
        record["tools"] = tools

    return record


def update_dataset_info(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "system": "system",
            "tools": "tools",
        },
    }
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Exact-state branch/pair JSONL.")
    parser.add_argument("--llamafactory-dir", default="../LLaMA-Factory", type=Path)
    parser.add_argument("--dataset-name", default="decisearch_exact_state_dpo")
    parser.add_argument("--out-file", default="decisearch_exact_state_dpo.jsonl")
    parser.add_argument("--summary-file", default="decisearch_exact_state_dpo_summary.json")
    parser.add_argument("--min-score-gap", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-dataset-info", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    data_dir = args.llamafactory_dir.resolve() / "data"
    out_path = data_dir / args.out_file
    summary_path = data_dir / args.summary_file
    dataset_info_path = data_dir / "dataset_info.json"
    data_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in rows:
            stats["seen"] += 1
            chosen, rejected, gap, reason = choose_pair(row, args.min_score_gap)
            if reason is not None:
                reject_reasons[reason] += 1
                stats["rejected"] += 1
                continue
            record = build_record(row, chosen, rejected, gap)
            if not record["conversations"][0]["value"]:
                reject_reasons["empty_prompt"] += 1
                stats["rejected"] += 1
                continue
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "id": record["id"],
                        "component": record["metadata"].get("component"),
                        "score_gap": record["metadata"].get("score_gap"),
                    }
                )

    if not args.no_dataset_info:
        update_dataset_info(dataset_info_path, args.dataset_name, args.out_file)

    summary = {
        "dataset_name": args.dataset_name,
        "input": str(args.input),
        "output_file": str(out_path),
        "dataset_info": str(dataset_info_path),
        "min_score_gap": args.min_score_gap,
        "stats": dict(stats),
        "reject_reasons": dict(reject_reasons.most_common()),
        "keep_rate": stats["written"] / stats["seen"] if stats["seen"] else 0.0,
        "examples": examples,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
