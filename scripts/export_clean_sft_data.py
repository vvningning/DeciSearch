#!/usr/bin/env python3
"""Clean CodeSearch teacher traces and export SFT-ready JSONL files.

The exporter keeps the raw trace files untouched. It reads all generated
llm_messages_*.json files, filters incomplete or malformed trajectories, and
writes two common intermediate SFT formats:

1. OpenAI-style messages with assistant `reasoning_content` fields.
2. OpenAI-style messages with reasoning embedded as `<think>...</think>`.

Both formats keep structured tool calls and tool observations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tools_description import TOOLS_DESCRIPTION


RUN_PATTERNS = [
    re.compile(r"sft_teacher_deepseek_v4_flash_train_full_shard(?P<shard>\d+)_tl(?P<turns>\d+)_mode(?P<mode>\w+)"),
    re.compile(r"sft_teacher_deepseek_v4_flash_train_sample(?P<sample>\d+)_shard(?P<shard>\d+)_tl(?P<turns>\d+)_mode(?P<mode>\w+)"),
]


def parse_run_name(path: Path) -> dict[str, Any] | None:
    for pattern in RUN_PATTERNS:
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        groups = match.groupdict()
        return {
            "run_name": path.name,
            "sample": int(groups.get("sample") or 0),
            "shard": int(groups["shard"]),
            "tool_turns": int(groups["turns"]),
            "mode": groups["mode"],
        }
    return None


def instance_id_from_file(path: Path) -> str:
    name = path.name
    prefix = "llm_messages_"
    suffix = ".json"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def get_reasoning(message: dict[str, Any]) -> str:
    return (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
        or ""
    )


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments")
        if arguments is None:
            arguments = "{}"
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)

        normalized.append(
            {
                "id": tool_call.get("id") or "",
                "type": tool_call.get("type") or "function",
                "function": {
                    "name": function.get("name") or "",
                    "arguments": arguments,
                },
            }
        )
    return normalized


def normalize_messages(messages: list[dict[str, Any]], *, think_tags: bool) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")

        if role in {"system", "user"}:
            normalized.append({"role": role, "content": message.get("content") or ""})
            continue

        if role == "tool":
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id") or "",
                    "content": message.get("content") or "",
                }
            )
            continue

        if role == "assistant":
            content = message.get("content") or ""
            reasoning = get_reasoning(message)
            out: dict[str, Any] = {"role": "assistant", "content": content}

            if think_tags and reasoning:
                out["content"] = f"<think>\n{reasoning}\n</think>\n{content}"
            elif reasoning:
                out["reasoning_content"] = reasoning

            tool_calls = normalize_tool_calls(message.get("tool_calls") or [])
            if tool_calls:
                out["tool_calls"] = tool_calls
            normalized.append(out)
            continue

        normalized.append({"role": role or "unknown", "content": message.get("content") or ""})

    return normalized


def validate_tool_pairing(messages: list[dict[str, Any]]) -> tuple[bool, str]:
    pending: list[str] = []
    seen_tool_calls = False

    for message in messages:
        role = message.get("role")

        if role == "assistant":
            if pending:
                return False, "missing_tool_observation"
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                seen_tool_calls = True
                ids = [tc.get("id") for tc in tool_calls]
                if any(not tid for tid in ids):
                    return False, "empty_tool_call_id"
                if len(set(ids)) != len(ids):
                    return False, "duplicate_tool_call_id"
                pending = list(ids)

        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                return False, "empty_tool_observation_id"
            if tool_call_id not in pending:
                return False, "orphan_tool_observation"
            pending.remove(tool_call_id)

        elif role == "user":
            if pending:
                return False, "missing_tool_observation_before_user"

    if pending:
        return False, "missing_tool_observation_at_end"
    if not seen_tool_calls:
        return False, "no_tool_calls"
    return True, ""


def completion_status(messages: Any) -> tuple[bool, str]:
    if not isinstance(messages, list) or not messages:
        return False, "bad_message_shape"
    final = messages[-1]
    if not isinstance(final, dict):
        return False, "bad_final_message"
    if final.get("role") != "assistant":
        return False, "final_not_assistant"
    if final.get("tool_calls"):
        return False, "final_has_tool_calls"
    if not (final.get("content") or "").strip():
        return False, "empty_final_content"
    return True, ""


def has_locations(final_content: str) -> bool:
    return "<locations_to_modify>" in final_content and "</locations_to_modify>" in final_content


def has_related_context(final_content: str) -> bool:
    return "<related_context>" in final_content and "</related_context>" in final_content


def has_dsml_pollution(final_content: str) -> bool:
    return "<｜DSML｜" in final_content or "｜DSML｜tool_calls" in final_content


def detect_forced_summary(messages: list[dict[str, Any]], max_tool_turns: int) -> bool:
    if len(messages) < 2:
        return False
    tool_turns = sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))
    previous = messages[-2]
    return (
        tool_turns >= max_tool_turns
        and previous.get("role") == "user"
        and "Stage 2 - RERANK" in (previous.get("content") or "")
    )


def build_record(
    *,
    run_info: dict[str, Any],
    source_file: Path,
    messages: list[dict[str, Any]],
    think_tags: bool,
) -> dict[str, Any]:
    instance_id = instance_id_from_file(source_file)
    normalized_messages = normalize_messages(messages, think_tags=think_tags)
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    final_content = messages[-1].get("content") or ""
    tool_turns = sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))
    tool_calls = sum(len(m.get("tool_calls") or []) for m in messages if m.get("role") == "assistant")
    reasoning_turns = sum(1 for m in assistant_messages if get_reasoning(m))

    return {
        "id": f"sample{run_info['sample']}_shard{run_info['shard']}:{instance_id}",
        "instance_id": instance_id,
        "sample": run_info["sample"],
        "shard": run_info["shard"],
        "messages": normalized_messages,
        "tools": TOOLS_DESCRIPTION,
        "metadata": {
            "source_file": str(source_file),
            "run_name": run_info["run_name"],
            "tool_turns": tool_turns,
            "tool_calls": tool_calls,
            "message_count": len(messages),
            "assistant_turns": len(assistant_messages),
            "reasoning_turns": reasoning_turns,
            "forced_summary": detect_forced_summary(messages, run_info["tool_turns"]),
            "has_related_context": has_related_context(final_content),
            "char_count": sum(len(m.get("content") or "") + len(get_reasoning(m)) for m in messages),
        },
    }


def iter_run_dirs(output_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    runs = []
    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue
        run_info = parse_run_name(path)
        if run_info:
            runs.append((path, run_info))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--out-dir", default="data/sft/clean_deepseek_v4_flash")
    parser.add_argument("--require-related-context", action="store_true")
    parser.add_argument("--min-reasoning-turns", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_path = out_dir / "codesearch_sft_openai_messages.jsonl"
    think_path = out_dir / "codesearch_sft_openai_messages_think_tags.jsonl"
    reject_path = out_dir / "codesearch_sft_rejects.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"

    stats: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    per_run: dict[str, Counter[str]] = defaultdict(Counter)
    length_buckets: Counter[str] = Counter()

    with field_path.open("w", encoding="utf-8") as field_f, think_path.open("w", encoding="utf-8") as think_f, reject_path.open("w", encoding="utf-8") as reject_f:
        for run_dir, run_info in iter_run_dirs(output_root):
            run_key = run_info["run_name"]
            files = sorted(run_dir.glob("llm_messages_*.json"))
            per_run[run_key]["files"] = len(files)
            stats["files"] += len(files)

            for source_file in files:
                instance_id = instance_id_from_file(source_file)
                try:
                    messages = json.loads(source_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    reason = "json_decode_error"
                    reject_reasons[reason] += 1
                    per_run[run_key][f"reject:{reason}"] += 1
                    reject_f.write(json.dumps({"id": instance_id, "source_file": str(source_file), "reason": reason, "error": str(exc)}, ensure_ascii=False) + "\n")
                    continue

                ok, reason = completion_status(messages)
                if ok:
                    ok, reason = validate_tool_pairing(messages)

                final_content = messages[-1].get("content") if isinstance(messages, list) and messages and isinstance(messages[-1], dict) else ""
                assistant_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"] if isinstance(messages, list) else []
                reasoning_turns = sum(1 for m in assistant_messages if get_reasoning(m))

                if ok and not has_locations(final_content or ""):
                    ok, reason = False, "missing_locations_to_modify"
                if ok and args.require_related_context and not has_related_context(final_content or ""):
                    ok, reason = False, "missing_related_context"
                if ok and has_dsml_pollution(final_content or ""):
                    ok, reason = False, "final_dsml_pollution"
                if ok and reasoning_turns < args.min_reasoning_turns:
                    ok, reason = False, "insufficient_reasoning"

                if not ok:
                    reject_reasons[reason] += 1
                    per_run[run_key][f"reject:{reason}"] += 1
                    reject_f.write(
                        json.dumps(
                            {
                                "id": instance_id,
                                "sample": run_info["sample"],
                                "shard": run_info["shard"],
                                "source_file": str(source_file),
                                "reason": reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                field_record = build_record(run_info=run_info, source_file=source_file, messages=messages, think_tags=False)
                think_record = build_record(run_info=run_info, source_file=source_file, messages=messages, think_tags=True)
                field_f.write(json.dumps(field_record, ensure_ascii=False) + "\n")
                think_f.write(json.dumps(think_record, ensure_ascii=False) + "\n")

                stats["accepted"] += 1
                per_run[run_key]["accepted"] += 1
                tool_turns = field_record["metadata"]["tool_turns"]
                stats["tool_turns_sum"] += tool_turns
                stats["forced_summary"] += int(field_record["metadata"]["forced_summary"])
                if field_record["metadata"]["has_related_context"]:
                    stats["has_related_context"] += 1
                char_count = field_record["metadata"]["char_count"]
                if char_count < 50_000:
                    length_buckets["<50k_chars"] += 1
                elif char_count < 100_000:
                    length_buckets["50k-100k_chars"] += 1
                elif char_count < 250_000:
                    length_buckets["100k-250k_chars"] += 1
                elif char_count < 500_000:
                    length_buckets["250k-500k_chars"] += 1
                else:
                    length_buckets[">=500k_chars"] += 1

    accepted = stats["accepted"]
    summary = {
        "output_files": {
            "openai_messages": str(field_path),
            "openai_messages_think_tags": str(think_path),
            "rejects": str(reject_path),
        },
        "stats": dict(stats),
        "accept_rate": accepted / stats["files"] if stats["files"] else 0,
        "avg_tool_turns": stats["tool_turns_sum"] / accepted if accepted else 0,
        "reject_reasons": dict(reject_reasons),
        "length_buckets": dict(length_buckets),
        "per_run": {key: dict(value) for key, value in sorted(per_run.items())},
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        "# CodeSearch SFT Export Summary",
        "",
        f"- Raw trace files scanned: {stats['files']}",
        f"- Accepted clean traces: {accepted}",
        f"- Accept rate: {summary['accept_rate']:.2%}",
        f"- Average tool turns: {summary['avg_tool_turns']:.2f}",
        f"- Forced summaries: {stats['forced_summary']}",
        f"- Accepted with related_context tag: {stats['has_related_context']}",
        "",
        "## Output Files",
        "",
        f"- `{field_path}`",
        f"- `{think_path}`",
        f"- `{reject_path}`",
        "",
        "## Reject Reasons",
        "",
    ]
    if reject_reasons:
        for key, value in reject_reasons.most_common():
            md_lines.append(f"- {key}: {value}")
    else:
        md_lines.append("- None")
    md_lines.extend(["", "## Length Buckets", ""])
    for key, value in length_buckets.items():
        md_lines.append(f"- {key}: {value}")
    md_lines.extend(["", "## Per Run", ""])
    for key, value in sorted(per_run.items()):
        md_lines.append(f"- {key}: files={value.get('files', 0)}, accepted={value.get('accepted', 0)}")
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
