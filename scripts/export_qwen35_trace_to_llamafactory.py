#!/usr/bin/env python3
"""Export Qwen3.5 CodeSearch traces to LlamaFactory ShareGPT tool-call data.

The raw traces may contain either OpenAI-style structured tool_calls or
Qwen3.5 XML tool calls in assistant content. This exporter converts both into
the same ShareGPT `function_call` turns expected by LlamaFactory, then lets the
`qwen3_5` template render them back into native Qwen3.5 tool-call syntax.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tools_description import TOOLS_DESCRIPTION


ALLOWED_TOOLS = {"glob", "grep", "read_file"}
TOOL_BLOCK_RE = re.compile(r"<(?P<name>glob|grep|read_file)>\s*(?P<body>.*?)\s*</(?P=name)>", re.S)
PARAM_RE = re.compile(r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>", re.S)


def instance_id_from_file(path: Path) -> str:
    name = path.name
    prefix = "llm_messages_"
    suffix = ".json"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def load_success_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return set(obj.get("results") or [])


def extract_reasoning(message: dict[str, Any]) -> str:
    return message.get("reasoning_content") or message.get("reasoning") or message.get("thinking") or ""


def sanitize_text(text: str) -> str:
    return (text or "").replace("<image>", "[image]").replace("<video>", "[video]").replace("<audio>", "[audio]")


def sanitize_think_text(text: str) -> str:
    return sanitize_text(text).replace("<think>", "[think]").replace("</think>", "[/think]")


def with_think_block(reasoning: str, content: str = "") -> str:
    reasoning = sanitize_think_text((reasoning or "").strip())
    content = sanitize_text(content or "")
    if not reasoning:
        return content
    return f"<think>\n{reasoning}\n</think>\n\n{content}"


def parse_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    return {}


def structured_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        name = function.get("name") or ""
        if name not in ALLOWED_TOOLS:
            raise ValueError(f"illegal_tool_name:{name}")
        calls.append({"name": name, "arguments": parse_arguments(function.get("arguments"))})
    return calls


def xml_tool_calls(content: str) -> list[dict[str, Any]]:
    calls = []
    for match in TOOL_BLOCK_RE.finditer(content or ""):
        name = match.group("name")
        args: dict[str, Any] = {}
        for param_match in PARAM_RE.finditer(match.group("body")):
            key = param_match.group("key").strip()
            value = param_match.group("value").strip()
            if not key:
                continue
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        calls.append({"name": name, "arguments": args})
    return calls


def tool_calls_from_message(message: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    calls = structured_tool_calls(message)
    if calls:
        return calls, "structured"

    calls = xml_tool_calls(message.get("content") or "")
    if calls:
        return calls, "xml"

    return [], ""


def format_function_turn(message: dict[str, Any], calls: list[dict[str, Any]], source: str) -> str:
    reasoning_parts = []
    reasoning = extract_reasoning(message).strip()
    if reasoning:
        reasoning_parts.append(reasoning)

    content = (message.get("content") or "").strip()
    if source == "structured" and content:
        reasoning_parts.append(content)
    elif source == "xml":
        preamble = TOOL_BLOCK_RE.sub("", content).strip()
        if preamble:
            reasoning_parts.append(preamble)

    return with_think_block("\n\n".join(reasoning_parts), json.dumps(calls, ensure_ascii=False))


def format_observation(tool_messages: list[dict[str, Any]]) -> str:
    observations = []
    for message in tool_messages:
        observations.append(
            {
                "tool_call_id": message.get("tool_call_id") or "",
                "content": sanitize_text(message.get("content") or ""),
            }
        )
    return json.dumps(observations, ensure_ascii=False)


def append_odd(conversations: list[dict[str, str]], role: str, value: str) -> bool:
    if len(conversations) % 2 != 0:
        return False
    conversations.append({"from": role, "value": value})
    return True


def append_even(conversations: list[dict[str, str]], role: str, value: str) -> bool:
    if len(conversations) % 2 != 1:
        return False
    conversations.append({"from": role, "value": value})
    return True


def final_answer_ok(message: dict[str, Any], require_related_context: bool) -> tuple[bool, str]:
    if message.get("role") != "assistant":
        return False, "final_not_assistant"
    if message.get("tool_calls"):
        return False, "final_has_structured_tool_calls"
    if xml_tool_calls(message.get("content") or ""):
        return False, "final_has_xml_tool_calls"
    content = message.get("content") or ""
    if not content.strip():
        return False, "empty_final_content"
    if "<locations_to_modify>" not in content or "</locations_to_modify>" not in content:
        return False, "missing_locations_to_modify"
    if require_related_context and (
        "<related_context>" not in content or "</related_context>" not in content
    ):
        return False, "missing_related_context"
    return True, ""


def validate_conversations(conversations: list[dict[str, str]]) -> bool:
    if not conversations or len(conversations) % 2 != 0:
        return False
    if conversations[-1].get("from") != "gpt":
        return False

    odd_tags = {"human", "observation"}
    even_tags = {"function_call", "gpt"}
    for idx, message in enumerate(conversations):
        if message.get("from") not in (odd_tags if idx % 2 == 0 else even_tags):
            return False
        if not isinstance(message.get("value"), str):
            return False
    return True


def convert_messages(messages: list[dict[str, Any]], require_related_context: bool) -> tuple[dict[str, Any] | None, str, Counter[str]]:
    if not isinstance(messages, list) or not messages:
        return None, "bad_message_shape", Counter()

    ok, reason = final_answer_ok(messages[-1], require_related_context)
    if not ok:
        return None, reason, Counter()

    system_parts: list[str] = []
    conversations: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    idx = 0

    while idx < len(messages):
        message = messages[idx]
        role = message.get("role")

        if role == "system":
            content = sanitize_text(message.get("content") or "")
            if content:
                system_parts.append(content)
            idx += 1
            continue

        if role == "user":
            content = sanitize_text(message.get("content") or "")
            if not append_odd(conversations, "human", content):
                return None, "bad_user_position", stats
            idx += 1
            continue

        if role == "assistant":
            try:
                calls, source = tool_calls_from_message(message)
            except (TypeError, json.JSONDecodeError, ValueError) as exc:
                return None, str(exc), stats

            if calls:
                if not append_even(conversations, "function_call", format_function_turn(message, calls, source)):
                    return None, "bad_function_call_position", stats
                stats[f"{source}_tool_turns"] += 1
                stats["tool_calls"] += len(calls)
            else:
                value = with_think_block(extract_reasoning(message), message.get("content") or "")
                if not append_even(conversations, "gpt", value):
                    return None, "bad_gpt_position", stats
            idx += 1
            continue

        if role == "tool":
            tool_messages = []
            while idx < len(messages) and messages[idx].get("role") == "tool":
                tool_messages.append(messages[idx])
                idx += 1
            if not append_odd(conversations, "observation", format_observation(tool_messages)):
                return None, "bad_observation_position", stats
            stats["observation_turns"] += 1
            continue

        return None, f"unknown_role:{role}", stats

    if not validate_conversations(conversations):
        return None, "validation_failed", stats
    if stats["tool_calls"] == 0:
        return None, "no_tool_calls", stats

    return {"system": "\n\n".join(system_parts), "conversations": conversations}, "", stats


def normalize_tools(tools: list[dict[str, Any]]) -> str:
    normalized = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(tool["function"])
        else:
            normalized.append(tool)
    return json.dumps(normalized, ensure_ascii=False)


def update_dataset_info(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools",
        },
    }
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--llamafactory-dir", default="../LlamaFactory")
    parser.add_argument("--dataset-name", default="codesearch_sft_qwen35_397b_train_v2_t06_tool")
    parser.add_argument("--out-file", default="codesearch_sft_qwen35_397b_train_v2_t06_tool.jsonl")
    parser.add_argument("--summary-file", default="codesearch_sft_qwen35_397b_train_v2_t06_tool_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-related-context", action="store_true")
    parser.add_argument("--no-dataset-info", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    success_path = run_dir / "success.json"
    success_ids = load_success_ids(success_path if success_path.exists() else None)
    llamafactory_dir = Path(args.llamafactory_dir).resolve()
    data_dir = llamafactory_dir / "data"
    out_path = data_dir / args.out_file
    summary_path = data_dir / args.summary_file
    dataset_info_path = data_dir / "dataset_info.json"

    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not dataset_info_path.exists():
        raise FileNotFoundError(dataset_info_path)

    stats: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    tool_stats: Counter[str] = Counter()
    examples = []

    files = sorted(run_dir.glob("llm_messages_*.json"))
    data_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for source_file in files:
            if args.limit is not None and stats["seen"] >= args.limit:
                break

            instance_id = instance_id_from_file(source_file)
            if success_ids is not None and instance_id not in success_ids:
                stats["skip:not_success_id"] += 1
                continue

            stats["seen"] += 1
            try:
                messages = json.loads(source_file.read_text(encoding="utf-8"))
            except Exception as exc:
                reject_reasons["json_decode_error"] += 1
                stats["rejected"] += 1
                if len(examples) < 5:
                    examples.append({"instance_id": instance_id, "reject": "json_decode_error", "error": str(exc)})
                continue

            converted, reason, local_tool_stats = convert_messages(messages, args.require_related_context)
            if converted is None:
                reject_reasons[reason] += 1
                stats["rejected"] += 1
                continue

            tool_stats.update(local_tool_stats)
            assistant_turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
            record = {
                "id": f"{run_dir.name}:{instance_id}",
                "instance_id": instance_id,
                "run_name": run_dir.name,
                "system": converted["system"],
                "conversations": converted["conversations"],
                "tools": normalize_tools(TOOLS_DESCRIPTION),
                "metadata": {
                    "source_file": str(source_file),
                    "message_count": len(messages),
                    "conversation_turns": len(converted["conversations"]),
                    "assistant_turns": assistant_turns,
                    "tool_turns": local_tool_stats["structured_tool_turns"] + local_tool_stats["xml_tool_turns"],
                    "tool_calls": local_tool_stats["tool_calls"],
                    "structured_tool_turns": local_tool_stats["structured_tool_turns"],
                    "xml_tool_turns": local_tool_stats["xml_tool_turns"],
                    "observation_turns": local_tool_stats["observation_turns"],
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1

            if len(examples) < 5:
                examples.append(
                    {
                        "id": record["id"],
                        "conversation_turns": record["metadata"]["conversation_turns"],
                        "tool_turns": record["metadata"]["tool_turns"],
                        "tool_calls": record["metadata"]["tool_calls"],
                    }
                )

            if stats["seen"] % 1000 == 0:
                print(
                    f"processed={stats['seen']} written={stats['written']} rejected={stats['rejected']}",
                    file=sys.stderr,
                    flush=True,
                )

    if not args.no_dataset_info:
        update_dataset_info(dataset_info_path, args.dataset_name, args.out_file)

    summary = {
        "dataset_name": args.dataset_name,
        "output_file": str(out_path),
        "dataset_info": str(dataset_info_path),
        "source_run_dir": str(run_dir),
        "success_filter": str(success_path) if success_path.exists() else None,
        "require_related_context": args.require_related_context,
        "stats": dict(stats),
        "tool_stats": dict(tool_stats),
        "reject_reasons": dict(reject_reasons.most_common()),
        "keep_rate": stats["written"] / stats["seen"] if stats["seen"] else 0,
        "examples": examples,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
