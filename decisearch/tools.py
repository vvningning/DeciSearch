"""Tool execution helpers for DeciSearch."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from decisearch.json_utils import parse_jsonish
from tools.builtin import GlobTool, GrepTool, ReadFileTool


def truncate_tool_content(content: str, max_chars: int) -> str:
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    original_len = len(content)
    return (
        content[:max_chars]
        + f"\n\n[Tool output truncated to {max_chars} chars from {original_len} chars. "
        "Request a narrower follow-up if more context is needed.]"
    )


def tool_call_key(tool_call: dict[str, Any]) -> tuple[str, str]:
    function = tool_call.get("function") or {}
    args = parse_jsonish(function.get("arguments")) or {}
    return function.get("name") or "", json.dumps(args, sort_keys=True, ensure_ascii=False)


class ToolExecutor:
    def __init__(self, workspace_root: Path, *, max_tool_result_chars: int = 12000):
        self.workspace_root = workspace_root
        self.max_tool_result_chars = max_tool_result_chars
        self._tools = {
            "grep": GrepTool(workspace_root=workspace_root),
            "read_file": ReadFileTool(workspace_root=workspace_root),
            "glob": GlobTool(workspace_root=workspace_root),
        }

    def execute_one(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        name = function.get("name") or ""
        arguments = parse_jsonish(function.get("arguments")) or {}
        tool_call_id = tool_call.get("id") or ""

        try:
            tool = self._tools.get(name)
            if tool is None:
                content = f"Error: unknown tool {name}"
            else:
                result = tool.execute(**arguments)
                content = str(result)
            return {
                "role": "tool",
                "content": truncate_tool_content(content, self.max_tool_result_chars),
                "tool_call_id": tool_call_id,
            }
        except Exception as exc:
            return {
                "role": "tool",
                "content": f"Error: {exc}",
                "tool_call_id": tool_call_id,
            }

    def execute_many(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        seen_tool_call_keys: set[tuple[str, str]],
        max_calls: int,
        max_workers: int,
    ) -> list[dict[str, Any]]:
        executable: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            key = tool_call_key(tool_call)
            if key in seen_tool_call_keys:
                skipped.append(
                    {
                        "role": "tool",
                        "content": "Skipped duplicate tool call: the same command was already executed earlier.",
                        "tool_call_id": tool_call.get("id") or "",
                    }
                )
                continue
            if max_calls > 0 and len(executable) >= max_calls:
                skipped.append(
                    {
                        "role": "tool",
                        "content": (
                            f"Skipped tool call because max_calls={max_calls} was reached. "
                            "Use previous results or request a narrower follow-up."
                        ),
                        "tool_call_id": tool_call.get("id") or "",
                    }
                )
                continue
            seen_tool_call_keys.add(key)
            executable.append(tool_call)

        if not executable:
            return skipped

        results: list[dict[str, Any]] = []
        if max_workers <= 1 or len(executable) == 1:
            results = [self.execute_one(call) for call in executable]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.execute_one, call) for call in executable]
                for future in as_completed(futures):
                    results.append(future.result())

        return skipped + results
