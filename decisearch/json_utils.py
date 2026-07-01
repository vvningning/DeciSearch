"""JSON and tool-call parsing helpers used by DeciSearch."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


def escape_invalid_json_string_backslashes(text: str) -> str:
    """Escape backslashes that Python-string-like model outputs leave invalid."""
    out: list[str] = []
    in_string = False
    escaped = False
    quote = ""
    valid_json_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}

    for ch in text:
        if not in_string:
            out.append(ch)
            if ch in ("'", '"'):
                in_string = True
                quote = ch
            continue

        if escaped:
            if ch in valid_json_escapes:
                out.append(ch)
            else:
                out.append("\\")
                out.append(ch)
            escaped = False
            continue

        if ch == "\\":
            out.append("\\")
            escaped = True
            continue

        out.append(ch)
        if ch == quote:
            in_string = False
            quote = ""

    if escaped:
        out.append("\\")
    return "".join(out)


def parse_jsonish(raw: Any) -> Any:
    """Parse strict JSON first, then common Python literal variants."""
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    for candidate in (text, escape_invalid_json_string_backslashes(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            pass
    return None


def _balanced_json_span(text: str, start: int) -> str | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    quote = ""

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue

        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            continue
        if ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
            if not stack:
                return text[start : idx + 1]
    return None


def extract_json_object(text: str | None) -> Any:
    """Extract a JSON object/list from free-form model text."""
    if not text:
        return None

    candidates: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I):
        candidates.append(match.group(1))
    for match in re.finditer(r"<json>\s*(.*?)\s*</json>", text, re.S | re.I):
        candidates.append(match.group(1))

    stripped = text.strip()
    candidates.append(stripped)

    first_positions = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos >= 0]
    if first_positions:
        span = _balanced_json_span(stripped, min(first_positions))
        if span:
            candidates.append(span)

    for candidate in candidates:
        parsed = parse_jsonish(candidate)
        if isinstance(parsed, (dict, list)):
            return parsed
    return None
