"""Chat wrapper for DeciSearch.

The default path targets OpenAI-compatible chat APIs. The ZAI path uses the
Zhipu/Z.ai SDK and text-form tool calls so the workflow transcript remains
portable across providers.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from decisearch.json_utils import parse_jsonish
from tools.tools_description import TOOLS_DESCRIPTION


@dataclass
class ChatOptions:
    model: str
    provider: str = "openai"
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float = 0.0
    repetition_penalty: float | None = None
    thinking: bool = False
    reasoning_effort: str = "high"
    client_parse_tools: bool = False


def tool_prompt_block() -> str:
    return (
        "\n\n# Available Tools\n"
        "Use tools by replying only with one or more tool calls in this exact format:\n"
        '<tool_call>{"name": "tool_name", "arguments": {}}</tool_call>\n'
        "Tool call payloads should be valid JSON. Use lowercase true/false/null.\n"
        "Tool observations will be returned in <tool_result> blocks.\n"
        "Available tool schemas:\n"
        f"{json.dumps(TOOLS_DESCRIPTION, ensure_ascii=False)}"
    )


def parse_text_tool_calls(content: str | None) -> list[dict[str, Any]]:
    if not content or "<tool_call>" not in content:
        return []

    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S):
        payload = parse_jsonish(match.group(1))
        payloads = payload if isinstance(payload, list) else [payload]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = item.get("arguments") or {}
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
    return calls


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_jsonable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=False)
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


def _coerce_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("value")
                if value is not None:
                    parts.append(str(value))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _message_dump(chat_message: Any) -> dict[str, Any]:
    dumped = _jsonable(chat_message)
    if isinstance(dumped, dict):
        return dumped
    return {
        "role": _get_field(chat_message, "role"),
        "content": _coerce_content(_get_field(chat_message, "content")),
        "tool_calls": _jsonable(_get_field(chat_message, "tool_calls")),
    }


def extract_assistant_message(chat_message: Any) -> dict[str, Any]:
    raw_message = _message_dump(chat_message)
    content = _coerce_content(_get_field(chat_message, "content"))
    result = {
        "role": _get_field(chat_message, "role", "assistant"),
        "content": content,
        "tool_calls": [],
        "raw_message": raw_message,
    }

    for field in ("reasoning", "reasoning_content", "thinking"):
        value = raw_message.get(field)
        if value is not None:
            result[field] = value

    raw_tool_calls = _get_field(chat_message, "tool_calls") or []
    if raw_tool_calls:
        for tool_call in raw_tool_calls:
            function = _get_field(tool_call, "function") or {}
            result["tool_calls"].append(
                {
                    "id": _get_field(tool_call, "id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": _get_field(tool_call, "type", "function"),
                    "function": {
                        "name": _get_field(function, "name"),
                        "arguments": _get_field(function, "arguments") or "{}",
                    },
                }
            )
    else:
        result["tool_calls"] = parse_text_tool_calls(content)
        if result["tool_calls"]:
            result["content"] = None

    return result


def _response_metadata(response: Any, choice: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in ("id", "model", "created", "object", "system_fingerprint"):
        value = _get_field(response, field)
        if value is not None:
            metadata[field] = _jsonable(value)

    usage = _get_field(response, "usage")
    if usage is not None:
        metadata["usage"] = _jsonable(usage)

    finish_reason = _get_field(choice, "finish_reason")
    if finish_reason is not None:
        metadata["finish_reason"] = _jsonable(finish_reason)

    index = _get_field(choice, "index")
    if index is not None:
        metadata["choice_index"] = _jsonable(index)

    return metadata


class ChatClient:
    def __init__(self, client: Any, options: ChatOptions, *, close_client: bool = True):
        self.client = client
        self.options = options
        self.close_client = close_client

    def _extra_body(self) -> dict[str, Any] | None:
        body: dict[str, Any] = {}
        if self.options.thinking:
            if self.options.reasoning_effort == "xhigh":
                body["reasoning_effort"] = self.options.reasoning_effort
            else:
                body["chat_template_kwargs"] = {
                    "thinking": True,
                    "reasoning_effort": self.options.reasoning_effort,
                }
        if self.options.top_k is not None:
            body["top_k"] = self.options.top_k
        if self.options.min_p is not None:
            body["min_p"] = self.options.min_p
        if self.options.repetition_penalty is not None:
            body["repetition_penalty"] = self.options.repetition_penalty
        return body or None

    def _tool_call_text(self, tool_call: dict[str, Any]) -> str:
        function = tool_call.get("function") or {}
        payload = {
            "name": function.get("name") or "",
            "arguments": parse_jsonish(function.get("arguments")) or {},
        }
        return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"

    def _api_messages_text_tools(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []
        tool_names: dict[str, str] = {}
        for message in messages:
            role = message.get("role")
            if role in {"system", "user"}:
                api_messages.append({"role": role, "content": message.get("content") or ""})
            elif role == "assistant":
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    for tool_call in tool_calls:
                        function = tool_call.get("function") or {}
                        tool_names[tool_call.get("id") or ""] = function.get("name") or "tool"
                    content = message.get("content") or "\n".join(self._tool_call_text(call) for call in tool_calls)
                else:
                    content = message.get("content") or ""
                api_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                tool_call_id = message.get("tool_call_id") or ""
                name = tool_names.get(tool_call_id, "tool")
                content = (
                    f"<tool_result id=\"{tool_call_id}\" name=\"{name}\">\n"
                    f"{message.get('content') or ''}\n"
                    "</tool_result>"
                )
                api_messages.append({"role": "user", "content": content})
        return api_messages

    def _api_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.options.client_parse_tools:
            return self._api_messages_text_tools(messages)

        api_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role in {"system", "user"}:
                api_messages.append({"role": role, "content": message.get("content") or ""})
            elif role == "assistant":
                clean = {"role": "assistant", "content": message.get("content")}
                if message.get("tool_calls"):
                    clean["tool_calls"] = message["tool_calls"]
                api_messages.append(clean)
            elif role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "content": message.get("content") or "",
                        "tool_call_id": message.get("tool_call_id") or "",
                    }
                )
        return api_messages

    def _create_kwargs(self, messages: list[dict[str, Any]], *, use_tools: bool, parallel_tool_calls: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.options.model,
            "messages": self._api_messages(messages),
            "temperature": self.options.temperature,
            "top_p": self.options.top_p,
        }
        if self.options.max_tokens is not None:
            kwargs["max_tokens"] = self.options.max_tokens

        if self.options.provider == "zai":
            if self.options.thinking:
                kwargs["thinking"] = {"type": "enabled"}
                kwargs["reasoning_effort"] = self.options.reasoning_effort
            if use_tools and not self.options.client_parse_tools:
                kwargs["tools"] = TOOLS_DESCRIPTION
                kwargs["tool_choice"] = "auto"
            return kwargs

        kwargs["extra_body"] = self._extra_body()
        kwargs["presence_penalty"] = self.options.presence_penalty
        if use_tools and not self.options.client_parse_tools:
            kwargs["tools"] = TOOLS_DESCRIPTION
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        return kwargs

    def _should_retry(self, exc: Exception) -> bool:
        text = str(exc).lower()
        retry_fragments = (
            "already borrowed",
            "timeout",
            "timed out",
            "temporarily",
            "rate limit",
            "too many requests",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        return any(fragment in text for fragment in retry_fragments)

    def create(self, messages: list[dict[str, Any]], *, use_tools: bool, parallel_tool_calls: bool = False) -> dict[str, Any]:
        kwargs = self._create_kwargs(messages, use_tools=use_tools, parallel_tool_calls=parallel_tool_calls)
        response = None
        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                if not self._should_retry(exc) or attempt == 5:
                    raise
                time.sleep(min(2.0 * (attempt + 1), 30.0))
        assert response is not None
        choice = response.choices[0]
        message = extract_assistant_message(choice.message)
        api_metadata = _response_metadata(response, choice)
        if api_metadata:
            message["metadata"] = {"api": api_metadata}
        return message

    def close(self) -> None:
        if not self.close_client:
            return
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
