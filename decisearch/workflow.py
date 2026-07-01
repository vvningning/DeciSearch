"""DeciSearch dynamic evidence workflow runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decisearch.json_utils import extract_json_object
from decisearch.llm import ChatClient, tool_prompt_block
from decisearch.prompts import (
    DECISEARCH_SYSTEM_PROMPT,
    WORKER_RETURN_PROMPT,
    controller_prompt,
    final_prompt,
    verifier_prompt,
    worker_prompt,
)
from decisearch.tools import ToolExecutor
from decisearch.types import EvidenceBoard, EvidenceItem, clean_text, normalize_location, split_location


@dataclass
class DeciSearchConfig:
    controller_steps: int = 5
    worker_tool_turns: int = 3
    max_workers_per_step: int = 2
    max_tool_calls_per_round: int = 4
    max_tool_workers: int = 4
    max_tool_result_chars: int = 12000
    max_final_locations: int = 8
    max_related_context: int = 8
    mode: str = "par"
    client_parse_tools: bool = False


def workflow_message(
    *,
    role: str,
    content: str | None,
    component: str,
    event: str,
    metadata: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls or [],
        "metadata": {
            "workflow": "decisearch",
            "component": component,
            "event": event,
            **(metadata or {}),
        },
    }


def assistant_with_metadata(
    message: dict[str, Any],
    *,
    component: str,
    event: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = dict(message)
    existing = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    message["metadata"] = {
        **existing,
        "workflow": "decisearch",
        "component": component,
        "event": event,
        **(metadata or {}),
    }
    message.setdefault("tool_calls", [])
    return message


def _coerce_workers(decision: dict[str, Any], max_workers: int) -> list[dict[str, Any]]:
    workers = decision.get("workers") or []
    if isinstance(workers, dict):
        workers = [workers]
    if not isinstance(workers, list):
        workers = []

    normalized: list[dict[str, Any]] = []
    for worker in workers[:max_workers]:
        if not isinstance(worker, dict):
            continue
        queries = worker.get("queries") or []
        if isinstance(queries, str):
            queries = [queries]
        normalized.append(
            {
                "mode": str(worker.get("mode") or "fallback"),
                "objective": str(worker.get("objective") or worker.get("query") or ""),
                "queries": [str(q) for q in queries if str(q).strip()][:5],
                "seed_location": str(worker.get("seed_location") or worker.get("candidate") or ""),
            }
        )
    return normalized


def _locations_from_decision(decision: dict[str, Any], key: str) -> list[str]:
    values = decision.get(key) or []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


class DeciSearchAgent:
    def __init__(
        self,
        *,
        chat_client: ChatClient,
        workspace_root: Path,
        config: DeciSearchConfig,
    ):
        self.chat_client = chat_client
        self.workspace_root = workspace_root.resolve()
        self.config = config
        self.tool_executor = ToolExecutor(
            self.workspace_root,
            max_tool_result_chars=config.max_tool_result_chars,
        )

    def _system_content(self) -> str:
        suffix = tool_prompt_block() if self.config.client_parse_tools else ""
        return DECISEARCH_SYSTEM_PROMPT + suffix

    def _call_json_component(
        self,
        *,
        component: str,
        event: str,
        prompt: str,
        use_tools: bool = False,
        parallel_tool_calls: bool = False,
    ) -> tuple[dict[str, Any], Any]:
        messages = [
            {"role": "system", "content": self._system_content()},
            {"role": "user", "content": prompt},
        ]
        assistant = self.chat_client.create(
            messages,
            use_tools=use_tools,
            parallel_tool_calls=parallel_tool_calls,
        )
        assistant = assistant_with_metadata(assistant, component=component, event=event)
        parsed = extract_json_object(assistant.get("content"))
        return assistant, parsed

    def _controller_decision(
        self,
        *,
        issue: str,
        board: EvidenceBoard,
        recent_events: list[dict[str, Any]],
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        prompt = controller_prompt(
            issue=issue,
            board=board.compact_json(),
            recent_events=recent_events,
            step=step,
            remaining_steps=max(self.config.controller_steps - step - 1, 0),
            workspace_root=self.workspace_root,
            max_workers=self.config.max_workers_per_step,
        )
        prompt_message = workflow_message(
            role="user",
            content=prompt,
            component="controller",
            event="state",
            metadata={"controller_step": step},
        )
        assistant, parsed = self._call_json_component(component="controller", event="decision", prompt=prompt)
        if not isinstance(parsed, dict):
            parsed = {"action": "verify" if board.items else "spawn", "rationale": "fallback after unparsable controller output"}
        parsed["action"] = str(parsed.get("action") or "").lower()
        if parsed["action"] not in {"spawn", "expand", "verify", "finalize"}:
            parsed["action"] = "verify" if board.items else "spawn"
        return prompt_message, assistant, parsed

    def _default_initial_workers(self, issue: str) -> list[dict[str, Any]]:
        words: list[str] = []
        for token in issue.replace("`", " ").replace('"', " ").split():
            stripped = token.strip(".,:;()[]{}")
            if len(stripped) >= 4 and any(ch.isalpha() for ch in stripped):
                words.append(stripped)
        queries = list(dict.fromkeys(words[:6]))
        return [
            {
                "mode": "literal",
                "objective": "find exact error strings, public names, and config keys from the issue",
                "queries": queries[:3],
                "seed_location": "",
            },
            {
                "mode": "symbol",
                "objective": "find definitions and usages of likely symbols from the issue",
                "queries": queries[3:6] or queries[:3],
                "seed_location": "",
            },
        ][: self.config.max_workers_per_step]

    def _worker_messages(self, *, issue: str, board: EvidenceBoard, worker: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._system_content()},
            {"role": "user", "content": worker_prompt(issue=issue, board=board.compact_json(), worker=worker, workspace_root=self.workspace_root)},
        ]

    def _run_worker(
        self,
        *,
        issue: str,
        board: EvidenceBoard,
        worker: dict[str, Any],
        worker_id: str,
    ) -> tuple[list[dict[str, Any]], list[EvidenceItem], dict[str, Any]]:
        messages = self._worker_messages(issue=issue, board=board, worker=worker)
        seen_tool_call_keys: set[tuple[str, str]] = set()
        flat_messages: list[dict[str, Any]] = [
            workflow_message(
                role="user",
                content=messages[-1]["content"],
                component="worker",
                event="state",
                metadata={"worker_id": worker_id, "worker": worker},
            )
        ]

        final_assistant: dict[str, Any] | None = None
        for turn in range(self.config.worker_tool_turns):
            assistant = self.chat_client.create(
                messages,
                use_tools=True,
                parallel_tool_calls=self.config.mode == "par",
            )
            assistant = assistant_with_metadata(
                assistant,
                component="worker",
                event="tool_or_evidence",
                metadata={"worker_id": worker_id, "turn": turn, "worker": worker},
            )
            messages.append(assistant)
            flat_messages.append(assistant)

            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                final_assistant = assistant
                break

            tool_results = self.tool_executor.execute_many(
                tool_calls,
                seen_tool_call_keys=seen_tool_call_keys,
                max_calls=self.config.max_tool_calls_per_round,
                max_workers=self.config.max_tool_workers,
            )
            messages.extend(tool_results)
            for result in tool_results:
                result = dict(result)
                result["metadata"] = {
                    "workflow": "decisearch",
                    "component": "worker",
                    "event": "tool_result",
                    "worker_id": worker_id,
                    "worker": worker,
                }
                flat_messages.append(result)

        if final_assistant is None:
            messages.append({"role": "user", "content": WORKER_RETURN_PROMPT})
            flat_messages.append(
                workflow_message(
                    role="user",
                    content=WORKER_RETURN_PROMPT,
                    component="worker",
                    event="return_prompt",
                    metadata={"worker_id": worker_id, "worker": worker},
                )
            )
            final_assistant = self.chat_client.create(messages, use_tools=False)
            final_assistant = assistant_with_metadata(
                final_assistant,
                component="worker",
                event="evidence",
                metadata={"worker_id": worker_id, "worker": worker},
            )
            flat_messages.append(final_assistant)

        parsed = extract_json_object(final_assistant.get("content"))
        evidence_items = self._evidence_from_worker(parsed, worker=worker, worker_id=worker_id)
        event = {
            "component": "worker",
            "worker_id": worker_id,
            "mode": worker.get("mode"),
            "objective": worker.get("objective"),
            "new_evidence": len(evidence_items),
            "notes": clean_text(parsed.get("notes") if isinstance(parsed, dict) else "", 300),
        }
        return flat_messages, evidence_items, event

    def _evidence_from_worker(self, parsed: Any, *, worker: dict[str, Any], worker_id: str) -> list[EvidenceItem]:
        if not isinstance(parsed, dict):
            return []
        raw_items = parsed.get("evidence") or []
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            raw_items = []

        items: list[EvidenceItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            location = raw.get("location") or ""
            file_path = raw.get("file") or ""
            function = raw.get("function") or ""
            if location:
                location = normalize_location(str(location), self.workspace_root)
                file_path, parsed_func = split_location(location)
                function = function or parsed_func
            elif file_path:
                file_path = normalize_location(str(file_path), self.workspace_root)
                location = f"{file_path}:{function}" if function else file_path
            else:
                continue

            try:
                confidence = float(raw.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            items.append(
                EvidenceItem(
                    location=location,
                    file=file_path,
                    function=str(function or ""),
                    mode=str(worker.get("mode") or raw.get("mode") or ""),
                    evidence=clean_text(raw.get("evidence") or raw.get("reason") or "", 600),
                    source_tool=str(raw.get("source_tool") or ""),
                    support="new",
                    confidence=max(0.0, min(confidence, 1.0)),
                    worker_id=worker_id,
                )
            )

        related = parsed.get("related_context") or []
        if isinstance(related, str):
            related = [related]
        if isinstance(related, list):
            for loc in related[:8]:
                location = normalize_location(str(loc), self.workspace_root)
                file_path, function = split_location(location)
                if file_path:
                    items.append(
                        EvidenceItem(
                            location=location,
                            file=file_path,
                            function=function,
                            mode=str(worker.get("mode") or ""),
                            evidence="related context returned by worker",
                            source_tool="worker",
                            support="weak",
                            confidence=0.35,
                            worker_id=worker_id,
                        )
                    )
        return items

    def _run_verifier(
        self,
        *,
        issue: str,
        board: EvidenceBoard,
        candidates: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        prompt = verifier_prompt(
            issue=issue,
            board=board.compact_json(limit=16),
            candidates=candidates,
            workspace_root=self.workspace_root,
        )
        prompt_message = workflow_message(
            role="user",
            content=prompt,
            component="verifier",
            event="state",
            metadata={"candidates": candidates},
        )
        assistant, parsed = self._call_json_component(component="verifier", event="judgment", prompt=prompt)
        if not isinstance(parsed, dict):
            parsed = {"judgments": [], "needs_more_evidence": candidates}
        for judgment in parsed.get("judgments") or []:
            if not isinstance(judgment, dict):
                continue
            location = str(judgment.get("location") or "")
            support = str(judgment.get("support") or "weak").lower()
            if support not in {"verified", "weak", "rejected", "duplicate"}:
                support = "weak"
            try:
                confidence = float(judgment.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            board.update_support(location, support, confidence=max(0.0, min(confidence, 1.0)), reason=str(judgment.get("reason") or ""))
        return prompt_message, assistant, parsed

    def _finalize_with_model(
        self,
        *,
        issue: str,
        board: EvidenceBoard,
        recent_events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = final_prompt(
            issue=issue,
            board=board.compact_json(limit=20),
            recent_events=recent_events,
            workspace_root=self.workspace_root,
        )
        prompt_message = workflow_message(role="user", content=prompt, component="finalizer", event="state")
        assistant, _ = self._call_json_component(component="finalizer", event="final", prompt=prompt)
        content = assistant.get("content") or ""
        if "<locations_to_modify>" not in content or "</locations_to_modify>" not in content:
            assistant["content"] = board.to_xml(
                max_locations=self.config.max_final_locations,
                max_context=self.config.max_related_context,
            )
            assistant["metadata"]["fallback_final"] = True
        return prompt_message, assistant

    def _finalize_from_decision(self, decision: dict[str, Any], board: EvidenceBoard) -> tuple[dict[str, Any], dict[str, Any]]:
        locations = [normalize_location(loc, self.workspace_root) for loc in _locations_from_decision(decision, "locations")]
        related = [normalize_location(loc, self.workspace_root) for loc in _locations_from_decision(decision, "related_context")]
        prompt = (
            "# Component Prompt: Finalizer\n"
            "The controller selected FINALIZE. Convert the controller decision into the required XML output.\n"
            "Do not call tools. Do not include prose outside XML.\n\n"
            "# Controller Decision\n"
            f"{json.dumps(decision, ensure_ascii=False, indent=2)}\n\n"
            "# Evidence Board\n"
            f"{board.compact_json(limit=20)}"
        )
        prompt_message = workflow_message(role="user", content=prompt, component="finalizer", event="state", metadata={"source": "controller_finalize"})
        if not locations:
            content = board.to_xml(self.config.max_final_locations, self.config.max_related_context)
            metadata = {"fallback_final": True, "source": "controller_finalize_empty"}
        else:
            content = (
                "<locations_to_modify>\n"
                + "\n".join(locations[: self.config.max_final_locations])
                + "\n</locations_to_modify>\n\n<related_context>\n"
                + "\n".join(related[: self.config.max_related_context])
                + "\n</related_context>"
            )
            metadata = {"source": "controller_finalize"}
        return prompt_message, workflow_message(role="assistant", content=content, component="finalizer", event="final", metadata=metadata)

    def run(self, *, issue: str, instance_id: str) -> list[dict[str, Any]]:
        board = EvidenceBoard(workspace_root=self.workspace_root)
        recent_events: list[dict[str, Any]] = []
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_content()},
        ]

        final_message: dict[str, Any] | None = None
        for step in range(self.config.controller_steps):
            controller_prompt_msg, controller_msg, decision = self._controller_decision(
                issue=issue,
                board=board,
                recent_events=recent_events,
                step=step,
            )
            controller_prompt_msg["metadata"].update({"instance_id": instance_id, "workspace_root": str(self.workspace_root)})
            controller_msg["metadata"]["controller_step"] = step
            controller_msg["metadata"]["decision"] = decision
            transcript.append(controller_prompt_msg)
            transcript.append(controller_msg)
            recent_events.append(
                {
                    "component": "controller",
                    "step": step,
                    "action": decision.get("action"),
                    "rationale": clean_text(decision.get("rationale"), 300),
                }
            )

            action = decision.get("action")
            if action == "finalize":
                final_prompt_msg, final_message = self._finalize_from_decision(decision, board)
                transcript.append(final_prompt_msg)
                transcript.append(final_message)
                break

            if action in {"spawn", "expand"}:
                workers = _coerce_workers(decision, self.config.max_workers_per_step) or self._default_initial_workers(issue)
                for worker_idx, worker in enumerate(workers):
                    worker_id = f"c{step}_w{worker_idx}"
                    worker_messages, evidence_items, event = self._run_worker(
                        issue=issue,
                        board=board,
                        worker=worker,
                        worker_id=worker_id,
                    )
                    event["new_board_items"] = board.add_many(evidence_items)
                    transcript.extend(worker_messages)
                    recent_events.append(event)
                continue

            candidates = _locations_from_decision(decision, "candidates") or board.candidate_locations(limit=12)
            if candidates:
                verifier_prompt_msg, verifier_msg, verifier_result = self._run_verifier(issue=issue, board=board, candidates=candidates)
                transcript.append(verifier_prompt_msg)
                transcript.append(verifier_msg)
                recent_events.append(
                    {
                        "component": "verifier",
                        "candidates": len(candidates),
                        "judgments": len(verifier_result.get("judgments") or []),
                    }
                )
            elif step == 0:
                for worker_idx, worker in enumerate(self._default_initial_workers(issue)):
                    worker_id = f"c{step}_w{worker_idx}"
                    worker_messages, evidence_items, event = self._run_worker(
                        issue=issue,
                        board=board,
                        worker=worker,
                        worker_id=worker_id,
                    )
                    event["new_board_items"] = board.add_many(evidence_items)
                    transcript.extend(worker_messages)
                    recent_events.append(event)

        if final_message is None:
            final_prompt_msg, final_message = self._finalize_with_model(issue=issue, board=board, recent_events=recent_events)
            transcript.append(final_prompt_msg)
            transcript.append(final_message)

        return transcript

    def close(self) -> None:
        self.chat_client.close()
