"""Prompts for the DeciSearch scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DECISEARCH_SYSTEM_PROMPT = """You are DECISEARCH, a read-only repository localization workflow.

Your job is to identify files, functions, methods, or classes that likely need to be changed for a software issue. The workflow is evidence-driven: search branches collect evidence, the verifier judges candidate support, and the final answer reports concise localization results.

General rules:
- Use only the available read-only tools when the current component prompt allows tools.
- Prefer evidence from source code, tests, error strings, public APIs, config keys, and call/dependency relations.
- Avoid repeating the same search. If evidence is weak, request narrower follow-up search instead of guessing.
- Final locations must use absolute paths inside the current worktree.
- Final output must include <locations_to_modify> and <related_context> XML blocks.
"""


def env_block(workspace_root: Path) -> str:
    return f"""# Environment
- Working Directory: {workspace_root}
- Search tools are restricted to this worktree.
- The repository is read-only; do not propose patches or edit files."""


def controller_prompt(
    *,
    issue: str,
    board: str,
    recent_events: list[dict[str, Any]],
    step: int,
    remaining_steps: int,
    workspace_root: Path,
    max_workers: int,
) -> str:
    recent = json.dumps(recent_events[-6:], ensure_ascii=False, indent=2)
    return f"""# Component Prompt: Controller
Decide the next workflow action from the current evidence board. Do not call tools.

Allowed actions:
- spawn: start up to {max_workers} bounded evidence workers.
- expand: follow a relation from existing candidates using one or more workers.
- verify: ask the verifier to judge current candidates.
- finalize: stop search when enough evidence exists or budget is exhausted.

Return only JSON in one of these forms:
{{
  "action": "spawn",
  "workers": [
    {{"mode": "literal|symbol|test|dependency|fallback", "objective": "what to find", "queries": ["query1", "query2"]}}
  ],
  "rationale": "short reason"
}}
{{
  "action": "expand",
  "workers": [
    {{"mode": "dependency", "objective": "relation to follow", "seed_location": "absolute/path.py:Symbol", "queries": ["symbol"]}}
  ],
  "rationale": "short reason"
}}
{{
  "action": "verify",
  "candidates": ["absolute/path.py:Symbol"],
  "rationale": "short reason"
}}
{{
  "action": "finalize",
  "locations": ["absolute/path.py:Symbol"],
  "related_context": ["absolute/path.py:Helper"],
  "rationale": "short reason"
}}

Decision policy:
- At step 0, usually spawn literal/symbol/test workers unless the issue directly names exact files and functions.
- Prefer verify before finalizing when there are multiple plausible candidates.
- Do not finalize with an empty location set unless no evidence can be found.
- Use fallback only after literal/symbol/test searches fail.

{env_block(workspace_root)}

# Issue
{issue}

# Evidence Board
{board}

# Recent Workflow Events
{recent}

# Budget
- controller_step: {step}
- remaining_controller_steps: {remaining_steps}"""


def worker_prompt(
    *,
    issue: str,
    board: str,
    worker: dict[str, Any],
    workspace_root: Path,
) -> str:
    return f"""# Component Prompt: Evidence Worker
Run a bounded search branch using grep, read_file, and glob. You may call tools.

Worker spec:
{json.dumps(worker, ensure_ascii=False, indent=2)}

Search behavior:
- Start with the provided queries/objective, then narrow using tool observations.
- Prefer exact error strings, identifiers, function/class names, tests, imports, callers, callees, and nearby symbols.
- Use grep in files_with_matches mode to discover candidates, then read_file or grep content for evidence.
- Stop when you can return useful evidence; do not keep searching duplicates.

When done, return only JSON:
{{
  "evidence": [
    {{
      "location": "absolute/path.py:Class.method_or_function",
      "file": "absolute/path.py",
      "function": "Class.method_or_function",
      "evidence": "short snippet or reason",
      "source_tool": "grep|read_file|glob",
      "confidence": 0.0
    }}
  ],
  "related_context": ["absolute/path.py:Helper"],
  "notes": "short branch summary"
}}

{env_block(workspace_root)}

# Issue
{issue}

# Evidence Board Before This Branch
{board}"""


WORKER_RETURN_PROMPT = """Tool budget for this worker is exhausted. Return the best evidence JSON now. Do not call tools."""


def verifier_prompt(
    *,
    issue: str,
    board: str,
    candidates: list[str],
    workspace_root: Path,
) -> str:
    return f"""# Component Prompt: Verifier
Judge whether candidate locations are supported by the evidence board. Do not call tools.

Return only JSON:
{{
  "judgments": [
    {{
      "location": "absolute/path.py:Symbol",
      "support": "verified|weak|rejected|duplicate",
      "confidence": 0.0,
      "reason": "short reason grounded in evidence"
    }}
  ],
  "needs_more_evidence": ["absolute/path.py:Symbol"]
}}

Support labels:
- verified: evidence directly links the issue to this file/function.
- weak: plausible but still missing direct support.
- rejected: unsupported or likely distractor.
- duplicate: same code location already represented by a better candidate.

{env_block(workspace_root)}

# Issue
{issue}

# Candidates To Judge
{json.dumps(candidates, ensure_ascii=False, indent=2)}

# Evidence Board
{board}"""


def final_prompt(
    *,
    issue: str,
    board: str,
    recent_events: list[dict[str, Any]],
    workspace_root: Path,
) -> str:
    recent = json.dumps(recent_events[-8:], ensure_ascii=False, indent=2)
    return f"""# Component Prompt: Final Localizer
Tool invocations are disabled. Produce the final localization answer.

Output exactly two XML blocks:
<locations_to_modify>
/absolute/path/to/repo/src/file.py:ClassName.method_name
/absolute/path/to/repo/src/other.py:function_name
</locations_to_modify>

<related_context>
/absolute/path/to/repo/tests/test_file.py:test_case
/absolute/path/to/repo/src/helper.py:helper_function
</related_context>

Rules:
- Include only locations supported by the evidence board.
- Prefer function/method/class locations when available; file-only locations are allowed when function evidence is unavailable.
- Use absolute paths under {workspace_root}.
- Order locations by relevance.
- Do not include prose outside the XML blocks.

# Issue
{issue}

# Evidence Board
{board}

# Recent Workflow Events
{recent}"""
