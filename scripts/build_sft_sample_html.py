#!/usr/bin/env python3
"""Build a static HTML visualization for one CodeSearch SFT training example."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"^<think>\n?(.*?)\n?</think>\n?\n?(.*)$", re.S)


def escape(text: Any) -> str:
    return html.escape(str(text or ""), quote=False)


def clip(text: str, limit: int) -> tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n\n... [truncated]", True


def split_think(value: str) -> tuple[str, str]:
    match = THINK_RE.match(value or "")
    if not match:
        return "", value or ""
    return match.group(1).strip(), match.group(2).strip()


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_tools_prompt(system: str, tools_json: str) -> tuple[str, str]:
    try:
        tools = json.loads(tools_json or "[]")
    except json.JSONDecodeError:
        tools = []

    tool_cards = []
    for tool in tools:
        name = tool.get("name", "tool") if isinstance(tool, dict) else "tool"
        description = tool.get("description", "") if isinstance(tool, dict) else ""
        parameters = tool.get("parameters", {}) if isinstance(tool, dict) else {}
        tool_cards.append(
            f"""
            <div class="tool-schema-card">
              <div class="tool-schema-name">{escape(name)}</div>
              <div class="tool-schema-desc">{escape(description)}</div>
              <pre class="pre schema-pre">{escape(pretty_json(parameters))}</pre>
            </div>
            """
        )

    qwen_prompt = (
        "<|im_start|>system\n"
        + (system or "")
        + "\n\n[Tool definitions are injected by the qwen3_5 template. Rendered below as JSON schema.]\n"
        + (pretty_json(tools) if tools else "[]")
        + "<|im_end|>\n"
        "<|im_start|>user\n"
        "[User issue follows in the next section]"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return "".join(tool_cards), qwen_prompt


def load_sample(path: Path, sample_id: str | None) -> dict[str, Any]:
    fallback = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if sample_id and obj.get("id") == sample_id:
                return obj

            turns = len(obj.get("conversations") or [])
            tool_turns = obj.get("metadata", {}).get("tool_turns", 0)
            if fallback is None and 10 <= turns <= 18 and 3 <= tool_turns <= 8:
                fallback = obj

    if sample_id:
        raise RuntimeError(f"Sample id not found: {sample_id}")
    if fallback is None:
        raise RuntimeError("No suitable sample found.")
    return fallback


def load_gt_files(path: Path, instance_id: str) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("instance_id") == instance_id:
                return sorted(obj.get("localization_gt", {}).get("files") or [])
    return []


def extract_steps(conversations: list[dict[str, str]]) -> list[dict[str, Any]]:
    steps = []
    i = 0
    while i < len(conversations):
        turn = conversations[i]
        if turn.get("from") != "function_call":
            i += 1
            continue

        think, payload = split_think(turn.get("value", ""))
        try:
            calls = json.loads(payload)
        except json.JSONDecodeError:
            calls = [{"parse_error": payload}]
        if not isinstance(calls, list):
            calls = [calls]

        observation = None
        if i + 1 < len(conversations) and conversations[i + 1].get("from") == "observation":
            raw_obs = conversations[i + 1].get("value", "")
            try:
                observation = json.loads(raw_obs)
            except json.JSONDecodeError:
                observation = [{"content": raw_obs}]
            if not isinstance(observation, list):
                observation = [observation]

        steps.append({"index": len(steps) + 1, "think": think, "calls": calls, "observation": observation or []})
        i += 2
    return steps


def select_step_indices(total: int, max_steps: int) -> list[int]:
    if total <= max_steps:
        return list(range(total))

    selected = {0, 1, total - 2, total - 1}
    middle = total // 2
    selected.add(middle)
    cursor = 2
    while len(selected) < max_steps and cursor < total - 2:
        selected.add(cursor)
        cursor += 1
    return sorted(selected)


def extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content or "", re.S)
    return match.group(1).strip() if match else ""


def render_metric(label: str, value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    return f"""
      <div class="metric">
        <span>{escape(label)}</span>
        <strong>{escape(text)}</strong>
      </div>
    """


def render_final(content: str) -> str:
    locations = extract_tag(content, "locations_to_modify")
    related = extract_tag(content, "related_context")
    location_items = []
    for line in locations.splitlines():
        line = line.strip()
        if not line:
            continue
        path, _, symbol = line.partition(":")
        location_items.append(
            f"""
            <li>
              <code>{escape(path)}</code>
              {f'<span class="symbol">{escape(symbol)}</span>' if symbol else ''}
            </li>
            """
        )

    return f"""
      <section class="section">
        <div class="section-title">Final Answer</div>
        <div class="final-grid">
          <div>
            <div class="subhead">Locations To Modify</div>
            <ol class="locations">{''.join(location_items) or '<li>No locations parsed.</li>'}</ol>
          </div>
          <div>
            <div class="subhead">Related Context</div>
            <pre class="pre compact">{escape(related or 'No related_context block parsed.')}</pre>
          </div>
        </div>
        <details>
          <summary>Show full final message</summary>
          <pre class="pre">{escape(content)}</pre>
        </details>
      </section>
    """


def render_step(step: dict[str, Any], obs_limit: int) -> str:
    call_blocks = []
    for call in step["calls"]:
        name = call.get("name") if isinstance(call, dict) else None
        args = call.get("arguments") if isinstance(call, dict) else call
        call_blocks.append(
            f"""
            <div class="tool-call">
              <div class="tool-name">{escape(name or 'tool_call')}</div>
              <pre class="pre compact">{escape(pretty_json(args))}</pre>
            </div>
            """
        )

    obs_blocks = []
    for obs in step["observation"]:
        content = obs.get("content", obs) if isinstance(obs, dict) else obs
        clipped, truncated = clip(str(content), obs_limit)
        obs_blocks.append(
            f"""
            <div class="observation">
              <div class="obs-label">Observation{escape(' - truncated' if truncated else '')}</div>
              <pre class="pre compact">{escape(clipped)}</pre>
            </div>
            """
        )

    think_html = ""
    if step["think"]:
        think_html = f"""
          <div class="think">
            <div class="think-label">&lt;think&gt; reasoning before tool call &lt;/think&gt;</div>
            <pre class="pre compact">{escape(step['think'])}</pre>
          </div>
        """

    return f"""
      <details class="step" open>
        <summary>
          <span class="step-number">Step {step['index']}</span>
          <span class="step-tools">{escape(', '.join(call.get('name', 'tool') for call in step['calls'] if isinstance(call, dict)))}</span>
        </summary>
        {think_html}
        <div class="tool-grid">{''.join(call_blocks)}</div>
        <div class="obs-list">{''.join(obs_blocks)}</div>
      </details>
    """


def build_html(sample: dict[str, Any], max_steps: int, obs_limit: int, gt_files: list[str]) -> str:
    conversations = sample["conversations"]
    metadata = sample.get("metadata") or {}
    system = sample.get("system") or ""
    tool_schema_cards, qwen_prompt_preview = render_tools_prompt(system, sample.get("tools") or "[]")
    issue = conversations[0]["value"]
    final_think, final_content = split_think(conversations[-1]["value"])
    steps = extract_steps(conversations)
    indices = select_step_indices(len(steps), max_steps)

    rendered_steps = []
    previous = -1
    for idx in indices:
        if previous != -1 and idx > previous + 1:
            rendered_steps.append(f'<div class="omitted">{idx - previous - 1} intermediate step(s) omitted</div>')
        rendered_steps.append(render_step(steps[idx], obs_limit))
        previous = idx

    if not gt_files:
        maybe_gt_files = metadata.get("gt_files") or []
        gt_files = maybe_gt_files if isinstance(maybe_gt_files, list) else []
    pred_files = metadata.get("pred_files") or []
    metrics = "".join(
        [
            render_metric("Precision", metadata.get("precision")),
            render_metric("Recall", metadata.get("recall")),
            render_metric("F1", metadata.get("f1")),
            render_metric("Tool turns", metadata.get("tool_turns")),
            render_metric("Conversation turns", len(conversations)),
            render_metric("Filter", metadata.get("quality_filter")),
        ]
    )

    gt_list = "".join(f"<li><code>{escape(path)}</code></li>" for path in gt_files)
    pred_list = "".join(f"<li><code>{escape(path)}</code></li>" for path in pred_files)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CodeSearch SFT Sample Visualization</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1c2430;
      --muted: #667085;
      --line: #d8dee8;
      --blue: #2457c5;
      --green: #197a55;
      --amber: #8a5a00;
      --code: #101828;
      --code-bg: #f0f3f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .topbar {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 26px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 13px;
    }}
    .dataset-pill {{
      white-space: nowrap;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 18px;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0;
      padding: 16px;
    }}
    .section-title {{
      font-weight: 700;
      font-size: 17px;
      margin-bottom: 12px;
    }}
    .subhead {{
      color: var(--muted);
      font-weight: 700;
      margin: 8px 0 6px;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: .06em;
    }}
    .two-col, .final-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }}
    .tools-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .pre {{
      margin: 0;
      padding: 12px;
      overflow: auto;
      color: var(--code);
      background: var(--code-bg);
      border: 1px solid #e4e8ef;
      border-radius: 6px;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .compact {{
      max-height: 360px;
    }}
    code {{
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #243b66;
      overflow-wrap: anywhere;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      margin: 10px 0;
      padding: 0;
    }}
    summary {{
      cursor: pointer;
      list-style: none;
      padding: 11px 13px;
      font-weight: 700;
    }}
    summary::-webkit-details-marker {{ display: none; }}
    details[open] > summary {{
      border-bottom: 1px solid var(--line);
    }}
    .step {{
      background: var(--panel);
    }}
    .step-number {{
      display: inline-block;
      color: var(--blue);
      margin-right: 8px;
    }}
    .step-tools {{
      color: var(--muted);
      font-weight: 600;
    }}
    .think {{
      margin: 12px;
      border-color: #d9e3f7;
      background: #f7faff;
      border: 1px solid #d9e3f7;
      border-radius: 8px;
      padding: 12px;
    }}
    .think-label {{
      color: #4b3aa8;
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .tool-schema-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 12px;
    }}
    .tool-schema-name {{
      display: inline-block;
      margin-bottom: 6px;
      padding: 3px 8px;
      color: #fff;
      background: #344054;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .tool-schema-desc {{
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .schema-pre {{
      max-height: 220px;
    }}
    .tool-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      padding: 12px;
    }}
    .tool-call {{
      border-left: 3px solid var(--blue);
      padding-left: 10px;
    }}
    .tool-name {{
      display: inline-block;
      margin-bottom: 6px;
      padding: 3px 8px;
      color: #fff;
      background: var(--blue);
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .obs-list {{
      padding: 0 12px 12px;
    }}
    .observation {{
      border-left: 3px solid var(--green);
      margin-top: 10px;
      padding-left: 10px;
    }}
    .obs-label {{
      color: var(--green);
      font-weight: 700;
      margin-bottom: 6px;
      font-size: 12px;
    }}
    .omitted {{
      text-align: center;
      color: var(--amber);
      border: 1px dashed #d7b56d;
      background: #fff8e8;
      border-radius: 8px;
      padding: 9px 12px;
      margin: 10px 0;
      font-weight: 700;
    }}
    .locations {{
      margin: 0;
      padding-left: 22px;
    }}
    .locations li {{
      margin: 8px 0;
    }}
    .symbol {{
      display: block;
      color: var(--muted);
      margin-top: 2px;
      font-size: 12px;
    }}
    .file-list {{
      margin: 0;
      padding-left: 20px;
    }}
    .final-think {{
      border-left: 3px solid #7a5ac9;
      padding-left: 10px;
    }}
    @media (max-width: 860px) {{
      .topbar, .two-col, .final-grid {{
        display: block;
      }}
      .dataset-pill {{
        margin-top: 10px;
        white-space: normal;
      }}
      .metrics {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div>
        <h1>CodeSearch SFT Sample Visualization</h1>
        <div class="subtitle">
          <strong>{escape(sample.get('id'))}</strong> · instance: <code>{escape(sample.get('instance_id'))}</code>
        </div>
      </div>
      <div class="dataset-pill">Dataset: codesearch_sft_b_balanced_qwen35</div>
    </div>

    <div class="metrics">{metrics}</div>

    <section class="section">
      <div class="section-title">Training Prompt Header</div>
      <div class="subhead">System Prompt</div>
      <pre class="pre compact">{escape(system)}</pre>
      <div class="subhead">Tool Definitions</div>
      <div class="tools-grid">{tool_schema_cards}</div>
      <details>
        <summary>Show qwen3_5-style prompt header preview</summary>
        <pre class="pre">{escape(qwen_prompt_preview)}</pre>
      </details>
    </section>

    <section class="section">
      <div class="section-title">Quality Labels</div>
      <div class="two-col">
        <div>
          <div class="subhead">Ground Truth Files</div>
          <ul class="file-list">{gt_list}</ul>
        </div>
        <div>
          <div class="subhead">Predicted Files</div>
          <ul class="file-list">{pred_list}</ul>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">User Issue</div>
      <pre class="pre">{escape(issue)}</pre>
    </section>

    <section class="section">
      <div class="section-title">Tool-Use Trajectory</div>
      <div class="subtitle">Showing {len(indices)} of {len(steps)} tool steps. Observations are clipped for readability.</div>
      {''.join(rendered_steps)}
    </section>

    <section class="section">
      <div class="section-title">Final Reasoning</div>
      <div class="final-think">
        <pre class="pre compact">{escape(final_think or 'No final reasoning block parsed.')}</pre>
      </div>
    </section>

    {render_final(final_content)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/sft/codesearch_sft_sample.jsonl",
    )
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--out",
        default="data/sft/codesearch_sft_sample_visualization.html",
    )
    parser.add_argument(
        "--gt",
        default="data/sft_sources/swebench_train_bugfix_candidates.jsonl",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--observation-chars", type=int, default=2400)
    args = parser.parse_args()

    sample = load_sample(Path(args.data), args.sample_id)
    gt_files = load_gt_files(Path(args.gt), sample.get("instance_id") or "")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(sample, args.max_steps, args.observation_chars, gt_files), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out),
                "sample_id": sample.get("id"),
                "instance_id": sample.get("instance_id"),
                "conversation_turns": len(sample.get("conversations") or []),
                "tool_turns": sample.get("metadata", {}).get("tool_turns"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
