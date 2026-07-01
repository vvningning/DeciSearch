#!/usr/bin/env python3
"""Compute file/function-level quality stats for cleaned CodeSearch SFT traces."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import jsonlines


LOCATIONS_RE = re.compile(r"<locations_to_modify>(.*?)</locations_to_modify>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
WORKTREE_MARKER = "/worktree_sft_train_full/"


THRESHOLD_SETS = {
    "A_strict": {
        "recall_min": 1.0,
        "precision_min": 0.5,
        "f1_min": 0.67,
        "max_pred_files": 8,
    },
    "B_balanced": {
        "recall_min": 0.8,
        "precision_min": 0.4,
        "f1_min": 0.6,
        "max_pred_files": 10,
    },
    "C_loose": {
        "recall_min": 0.5,
        "precision_min": 0.4,
        "f1_min": 0.5,
        "max_pred_files": 10,
    },
    "D_f1_ge_0.7": {
        "recall_min": 0.0,
        "precision_min": 0.0,
        "f1_min": 0.7,
        "max_pred_files": None,
    },
    "E_f1_ge_0.6": {
        "recall_min": 0.0,
        "precision_min": 0.0,
        "f1_min": 0.6,
        "max_pred_files": None,
    },
    "F_recall1_pred_le_5": {
        "recall_min": 1.0,
        "precision_min": 0.0,
        "f1_min": 0.0,
        "max_pred_files": 5,
    },
}

FUNCTION_THRESHOLD_SETS = {
    **THRESHOLD_SETS,
    # Anchored to Qwen3.5-397B SWE-bench Verified function PRF:
    # precision=0.6595, recall=0.6647, f1=0.6621.
    "func_verified_relaxed": {
        "recall_min": 0.5,
        "precision_min": 0.5,
        "f1_min": 0.5,
        "max_pred_files": 10,
    },
    "func_verified_balanced": {
        "recall_min": 0.6,
        "precision_min": 0.5,
        "f1_min": 0.55,
        "max_pred_files": 10,
    },
    "func_verified_near": {
        "recall_min": 0.6,
        "precision_min": 0.6,
        "f1_min": 0.6,
        "max_pred_files": 10,
    },
    "func_verified_strict": {
        "recall_min": 0.67,
        "precision_min": 0.6,
        "f1_min": 0.6,
        "max_pred_files": 10,
    },
}


def load_gt(path: Path) -> dict[str, dict[str, set[str]]]:
    rows: dict[str, dict[str, set[str]]] = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            localization_gt = row.get("localization_gt", {}) or {}
            rows[row["instance_id"]] = {
                "files": set(localization_gt.get("files") or []),
                "functions": set(localization_gt.get("functions") or []),
            }
    return rows


def final_content(obj: dict[str, Any]) -> str:
    messages = obj.get("messages")
    if isinstance(messages, list) and messages:
        return messages[-1].get("content") or ""
    conversations = obj.get("conversations")
    if isinstance(conversations, list):
        for message in reversed(conversations):
            if message.get("from") == "gpt":
                return message.get("value") or ""
    return ""


def normalize_pred_path(path_part: str) -> str:
    path_part = path_part.strip()
    if WORKTREE_MARKER in path_part:
        rest = path_part.split(WORKTREE_MARKER, 1)[1]
        pieces = rest.split("/", 1)
        return pieces[1] if len(pieces) == 2 else rest
    return path_part.lstrip("/")


def pred_locations_from_content(content: str) -> dict[str, set[str]]:
    content = THINK_RE.sub("", content or "")
    match = LOCATIONS_RE.search(content)
    files: set[str] = set()
    functions: set[str] = set()
    if not match:
        return {"files": files, "functions": functions}

    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line:
            continue
        function_name = ""
        if "::" in line:
            path_part, function_name = line.split("::", 1)
        elif ":" in line:
            path_part, function_name = line.split(":", 1)
        else:
            path_part = line
        file_path = normalize_pred_path(path_part)
        files.add(file_path)
        function_name = function_name.strip()
        if function_name:
            functions.add(f"{file_path}:{function_name}")
    return {"files": files, "functions": functions}


def compute_prf(gt_items: set[str], pred_items: set[str]) -> dict[str, float | int]:
    hits = len(gt_items & pred_items)
    recall = hits / len(gt_items) if gt_items else 0.0
    precision = hits / len(pred_items) if pred_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hits": hits,
        "gt": len(gt_items),
        "pred": len(pred_items),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def passes(metrics: dict[str, float | int], threshold: dict[str, float | int | None]) -> bool:
    threshold_max_pred = threshold["max_pred_files"]
    return (
        metrics["recall"] >= threshold["recall_min"]
        and metrics["precision"] >= threshold["precision_min"]
        and metrics["f1"] >= threshold["f1_min"]
        and (threshold_max_pred is None or metrics["pred"] <= threshold_max_pred)
    )


def bump_hist(hist: dict[str, int], value: float) -> None:
    key = f"{value:.2f}"
    hist[key] = hist.get(key, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="data/sft/clean_deepseek_v4_flash/codesearch_sft_openai_messages.jsonl")
    parser.add_argument("--gt", default="data/sft_sources/swebench_train_bugfix_candidates.jsonl")
    parser.add_argument("--out", default="data/sft/clean_deepseek_v4_flash/file_recall_filter_stats.json")
    parser.add_argument("--max-pred-files", type=int, default=5)
    args = parser.parse_args()

    gt = load_gt(Path(args.gt))
    stats: dict[str, Any] = {
        "total_clean": 0,
        "has_gt": 0,
        "file_gt_empty": 0,
        "function_gt_empty": 0,
        "both_file_and_function_gt": 0,
        "file_pred_empty": 0,
        "function_pred_empty": 0,
        "file_recall_1": 0,
        "file_pred_le_max": 0,
        "file_keep_recall1_pred_le_max": 0,
        "max_pred_files": args.max_pred_files,
    }
    pred_size_hist: dict[str, dict[str, int]] = {"file": {}, "function": {}}
    metric_hist: dict[str, dict[str, dict[str, int]]] = {
        "file": {"recall": {}, "precision": {}, "f1": {}},
        "function": {"recall": {}, "precision": {}, "f1": {}},
    }
    threshold_counts = {
        "file": {key: 0 for key in THRESHOLD_SETS},
        "function": {key: 0 for key in FUNCTION_THRESHOLD_SETS},
        "file_and_function": {key: 0 for key in THRESHOLD_SETS},
        "file_function_strict": {
            file_key: {function_key: 0 for function_key in FUNCTION_THRESHOLD_SETS}
            for file_key in THRESHOLD_SETS
        },
        "file_function_fallback": {
            file_key: {function_key: 0 for function_key in FUNCTION_THRESHOLD_SETS}
            for file_key in THRESHOLD_SETS
        },
    }
    examples_keep = []
    examples_drop = []

    with Path(args.sft).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            stats["total_clean"] += 1
            instance_id = obj["instance_id"]
            gt_entry = gt.get(instance_id)
            if gt_entry is None:
                continue

            stats["has_gt"] += 1
            gt_files = gt_entry["files"]
            gt_functions = gt_entry["functions"]
            if not gt_files:
                stats["file_gt_empty"] += 1
            if not gt_functions:
                stats["function_gt_empty"] += 1
            if gt_files and gt_functions:
                stats["both_file_and_function_gt"] += 1

            pred = pred_locations_from_content(final_content(obj))
            pred_files = pred["files"]
            pred_functions = pred["functions"]
            if not pred_files:
                stats["file_pred_empty"] += 1
            if not pred_functions:
                stats["function_pred_empty"] += 1

            file_metrics = compute_prf(gt_files, pred_files)
            function_metrics = compute_prf(gt_functions, pred_functions)

            pred_size_hist["file"][str(file_metrics["pred"])] = pred_size_hist["file"].get(str(file_metrics["pred"]), 0) + 1
            pred_size_hist["function"][str(function_metrics["pred"])] = pred_size_hist["function"].get(str(function_metrics["pred"]), 0) + 1
            for level, metrics in (("file", file_metrics), ("function", function_metrics)):
                bump_hist(metric_hist[level]["recall"], float(metrics["recall"]))
                bump_hist(metric_hist[level]["precision"], float(metrics["precision"]))
                bump_hist(metric_hist[level]["f1"], float(metrics["f1"]))

            file_recall_ok = file_metrics["recall"] == 1.0
            file_size_ok = file_metrics["pred"] <= args.max_pred_files
            file_keep = file_recall_ok and file_size_ok
            stats["file_recall_1"] += int(file_recall_ok)
            stats["file_pred_le_max"] += int(file_size_ok)
            stats["file_keep_recall1_pred_le_max"] += int(file_keep)

            file_passes = {
                threshold_key: bool(gt_files) and passes(file_metrics, threshold)
                for threshold_key, threshold in THRESHOLD_SETS.items()
            }
            function_passes = {
                threshold_key: bool(gt_functions) and passes(function_metrics, threshold)
                for threshold_key, threshold in FUNCTION_THRESHOLD_SETS.items()
            }
            for threshold_key, file_ok in file_passes.items():
                threshold_counts["file"][threshold_key] += int(file_ok)
                same_named_function_ok = function_passes.get(threshold_key, False)
                threshold_counts["file_and_function"][threshold_key] += int(file_ok and same_named_function_ok)
            for threshold_key, function_ok in function_passes.items():
                threshold_counts["function"][threshold_key] += int(function_ok)
            for file_key, file_ok in file_passes.items():
                for function_key, function_ok in function_passes.items():
                    threshold_counts["file_function_strict"][file_key][function_key] += int(file_ok and function_ok)
                    fallback_ok = file_ok and (not gt_functions or function_ok)
                    threshold_counts["file_function_fallback"][file_key][function_key] += int(fallback_ok)

            example = {
                "id": obj.get("id"),
                "instance_id": instance_id,
                "gt_files": sorted(gt_files),
                "pred_files": sorted(pred_files),
                "gt_functions": sorted(gt_functions),
                "pred_functions": sorted(pred_functions),
                "file": file_metrics,
                "function": function_metrics,
            }
            if file_keep and len(examples_keep) < 5:
                examples_keep.append(example)
            elif not file_keep and len(examples_drop) < 8:
                examples_drop.append(example)

    def rates(counts: dict[str, int], denominator: int) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "count": count,
                "rate": count / denominator if denominator else 0,
            }
            for key, count in counts.items()
        }

    def matrix_rates(matrix: dict[str, dict[str, int]], denominator: int) -> dict[str, dict[str, dict[str, float | int]]]:
        return {
            file_key: rates(function_counts, denominator)
            for file_key, function_counts in matrix.items()
        }

    summary = {
        "stats": stats,
        "keep_rate_total_clean": stats["file_keep_recall1_pred_le_max"] / stats["total_clean"] if stats["total_clean"] else 0,
        "keep_rate_has_gt": stats["file_keep_recall1_pred_le_max"] / stats["has_gt"] if stats["has_gt"] else 0,
        "pred_size_hist": pred_size_hist,
        "metric_hist": metric_hist,
        "threshold_sets": THRESHOLD_SETS,
        "function_threshold_sets": FUNCTION_THRESHOLD_SETS,
        "threshold_counts": {
            "file": rates(threshold_counts["file"], stats["has_gt"]),
            "function": rates(threshold_counts["function"], stats["both_file_and_function_gt"]),
            "file_and_function": rates(threshold_counts["file_and_function"], stats["both_file_and_function_gt"]),
            "file_function_strict": matrix_rates(
                threshold_counts["file_function_strict"],
                stats["both_file_and_function_gt"],
            ),
            "file_function_fallback": matrix_rates(
                threshold_counts["file_function_fallback"],
                stats["has_gt"],
            ),
        },
        "examples_keep": examples_keep,
        "examples_drop": examples_drop,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
