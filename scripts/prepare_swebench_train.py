#!/usr/bin/env python3
"""Prepare SWE-bench train split for code-localization trajectory generation.

This script downloads SWE-bench metadata, checks overlap with Verified/local eval
sets, filters patch complexity, and writes JSONL files compatible with run.py and
create_worktree.py in this repository.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def changed_files_from_patch(patch: str) -> tuple[list[str], bool, bool]:
    files = []
    added_file = False
    deleted_file = False
    current_file = None

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                current_file = b_path[2:] if b_path.startswith("b/") else b_path
                files.append(current_file)
        elif line.startswith("new file mode"):
            added_file = True
        elif line.startswith("deleted file mode"):
            deleted_file = True
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                deleted_file = True
            elif target.startswith("b/"):
                current_file = target[2:]
                if not files or files[-1] != current_file:
                    files.append(current_file)
        elif line.startswith("--- ") and line[4:].strip() == "/dev/null":
            added_file = True

    return sorted(set(files)), added_file, deleted_file


def patch_line_count(patch: str) -> int:
    count = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---", "diff --git", "index ", "@@")):
            continue
        if line.startswith(("+", "-")):
            count += 1
    return count


def is_test_or_doc_path(path: str) -> bool:
    parts = path.split("/")
    lowered = path.lower()
    return (
        "test" in parts
        or "tests" in parts
        or lowered.startswith("test_")
        or "/test_" in lowered
        or lowered.endswith("_test.py")
        or "/docs/" in lowered
        or lowered.startswith("docs/")
        or "/doc/" in lowered
        or lowered.startswith("doc/")
        or "/examples/" in lowered
        or lowered.startswith("examples/")
    )


def load_instance_ids_from_hf(dataset_name: str, split: str) -> set[str]:
    ds = load_dataset(dataset_name, split=split)
    return set(ds["instance_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--verified-dataset", default="SWE-bench/SWE-bench_Verified")
    parser.add_argument("--verified-split", default="test")
    parser.add_argument("--local-eval", default="data/swe-bench_verified.jsonl")
    parser.add_argument("--out-full", default="data/sft_sources/swebench_train_bugfix_candidates.jsonl")
    parser.add_argument("--out-pilot", default="data/sft_sources/swebench_train_bugfix_candidates_200.jsonl")
    parser.add_argument("--pilot-size", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--max-patch-lines", type=int, default=800)
    parser.add_argument("--allow-test-doc-only", action="store_true")
    args = parser.parse_args()

    local_eval_path = Path(args.local_eval)

    print(f"Loading {args.dataset} split={args.split} ...")
    train_ds = load_dataset(args.dataset, split=args.split)
    print(f"Raw train rows: {len(train_ds)}")

    print(f"Loading official verified ids from {args.verified_dataset} split={args.verified_split} ...")
    official_verified_ids = load_instance_ids_from_hf(args.verified_dataset, args.verified_split)

    local_eval_ids = set()
    if local_eval_path.exists():
        local_eval_ids = {row["instance_id"] for row in read_jsonl(local_eval_path)}

    train_ids = set(train_ds["instance_id"])
    print(f"Official verified ids: {len(official_verified_ids)}")
    print(f"Local eval ids: {len(local_eval_ids)}")
    print(f"Train ∩ official verified: {len(train_ids & official_verified_ids)}")
    print(f"Train ∩ local eval: {len(train_ids & local_eval_ids)}")

    excluded_ids = official_verified_ids | local_eval_ids
    repo_counter = Counter()
    drop_reasons = Counter()
    candidates: list[dict[str, Any]] = []

    for row in train_ds:
        instance_id = row["instance_id"]
        repo = row["repo"]
        patch = row.get("patch") or ""
        problem_statement = row.get("problem_statement") or ""

        if instance_id in excluded_ids:
            drop_reasons["verified_overlap"] += 1
            continue
        if not problem_statement.strip():
            drop_reasons["empty_problem_statement"] += 1
            continue

        files, added_file, deleted_file = changed_files_from_patch(patch)
        py_files = [p for p in files if p.endswith(".py")]
        if added_file:
            drop_reasons["adds_file"] += 1
            continue
        if deleted_file:
            drop_reasons["deletes_file"] += 1
            continue
        if not py_files or len(py_files) != len(files):
            drop_reasons["non_python_or_no_python"] += 1
            continue
        if len(py_files) > args.max_files:
            drop_reasons["too_many_files"] += 1
            continue
        if patch_line_count(patch) > args.max_patch_lines:
            drop_reasons["patch_too_large"] += 1
            continue
        if not args.allow_test_doc_only and all(is_test_or_doc_path(p) for p in py_files):
            drop_reasons["test_doc_only"] += 1
            continue

        out = {
            "instance_id": instance_id,
            "repo": repo,
            "repo_path": repo,
            "repo_url": f"https://github.com/{repo}.git",
            "base_commit": row["base_commit"],
            "problem_statement": problem_statement,
            "patch": patch,
            "localization_gt": {
                "files": py_files,
                "functions": [],
            },
            "data_type": "sft_source",
            "data_source": f"{args.dataset}:{args.split}",
        }
        candidates.append(out)
        repo_counter[repo] += 1

    candidates.sort(key=lambda x: (x["repo"], x["instance_id"]))
    full_count = write_jsonl(Path(args.out_full), candidates)
    pilot_count = write_jsonl(Path(args.out_pilot), candidates[: args.pilot_size])

    print("\nDrop reasons:")
    for key, value in drop_reasons.most_common():
        print(f"  {key}: {value}")

    print("\nCandidate repos:")
    for key, value in repo_counter.most_common():
        print(f"  {key}: {value}")

    print("\nWrote:")
    print(f"  full:  {args.out_full} ({full_count})")
    print(f"  pilot: {args.out_pilot} ({pilot_count})")


if __name__ == "__main__":
    main()
