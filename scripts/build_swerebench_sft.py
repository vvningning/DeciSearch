#!/usr/bin/env python3
"""Build a SWE-rebench-derived SFT source jsonl in this repo's schema.

Pipeline:

1. Load nebius/SWE-rebench (test split, 21,336 Python instances).
2. Drop any (repo, base_commit) that appears in our eval sets (Verified,
   Loc-Bench V1, SWE-bench-Live verified) — eval-leak protection.
3. Drop any (repo, base_commit) already in our existing SFT source v2 — dedup.
4. Filter by patch shape:
   - non-empty patch parseable as unified diff
   - only Python files touched (no test-only edits unless --allow-test-doc-only)
   - <= --max-files (default 5)
   - <= --max-patch-lines (default 1500)
   - no file additions/deletions (keeps post-image worktrees meaningful)
5. Filter by license to a copyleft-safe allowlist (--license-allowlist).
6. Sample down to --target-size with diversity:
   - keep at most --max-per-repo per repo (default 20) so a few mega-repos
     don't dominate
   - shuffle deterministically (--seed) then take target-size
7. Emit rows in our normalized SFT schema, drop helper sidecar lists.

Output:
    data/sft_sources/swerebench_sft.jsonl                — main SFT-ready jsonl
    data/sft_sources/swerebench_sft_full.jsonl           — pre-sample full set (for re-pick)
    data/sft_sources/swerebench_sft_repos.json           — unique repos + counts
    data/sft_sources/swerebench_sft.dropped.json         — stats on what got dropped

`localization_gt.functions` is left empty here; downstream call
`scripts/derive_function_gt_from_patch.py` once worktrees exist.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


# Permissive licenses that allow downstream redistribution + commercial use.
# We deliberately exclude GPL/AGPL/LGPL since teacher distillation might
# create a derivative obligation under copyleft if we open-source the
# resulting model. None / Unknown is also excluded to be safe.
DEFAULT_LICENSE_ALLOWLIST = {
    # MIT family
    "MIT License", "MIT",
    "MIT-CMU License",
    # Apache 2.0
    "Apache License 2.0", "Apache-2.0",
    "Apache License 2.0 or MIT License",
    "MIT/Apache-2.0 Dual License",
    # BSD family — all variants are permissive
    "BSD 3-Clause \"New\" or \"Revised\" License", "BSD-3-Clause",
    "BSD 3-Clause", "BSD 3-Clause License", "3-Clause BSD License",
    "3-Clause BSD license", "New BSD License", "Modified BSD License",
    "BSD 2-Clause \"Simplified\" License", "BSD-2-Clause",
    "BSD 2-clause license", "BSD-2-Clause-Patent",
    "BSD 4-Clause \"Original\" or \"Old\" License",
    "BSD", "BSD License", "BSD-3-Clause-Modification",
    # Mozilla — weak copyleft, file-level only
    "Mozilla Public License 2.0 (MPL 2.0)",
    # ISC, Python, public-domain-like
    "ISC License",
    "Python Software Foundation License",
    "Unlicense", "The Unlicense",
    "Creative Commons Zero v1.0 Universal",
    "Creative Commons Zero v1.0 Universal license (CC0 1.0)",
    # Zope (BSD-style)
    "Zope Public License 2.1",
}


def changed_files_from_patch(patch: str) -> tuple[list[str], bool, bool]:
    """Same as prepare_swebench_train.py — kept here for self-containment."""
    files: list[str] = []
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


def files_and_lines_from_patch(patch: str) -> tuple[list[str], list[str]]:
    """Mirror of build_eval_extensions.files_and_lines_from_patch."""
    files: set[str] = set()
    lines: list[str] = []
    current: str | None = None
    cursor = 0
    in_hunk = False
    hunk_lines: list[int] = []

    def flush():
        nonlocal hunk_lines, current
        if hunk_lines and current is not None:
            lines.append(f"{current}:{min(hunk_lines)}:{max(hunk_lines)}")
        hunk_lines = []

    for line in patch.splitlines():
        if line.startswith("+++ "):
            flush()
            in_hunk = False
            target = line[4:].strip()
            if target == "/dev/null":
                current = None
            elif target.startswith("b/"):
                current = target[2:]
            else:
                current = target
            if current:
                files.add(current)
            continue

        if line.startswith("@@"):
            flush()
            in_hunk = True
            if current is None:
                continue
            m = HUNK_HEADER_RE.match(line)
            if not m:
                continue
            cursor = int(m.group(1))
            continue

        if not in_hunk or current is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            hunk_lines.append(cursor); cursor += 1
        elif line.startswith("-") and not line.startswith("---"):
            hunk_lines.append(cursor)
        elif line.startswith("\\"):
            continue
        else:
            cursor += 1

    flush()
    return sorted(files), lines


def is_test_or_doc_path(path: str) -> bool:
    parts = path.split("/")
    lowered = path.lower()
    return (
        "test" in parts or "tests" in parts
        or lowered.startswith("test_") or "/test_" in lowered
        or lowered.endswith("_test.py")
        or "/docs/" in lowered or lowered.startswith("docs/")
        or "/doc/" in lowered or lowered.startswith("doc/")
        or "/examples/" in lowered or lowered.startswith("examples/")
    )


def load_dedup_pairs(jsonl_path: Path) -> set[tuple[str, str]]:
    """Return {(repo_path, base_commit)} from a jsonl."""
    out: set[tuple[str, str]] = set()
    if not jsonl_path.exists():
        return out
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            repo = d.get("repo_path") or d.get("repo")
            if repo and d.get("base_commit"):
                out.add((repo, d["base_commit"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nebius/SWE-rebench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="data/sft_sources/swerebench_sft.jsonl", type=Path)
    parser.add_argument("--out-full", default="data/sft_sources/swerebench_sft_full.jsonl", type=Path,
                        help="Pre-sample full filtered set, for re-sampling later")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-patch-lines", type=int, default=1500)
    parser.add_argument("--allow-test-doc-only", action="store_true")
    parser.add_argument(
        "--existing-sources",
        nargs="*",
        default=[
            "data/sft_sources/swebench_train_bugfix_candidates_v2.jsonl",
        ],
        help="SFT jsonl(s) to dedupe against (by repo+commit).",
    )
    parser.add_argument(
        "--eval-sources",
        nargs="*",
        default=[
            "data/swe-bench_verified.jsonl",
            "data/swe-bench_locbench_v1.jsonl",
            "data/swe-bench_live_verified.jsonl",
        ],
        help="Eval jsonl(s) to dedupe against (eval-leak protection).",
    )
    parser.add_argument(
        "--license-allowlist",
        nargs="*",
        default=None,
        help="License names to accept. Default is a permissive non-copyleft allowlist.",
    )
    parser.add_argument("--target-size", type=int, default=5000,
                        help="Final sample size after diversity capping. Use 0 to skip sampling.")
    parser.add_argument("--max-per-repo", type=int, default=20,
                        help="Diversity cap: at most this many instances per repo (pre-sample).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    allowlist = set(args.license_allowlist) if args.license_allowlist else DEFAULT_LICENSE_ALLOWLIST

    print(f"loading {args.dataset} split={args.split} ...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"raw rows: {len(ds)}")

    eval_pairs: set[tuple[str, str]] = set()
    for p in args.eval_sources:
        added = load_dedup_pairs(Path(p))
        print(f"eval dedup base {p}: {len(added)}")
        eval_pairs |= added

    existing_pairs: set[tuple[str, str]] = set()
    for p in args.existing_sources:
        added = load_dedup_pairs(Path(p))
        print(f"existing-source dedup base {p}: {len(added)}")
        existing_pairs |= added

    drop_reasons: Counter[str] = Counter()
    kept: list[dict] = []

    for row in ds:
        repo = row["repo"]
        commit = row["base_commit"]
        patch = row.get("patch") or ""
        ps = row.get("problem_statement") or ""

        if (repo, commit) in eval_pairs:
            drop_reasons["eval_leak"] += 1
            continue
        if (repo, commit) in existing_pairs:
            drop_reasons["existing_source_dup"] += 1
            continue

        lic = (row.get("license_name") or "").strip()
        if lic not in allowlist:
            drop_reasons[f"license:{lic or 'unknown'}"] += 1
            continue

        if not ps.strip():
            drop_reasons["empty_problem_statement"] += 1
            continue
        if not patch.strip():
            drop_reasons["empty_patch"] += 1
            continue

        files_in_diff, added_file, deleted_file = changed_files_from_patch(patch)
        py_files = [p for p in files_in_diff if p.endswith(".py")]
        if added_file:
            drop_reasons["adds_file"] += 1; continue
        if deleted_file:
            drop_reasons["deletes_file"] += 1; continue
        if not py_files or len(py_files) != len(files_in_diff):
            drop_reasons["non_python_or_mixed"] += 1; continue
        if len(py_files) > args.max_files:
            drop_reasons["too_many_files"] += 1; continue
        if patch_line_count(patch) > args.max_patch_lines:
            drop_reasons["patch_too_large"] += 1; continue
        if not args.allow_test_doc_only and all(is_test_or_doc_path(p) for p in py_files):
            drop_reasons["test_doc_only"] += 1; continue

        files, lines = files_and_lines_from_patch(patch)

        kept.append({
            "instance_id": row["instance_id"],
            "repo": repo,
            "repo_path": repo,
            "repo_url": f"https://github.com/{repo}.git",
            "base_commit": commit,
            "problem_statement": ps,
            "patch": patch,
            "localization_gt": {
                "files": files,
                "functions": [],  # filled later by derive_function_gt_from_patch.py
                "lines": lines,
            },
            "data_type": "sft_source",
            "data_source": f"{args.dataset}:{args.split}",
            "license_name": lic,
            "created_at": str(row.get("created_at") or ""),
        })

    print(f"\nafter filtering: {len(kept)} kept")
    print("drop reasons (top 20):")
    for k, v in drop_reasons.most_common(20):
        print(f"  {v:6d}  {k}")

    # Write pre-sample full set (for re-pick later w/o re-filtering)
    args.out_full.parent.mkdir(parents=True, exist_ok=True)
    with args.out_full.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote full filtered set: {args.out_full} ({len(kept)} rows)")

    # Diversity cap + sample
    if args.target_size and args.target_size < len(kept):
        rng = random.Random(args.seed)
        rng.shuffle(kept)
        per_repo: dict[str, int] = defaultdict(int)
        capped: list[dict] = []
        for row in kept:
            if per_repo[row["repo"]] >= args.max_per_repo:
                continue
            per_repo[row["repo"]] += 1
            capped.append(row)
            if len(capped) >= args.target_size:
                break
        sampled = capped
        # If we didn't fill target after per-repo cap, top up from leftovers
        if len(sampled) < args.target_size:
            sampled_ids = {r["instance_id"] for r in sampled}
            for row in kept:
                if row["instance_id"] in sampled_ids:
                    continue
                sampled.append(row)
                if len(sampled) >= args.target_size:
                    break
    else:
        sampled = kept

    print(f"\nfinal sample: {len(sampled)}")

    with args.out.open("w") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sampled_repos = Counter(r["repo"] for r in sampled)
    print(f"unique repos in final sample: {len(sampled_repos)}")
    print("top 15 repos by sampled count:")
    for r, n in sampled_repos.most_common(15):
        print(f"  {n:4d}  {r}")

    args.out.with_suffix(".repos.json").write_text(
        json.dumps({"unique_repos": len(sampled_repos),
                    "repo_counts": dict(sampled_repos.most_common())}, indent=2)
    )
    args.out.with_suffix(".dropped.json").write_text(
        json.dumps({"total_raw": len(ds), "drop_reasons": dict(drop_reasons.most_common())},
                   indent=2)
    )
    print(f"\nwrote sft jsonl: {args.out}")


if __name__ == "__main__":
    main()
