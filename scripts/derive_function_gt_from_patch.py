#!/usr/bin/env python3
"""Derive `localization_gt.functions` and `localization_gt.lines` from a unified
diff patch + a checked-out repository worktree at the corresponding base commit.

Output schema matches what `evaluation/eval_prf.py` expects (same as
`data/swe-bench_verified.jsonl`):

    localization_gt = {
        "files":     ["path/to/file.py", ...],
        "functions": ["path/to/file.py:func_name",
                      "path/to/file.py:ClassName.method_name", ...],
        "lines":     ["path/to/file.py:start:end", ...],
    }

Only Python files are handled (uses stdlib `ast`). Non-Python files contribute
to `files` only — `functions` for them are skipped silently so the caller can
keep them or filter them out.

**Granularity policy:** for nested function definitions (e.g. a decorator's
inner `wrapper`), this script reports the *most deeply nested* function that
encloses each changed line. Validation against SWE-bench Verified's
human-curated GT (n=386) gives file-set exact match 100%, function-set
gold⊆pred 90.4%, function-level micro PRF P=0.73 / R=0.89 / F1=0.80. Most of
the remaining ~10% are cases where the curated GT names an outer
`Class.__call__` while the patch's `+`/`-` lines are actually inside a nested
`wrapper`; both choices are defensible and the curated set tends to be
more conservative (fewer reported functions).

Typical use:

    python scripts/derive_function_gt_from_patch.py \
        --in  data/sft_sources/some_dataset.jsonl \
        --out data/sft_sources/some_dataset_with_gt.jsonl \
        --repos-base ./repos

Each input row must have at least `instance_id`, `repo_path` (owner/repo),
`base_commit`, and `patch`. `repos-base` is the directory containing checked-out
repositories at the right commit; `<repos-base>/<owner>/<repo>/...` must resolve
to a working tree where files at `base_commit` can be read. Pass `--worktree-base`
instead if the rows live in per-instance worktrees (e.g.
`worktree/<instance_id>/...`).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    """The post-image span of one unified diff hunk plus the precise line
    numbers that were actually modified (+ in the post-image, − anchored to the
    line that was deleted before).

    `changed_lines` holds post-image line numbers we treat as "the edit". For a
    `+` line that's the new line number; for a `-` line we use the next
    surviving post-image line, since a deletion's "location" in the post-image
    is the gap between the surrounding lines.
    """
    file_path: str
    start_line: int                 # post-image hunk start (1-indexed)
    end_line: int                   # post-image hunk end (inclusive)
    changed_lines: list[int]        # post-image lines actually edited


def parse_patch_hunks(patch: str) -> list[Hunk]:
    """Walk a unified diff and yield one Hunk per `@@` block.

    File path comes from the `+++ b/<path>` header (consistent with
    `prepare_swebench_train.py::changed_files_from_patch`).
    """
    hunks: list[Hunk] = []
    current_file: str | None = None
    cur: Hunk | None = None
    cursor = 0  # post-image line being walked inside the current hunk

    def flush():
        nonlocal cur
        if cur is not None:
            hunks.append(cur)
        cur = None

    for line in patch.splitlines():
        if line.startswith("+++ "):
            flush()
            target = line[4:].strip()
            if target == "/dev/null":
                current_file = None
            elif target.startswith("b/"):
                current_file = target[2:]
            else:
                current_file = target
            continue

        if line.startswith("@@"):
            flush()
            if current_file is None:
                continue
            m = HUNK_HEADER_RE.match(line)
            if not m:
                continue
            start = int(m.group(1))
            length = int(m.group(2)) if m.group(2) else 1
            if length == 0:
                # Pure-deletion hunk at the file boundary; nothing to locate in
                # the post-image. Skip.
                continue
            cur = Hunk(
                file_path=current_file,
                start_line=start,
                end_line=start + length - 1,
                changed_lines=[],
            )
            cursor = start
            continue

        if cur is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            cur.changed_lines.append(cursor)
            cursor += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Deletion: location is the post-image cursor (the next surviving
            # line). Use cursor without advancing it.
            cur.changed_lines.append(cursor)
        else:
            # Context line (or `\ No newline at end of file`).
            if line.startswith("\\"):
                continue
            cursor += 1

    flush()
    return hunks


def _walk_with_parents(tree: ast.AST) -> Iterable[tuple[ast.AST, list[ast.AST]]]:
    """DFS over AST nodes, yielding (node, ancestor_stack)."""
    stack: list[tuple[ast.AST, list[ast.AST]]] = [(tree, [])]
    while stack:
        node, parents = stack.pop()
        yield node, parents
        for child in ast.iter_child_nodes(node):
            stack.append((child, parents + [node]))


def _qualified_name(node: ast.AST, parents: list[ast.AST]) -> str:
    """Build `Class.method` / `Outer.Inner.method` for a function definition.

    Falls back to bare function name when not nested inside a class.
    """
    parts: list[str] = []
    for ancestor in parents:
        if isinstance(ancestor, (ast.ClassDef,)):
            parts.append(ancestor.name)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parts.append(node.name)
    return ".".join(parts)


def functions_for_lines(
    source: str,
    target_lines: Iterable[int],
) -> list[str]:
    """For each post-image line in `target_lines`, find the most deeply nested
    function/method that contains it, and return the deduplicated list of
    qualified names (preserving first-seen order).

    A function "contains" a line iff `lineno <= line <= end_lineno`.

    Lines that fall outside every function (top-level statements, class body
    statements that aren't inside a method) are silently dropped — for those
    the caller still has file-level GT.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Pre-collect (start, end, depth, qualname) for every function.
    funcs: list[tuple[int, int, int, str]] = []
    for node, parents in _walk_with_parents(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        node_end = getattr(node, "end_lineno", None)
        if node_end is None:
            continue
        depth = sum(
            1 for a in parents
            if isinstance(a, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
        qname = _qualified_name(node, parents)
        if qname:
            funcs.append((node.lineno, node_end, depth, qname))

    seen: set[str] = set()
    out: list[str] = []
    for line in target_lines:
        best: tuple[int, str] | None = None  # (depth, qualname)
        for start, end, depth, qname in funcs:
            if start <= line <= end and (best is None or depth > best[0]):
                best = (depth, qname)
        if best is not None and best[1] not in seen:
            seen.add(best[1])
            out.append(best[1])
    return out


def derive_gt(
    patch: str,
    repo_root: Path,
) -> dict:
    """Return a `localization_gt` dict for a single sample.

    - `files`: sorted unique post-image paths touched by the patch.
    - `functions`: `path:qualname` for hunks that cleanly sit inside one
      Python function/method.
    - `lines`: `path:start:end` for every hunk (regardless of language).

    Files that aren't Python or that fail to AST-parse only contribute to
    `files` and `lines`, not to `functions`.
    """
    hunks = parse_patch_hunks(patch)

    files: set[str] = set()
    functions: list[str] = []
    seen_functions: set[str] = set()
    lines: list[str] = []

    file_source_cache: dict[str, str | None] = {}

    for h in hunks:
        files.add(h.file_path)
        if h.changed_lines:
            lines.append(
                f"{h.file_path}:{min(h.changed_lines)}:{max(h.changed_lines)}"
            )

        if not h.file_path.endswith(".py"):
            continue

        if h.file_path not in file_source_cache:
            file_path = repo_root / h.file_path
            try:
                file_source_cache[h.file_path] = file_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except (FileNotFoundError, IsADirectoryError, OSError):
                file_source_cache[h.file_path] = None

        source = file_source_cache[h.file_path]
        if source is None:
            continue

        for qname in functions_for_lines(source, h.changed_lines):
            entry = f"{h.file_path}:{qname}"
            if entry not in seen_functions:
                seen_functions.add(entry)
                functions.append(entry)

    return {
        "files": sorted(files),
        "functions": functions,
        "lines": lines,
    }


def resolve_repo_root(
    row: dict,
    repos_base: Path | None,
    worktree_base: Path | None,
) -> Path | None:
    """Resolve where to read the post-image source from for one sample.

    - If `worktree_base` is given, look at `<worktree_base>/<instance_id>/`.
      This matches how `create_worktree.py` lays out per-instance worktrees.
    - Else fall back to `<repos_base>/<repo_path>/`, assuming the repo was
      already checked out at `base_commit` (caller's responsibility).
    """
    instance_id = row.get("instance_id")
    if worktree_base is not None and instance_id:
        wt = worktree_base / instance_id
        if wt.is_dir():
            return wt

    repo_path = row.get("repo_path")
    if repos_base is not None and repo_path:
        rp = repos_base / repo_path
        if rp.is_dir():
            return rp

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    parser.add_argument(
        "--repos-base",
        type=Path,
        help="Directory containing checked-out repos at base_commit "
             "(layout: <repos-base>/<owner>/<repo>/...).",
    )
    parser.add_argument(
        "--worktree-base",
        type=Path,
        help="Directory containing per-instance worktrees "
             "(layout: <worktree-base>/<instance_id>/...). Takes priority over --repos-base.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing localization_gt instead of skipping rows that already have one.",
    )
    args = parser.parse_args()

    if args.repos_base is None and args.worktree_base is None:
        parser.error("must give --repos-base and/or --worktree-base")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_written = 0
    n_skipped_existing = 0
    n_no_repo = 0
    n_no_functions = 0
    function_counts: list[int] = []

    with args.in_path.open("r", encoding="utf-8") as fin, \
            args.out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1

            row = json.loads(line)
            existing = row.get("localization_gt") or {}
            if (
                not args.overwrite
                and existing.get("functions")
                and existing.get("files")
            ):
                n_skipped_existing += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                continue

            repo_root = resolve_repo_root(row, args.repos_base, args.worktree_base)
            if repo_root is None:
                n_no_repo += 1
                # still emit file-level GT from the patch alone
                from_patch = derive_gt(row.get("patch", ""), Path("/dev/null"))
                row["localization_gt"] = {
                    "files": from_patch["files"],
                    "functions": [],
                    "lines": from_patch["lines"],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                continue

            gt = derive_gt(row.get("patch", ""), repo_root)
            row["localization_gt"] = gt
            if not gt["functions"]:
                n_no_functions += 1
            else:
                function_counts.append(len(gt["functions"]))

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    avg_funcs = (
        sum(function_counts) / len(function_counts) if function_counts else 0.0
    )
    print(f"input rows:                  {n_in}")
    print(f"written:                     {n_written}")
    print(f"already had function GT:     {n_skipped_existing}")
    print(f"no worktree/repo found:      {n_no_repo}")
    print(f"no function GT (parse fail / non-Python): {n_no_functions}")
    print(f"avg functions per sample (when present): {avg_funcs:.2f}")


if __name__ == "__main__":
    main()
