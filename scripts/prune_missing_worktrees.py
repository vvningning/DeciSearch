#!/usr/bin/env python3
"""Prune jsonl rows whose worktree directory does not exist.

Some upstream PRs reference commits that no longer exist in the repo's git
history (force-push, rebase, deleted branch), so `git worktree add` fails for
those instances. Running run.py on them would crash inside the tool layer
with "Path not found". We strip them out and write the pruned jsonl back
in-place, plus a sidecar .pruned.jsonl listing what was dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--worktree-base", required=True, type=Path)
    args = parser.parse_args()

    kept: list[dict] = []
    dropped: list[dict] = []
    with args.in_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            wt = args.worktree_base / row["instance_id"]
            if wt.is_dir():
                kept.append(row)
            else:
                dropped.append({
                    "instance_id": row["instance_id"],
                    "repo_path": row.get("repo_path"),
                    "base_commit": row.get("base_commit"),
                })

    print(f"{args.in_path.name}: kept {len(kept)}, dropped {len(dropped)}")
    if dropped:
        # show top-10 dropped reasons by repo
        from collections import Counter
        by_repo = Counter(d["repo_path"] for d in dropped)
        for repo, n in by_repo.most_common(10):
            print(f"  - {repo}: {n}")

    with args.in_path.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sidecar = args.in_path.with_suffix(args.in_path.suffix + ".pruned.jsonl")
    with sidecar.open("w") as f:
        for row in dropped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  → wrote {sidecar.name} with {len(dropped)} dropped ids")


if __name__ == "__main__":
    main()
