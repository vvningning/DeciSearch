#!/usr/bin/env python3
"""Clone GitHub repositories referenced by a SWE-bench-style JSONL file."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def read_repos(jsonl_path: Path) -> dict[str, str]:
    repos: dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            repo = row["repo"]
            repos[repo] = row.get("repo_url") or f"https://github.com/{repo}.git"
    return repos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--repos-base", required=True)
    args = parser.parse_args()

    repos = read_repos(Path(args.jsonl))
    repos_base = Path(args.repos_base)
    repos_base.mkdir(parents=True, exist_ok=True)

    print(f"Repositories to clone/check: {len(repos)}")
    for repo, url in sorted(repos.items()):
        dest = repos_base / repo
        if dest.exists():
            print(f"exists: {repo} -> {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"clone:  {repo} -> {dest}")
        subprocess.run(["git", "clone", url, str(dest)], check=True)


if __name__ == "__main__":
    main()
