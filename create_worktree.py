#!/usr/bin/env python3
"""
Batch create git worktrees
Read data and create independent worktrees for each instance, determining whether to place in current container based on instance_id hash
"""

import json
import jsonlines
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import shutil
import os

class WorktreeCreator:
    def __init__(self, repos_base="/home/repos", worktree_base="/home/worktree"):
        self.repos_base = Path(repos_base)
        self.worktree_base = Path(worktree_base)
        self.worktree_base.mkdir(parents=True, exist_ok=True)
    
    def create_worktree(self, repo_name, base_commit, instance_id):
        """
        Create worktree for a single instance
        
        Args:
            repo_name: Repository name (last part from repo field)
            base_commit: Commit hash
            instance_id: Instance ID, used as worktree directory name
        
        Returns:
            dict: Result information
        """
        repo_path = self.repos_base / repo_name
        worktree_path = self.worktree_base / instance_id
        
        # Check if source repository exists
        if not repo_path.exists():
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": f"Repository not found: {repo_path}"
            }
        
        # Add to safe.directory (avoid dubious ownership error)
        self._add_safe_directory(repo_path)
        
        # Clean up if worktree already exists
        if worktree_path.exists():
            print(f"Worktree already exists: {worktree_path}")
            return {
                "instance_id": instance_id,
                "status": "success",
                "path": str(worktree_path)
            }
        
        # Create worktree
        try:
            cmd = ["git", "worktree", "add", "--detach", str(worktree_path), base_commit]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "instance_id": instance_id,
                "status": "success",
                "path": str(worktree_path)
            }
        
        except subprocess.CalledProcessError as e:
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": f"Git command failed: {e.stderr}"
            }
        except Exception as e:
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": str(e)
            }
    
    def _add_safe_directory(self, repo_path):
        """Add repository to git safe.directory configuration"""
        try:
            # Check if already added
            result = subprocess.run(
                ["git", "config", "--global", "--get-all", "safe.directory"],
                capture_output=True,
                text=True
            )
            
            repo_path_str = str(repo_path)
            if repo_path_str not in result.stdout:
                # Add to safe.directory
                subprocess.run(
                    ["git", "config", "--global", "--add", "safe.directory", repo_path_str],
                    capture_output=True,
                    check=True
                )
        except Exception:
            # Ignore errors and continue execution
            pass
    
    def _remove_existing_worktree(self, repo_path, worktree_path):
        """Clean up existing worktree"""
        # First try git worktree remove
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_path,
            capture_output=True
        )
        
        # If still exists, force delete directory
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

def load_tasks(jsonl_path, repo_path_mode="basename"):
    """
    Load task list from jsonl file, select corresponding data source

    Args:
        jsonl_path: jsonl with rows that have repo_path (owner/repo).
        repo_path_mode:
            "basename" (default, legacy)  → use last path segment, matching the
                                            flat layout under repos/.
            "full"                        → use the full owner/repo path,
                                            matching scripts/clone_repos_from_jsonl.py
                                            and repos_sft_train/ layout. Use this
                                            for new evaluation datasets that may
                                            include name collisions across owners.

    Returns:
        list of (repo_name, base_commit, instance_id)
    """
    tasks = []
    with jsonlines.open(jsonl_path, 'r') as reader:
        for data in reader:
            repo_full = data['repo_path']  # e.g., "astropy/astropy"
            if repo_path_mode == "full":
                repo_name = repo_full
            else:
                repo_name = repo_full.split('/')[-1]
            base_commit = data['base_commit']
            instance_id = data['instance_id']
            tasks.append((repo_name, base_commit, instance_id))
    return tasks


def process_batch(creator, tasks, max_workers=32):
    """
    Process all tasks concurrently
    
    Args:
        creator: WorktreeCreator instance
        tasks: Task list
        max_workers: Maximum concurrency
    
    Returns:
        list: All results
    """
    results = []
    total = len(tasks)
    
    print(f"Starting to create {total} worktrees, max concurrency: {max_workers}")
    print("=" * 60)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(creator.create_worktree, repo, commit, instance): instance
            for repo, commit, instance in tasks
        }
        
        # Collect results
        completed = 0
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                completed += 1
                status_symbol = "✓" if result["status"] == "success" else "✗"
                
                if result["status"] == "success":
                    print(f"[{completed}/{total}] {status_symbol} {instance_id}")
                else:
                    print(f"[{completed}/{total}] {status_symbol} {instance_id}: {result['error']}")
                
            except Exception as e:
                print(f"[{completed}/{total}] ✗ {instance_id}: Exception: {e}")
                results.append({
                    "instance_id": instance_id,
                    "status": "error",
                    "error": str(e)
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch create git worktree for swe-bench instances"
    )
    parser.add_argument(
        "--jsonl",
        default="./data/swe-bench_verified.jsonl",
        help="JSONL data file path"
    )
    parser.add_argument(
        "--repos-base",
        default="",
        help="Original repository base directory (absolute path required)",
        required=True
    )
    parser.add_argument(
        "--worktree-base",
        default="",
        help="Worktree output directory (absolute path required)",
        required=True
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum concurrency"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit processing quantity (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show worktrees to be created, do not actually execute"
    )
    parser.add_argument(
        "--repo-path-mode",
        choices=["basename", "full"],
        default="basename",
        help='How to locate the source repo under --repos-base. '
             '"basename" (legacy, default) uses repo_path.split("/")[-1]. '
             '"full" uses owner/repo, matching clone_repos_from_jsonl.py.',
    )
    args = parser.parse_args()

    # Load tasks
    print(f"Reading data: {args.jsonl}")
    tasks = load_tasks(args.jsonl, repo_path_mode=args.repo_path_mode)
    
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"Limit processing quantity: {args.limit}")
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Repository directory: {args.repos_base}")
    print(f"Worktree directory: {args.worktree_base}")
    
    if args.dry_run:
        print("\n[Dry Run Mode] Will create the following worktrees:")
        for repo, commit, instance in tasks[:10]:  # Show only first 10
            print(f"  {instance}: {repo}@{commit[:8]}")
        if len(tasks) > 10:
            print(f"  ... {len(tasks) - 10} more")
        return
    
    # Create WorktreeCreator
    creator = WorktreeCreator(
        repos_base=args.repos_base,
        worktree_base=args.worktree_base
    )
    
    # Execute batch processing
    results = process_batch(creator, tasks, max_workers=args.max_workers)
    
    # Statistics results
    print("\n" + "=" * 60)
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    
    print(f"Completed: {success} successful, {failed} failed")
    
    if failed > 0:
        print("\nFailed instances:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['instance_id']}: {r['error']}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())