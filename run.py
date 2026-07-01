#!/usr/bin/env python3
"""Run DeciSearch on SWE-style code-localization datasets."""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from decisearch import DeciSearchAgent, DeciSearchConfig
from decisearch.llm import ChatClient, ChatOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Kimi-K2-Instruct")
    parser.add_argument("--exp_name", default="decisearch")
    parser.add_argument("--provider", default="openai", choices=["openai", "zai"], help="Chat API provider")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--max_workers", default=8, type=int, help="Parallel dataset workers")
    parser.add_argument("--max_tool_workers", default=None, type=int, help="Parallel tool workers inside one agent; defaults to max_workers")
    parser.add_argument("--data_path", default="./data/swe-bench_verified.jsonl", type=str)
    parser.add_argument("--agent", default="decisearch", type=str, help="Kept for CLI compatibility; DeciSearch is always used")
    parser.add_argument("--worktree_base", default="", help="Worktree directory (absolute path required)", required=True)
    parser.add_argument("--tool_turns", default=6, type=int, help="Controller steps in DeciSearch")
    parser.add_argument("--worker_tool_turns", default=4, type=int)
    parser.add_argument("--max_workflow_workers", default=2, type=int, help="Max evidence workers per controller step")
    parser.add_argument("--mode", default="par", type=str, choices=["par", "ser"])
    parser.add_argument("--thinking", action="store_true", help="Enable model thinking via chat_template_kwargs")
    parser.add_argument("--reasoning_effort", default="high", choices=["low", "medium", "high", "max", "xhigh"])
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=None, type=int, help="Maximum generated tokens per chat completion")
    parser.add_argument("--top_k", default=None, type=int)
    parser.add_argument("--min_p", default=None, type=float)
    parser.add_argument("--presence_penalty", default=0.0, type=float)
    parser.add_argument("--repetition_penalty", default=None, type=float)
    parser.add_argument("--max_tool_result_chars", default=int(os.environ.get("MAX_TOOL_RESULT_CHARS", "12000")), type=int)
    parser.add_argument("--max_tool_calls_per_round", default=int(os.environ.get("MAX_TOOL_CALLS_PER_ROUND", "4")), type=int)
    parser.add_argument("--client_parse_tools", action="store_true", help="Parse raw <tool_call> blocks client-side")
    parser.add_argument("--disable_client_reuse", action="store_true", help="Create and close a fresh API client for every instance")
    parser.add_argument("--gc_interval", default=100, type=int, help="Run Python garbage collection every N completed instances; 0 disables")
    parser.add_argument("--fd_log_interval", default=250, type=int, help="Log process/system file descriptor counts every N completed instances; 0 disables")
    parser.add_argument("--max_final_locations", default=8, type=int)
    parser.add_argument("--max_related_context", default=8, type=int)
    return parser.parse_args()


def is_completed_output(path: Path) -> bool:
    try:
        messages = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(messages, list) or not messages:
        return False
    final_message = messages[-1]
    return (
        isinstance(final_message, dict)
        and final_message.get("role") == "assistant"
        and not final_message.get("tool_calls")
        and bool((final_message.get("content") or "").strip())
        and "<locations_to_modify>" in (final_message.get("content") or "")
    )


def load_instances(path: str) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                instances.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to parse {path}:{line_no}: {exc}") from exc
    return instances


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_thread_local = threading.local()
_shared_clients: list[Any] = []
_shared_clients_lock = threading.Lock()


def _close_api_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def close_shared_clients() -> None:
    with _shared_clients_lock:
        clients = list(_shared_clients)
        _shared_clients.clear()
    for client in clients:
        try:
            _close_api_client(client)
        except Exception:
            pass


atexit.register(close_shared_clients)


def fd_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        snapshot["process_open_fds"] = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception as exc:
        snapshot["process_open_fds_error"] = str(exc)
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        snapshot["rlimit_nofile"] = [soft, hard]
    except Exception as exc:
        snapshot["rlimit_nofile_error"] = str(exc)
    try:
        snapshot["system_file_nr"] = Path("/proc/sys/fs/file-nr").read_text(encoding="utf-8").strip()
    except Exception as exc:
        snapshot["system_file_nr_error"] = str(exc)
    return snapshot


def _client_cache_key(args: argparse.Namespace) -> tuple[str, str, str, str]:
    return (
        args.provider,
        args.api_key or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ZAI_API_KEY", ""),
        args.base_url or "",
        os.environ.get("OPENAI_HOST_HEADER", ""),
    )


def build_api_client(args: argparse.Namespace) -> Any:
    if args.provider == "zai":
        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise RuntimeError("zai-sdk is required for --provider zai; install it with `pip install zai-sdk`.") from exc
        return ZhipuAiClient(api_key=args.api_key or os.environ.get("ZAI_API_KEY", ""))
    default_headers = {}
    host_header = os.environ.get("OPENAI_HOST_HEADER")
    if host_header:
        default_headers["Host"] = host_header
    http_client = None
    try:
        import httpx

        timeout = float(os.environ.get("OPENAI_HTTP_TIMEOUT", "180"))
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(30.0, timeout)),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    except Exception:
        http_client = None
    client_kwargs = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "default_headers": default_headers or None,
    }
    if http_client is not None:
        client_kwargs["http_client"] = http_client
    return OpenAI(**client_kwargs)


def get_chat_api_client(args: argparse.Namespace) -> tuple[Any, bool]:
    if args.disable_client_reuse:
        return build_api_client(args), True

    key = _client_cache_key(args)
    cached = getattr(_thread_local, "chat_api_client", None)
    cached_key = getattr(_thread_local, "chat_api_client_key", None)
    if cached is not None and cached_key == key:
        return cached, False

    if cached is not None:
        try:
            _close_api_client(cached)
        finally:
            with _shared_clients_lock:
                if cached in _shared_clients:
                    _shared_clients.remove(cached)

    client = build_api_client(args)
    _thread_local.chat_api_client = client
    _thread_local.chat_api_client_key = key
    with _shared_clients_lock:
        _shared_clients.append(client)
    return client, False


def build_agent(args: argparse.Namespace, workspace_root: Path) -> DeciSearchAgent:
    client, close_client = get_chat_api_client(args)
    chat_options = ChatOptions(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        client_parse_tools=args.client_parse_tools,
    )
    config = DeciSearchConfig(
        controller_steps=args.tool_turns,
        worker_tool_turns=args.worker_tool_turns,
        max_workers_per_step=args.max_workflow_workers,
        max_tool_calls_per_round=args.max_tool_calls_per_round,
        max_tool_workers=(args.max_tool_workers if args.max_tool_workers is not None else (args.max_workers if args.mode == "par" else 1)),
        max_tool_result_chars=args.max_tool_result_chars,
        max_final_locations=args.max_final_locations,
        max_related_context=args.max_related_context,
        mode=args.mode,
        client_parse_tools=args.client_parse_tools,
    )
    return DeciSearchAgent(
        chat_client=ChatClient(client, chat_options, close_client=close_client),
        workspace_root=workspace_root,
        config=config,
    )


def process_single_instance(obj: dict[str, Any], args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    start_time = time.time()
    instance_id = obj["instance_id"]
    instance_output_path = output_path / f"llm_messages_{instance_id}.json"

    if instance_output_path.exists() and is_completed_output(instance_output_path):
        return {
            "instance_id": instance_id,
            "elapsed_time": 0.0,
            "status": "success",
            "skipped": True,
        }

    agent = None
    try:
        workspace_root = (Path(args.worktree_base) / instance_id).resolve()
        if not workspace_root.is_dir():
            raise FileNotFoundError(f"worktree not found: {workspace_root}")

        agent = build_agent(args, workspace_root)
        messages = agent.run(issue=obj["problem_statement"], instance_id=instance_id)
        write_json(instance_output_path, messages)

        return {
            "instance_id": instance_id,
            "elapsed_time": time.time() - start_time,
            "status": "success",
        }
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "elapsed_time": time.time() - start_time,
            "status": "failed",
            "error": str(exc),
        }
    finally:
        if agent is not None:
            agent.close()


def process_all_parallel(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    instances = load_instances(args.data_path)
    total = len(instances)
    logger.info("Total instances to process: %s", total)
    logger.info("Initial FD snapshot: %s", fd_snapshot())

    results: dict[str, list[Any]] = {"success": [], "failed": [], "timing": []}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_instance, obj, args, output_path): obj.get("instance_id", f"unknown_{i}")
            for i, obj in enumerate(instances)
        }

        with tqdm(total=total, desc="Processing", ncols=100) as pbar:
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    result = future.result()
                    timing_record = {
                        "instance_id": result["instance_id"],
                        "elapsed_time": result["elapsed_time"],
                        "status": result["status"],
                        "skipped": result.get("skipped", False),
                    }
                    if result.get("error"):
                        timing_record["error"] = result["error"]
                    results["timing"].append(timing_record)
                    if result["status"] == "success":
                        results["success"].append(result["instance_id"])
                    else:
                        results["failed"].append({"id": result["instance_id"], "error": result.get("error", "Unknown error")})
                except Exception as exc:
                    results["failed"].append({"id": instance_id, "error": str(exc)})
                    results["timing"].append(
                        {
                            "instance_id": instance_id,
                            "elapsed_time": 0.0,
                            "status": "failed",
                            "skipped": False,
                            "error": str(exc),
                        }
                    )
                    logger.error("%s failed with exception: %s", instance_id, exc)
                finally:
                    completed = len(results["success"]) + len(results["failed"])
                    if args.gc_interval > 0 and completed % args.gc_interval == 0:
                        gc.collect()
                    if args.fd_log_interval > 0 and completed % args.fd_log_interval == 0:
                        logger.info("FD snapshot at %s/%s: %s", completed, total, fd_snapshot())
                    pbar.set_postfix({"Success": len(results["success"]), "Failed": len(results["failed"])})
                    pbar.update(1)

    logger.info("Completed: %s/%s", len(results["success"]), total)
    logger.info("Failed: %s/%s", len(results["failed"]), total)
    successful_times = [t["elapsed_time"] for t in results["timing"] if t["status"] == "success" and not t.get("skipped")]
    results["timing"].append(
        {
            "instance_id": "summary",
            "elapsed_time_mean": float(np.mean(successful_times)) if successful_times else 0.0,
        }
    )
    return results


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("zai").setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    output_path = Path(f"output/{args.exp_name}_tl{args.tool_turns}_mode{args.mode}")
    output_path.mkdir(parents=True, exist_ok=True)

    if args.agent != "decisearch":
        logger.warning("--agent=%s was requested, but this codebase now runs DeciSearch only.", args.agent)

    try:
        results = process_all_parallel(args, output_path)
    finally:
        close_shared_clients()
        gc.collect()
    logger.info("Final FD snapshot: %s", fd_snapshot())
    write_json(output_path / "timing.json", {"total": len(results["timing"]), "results": results["timing"]})
    write_json(output_path / "success.json", {"total": len(results["success"]), "results": results["success"]})
    write_json(output_path / "error.json", {"total": len(results["failed"]), "errors": results["failed"]})


if __name__ == "__main__":
    main()
