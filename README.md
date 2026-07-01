# DeciSearch

DeciSearch is the code release for **DeciSearch: A Decision-Trained Dynamic Evidence Workflow for Repository-Level Code Localization**.

The system localizes files and functions for SWE-bench-style issues. It does not generate patches. Instead, it produces a compact set of locations that can be used by a downstream repair agent.

## What Is Included

- `decisearch/`: dynamic evidence workflow runtime.
- `tools/`: read-only repository tools: `grep`, `glob`, and `read_file`.
- `run.py`: batch inference entrypoint for SWE-bench-style JSONL tasks.
- `extract_res.py`: extract XML final answers into JSONL predictions.
- `evaluation/`: file/function localization metrics and tool-efficiency analysis.
- `scripts/`: repository/worktree preparation and SFT trace export utilities.
- `data/samples/smoke_verified_one.jsonl`: one small smoke-test task in the expected schema.

Large local artifacts are intentionally not included: cloned repositories, per-instance worktrees, model outputs, logs, full benchmark JSONL files, SFT traces, and model checkpoints.

## Method Overview

DeciSearch formulates repository-level localization as a bounded decision process over an explicit evidence board. At every controller step, the model selects one action:

- `spawn`: start bounded evidence workers for new search intents.
- `expand`: follow relations from existing candidates.
- `verify`: judge candidate support against the evidence board.
- `finalize`: return the supported locations.

Evidence workers are restricted to read-only tools. The final output uses two XML blocks:

```xml
<locations_to_modify>
/abs/worktree/src/file.py:ClassName.method
</locations_to_modify>

<related_context>
/abs/worktree/tests/test_file.py:test_case
</related_context>
```

The paper trains the workflow with cold-start workflow SFT and exact-state preference optimization. This repository includes the inference workflow and SFT trace export/cleaning utilities. Full preference-pair generation code and trained checkpoints are not included in this initial public package.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install ripgrep as a system binary, or install a Python package that provides `rg`:

```bash
# Ubuntu/Debian
sudo apt-get install ripgrep

# macOS
brew install ripgrep
```

## Data Schema

Each input JSONL row should contain at least:

```json
{
  "instance_id": "owner__repo-12345",
  "repo_path": "owner/repo",
  "repo_url": "https://github.com/owner/repo.git",
  "base_commit": "abcdef...",
  "problem_statement": "Issue text...",
  "patch": "diff --git ...",
  "localization_gt": {
    "files": ["src/file.py"],
    "functions": ["src/file.py:function_name"],
    "lines": ["src/file.py:10:30"]
  }
}
```

`localization_gt` is used only for evaluation, not as model input.

## Quick Start

The smoke sample references `astropy/astropy`. Clone the repository, create the detached worktree, then run DeciSearch against an OpenAI-compatible chat API.

```bash
python scripts/clone_repos_from_jsonl.py \
  --jsonl data/samples/smoke_verified_one.jsonl \
  --repos-base repos

python create_worktree.py \
  --jsonl data/samples/smoke_verified_one.jsonl \
  --repos-base repos \
  --worktree-base worktree \
  --repo-path-mode full \
  --max-workers 1

python run.py \
  --model YOUR_MODEL_NAME \
  --api_key "$OPENAI_API_KEY" \
  --base_url http://127.0.0.1:8001/v1 \
  --data_path data/samples/smoke_verified_one.jsonl \
  --worktree_base "$(pwd)/worktree" \
  --exp_name smoke \
  --tool_turns 2 \
  --worker_tool_turns 2 \
  --max_workflow_workers 2 \
  --mode par \
  --max_workers 1 \
  --temperature 0.0 \
  --top_p 1.0
```

Parse and evaluate the run:

```bash
python extract_res.py \
  --res_path output/smoke_tl2_modepar \
  --worktree_base "$(pwd)/worktree" \
  --output_path output/smoke_predictions.jsonl

python evaluation/eval_prf.py \
  --pred_file output/smoke_predictions.jsonl \
  --gt_path data/samples/smoke_verified_one.jsonl \
  --pred_only
```

## Running Full Evaluations

Prepare a benchmark JSONL in the schema above, then use the same three-stage pipeline:

1. Clone repositories with `scripts/clone_repos_from_jsonl.py`.
2. Create detached worktrees with `create_worktree.py`.
3. Run `run.py`, parse with `extract_res.py`, and score with `evaluation/eval_prf.py`.

For datasets with `owner/repo` layouts, pass `--repo-path-mode full`. For legacy flat layouts, use the default `basename` mode.

## SFT Data Utilities

The scripts under `scripts/` support the workflow-SFT path used in the paper:

```bash
python scripts/prepare_swebench_train.py \
  --out-full data/sft_sources/swebench_train_bugfix_candidates.jsonl

python scripts/derive_function_gt_from_patch.py \
  --in data/sft_sources/swebench_train_bugfix_candidates.jsonl \
  --out data/sft_sources/swebench_train_bugfix_candidates_with_function_gt.jsonl \
  --worktree-base worktree_sft

python scripts/export_qwen35_trace_to_llamafactory.py \
  --run-dir output/YOUR_TEACHER_RUN \
  --llamafactory-dir ../LlamaFactory \
  --dataset-name decisearch_sft \
  --out-file decisearch_sft.jsonl
```

These utilities expect local teacher trajectories generated by `run.py`. They do not download or include teacher outputs.

## Paper Results Snapshot

The paper reports that DeciSearch improves repository-level file/function localization on Loc-Bench and SWE-bench-Live, and improves downstream SWE-bench Verified repair handoff quality with a fixed repair agent. On Loc-Bench, the reported DS-30B file-level result is `F1=57.0` and `F0.5=62.5`; on SWE-bench-Live, it is `F1=43.5` and `F0.5=55.1`.

## Citation

```bibtex
@misc{decisearch2026,
  title = {DeciSearch: A Decision-Trained Dynamic Evidence Workflow for Repository-Level Code Localization},
  author = {Anonymous Author(s)},
  year = {2026},
  note = {Code release}
}
```
