# DeciSearch Training with LLaMA-Factory

This directory contains LLaMA-Factory dataset descriptors and LoRA config templates for the two training stages described in the paper:

1. **Workflow SFT**: imitate high-quality DeciSearch teacher trajectories at the controller, worker, verifier, and finalizer turns.
2. **Exact-State DPO**: compare alternative next responses under the same rendered workflow state.

The configs target Qwen3-4B by default. The SFT template uses `learning_rate: 1.0e-5`, which is conservative for preserving tool-call format and reasoning structure before DPO. For DS-30B-style runs, change `model_name_or_path`, output directories, and distributed-training launcher settings.

## 1. Prepare LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git ../LLaMA-Factory
cd ../LLaMA-Factory
pip install -e ".[torch,metrics]"
```

LLaMA-Factory expects custom dataset entries in `data/dataset_info.json`. Merge `training/llamafactory_dataset_info.json` into that file, or let the export scripts update it automatically.

## 2. Workflow SFT

Generate teacher trajectories with `run.py`, then export them to ShareGPT tool-call format:

```bash
python scripts/export_qwen35_trace_to_llamafactory.py \
  --run-dir output/YOUR_TEACHER_RUN \
  --llamafactory-dir ../LLaMA-Factory \
  --dataset-name decisearch_sft \
  --out-file decisearch_sft.jsonl \
  --require-related-context
```

Train with:

```bash
cd ../LLaMA-Factory
llamafactory-cli train ../DeciSearch-pub/training/qwen3_lora_sft_decisearch.yaml
```

## 3. Exact-State DPO

First extract fixed workflow prompts from the SFT policy trajectories:

```bash
python scripts/extract_workflow_snapshots.py \
  --run-dir output/YOUR_SFT_POLICY_RUN \
  --out data/dpo/snapshots.jsonl
```

For each selected snapshot, sample alternative responses from the same prompt, resume the deterministic DeciSearch runtime, and score the branch by final localization utility plus component-specific checks. The branch-scoring JSONL should look like:

```json
{
  "id": "snapshot-id",
  "instance_id": "owner__repo-12345",
  "component": "controller",
  "system": "system prompt",
  "prompt": "rendered component prompt",
  "branches": [
    {"response": "{\"action\": \"verify\", ...}", "score": 0.82, "valid": true},
    {"response": "{\"action\": \"finalize\", ...}", "score": 0.21, "valid": true}
  ]
}
```

Then convert branch scores to LLaMA-Factory DPO data:

```bash
python scripts/export_exact_state_dpo_to_llamafactory.py \
  --input data/dpo/exact_state_branches.jsonl \
  --llamafactory-dir ../LLaMA-Factory \
  --dataset-name decisearch_exact_state_dpo \
  --out-file decisearch_exact_state_dpo.jsonl \
  --min-score-gap 0.05
```

Train DPO from the SFT adapter:

```bash
cd ../LLaMA-Factory
llamafactory-cli train ../DeciSearch-pub/training/qwen3_lora_dpo_decisearch.yaml
```

## Notes

- SFT data is ShareGPT format with `human`, `function_call`, `observation`, and `gpt` turns. Tool-call turns are learned by the model.
- DPO data is ShareGPT preference format: a fixed prompt in `conversations`, a better next response in `chosen`, and a worse next response in `rejected`.
- Keep `template`, `enable_thinking`, and inference-time chat template settings consistent between trajectory generation, SFT, DPO, and final evaluation.
- The repository ships formatters and config templates. It does not ship teacher trajectories, branch-rollout results, or model checkpoints.
