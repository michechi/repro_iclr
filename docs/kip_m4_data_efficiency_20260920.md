# KIP M4 data efficiency — 20 September 2026

## Stopping-rule amendment — 21 September 2026

The amended protocol stops an arm once its full test AUROC is exactly 1.0 and
moves on to the remaining models. It uses
`--stop-on-perfect-test`: larger training fractions for a saturated arm are
explicitly marked `skipped_saturated` with the measured result as evidence.
Skipped fractions have no imputed metrics. All four arms follow this rule.

For future fits, each new minimum-validation-loss checkpoint with validation
AUROC exactly 1.0 triggers a check on the complete fixed 50,000-row test set.
Exact test AUROC 1.0 stops that fit immediately; rounded values such as 0.999999
do not. Test checks that fail the criterion preserve RNG and model mode, and
training continues under the original schedule. The successful diagnostic
predictions become the final test predictions without another test evaluation.
Smoke tests never trigger production stopping or skipping.

This changes the budget and stopping protocol: **test results now inform stopping
and which larger fractions are run**. Report the resulting scores and curves as
test-informed, not as an independent final held-out estimate or an unbiased
sample-complexity threshold. Validation still selects checkpoints and the F1
decision threshold. The original 1%, 10% and 30% Qwen27 results retain their
original manifests and stopping protocol.

The saved 200K Qwen27 epoch 9 checkpoint achieved test AUROC, accuracy, and
F1 exactly 1.0 on 50,000 rows. Its 300K and 400K configurations are marked
`skipped_saturated`; no metrics are imputed for them.

The rest of this document records the original experiment setup. Where the
original full-grid or test-isolation descriptions differ, this amendment applies
to the resumed queue.

Requested run order on the local NVIDIA H200:

1. Qwen/Qwen3.8-27B QLoRA, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
2. Qwen/Qwen2.5-14B QLoRA, revision `97e1e76335b7017d8f67c08a19d103c0504298c9`.
3. meta-llama/Llama-3.1-8B LoRA, revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
4. The same LLaMA 8B base checkpoint, full fine-tuning.

Each arm starts fresh pretrained weights for each fraction, in ascending order:
1%, 10%, 30%, 50%, 75%, 100% = 4,000, 40,000, 120,000, 200,000, 300,000,
400,000 training examples. This is 24 production fits plus small GPU preflights.
These fractions come from `README.md` and `src/experiments/LLM_fraction_experiment.py`.

## Data and protocol

- Task: KIP M4, original ordered sequences, seed 9950.
- CSVs: `data/simulation/kip_ordered_recovered/{X,y}_{train,val,test}_kip_m4.csv`.
- Only training is subsampled. All production fits use the same 50,000 validation
  and 50,000 test examples. Originally test results did not control training;
  see the 21 September amendment for the new stopping rule. F1 thresholds remain
  selected using validation only.
- Stratified sampling uses the repository's sklearn `train_test_split` helper with
  seed 9950. Selected original indices are sorted and saved, with SHA256 digests.
  All arms at a given fraction use the same rows; different fractions are not
  guaranteed to be nested.
- Prompt: `Sequential events: ` followed by the 20 space-separated event letters.
  No instruction template or reasoning generation. Token limit 64; truncation forbidden.
- Final unmasked text hidden state, linear two-class head without bias, pure outcome
  cross entropy in FP32. Qwen vision and vocabulary heads do not participate.
- AdamW, LR 2e-5, weight decay 0.01, effective batch 256, maximum 20 epochs,
  linear schedule with 6% warmup, minimum validation loss, patience 3.
- BF16 computation, FP32 trainable parameters and Adam states; gradient checkpointing.
  GPU microbatches vary by arm and may shrink after OOM while preserving effective batch.
  Full tuning may fall back to the repository's CPU AdamW implementation if needed.
- QLoRA: NF4 four-bit weights, double quantization, BF16 computation. LoRA rank 8,
  alpha 16, dropout 0.1, q/k/v/output projections. Qwen27 hybrid linear-attention
  layers additionally adapt QKV, z/b/a gates and output projections. Exact paths
  and trainable counts are recorded in every manifest.
- F1 threshold is selected on validation only over 0.10..0.89 in steps of 0.01.
  Metrics also include accuracy/F1 at threshold 0.5 and threshold-independent AUROC.

This document records the training protocol. Released completed results are
listed in the aggregate result inventory; omitted or skipped runs have no
imputed metrics.

## Execution and artifacts

Launcher: `python -B scripts/run_kip_data_efficiency.py --background`.
Quantized runs require bitsandbytes; the recorded experiment used 0.50.2.

Run root: `results/kip_m4_data_efficiency_20260920_seed9950/`.

Initial microbatches were 128 for adapter arms and 16 for full tuning, with
effective batch size 256 and evaluation batch size 256. The following list
describes locally generated artifacts; the release includes only the selected
aggregate subset. Not distributed in this release: `supervisor.pid`,
`supervisor.log`, `queue_status.json`, `launch_plan.json`, `protocol/`, the
per-attempt `manifest.json`, `status.json`, `history.jsonl`, row indices,
`val_predictions.csv`, `test_predictions.csv` and `best_trainable.pt`.
Retained: `summary.csv` and one `metrics.json` per completed attempt
(`<arm>/fraction_<f>/attempt_001/metrics.json`, 16 files).

- `supervisor.pid`, `supervisor.log`, `queue_status.json`: queue identity and progress.
- `launch_plan.json`, `protocol/`: immutable plan and copied source files.
- `summary.csv`: production results and explicit pending/failed statuses.
- Per arm/fraction/attempt: `manifest.json`, `status.json`, `history.jsonl`, selected
  row indices, `val_predictions.csv`, `test_predictions.csv`, and `metrics.json`.
- LoRA/QLoRA also save the selected adapter and head tensors in `best_trainable.pt`.
  Full 8B selected weights are held in CPU RAM until final evaluation and are not
  exported because available disk space is insufficient. Metrics and predictions
  are saved for all completed fits. An unfinished fit restarts from the base model;
  no optimizer checkpoint resume is provided.
- `scripts/plot_kip_data_efficiency.py --run-root <root>` exports PNG/PDF/SVG curves
  from completed rows only. No seed error bars are inferred from this single-seed run.

The queue validates exit status and completion metrics before advancing, retries
OOM failures with smaller microbatches, and preserves failed attempts. An arm's
production fits require its GPU preflight to pass. A failed point is reported as
failed rather than filled in with a score. Global locks prevent duplicate queue
processes and are inherited by active training children.
