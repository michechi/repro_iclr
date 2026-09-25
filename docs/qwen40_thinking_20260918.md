# Qwen3.8-27B with 40 demonstrations and thinking enabled

This extension was requested after the completed [0/5/20-shot nonthinking experiment](qwen38_fewshot_20260918.md). It uses the same frozen **Qwen/Qwen3.8-27B** checkpoint at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, full BF16 weights, and the same H200. Previous results remain separate and unchanged.

The full plan covers OC deterministic, OC with 30% label noise, GPC, KIP4, and KIP6, with **40 demonstrations, seeds 0/1/2, and the same 1,000 test rows per task**: 15 configurations and 15,000 predictions. Each demonstration set contains 20 positives and 20 negatives, sampled from training data without replacement. Complete sequence matches to the fixed test sample are excluded. The new 40-shot sets follow the existing sampler and are **not nested extensions** of the old 20-shot sets; observed same-seed intersections are only 0–2 training examples.

**Status:** the full main run has been launched after operational pilots. Expected runtime from pilot throughput is approximately 50–60 hours, subject to actual generation lengths and throughput. This is an estimate, not a completed result. See [main status](../results/qwen38_27b_fewshot40_thinking_20260918/run/status.json) and [launcher status](../results/qwen38_27b_fewshot40_thinking_20260918/run_launcher.json).

## Genuine reasoning generation

The official chat template uses `enable_thinking=True` and explicit **`reasoning_effort="medium"`**. The user instruction and sequence-example format are unchanged; no task identities, key sets, or hidden rules are disclosed. Medium-effort prompts contain **1,435 tokens** for all 15,000 main queries. This differs from the checkpoint's default xhigh effort, which adds a system instruction and yields 1,477 tokens. The selected effort is recorded explicitly.

The model generates reasoning with its native cache enabled and the official thinking sampling settings: temperature 1.0, top-p 0.95, top-k 20, min-p 0, and repetition penalty 1.0. The [pinned official model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md) documents the available effort levels and sampling defaults. The experiment limits each query to **4,096 generated thinking tokens** and reserves a total context limit of 8,192 tokens. No input truncation is permitted.

Generation stops at the first emitted `</think>` token. The exact generated token IDs are retained; decoded reasoning is archived for inspection but is never retokenized for scoring. The evaluator appends the canonical two-newline separator, then runs an ordinary forward pass to obtain the raw logits for `0` and `1`. It does not use temperature/top-k/top-p-processed generation scores for classification.

When the reasoning budget is exhausted, the evaluator explicitly adds the closing thinking token and separator before scoring. Premature ordinary EOS is separately marked and handled with the same explicit closure. All queries remain in the evaluation, including these cases. Natural completion, budget exhaustion, early EOS, forced closure, and token counts are recorded per query and summarized in the report. This is a **bounded thinking protocol**; a forced close must never be described as naturally finished reasoning.

## Scoring and reproducibility

Primary AUROC uses `logit(1) - logit(0)`, computed after converting the two BF16 output logits to FP32. Its sigmoid is conditional on the two label tokens **and one sampled reasoning trace**. It is neither a probability marginalized over reasoning paths nor a freely generated final-answer measure. Accuracy, precision, recall, F1, and confusion matrices use threshold 0.5. Raw-score and probability AUROC, tied-score diagnostics, and class counts are retained.

Every generation batch receives a seed derived from the separate generation seed 9951, cell identity fields, and batch start. The exact batch size and ordered inputs are part of the configuration. Completed batches are saved atomically with hashes, so an interrupted run can reuse verified batches without changing the RNG streams of later batches. Batch-size changes require a different run configuration.

Main generation uses batches of 32, while final-label scoring uses batches of four to limit memory consumption with long reasoning prefixes. Reasoning is checkpointed before scoring, so an interrupted scoring step can reuse it. Completed batches retain the full reasoning and score evidence. The original medium-effort pilot used a 32-row scoring batch and encountered a CUDA allocator warning that recovered; the later small scoring batches passed GPU validation and fault-injection/resume tests.

Across-seed means and sample SDs summarize three demonstration configurations. Because each configuration also uses a distinct recorded reasoning RNG stream, the SD includes both demonstration changes and stochastic reasoning variation. It is not an isolated estimate of either source, a confidence interval, or variation across independent datasets.

The previous nonthinking 20-shot results are a descriptive reference. Both demonstration count/set and thinking mode change, and sampled reasoning introduces another source of variation. Differences cannot be attributed solely to more examples or solely to thinking. No 20-shot thinking control is implied.

## Code, validation, and execution

- [Thinking evaluator](../src/experiments/qwen_thinking_fewshot_eval.py)
- [Native generation and termination helper](../src/experiments/qwen_thinking_generation.py)
- [Supervised launcher](../scripts/run_qwen_thinking_fewshot.py)
- [Independent report generator](../scripts/summarize_qwen_thinking.py)
- [Data and tokenizer audit](../results/qwen38_27b_fewshot40_thinking_20260918/evidence/forty_shot_data_tokenizer_audit.json)
- [Generation tests](../tests/test_qwen_thinking_generation.py) and [reference/sample tests](../tests/test_qwen_thinking_fewshot_eval.py)

The pilot evaluates 32 proportionally stratified KIP6 examples selected from the original fixed 1,000-row sample, with demonstration seed 0 and the production generation batch size. Its demonstrations still exclude the full original sample. It checks actual cached GPU generation, natural versus forced completion, final-label scoring, memory, and runtime. Pilot metrics are diagnostic and remain separate from the main study.

The medium-effort pilot completed in 437.63 seconds. **All 32 traces reached the 4,096-token cap; none closed naturally.** Its diagnostic AUROC was 0.6035 on only 32 queries. Inspection and independent decoding checks found coherent, verbose pattern searches and counting errors, rather than corrupted or mechanically looping token output. A separate eight-query low-effort diagnostic also reached the cap on all eight queries. Since low effort did not resolve truncation, the main protocol retains the originally planned medium effort; settings were not selected by pilot classification accuracy. These observations make the reasoning-budget limitation material: the main results must be described as budgeted reasoning followed by forced closure where needed.

```bash
python -B scripts/run_qwen_thinking_fewshot.py --pilot --background
python -B scripts/run_qwen_thinking_fewshot.py --background
```

The second command starts the full plan after a successful pilot. `--n-test 100` instead selects a reproducible, proportionally stratified subset of each original test sample; such a reduced run must be described as a smaller experiment. `--dry-run` checks data and sampling without model loading.

The launcher supervises the evaluator and invokes the independent report after successful completion. Main outputs are under `results/qwen38_27b_fewshot40_thinking_20260918/run/`, with reports under `run_analysis/`; pilot artifacts use `pilot/` and `pilot_analysis/`. Status is incremental, and a partial result must not be presented as the completed grid. Check status timestamps and process liveness together when a run is unfinished. Not distributed in this release: `run/`, `pilot/` and `pilot_analysis/`; retained: `run_analysis/summary.csv`.

Checkpoint weights remain in the existing volatile host RAM cache; no duplicate checkpoint download or deletion is needed. Configuration, source snapshots, predictions, generated reasoning, per-batch hashes, logs, and reports are saved persistently in the experiment directory.
