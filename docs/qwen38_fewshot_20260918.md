# Qwen3.8-27B few-shot evaluation

This experiment uses the requested **Qwen/Qwen3.8-27B**, pinned to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The [official model card](https://huggingface.co/Qwen/Qwen3.8-27B) describes the checkpoint; its local configuration selects Transformers' native `Qwen3_5ForConditionalGeneration` architecture. The model runs with frozen BF16 weights and no quantization or task-specific training. Inputs are text only.

**Completed:** all 35 main configurations and 35,000 predictions passed independent metric recomputation and artifact checks. The main grid took **1,741.05 seconds (29.02 minutes)** on one H200. Five-shot mean AUROC ranges from 0.4917 to 0.5260; twenty-shot means range from 0.4780 to 0.5209. See the [full results](../results/qwen38_27b_fewshot_20260918/analysis/RESULTS.md), [paper paragraph](../results/qwen38_27b_fewshot_20260918/analysis/PAPER_SUMMARY.md), and [verified comparison with Qwen2.5](../results/qwen38_27b_fewshot_20260918/comparison/COMPARISON.md).

The experiment repeats the five tasks, shot counts, demonstration seeds, and test samples from the [Qwen2.5-14B evaluation](qwen_fewshot_20260918.md). It uses the official chat template with **thinking disabled** and scores the next-token alternatives `0` and `1`. It does not generate reasoning or answers. Qwen2.5 was a base checkpoint evaluated with raw completion prompts, whereas this is a post-trained checkpoint using chat. Consequently, differences combine checkpoint, model size, training-stage, and prompt-format effects; they do not isolate a parameter-count effect or compare thinking abilities.

## Data and sampling

| Task | Data | Target | Full train / test rows |
|---|---|---|---:|
| OC deterministic | Canonical corrected OC exports from the previous Qwen experiment | Clean ordered-compliance label `Y_star` | 400,000 / 50,000 |
| OC noise | The same canonical full sequence pool | `Y_pi_0p3`, independent symmetric label flips with probability 0.3 | 400,000 / 50,000 |
| GPC | `data/simulation/tested`, tag `test_just_pair` | Stored global key-count parity label | 800,000 / 100,000 |
| KIP4 | `data/simulation/kip_ordered_recovered`, tag `kip_m4` | Stored even inversion parity for four keys | 400,000 / 50,000 |
| KIP6 | The same recovered directory, tag `kip_m6` | Stored even inversion parity for six keys | 400,000 / 50,000 |

The original data files remain unchanged. Corrected OC exports and their audit are reused from `results/qwen25_14b_fewshot_20260918/data/oc_corrected/`. Definitions and generation settings are documented in the earlier experiment. In particular, the clean/noisy OC tasks each use their own proportionally stratified sample from the common sequence pool, rather than identical selected clean/noisy rows.

- One fixed sample of **1,000 test rows per task**, approximately proportionally stratified with seed 9950.
- **0, 5, and 20 demonstrations**. Nonzero-shot configurations use demonstration seeds **0, 1, and 2**. Zero-shot runs once per task.
- **35 configurations / 35,000 query predictions**. Repeated evaluations of the fixed samples are not 35,000 independent test examples.
- Demonstrations are sampled without replacement from training data. Five-shot uses three negatives and two positives; twenty-shot uses ten of each. The ordered demonstration set is fixed across queries within a configuration. Different shot counts do not require nested exemplar sets.
- Exact sequence matches to the sampled test queries are excluded from demonstration candidates.
- Before model loading, the evaluator verifies test indices, labels, sequence text, data file hashes, and complete ordered demonstration payloads against each corresponding completed Qwen2.5 configuration.

A separate 32-query-per-task pilot covers all five tasks and all three shot counts with demonstration seed 0. Its 15 cells are operational checks and are excluded from main results.

## Prompt and scoring

The single user message begins with this neutral instruction:

> Predict the binary outcome for the final sequence. Use the labeled examples when provided. Reply with exactly one digit, 0 or 1, and no other text.

Labeled examples and the final unlabeled query follow in the previous experiment's `Sequential events: <20 space-separated letters>` / `Outcome (0 or 1):` format. No task name, hidden rule, key set, rank order, or parity formula appears in the prompt. Therefore zero-shot is an uninformed control for an undisclosed rule, and poor few-shot performance would not show inability to execute a supplied rule.

The official `apply_chat_template` uses `add_generation_prompt=True` and `enable_thinking=False`. This appends the official assistant prefix containing an empty closed thinking block. The template trims the trailing whitespace from the query. The evaluator tests exact prompt-prefix preservation for both numeral continuations, rather than assuming whitespace behavior. For this tokenizer, label IDs are **15 / 16**, and prompts have **77 / 247 / 757 tokens** for 0 / 5 / 20 shots. Truncation is forbidden; the configured limit is 4,096 tokens.

Only the final-position logits for those two labels are extracted. They are promoted to FP32 before computing `logit(1) - logit(0)` and its sigmoid. Primary AUROC uses raw log-odds to avoid extra ranking ties from sigmoid saturation. The sigmoid is a probability conditional on choosing between these two tokens, not full-vocabulary probability mass or established calibration. Secondary AUROC uses probabilities; accuracy, precision, recall, F1, and confusion matrices use threshold 0.5. Undefined precision/F1 use `zero_division=0`.

Grouped results report the mean and sample SD across the three demonstration seeds. This SD measures demonstration sensitivity on a fixed test sample; it is not a confidence interval or variability across independent task generators.

BF16 output logits can produce tied label scores. Promoting those logits to FP32 before subtraction avoids further arithmetic loss but does not restore information already rounded in the logits. Raw-log-odds AUROC removes only additional ties caused by sigmoid saturation. Distinct score counts are retained in the results.

## Implementation and provenance

The [Qwen3.8 evaluator](../src/experiments/qwen38_fewshot_eval.py) uses native Transformers loading and shares pure data, sampling, scoring, and metric helpers with the unchanged [previous evaluator](../src/experiments/qwen_fewshot_eval.py). Both source files are frozen by the [launcher](../scripts/run_qwen38_fewshot.py) for each stage. Runtime records include package versions, actual kernel bindings, parameter count and dtype, loading diagnostics, prompt/template hashes, exact prompts, and label tokenization. Model loading rejects missing or mismatched parameters; unused auxiliary `mtp.*` checkpoint tensors are explicitly recorded.

The checkpoint has 18 BF16 safetensors shards totaling 55,563,006,776 bytes.
Download the pinned revision into a writable local Hugging Face cache. Base
weights are not distributed with this release. Content-based checkpoint
identity permits a pinned redownload independently of cache modification times.

The execution environment uses one H200 GPU, batch size 8, BF16 weights, SDPA full attention, and Flash Linear Attention kernels for GatedDeltaNet where available. The small `fla-core==0.5.2` and `einops==0.8.2` dependencies are pinned by package hash in the experiment's `protocol/kernel_requirements.txt` (not distributed in this release: the `protocol/` directory; the two pins stated here are the recorded versions). Causal convolution retains its PyTorch implementation. Runtime metadata distinguishes individual bindings from the architecture's aggregate optional-fast-path flag.

From the repository root, after the checkpoint download, integrity audit, and kernel validation:

```bash
python -B scripts/run_qwen38_fewshot.py --pilot --background
python -B scripts/run_qwen38_fewshot.py --background
```

Run the main command only after the pilot completes successfully. `--dry-run` validates sampling and cross-model correspondence without loading the checkpoint. Completed cells can be resumed only if their configuration identities and artifact hashes match; incomplete cells restart. The output directory is locked against concurrent writers.

The [summarizer](../scripts/summarize_qwen_fewshot.py) independently recomputes metrics from saved predictions and verifies artifact hashes, indices, labels, and score arithmetic before producing tables and figures:

```bash
python -B scripts/summarize_qwen_fewshot.py \
  --run-dir results/qwen38_27b_fewshot_20260918/run \
  --output-dir results/qwen38_27b_fewshot_20260918/analysis
```

Not distributed in this release: `run/` (per-cell predictions, logs and status). Retained under `results/qwen38_27b_fewshot_20260918/`: `analysis/summary.csv` and `gpc_recheck_20260924/summary.csv`.

Completion is established by verified cell artifacts. A live or interrupted `status.json` is interpreted alongside process liveness. Few-shot results should remain distinct from models trained on hundreds of thousands of task examples.
