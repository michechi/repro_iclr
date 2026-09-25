# Qwen few-shot evaluation across five tasks

This experiment was requested on 18 September 2026. The repository references Qwen2.5-14B and Qwen3-4B, but neither checkpoint was cached when the experiment started. The chosen checkpoint is **Qwen/Qwen2.5-14B base**, matching the existing few-shot model choice. This is inference with a frozen pretrained causal language model; there is no fine-tuning, task-specific classification head, chat template, or rule disclosure.

**Completed:** all 35 main configurations and 35,000 query predictions, with metrics independently recomputed from saved predictions. The main grid took 830.48 seconds on the H200. See the [full results](../results/qwen25_14b_fewshot_20260918/analysis/RESULTS.md), [summary table](../results/qwen25_14b_fewshot_20260918/analysis/summary.csv), and [comparison figure](../results/qwen25_14b_fewshot_20260918/analysis/qwen_fewshot_auroc.pdf). Five-shot mean AUROC ranges from 0.5001 to 0.5024; twenty-shot mean AUROC ranges from 0.4927 to 0.5156. These are near-chance outcomes for this prompt, checkpoint, and demonstration budget.

The [official model card](https://huggingface.co/Qwen/Qwen2.5-14B) identifies this as the pretrained/base model with approximately 14.7 billion parameters. We pin revision `97e1e76335b7017d8f67c08a19d103c0504298c9`. Its eight BF16 safetensors shards total 29,540,133,920 bytes. Only one model copy is downloaded, without quantization or a second checkpoint export.

## Tasks and data

| Task | Input source | Outcome | Full train / test rows |
| --- | --- | --- | --- |
| OC deterministic | Canonical `data/simulation/oc_hole_audit/splits/` | `Y_star` | 400,000 / 50,000 |
| OC noise | Same canonical sequences; aligned `noise/` tables | `Y_pi_0p3`, symmetric label-flip probability 0.3 | 400,000 / 50,000 |
| GPC | `data/simulation/tested/`, tag `test_just_pair` | Stored global key-count parity label | 800,000 / 100,000 |
| KIP4 | `data/simulation/kip_ordered_recovered/`, tag `kip_m4` | Stored even-inversion-parity label | 400,000 / 50,000 |
| KIP6 | Same recovered directory, tag `kip_m6` | Stored even-inversion-parity label | 400,000 / 50,000 |

Corrected OC uses the canonical ordered-compliance rule. Its dataset seed is 9550 and noise seed is 9650. These data are distinct from the older tag-6/tag-9 comparison: the legacy tasks have a rule difference beyond label noise. The corrected OC CSVs exported for this evaluation preserve source row order and IDs, with explicit source-file hashes and observed-label column definitions. No original dataset is modified. Optional evaluator tasks `oc_det_legacy` and `oc_noise_legacy` remain separately named.

The underlying OC rule uses key letters W, D, Q, J, X, U in that reference order, lag 7, and the repository's canonical `rule_greedy_monotone_impl(tolerance=False, min_chain_length=2)`. A clean positive contains a qualifying maximal lag-7 key-symbol run whose reference ranks are non-decreasing. The noisy observed label is the clean label XOR an independently sampled flip indicator. GPC instead labels an example positive when its total count of W, D, Q, J, X, N is even. KIP4/KIP6 label even inversion counts in their fixed key-order rankings, with keys C,O,T,A and G,H,L,Y,I,Z respectively. These definitions are recorded for interpretation; none is supplied in the model's prompt.

The OC clean/noisy test samples are each proportionally stratified by their own outcome; they come from the same full sequence pool but are not necessarily the same selected rows. This is a task comparison, not a paired estimate of the effect of label corruption on identical prompts. GPC/KIP labels and the canonical OC data were independently audited before evaluation. All sequences contain 20 letters separated by U+001F in the CSV; the evaluator renders them as 20 space-separated letters.

## Evaluation protocol

- Shot counts: **0, 5, 20**.
- Demonstration seeds: **0, 1, 2** for nonzero shot counts. Zero-shot is evaluated once per task because no demonstration sampling occurs.
- **1,000 test examples per task**, selected once by proportional class stratification with seed 9950. Identical test rows are reused across shot counts and demonstration seeds for a task.
- **35 configurations and 35,000 query predictions** in the main grid. Reused test rows are not independent additional test examples.
- Demonstrations are drawn only from training data, without replacement, and balanced by label: three negatives/two positives for five shots; ten of each for twenty shots. They remain fixed for every query in a configuration. Changing shot count does not guarantee nested demonstration sets.
- Demonstration candidates whose complete sequence matches a sampled test sequence are excluded. All selected training and test indices are saved.
- Prompt for each demonstration: `Sequential events: <20 letters>\nOutcome (0 or 1): <label>`. The query uses the same form with its label omitted and a trailing space included.
- The prompt does not reveal task identity, key sets, rank rules, or parity formulas. Zero-shot is therefore an uninformed baseline for an arbitrary hidden rule, not an explicit reasoning test.
- The frozen model runs in BF16 on one H200 GPU with SDPA attention. The final two label logits are converted to FP32 before scoring. Batch size is recorded in the run configuration.
- Full prompt-plus-label tokenization must equal the prompt tokens followed by exactly one distinct token for 0 or 1. Unsupported tokenization fails explicitly. No prompt truncation is allowed; the configured limit is 4,096 tokens.
- The binary probability is `sigmoid(logit_1 - logit_0)`, conditional on the two permitted label tokens. It is not the model's full-vocabulary probability mass for the numeral and is not established to be calibrated.
- Primary AUROC uses the raw FP32 label log-odds, avoiding artificial ties caused by a saturated sigmoid. Probability-based AUROC is also saved as a diagnostic. Accuracy/F1/precision/recall use threshold 0.5, with class counts and confusion matrices. Demonstration-seed SD measures sensitivity to exemplar selection on the fixed test sample, not uncertainty across generated tasks.

A separate small pilot checks real model loading, every task, and all three prompt lengths before the main grid. Pilot metrics are diagnostic and must not be pooled with main-grid results.

## Code and commands

The new [evaluator](../src/experiments/qwen_fewshot_eval.py) is separate from the historical generic few-shot code, preserving that experiment's source. Twenty [focused tests](../tests/test_qwen_fewshot_eval.py) cover data normalization, sampling, label scoring, sigmoid saturation, truncation rejection, and completed-run validation. The [launcher](../scripts/run_qwen_fewshot.py) freezes a source copy and saves persistent logs; `--pilot --background` runs the small pilot, and `--background` starts the main grid with the protocol below.

From the repository root, after preparing the corrected OC exports and downloading the pinned model:

```bash
python -B -u -m src.experiments.qwen_fewshot_eval \
  --model-path /path/to/local/model/snapshot \
  --oc-data-dir results/qwen25_14b_fewshot_20260918/data/oc_corrected \
  --oc-det-tag ocdet --oc-noise-tag ocnoise \
  --tasks oc_det,oc_noise,gpc,kip4,kip6 \
  --shots 0,5,20 --demo-seeds 0,1,2 \
  --test-seed 9950 --n-test 1000 --batch-size 8 --max-length 4096 \
  --output-dir results/qwen25_14b_fewshot_20260918/run
```

`--dry-run` validates data/sampling without requiring model weights; it does not claim to check tokenizer behavior. A real pilot additionally exercises GPU inference and tokenization. Completed configurations can be skipped only when configuration identity and saved artifact hashes match. An interrupted incomplete configuration restarts. The evaluator locks its output directory against concurrent writers.

The [experiment directory](../results/qwen25_14b_fewshot_20260918/) in this release contains the corrected OC data export with its provenance (`data/oc_corrected/`, including `export_manifest.json` and `prepare.py`) and the recomputed summary (`analysis/summary.csv`). Not distributed in this release: the frozen evaluator copy, pilot/main logs, per-configuration demonstrations and predictions, and the incremental `run/` status and summary JSON (the `--output-dir` above names that historical location; use a directory under `runs/` for your own reruns). When assessing an unfinished rerun, read its `run/status.json` alongside process liveness.

## Reporting limits

Few-shot results concern a frozen model receiving at most 20 examples, and must not be presented as directly compute/data-matched comparisons to models trained on hundreds of thousands of examples. A single selected pretrained checkpoint, one prompt format, one test sample, and three demonstration seeds do not establish general Qwen performance or architectural inability. Arbitrary hidden-rule tasks can remain near chance without showing that the language model cannot apply an explicitly supplied rule.

No main-grid result is claimed until its configuration has completed and the saved predictions have been checked. The KIP full-training review remains a separate experiment report.

The independently verified weight hashes (`evidence/qwen_model_weight_hashes.json`, not distributed in this release) cover all eight safetensors shards, in addition to the evaluator's per-run metadata and size/mtime checks. The actual GPU-loaded causal model has **14,770,033,664 parameters**, all frozen.
