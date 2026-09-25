# Llama 3.2 1B full fine-tuning on KIP, seed 9950

Four independent runs use KIP M4/M6 with ordered and shuffled sequences.
Every run starts from `meta-llama/Llama-3.2-1B`, revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`, already cached locally.

The task is binary outcome classification using
`LlamaForSequenceClassification`, with all 1,235,818,496 parameters trainable.
The prompt is the same as in the BERT/RoBERTa seed-9950 comparison:
`Sequential events: ` followed by the 20 space-separated events. This is
pure outcome cross-entropy, rather than the combined language-model and
classification loss in the historical Llama fraction experiment.

## Fixed training settings

| Setting | Value |
| --- | --- |
| Seed | 9950 |
| Data | 400,000 training / 50,000 validation / 50,000 test per dataset |
| Batch size | 256 |
| Gradient accumulation | 1 |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Schedule | 6% linear warmup, linear decay |
| Epoch cap / patience | 20 / 3, minimum validation loss |
| Precision | FP32 trainable weights and Adam states; BF16 network autocast |
| Loss | Explicit FP32 binary outcome cross-entropy |
| Input length | Exactly 24 tokens for all audited sequences; no truncation |
| Padding | Right padding, EOS reused as PAD; EOS absent from real content |
| Attention | Causal Llama attention; KV cache disabled |

Original labels and partitions are retained in the shuffled condition.
The ordered CSVs are the previously verified exact recovered originals.
F1 thresholds are selected on validation data only. Ordered runs also
evaluate their selected model on the paired shuffled test data using the
same threshold.

## Checkpoints and storage

The best validation-selected state is kept in CPU RAM during each run.
One complete FP32 Hugging Face model is exported after training, with its
tokenizer, metrics, and test predictions. Four selected models require
approximately 19.8 GB, fitting the available disk without modifying prior
BERT/RoBERTa artifacts.

There are no persistent optimizer or epoch checkpoints. An interrupted
unfinished run must restart in a new empty output directory. Completed runs
can be skipped after configuration and artifact hash verification.

## Commands and outputs

```bash
# Small GPU pipeline check; model export covered by the tiny CPU tests.
python -B scripts/run_kip_llama.py --smoke

# Detach the four full-data runs.
python -B scripts/run_kip_llama.py --background

# Inspect progress.
python -B scripts/run_kip_llama.py --status
```

These GPU commands require execution outside the restricted tool sandbox.
Output root: `results/kip_llama1b_full_20260916_seed9950/`.

`matrix.json` records commands and source hashes, `matrix_status.json` tracks
the queue, `logs/` contains per-run logs, and `summary.csv` accumulates
completed results. Individual artifacts are under
`runs/full/llama1b/{kip_m4,kip_m6}/{ordered,shuffled_train}/seed9950/`.

Not distributed in this release: `matrix.json`, `matrix_status.json`, `logs/`,
the smoke directory, and per-run artifacts other than metrics. Retained:
`summary.csv` and one `metrics.json` per run under `runs/` (four files).

The smoke test has a separate directory and is excluded from production
results. It deliberately omits the large model export.
