# KIP BERT / RoBERTa, seed 9950

The September 16, 2026 matrix trains BERT-base and RoBERTa-base on KIP M4
and M6, using original ordered sequences and the existing shuffled sequences.
There is one training seed, **9950**. All eight LoRA runs precede all eight
full fine-tuning runs. Every run starts from the pretrained base checkpoint;
full fine-tuning does not continue from the LoRA model.

## Data and shuffle meaning

Each dataset contains 400,000 training, 50,000 validation, and 50,000 test
examples, each with 20 events. Shuffling permutes events within each example
and retains the original labels. It is a control for the sequence order.
It does not shuffle only the minibatch order or recompute parity labels.
Training minibatch order is randomized in both conditions.

The original ordered CSVs were absent from this checkout. The recovery script
`scripts/recover_kip_ordered.py` exactly inverted the recorded per-example
shuffle with seed 777. All recovered rows were checked against the rule,
and forward shuffling reproduced the existing CSV bytes. Independent
regeneration using the surviving historical generator bytecode matched all
12 recovered CSV files byte-for-byte.

Data locations:

- Ordered: `data/simulation/kip_ordered_recovered/`
- Shuffled: `data/simulation/kip_shuffled/`
- Recovery evidence: `recovery_manifest.json` and
  `independent_generator_audit.json` in the recovered directory.

There are no duplicate ordered sequences within a split or across splits.
The recovery preserves the original train/validation/test partitions.

## Training settings

| Setting | Value |
| --- | --- |
| Models | `bert-base-uncased`, `FacebookAI/roberta-base` |
| BERT revision | `86b5e0934494bd15c9632b12f734a8a67f723594` |
| RoBERTa revision | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| Seed | 9950 |
| Batch size | 256 |
| Optimizer | AdamW, learning rate 2e-5 |
| Schedule | 6% linear warmup, linear decay |
| Epoch cap | 20 |
| Early stopping | Validation loss, patience 3 |
| LoRA | Rank 8, alpha 16, dropout 0.1, query/key/value |
| Precision / hardware | BF16 autocast, NVIDIA H100 80GB |
| Input | `Sequential events: ` followed by space-separated events |
| Tokenization | Pretokenized, dynamic padding, maximum 64; truncation forbidden |

The batch size differs from historical KIP BERT runs (16). It was selected
using a short H100 throughput check and is held constant across this matrix.
These are new experiments; existing seed-9550 outputs remain separate.

Checkpoints are selected only using validation loss. The F1 threshold is
selected on validation data; test labels do not select weights or thresholds.
Ordered models also receive a shuffled-test evaluation with the same selected
checkpoint and threshold.

## Execution and artifacts

The interpreter is `python`. Model files are local, and
the supervisor runs with Hugging Face offline mode. The training process
requires direct GPU access outside the restricted tool sandbox.

```bash
# Four short checks: both model families and both fine-tuning methods.
python -B scripts/run_kip_encoders.py --smoke

# Launch the complete matrix detached from the terminal.
python -B scripts/run_kip_encoders.py --background

# Inspect overall and current-run progress.
python -B scripts/run_kip_encoders.py --status
```

Output root: `results/kip_encoders_20260916_seed9950/`.

- `matrix.json`: all 16 commands and pinned model paths.
- `matrix_status.json`: supervisor state, PID, active run, completed count.
- `supervisor.log` and `logs/`: overall and individual training logs.
- `runs/{lora,full}/{bert,roberta}/{kip_m4,kip_m6}/{ordered,shuffled_train}/seed9950/`:
  checkpoint, run configuration, epoch history, status, and test predictions.
- `summary.csv`: accumulated results for completed runs only.

Not distributed in this release: `matrix.json`, `matrix_status.json`,
`supervisor.log`, `logs/`, and the per-run checkpoints, configuration, epoch
history, status and test predictions. Retained: `summary.csv` and one
`metrics.json` per completed run under `runs/` (16 files).

The supervisor stops on a failed run and does not launch the next stage.
Re-running the same command resumes unfinished work using the last saved
epoch. Only a completed run with saved metrics counts toward the total.
