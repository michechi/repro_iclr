# Reproducing the released artifacts

Run top-level commands from the directory containing this file. Python 3.12 is the validated interpreter family. Install `requirements.txt` for plotting and classical/scratch experiments. Install `requirements-models.txt` for pretrained models and `requirements-dev.txt` for tests. Choose a PyTorch build appropriate to your own GPU/driver. The observed environment is recorded in [release/ENVIRONMENT.md](release/ENVIRONMENT.md); minimum dependency ranges are not a claim of bitwise reproducibility across versions or devices.

## 1. Verify the snapshot and datasets

```bash
python -B scripts/verify_release.py --require-manifest
python -B scripts/check_datasets.py
```

The integrity manifest (`MANIFEST.sha256`, generated at packaging time as the final step) covers the released bytes. Run the release verifier on the freshly extracted snapshot before creating local environments or run outputs inside it. The dataset checker verifies the recovery-manifest hashes, split sizes, exact-once KIP keys and parity labels for all 1,000,000 KIP examples; it also verifies the corrected OC rule and aligned deterministic/noisy inputs for all 500,000 OC examples. It reports the observed noise rate, not a forced exactly 30% noise count.

See [data/README.md](data/README.md) for the distinction between historical `_6`/`_9` labels and corrected OC. Do not substitute one dataset for the other when comparing metrics.

## 2. Replot attention without inference

```bash
python -B scripts/reproduce_figures.py
```

This regenerates PDF, PNG and SVG for OC, KIP-M4 and CKD using the exact aggregate CSVs in `paper/figures/`. It writes to `runs/figures/`; released figures remain unchanged. Numerical source hashes are recorded in `figure_inputs.json`. The CKD command needs no clinical data. The original `build_*attention*figure.py` scripts reconstruct aggregates from audit outputs; the CKD version requires private per-patient audit files and cannot run from this public release alone.

Read [paper/attention_appendix.tex](paper/attention_appendix.tex) for the measurement definitions and limits. A standalone LaTeX wrapper, `paper/attention_preview.tex`, is provided; to compile it:

```bash
cd paper
pdflatex attention_preview.tex
pdflatex attention_preview.tex
```

The full working manuscript is not a complete LaTeX build; see [paper/README.md](paper/README.md).

The KIP-M4 data-efficiency plot can also be regenerated from completed fits:

```bash
python -B scripts/plot_kip_data_efficiency.py \
  --run-root results/kip_m4_data_efficiency_20260920_seed9950 \
  --output-dir runs/figures/kip_data_efficiency
```

The released version is in `paper/figures/kip_data_efficiency/`. Its caption records the single seed and test-informed stopping. The 300K/400K fractions have no plotted values.

## 3. Small CPU training check

```bash
python -B -m src.experiments.kip_rnn_transformer_training \
  --tag kip_m4 --seed 42 --device cpu --smoke \
  --output-dir runs/kip_m4_cpu_smoke
```

This performs one short training epoch on the first 256 training, 512 validation and 512 test rows of the canonical splits. It validates the full input before selecting the smoke subset. Smoke metrics are not paper results. Use a new output directory for each fit.

For the six full RNNTransformer runs, use `--device cuda`, omit `--smoke`, and run each combination of `--tag kip_m4/kip_m6` and `--seed 9550/9551/9552`. The protocol is recorded in [docs/kip_rnn_transformer_20260924.md](docs/kip_rnn_transformer_20260924.md). The trainer defaults to the recorded 30-epoch cap, strict validation-F1 early stopping and patience 5.

## 4. Pretrained KIP experiments

Training modules take explicit local model snapshots. Download the recorded model revision yourself before running. Example for one independent Qwen27B QLoRA fit:

```bash
python -B -m src.experiments.kip_data_efficiency \
  --arm qwen27b_qlora --model-path /path/to/local/model/snapshot \
  --fraction 0.50 --seed 9950 --stop-on-perfect-test \
  --output-dir runs/qwen27b_kip_m4_200k
```

Use fractions `0.01`, `0.10`, `0.30`, `0.50` for 4K, 40K, 120K and 200K. Arms are `qwen27b_qlora`, `qwen14b_qlora`, `llama8b_lora`, `llama8b_full`. Every fit starts from the pretrained base. Full Llama-8B can use `--optimizer-offload cpu`; it needs substantial host RAM. The released sweep adopted a test-informed stopping rule; enabling `--stop-on-perfect-test` reproduces that rule, and the stopped larger fractions are not independent observations. Defaults and exact per-run settings are documented in [docs/kip_m4_data_efficiency_20260920.md](docs/kip_m4_data_efficiency_20260920.md) and the saved metrics.

Encoder and Llama-1B experiments use `src.experiments.kip_encoder_training` and `src.experiments.kip_llama_training`. Run each module with `--help`; supply `--ordered-dir`, `--shuffled-dir`, and `--output-dir` explicitly, with inputs under `data/simulation/`, a separate cache, and outputs under `runs/`. The encoder trainer exposes `--tuning full` and `--tuning lora`; the Llama-1B trainer performs full fine-tuning. Both accept M4/M6 tags. The experiment-specific protocol documents record seeds and tuning settings.

## 5. OC, CKD and attention re-extraction

Task-specific trainers are `src.experiments.encoder_oc_ckd`, `src.experiments.llama8_oc_ckd`, `src.experiments.qwen_oc_training`, and `src.mimic.qwen_ckd_training`. The corresponding audit modules are under `src/analysis/`. The encoder and Llama-8B trainers require `--run-root` and `--job`, which select an entry in a locally prepared `launch_plan.json`; dataset paths, model snapshots and recipes are taken from that plan. They do not accept a direct `--data-dir` argument. The Qwen trainers accept `--model-path`, `--data-dir` and `--output-dir` directly. For OC, use the corrected CSV export directory documented in `data/README.md`; for CKD, use your own prepared cohort directory. No KIP-tuned checkpoint is used to initialize OC or CKD fits.

Qwen OC runs were externally capped after epoch 6 while retaining the learning-rate schedule configured for 20 epochs. A new run using `--epochs 6` changes that schedule and is not an exact reproduction. The selected checkpoints and protocol are described in the inventory; the private checkpoints and operational stop/recovery scripts are not distributed. Retaining the 20-epoch schedule and selecting the corresponding saved epoch is necessary for a matched re-run.

Fresh attention extraction requires the saved fine-tuned checkpoint, matching tokenizer, dataset and checkpoint-specific manifest. Those weights and clinical inputs are not bundled. Replotting released aggregate figures is independently supported.

Historical queue supervisors in `scripts/run_*.py` coordinate specific original run layouts and process states. Use direct trainer modules for a new machine; do not assume a historical queue is a general-purpose workflow. The two supervisors that pause or replace other queues (`run_kip_checkpoint_attention.py` and `run_qwen_oc_clinical_audits.py`) have disabled CLI entry points in this release; their source remains for review. No training is started by the quick start.

## 6. Separate ordered-completion and mechanism package

Run the nested package in a separate process/environment:

```bash
cd repro
python -m pip install -e '.[dev]'
export DATA_DIR="$(cd .. && pwd)/data"
export RESULTS_DIR="$(cd .. && pwd)/runs/repro"
python -m pytest -q tests/test_oc_completion.py tests/test_ordered_metrics.py
```

To generate small new synthetic samples without overwriting the supplied data, set `DATA_DIR` to a new directory under `runs/` before invoking `python -m src.oc_completion.gen_datasets --smoke --n_workers 1`. See [repro/README.md](repro/README.md) for runnable package examples. [repro/REPRODUCING.md](repro/REPRODUCING.md) preserves historical workflow notes; it is not a complete executable checklist for this snapshot. The scheduler templates it cites under `repro/scripts/slurm/` are institution-neutral and need your own `--account`, container image and paths before submission; the one mechanism-audit wrapper it mentions is not distributed. Ordered-completion modules under `repro/src/oc_completion/` locate this outer directory through `RELEASE_ROOT` (default: the parent of `repro/`), and legacy benchmark helpers use `REPRO_ROOT` (default: `repro/`); override them only if the package is moved relative to its data. New generator output is not automatically byte-identical to the canonical exported datasets.

## 7. Tests and limits

```bash
python -m pytest -q tests
```

Run nested tests separately from `repro/`. Some integration tests require CUDA, optional dependencies, or external tokenizers/data; skips and the exact executed suites are recorded in [release/VALIDATION.md](release/VALIDATION.md), generated at packaging time. Compilation, CPU checks and small training validation were performed for packaging. The expensive GPU experiments were not rerun to build this release.

Clinical preparation code is included for review, with public diagnosis mappings. Patient data and row-level clinical artifacts must be obtained and kept in your own authorized environment. Source for additional historical pipelines is included even when this release does not contain a complete evaluated run for that pipeline.

Legacy modules: a few historical top-level modules kept for review (`src/analysis/data_exploration_top.py` and related exploration scripts, `src/data/prompts.py`, `src/analysis/complexity_graph.py`, `src/models/training_time.py`) import `icdmappings`, `num2words`, `seaborn` or `wandb`. These are optional dependencies of those legacy modules only; none of the entry points documented above needs them, and none of the requirement files above installs them. They are listed separately in [requirements-legacy.txt](requirements-legacy.txt).
