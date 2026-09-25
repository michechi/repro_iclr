# Sequence benchmark reproduction package

This package contains the synthetic sequence generators, mechanism audits,
ordered-completion experiments, recommendation audit, clinical cohort code,
and tests used by the anonymous submission. It is a separate Python import
root from the top-level `src` package: run these commands from `ICLR/repro`.
Do not combine both `src` packages in one Python process.

```bash
python -m pip install -e '.[dev]'
export DATA_DIR="$(cd .. && pwd)/data"
export RESULTS_DIR="$(cd .. && pwd)/runs/repro"
python -m pytest -q tests/test_oc_completion.py tests/test_ordered_metrics.py
```

Only source is shipped here. Synthetic generators can create the inputs;
clinical data and pretrained model weights require separate authorized access.
Set `HF_TOKEN` in your environment when a gated model requires authentication.
The environment file is an optional template; the code does not automatically
load `.env`.

For a small CPU synthetic-data generation run (no model training):

```bash
export DATA_DIR="$(cd .. && pwd)/runs/repro/data"
export RESULTS_DIR="$(cd .. && pwd)/runs/repro"
python -m src.oc_completion.gen_datasets --smoke --n_workers 1
python -m src.oc_completion.ordered_data --smoke --n-workers 1 \
  --results-root "$RESULTS_DIR/oc_hole_audit"
```

`RELEASE_ROOT` defaults to the outer `ICLR` directory for ordered-completion
code; `REPRO_ROOT` defaults to this package for legacy benchmark code. Set
`DATA_DIR`, `RESULTS_DIR`, `CACHE_DIR`, and `HF_HUB_CACHE` explicitly when sharing
artifacts between them. These overrides are module-specific: several legacy
trainers require their `--path_csv` and `--output_dir` arguments instead, and
some OC trainers derive output paths from `RELEASE_ROOT`. Check the selected
module before running it; keep new outputs under `runs/`.
Clinical overrides are defined in `src/common.py`
(`MIMIC_RAW`, `MIMIC_PROCESSED`, `MIMIC_TRAINING`, `MIMIC_CODES`, `MIMIC_RESULTS`).

See [CODE_PORTABILITY.md](../docs/CODE_PORTABILITY.md) for copied-code scope and
limitations, and the outer repository documentation for current paper recipes.
Historical `_6` labels do not implement the canonical ordered-composition
rule; `src.oc_completion.gen_datasets` regenerates the audited OC tasks. The
legacy Parity generator uses the hidden subset `{W,D,Q,J,X,N}`. Preserve these
distinctions when comparing old artifacts and new runs.

The original [MIT license](LICENSE) is retained.
