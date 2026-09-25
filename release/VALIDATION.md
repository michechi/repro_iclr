# Validation record

Checks executed on the assembled snapshot, on the packaging machine, on
2026-09-25. All commands ran with `CUDA_VISIBLE_DEVICES=''`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `PYTHONDONTWRITEBYTECODE=1`;
no model weights were downloaded, no GPU was used, no clinical record was read,
and no paper experiment was rerun. Test suites ran on a scratch copy of the
release so that no run output entered the packaged tree.

## Structural checks

| Check | Command | Outcome |
| --- | --- | --- |
| Release verifier | `python -B scripts/verify_release.py --require-manifest` | status `passed`; every packaged file is covered by `MANIFEST.sha256` (933 entries), no forbidden artifact, link, credential pattern, private path, non-anonymous author label or clinical patient-level field detected |
| Dataset integrity | `python -B scripts/check_datasets.py` | passed: all 1,000,000 KIP rows (M4/M6, train/val/test) verified against the recovery manifest hashes, exact-once keys and inversion-parity labels; all 500,000 corrected OC rows verified against the canonical rule; observed noise rates 0.300155 / 0.30202 / 0.3011 |
| Figure regeneration | `python -B scripts/reproduce_figures.py` | wrote the OC, KIP-M4 and CKD attention figures (PDF/PNG/SVG) and `figure_inputs.json` from the bundled aggregate CSVs in about five seconds |
| Python syntax | `compile()` over every `.py` file | 345 files parse; no `__pycache__`, `.pyc`, `.DS_Store`, symlink or cache directory present |
| Manuscript snapshot | label/reference scan of `paper/manuscript_source.tex` | 87 labels, 54 references, none undefined; no comment lines remain; seven citation keys are still absent from `references.bib` (listed in `docs/MANUSCRIPT_GAPS.md`) |

## Test suites

From the release root with `PYTHONPATH=.` (the root `pyproject.toml` adds `src`):

```bash
python -B -m pytest -q -p no:cacheprovider tests
# 391 passed, 3 skipped, 72 subtests passed
```

The three skips are environment-dependent: two encoder tokenizer checks that
need `ICLR_TEST_BERT_TOKENIZER`, and the bitsandbytes NF4 deserializer check
in `tests/test_kip_llama70_models.py` (bitsandbytes was not installed).

From `repro/` with `PYTHONPATH=.` and `DATA_DIR=../data`:

```bash
python -B -m pytest -q -p no:cacheprovider tests
# 182 passed
```

CPU smoke training (`src.experiments.kip_rnn_transformer_training --tag kip_m4
--seed 42 --device cpu --smoke`) completed one epoch on the 256/512/512-row
subset and wrote its metrics and manifest. Smoke metrics are not paper results.

## Sanitation checks

- Identity, venue and hostname denylist grep over every text file and every
  file name: zero hits after the fix pass.
- PNG/PDF/SVG/parquet metadata parsed: only plotting-software and anonymous
  creator fields.
- 47 translated legacy modules compared with their originals by a masked-AST
  check (`ast.dump` with string constants replaced): structure identical in
  every file; only comments, docstrings and natural-language strings differ.
- Legacy modules that execute processing on import (for example
  `src/data/tabular_evo.py`, `src/data/splitting_data.py`) and modules that
  need optional legacy packages (for example `num2words` in `src/data/prompts.py`)
  were not imported; this is documented in REPRODUCING.md.

## Environment

Python 3.12.3 on Linux; PyTorch 2.13.0, Transformers 5.14.1, PEFT 0.19.1,
Accelerate 1.14.0, NumPy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0, XGBoost 3.3.0,
pyarrow 25.0.1, Matplotlib 3.11.1; pytest 9.1.1. See ENVIRONMENT.md for the
original assembly environment. These are observations, not a lockfile.

## Not validated

LaTeX compilation (no TeX installation), GPU training, gated model downloads,
quantized inference, clinical cohort reconstruction, and replay of historical
run outputs.
