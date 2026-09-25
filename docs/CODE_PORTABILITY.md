# Code scope, portability, and CPU validation

The release preserves the scientific source layouts and retains independent
top-level and nested `repro` Python import roots. No original working-tree
source was modified while preparing this copy.

## Included code

| Directory | Included files | Purpose |
|---|---:|---|
| `src/` | 110 Python files | Models, training recipes, clinical preprocessing, feature and attention audits |
| `repro/src/` | 90 Python files and one README | OC generators/oracle, ordered completion, mechanism audit, legacy reproduction code |
| `tests/` | 39 Python files | Main scientific and orchestration contracts |
| `repro/tests/` | 12 Python files | OC rules, convergence, encoding, metrics, checkpoint, and report contracts |
| `configs/` | Four YAML protocols | AKI scientific protocol variants |
| `simulation/` | 33 Python files, README, project metadata | Historical synthetic generators and diagnostics |
| `mimic_analysis/` | 29 source/configuration/documentation files | Cancer trajectory preprocessing and analysis |

The nested package also includes its project metadata, test import setup,
environment template, and original MIT license. The mechanism audit's Python
scripts are retained because the OC oracle and attention audits depend on
their rule implementation. The original tree has no `repro/configs/` directory.

This code-copy scope excludes patient data, model weights, checkpoints,
notebooks, caches, generated mechanism plots/tables, and institution-specific
scheduler wrappers. Release data/results and root orchestration scripts are
documented separately. No symlinks were copied.

## Changes made to the copy

- Machine-specific repository, model-cache, input, and output paths became
  relative defaults. Run legacy top-level modules from the release root;
  their existing CLI path arguments remain available.
- OC modules that formerly hardcoded the repository root now derive it from
  their file location, with `RELEASE_ROOT` still available as an override.
- Personal author annotations became `Anonymous`; private account and
  filesystem names were removed. Third-party model identifiers and license
  notices remain intact.
- Cancer-pipeline YAML paths now resolve relative to `mimic_analysis/`,
  independently of the caller's working directory. Every key supports a
  `MIMIC_CANCER_<UPPERCASE_KEY>` environment override, for example
  `MIMIC_CANCER_MIMIC_HOSP`.
- The nested package declares the PyArrow and PyYAML dependencies used by its
  code. Its metadata and README identify the anonymous release. The synthetic
  generator metadata now lists its existing imported dependencies.
- Two unambiguous syntax errors in legacy source were repaired: concatenated
  assignments in `src/data/sript_text_generation.py`, and a duplicate comma
  in `src/experiments/XGBoost_4_seq.py`.
- Three root test modules tied to omitted emergency/cancelled launchers were
  omitted: `test_qwen_oc_epoch_cap.py`, `test_qwen_oc_epoch_cap_recovery.py`,
  and `test_kip_llama70_launcher.py`. Seven nested test functions that inspect
  omitted institution-specific scheduler wrappers were removed. Their
  numerical convergence and recipe checks remain.
- Encoder tokenizer integration tests accept `ICLR_TEST_BERT_TOKENIZER` and
  `ICLR_TEST_ROBERTA_TOKENIZER` local directories. Without them, those checks
  skip rather than depend on a private historical result manifest.

- Second sanitation pass (release audit): the `host`/`hardware` columns of
  `results/matched_completion/training_results.csv` and
  `results/oc_hole_audit/baseline_results.csv` now read `redacted` (they held
  cloud-VM hostnames), and every ordered-completion trainer that recorded
  `os.uname().nodename` or `socket.gethostname()` now records
  `RESULT_HOST_LABEL` (default `redacted`) instead. The recipe column strings
  that named an earlier manuscript venue now read `paper appendix tab:dl_optimal`.
- Three files were renamed to remove a cluster or venue name from the path:
  `src/experiments/Finetuning_time_simulation_tiny.py`,
  `simulation/stat_test_ngram.py`, and
  `mimic_analysis/scripts/12_make_summary_figure.py` (its logger name and output
  file names now use the `summary_figure` stem). Nothing
  imports these modules.
- The override variable for the ordered-completion root is now `RELEASE_ROOT`
  (it previously carried the private project name). Module docstrings no longer
  name a private virtual environment, and the historical KIP/Qwen runners anchor
  their pinned model-cache paths to the release root instead of the working
  directory.
- Italian comments, docstrings, log messages and argparse help strings in 47
  legacy modules under `src/`, `simulation/` and `repro/src/generators/` were
  translated to English. Every translated file was checked against its original
  with an AST comparison that masks string constants: the program structure is
  identical, and only comments, docstrings and natural-language strings differ.
- A root `pyproject.toml` (pytest configuration only: `pythonpath = [".", "src"]`
  and the `slow` marker) was added so that `tests/test_aki_cohort.py` collects;
  `tests/test_kip_llama70_models.py` skips its bitsandbytes deserializer check
  when bitsandbytes is not installed instead of failing.
- Added evidence and templates: `results/mechanism_id_reference/report.md` and
  `plots/`, `repro/src/analysis/mechanism_id/results/*.csv`,
  `results/sensitivity/sensitivity_{BERT,DL}.md`,
  `results/mimic_legacy/ckd_diagnostic_summary.md`,
  `paper_tables/counterfactual_pairs_tricky_rnd.csv`, and the institution-neutral
  scheduler templates under `repro/scripts/slurm/`.
- `paper/manuscript_source.tex` is derived from the working manuscript by a
  private script that drops comment-only lines, blanks inline comments (keeping
  the `%`), points `\graphicspath` at `figures/`, and removes the placeholder
  repository footnote.

Scientific rules, model settings, cohort criteria, random seeds, split logic,
metrics, and numerical protocol values were retained. The H200 AKI protocol
is named `configs/mimic_aki_h200_smd015_amendment_v3.yaml`; the corresponding
integrity-gate test passes against that file.

## Working directories and paths

The main tree and nested package both define a package named `src`. Use
separate Python processes and working directories. Install their dependencies
in separate environments if installing both packages in editable mode.

From the release root, use `python -m src.<module>`. For the nested package:

```bash
cd repro
export DATA_DIR="$(cd .. && pwd)/data"
export RESULTS_DIR="$(cd .. && pwd)/runs/repro"
python -m src.oc_completion.gen_datasets --help
```

`REPRO_ROOT` defaults to `repro/` for legacy benchmark helpers. `RELEASE_ROOT`
defaults to the outer release root for OC modules. Explicit `DATA_DIR` and
`RESULTS_DIR` avoid ambiguity when sharing artifacts in modules that honor
them. Several legacy trainers use `--path_csv` and `--output_dir` instead;
some OC trainers derive results/checkpoint paths directly from `RELEASE_ROOT`.
Inspect the selected module's CLI and path constants and use a new location
under `runs/` for reruns. `CACHE_DIR` and `HF_HUB_CACHE` control caches where
supported; legacy programs also expose their original `--cache_dir` argument.

Clinical inputs and model weights must be supplied separately. CKD helper
overrides are defined in `repro/src/common.py`; AKI commands expose explicit
input/output paths. The `.env.example` file is a template, not an automatically
loaded configuration file.

## CPU checks performed

All checks below used CPU execution with GPU visibility disabled. No paper
experiment was launched and no real patient records were processed. Some
tests construct tiny synthetic fixtures, optimizer states, or baseline fits
as part of their existing contracts. Offline settings prevented model downloads.

From the release root, equivalent commands are:

```bash
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export PYTHONPATH=.

python -B -m pytest -q -p no:cacheprovider \
  tests/mimic/aki tests/test_aki_cohort.py -m 'not slow'
# 65 passed, 3 deselected

python -B -m pytest -q -p no:cacheprovider \
  tests/test_qwen_fewshot_eval.py tests/test_qwen38_fewshot_eval.py \
  tests/test_qwen_thinking_generation.py tests/test_qwen_thinking_fewshot_eval.py \
  tests/test_compare_qwen_fewshot.py tests/test_summarize_qwen_thinking.py \
  tests/test_kip_cpu_offload.py tests/test_kip_efficiency_models.py
# 102 passed

python -B -m pytest -q -p no:cacheprovider \
  tests/test_encoder_oc_ckd_attention.py tests/test_kip_checkpoint_attention.py \
  tests/test_llama8_oc_ckd_attention.py tests/test_qwen_oc_clinical_audits.py \
  -k 'not tiny_real_encoders and not end_to_end and not replace and not resume_only and not stop'
# 39 passed, 1 optional-tokenizer skip, 6 deselected, 4 subtests passed
```

From `repro/`, with `DATA_DIR` pointing at the outer release's synthetic data:

```bash
export DATA_DIR=../data
python -B -m pytest -q -p no:cacheprovider \
  tests/test_oc_completion.py tests/test_ordered_metrics.py \
  tests/test_ordered_data.py tests/test_oc_strict_pairs.py
# 32 passing checks: 31 passed initially; the stored-label check then passed
# separately after setting DATA_DIR=../data.

python -B -m pytest -q -p no:cacheprovider \
  tests/test_oc_scratch_encoding.py tests/test_general_convergence.py \
  tests/test_baseline_convergence.py tests/test_bert_convergence.py \
  tests/test_ordered_report.py
# 134 passed
```

Across these disjoint selections: **372 passed**, one optional-tokenizer skip,
nine deselected integration checks. Collection also succeeded for all 394
root and 182 nested tests; the whole collected suite was not executed.
All 307 Python files in the copied source/test trees parsed successfully.
Cancer-path default resolution and its environment override were checked
directly without opening clinical files.

Validation used Python 3.12.3, pytest 9.1.1, NumPy 2.5.1, pandas 3.0.5, and
scikit-learn 1.9.0. The initial CPU-only dependency environment used
PyTorch 2.12.1 and PyArrow 25.0.0. Transformer contract checks used PyTorch
2.13.0, Transformers 5.14.1, PEFT 0.19.1, Accelerate 1.14.0, and PyArrow
25.0.1, with pytest supplied from the existing CPU environment. These are
observed validation versions, not a claim that all dependency versions in the
broad original minimum-version ranges have been tested.

## Remaining limitations

Legacy exploratory scripts can execute processing immediately on import;
inspect them before running and prefer documented `python -m` entrypoints.
Full training, gated model downloads, GPU/quantization compatibility, clinical
cohort reconstruction, and historical artifact replay were not rerun during
release preparation. Optional dependencies and authorized external datasets
remain necessary for those workflows. The source copies differ from the
original bytes after anonymization, so original-run source hashes describe
historical provenance and must not be treated as hashes of this release.

The older tag `_6` data and canonical OC regeneration are distinct protocols.
The legacy Parity generator uses `{W,D,Q,J,X,N}`; the canonical OC oracle uses
the ordered key set `{W,D,Q,J,X,U}`. Keep those identities explicit when
comparing historical reference outputs.
