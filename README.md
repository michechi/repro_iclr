# Anonymous sequence-learning benchmark

Code, canonical synthetic data, aggregate experiment results, and attention figures for reviewing experiments on ordered compliance (OC), global parity counting (GPC), key-inversion parity (KIP-M4/M6), and MIMIC-IV CKD progression to ESRD.

This is a self-contained release snapshot. Start with the measured results in [RESULTS_INVENTORY.md](docs/RESULTS_INVENTORY.md). The included manuscript is a working draft: several table entries remain explicitly simulated or unsupported by the bundled evidence, and some manuscript assets are missing. [MANUSCRIPT_GAPS.md](docs/MANUSCRIPT_GAPS.md) identifies those items.

## Quick start

Use Python 3.12 in a fresh virtual environment. From this directory:

```bash
python -B scripts/verify_release.py --require-manifest
python -m venv ../iclr-venv
source ../iclr-venv/bin/activate
python -m pip install -r requirements.txt
python -B scripts/check_datasets.py
python -B scripts/reproduce_figures.py
```

`MANIFEST.sha256`, read by the first command from this directory, is generated at packaging time as the final step. The last command writes the three attention figures to `runs/figures/`, without training or downloading a model. The released PDF, PNG and SVG versions are in [paper/figures/](paper/figures/). Dataset checks verify every KIP row and every corrected OC clean label.

For training, testing, data preparation, and model access, see [REPRODUCING.md](REPRODUCING.md). A small CPU training example is included. GPU training and clinical reanalysis require additional dependencies and inputs.

## Contents

| Directory | Contents |
| --- | --- |
| `src/` | Classification training, KIP experiments, attention audits, CKD and AKI pipeline code |
| `repro/` | Separate import root for synthetic generators, mechanism audits and ordered completion |
| `data/` | Canonical synthetic datasets and public diagnosis-code mappings |
| `codes/` | Public CCS diagnosis-category mappings (`CCS_DX_mapping.csv`, `CCS_DX_categories.csv`) used by the clinical pipelines; no patient records |
| `results/` | Selected aggregate results; corrected synthetic OC CSV exports remain at their historical path |
| `paper/` | Attention appendix, standalone preview source, figures and working manuscript source |
| `paper_tables/` | Historical aggregate tables; interpretation in the results inventory |
| `scripts/` | Release checks, plotting, scientific audits and historical experiment runners |
| `tests/`, `repro/tests/` | Numerical, data-integrity and model-interface tests |
| `configs/`, `simulation/`, `mimic_analysis/` | Additional pipeline configuration and source |
| `docs/`, `release/` | Protocols, evidence inventory, limitations and validation records |

`src/` and `repro/src/` are distinct Python packages with the same name. Run their commands from their respective directories in separate processes. Do not install both as editable packages in one environment. Ordered-completion modules under `repro/src/` locate this outer directory through the `RELEASE_ROOT` environment variable (default: the parent of `repro/`); legacy helpers in `repro/` use `REPRO_ROOT`. See [repro/README.md](repro/README.md).

## Data and model access

Synthetic data are included. Clinical patient data, patient-level predictions, per-patient attention exports, credentials and model weights are excluded. CKD figures use aggregate statistics only. Clinical reruns require your own authorized MIMIC-IV access and cohort preparation. Obtain pretrained model weights from their respective providers and observe their access terms; gated models may require a locally supplied `HF_TOKEN`.

Preserved model identifiers follow the recorded artifacts, including Qwen3.8-27B and Qwen2.5-14B. Their precise revisions and recipes are recorded in the experiment documents. The quick-start commands do not download model weights.

## Evidence and release status

The results include completed KIP data-efficiency experiments, M4/M6 RNNTransformer repeats, OC/CKD training and attention audits, and additional historical benchmark aggregates. Missing runs are not interpolated. Large-model KIP sweeps stopped after success at 200K under a test-informed budget rule; the larger fractions were not measured. The requested Llama-70B job was cancelled before training and has no completed result here.

Use [RESULTS_INVENTORY.md](docs/RESULTS_INVENTORY.md) to distinguish tasks, recipes, seeds, subsets and completion status. [ANONYMIZATION_REVIEW.md](docs/ANONYMIZATION_REVIEW.md) describes the export review; [VALIDATION.md](release/VALIDATION.md), generated at packaging time, records the checks actually run. Historical protocol notes describe original experiments and may reference private checkpoints or outputs that are intentionally absent.

The repository history starts at the anonymous release commit; no earlier history or author remotes are included. Keep local reruns in the ignored `runs/` directory and never add credentials, patient data, or model caches. See [LICENSE](LICENSE) for the scope of the retained license.
