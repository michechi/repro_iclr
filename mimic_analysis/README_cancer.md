# MIMIC-IV cancer trajectory preprocessing pipeline

Implements the preprocessing contract in [`docs/mimic_cancer.md`](docs/mimic_cancer.md)
end-to-end. Produces two datasets from MIMIC-IV + MIMIC-IV-Note using the
DFCI-imaging-student model:

- **Dataset A** — strict order-only cancer trajectory task (`y_order`)
- **Dataset B** — 1-year mortality anchored on the last cancer-related radiology
  admission (`y_death_365`)

## One-time setup

```bash
# Install system prereqs (Ubuntu 24.04)
sudo apt-get install -y python3-venv python3-pip

# Create venv
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install -U pip wheel

# Install torch CPU build FIRST (from the pytorch index)
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install the rest
cd ./mimic_analysis
pip install -r requirements.txt

# Pre-cache bert-base-uncased (required by LabeledModel.__init__)
python -c "from transformers import AutoTokenizer, AutoModel; \
  AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side='left'); \
  AutoModel.from_pretrained('bert-base-uncased')"
```

After the initial cache warm-up you can run inference fully offline by
exporting `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`.

## Raw data layout expected

Paths in `configs/paths.yaml` resolve relative to `mimic_analysis/`. Override a
value with `MIMIC_CANCER_<UPPERCASE_KEY>`, such as `MIMIC_CANCER_MIMIC_HOSP`, or
edit the YAML if your authorized local data layout differs:

```
data/hosp/{patients,admissions,diagnoses_icd,d_icd_diagnoses}.csv.gz
data/mimiciv_note/physionet.org/files/mimic-iv-note/2.2/note/{radiology,radiology_detail}.csv.gz
data/dfci_models/physionet.org/files/dfci-cancer-outcomes-ehr/1.0.0/models/DFCI-student-imaging/dfci-student-imaging.pt
```

## Pipeline

Each script is idempotent and writes to `data/intermediate/` or
`data/processed/`. Run them in order from the `mimic_analysis/` directory
with the venv activated:

```bash
cd ./mimic_analysis
source ../.venv/bin/activate

python scripts/00_validate_inputs.py         # ~30s, loads DFCI model and runs 1 forward pass
python scripts/01_build_base_tables.py       # minutes — streams radiology metadata only
python scripts/02_prefilter_candidate_notes.py  # tens of minutes — streams 746 MB radiology.csv.gz
python scripts/03_run_radiology_model.py     # HOURS on CPU; minutes with a GPU
python scripts/04_build_note_sequences.py    # minutes
python scripts/05_make_order_task.py         # seconds
python scripts/06_make_mortality_task.py     # seconds
python scripts/07_make_baselines_features.py # seconds
python scripts/08_qc_reports.py              # seconds; exits non-zero on QC failure
```

Running on GPU: pass `--device cuda` to `03_run_radiology_model.py` (the rest
are CPU-only). Batch size defaults to 16; increase on GPU.

## Smoke test on a slice

Each data-processing script accepts `--max-notes N` to cap work to N notes.
End-to-end dry run:

```bash
python scripts/01_build_base_tables.py        --max-notes 5000
python scripts/02_prefilter_candidate_notes.py --max-notes 500
python scripts/03_run_radiology_model.py      --max-notes 500
python scripts/04_build_note_sequences.py
python scripts/05_make_order_task.py
python scripts/06_make_mortality_task.py
python scripts/07_make_baselines_features.py
python scripts/08_qc_reports.py
```

This completes in a few minutes on CPU and validates the full wiring.

## Outputs

See `docs/mimic_cancer.md` §12 for the full list. Highlights:

| Path | Purpose |
|------|---------|
| `data/intermediate/sequences/note_level_trajectories.parquet` | One row per cancer-positive note with 10 multi-hot labels + probs + event_time |
| `data/processed/dataset_order_only/labels.parquet` | Patient-level Dataset A labels with `y_order`, `t_prog1`, `t_resp1`, `t_cut`, `split_order_task` |
| `data/processed/dataset_order_only/X_seq_{context,core}.parquet` | Ordered sequence representations for Dataset A |
| `data/processed/dataset_order_only/X_bag_context.parquet` | Bag-of-events baseline for Dataset A |
| `data/processed/dataset_mortality365/labels.parquet` | Patient-level Dataset B labels with `y_death_365`, `t0`, `hadm_last`, `split_mortality_task` |
| `data/processed/dataset_mortality365/X_seq.parquet` | Ordered sequence representation for Dataset B |
| `data/intermediate/qc/preprocessing_report.json` | Counts + sanity + leakage + addenda summary |

Raw radiology text never appears in any processed artifact. The only place
text lives on disk is `data/intermediate/notes/candidate_radiology_notes_with_text.parquet`,
which is gitignored and used only as input to step 3.

## Design notes (the things that will bite you)

- The DFCI imaging model's **logit-2 head is "progression-OR-mixed-response"**
  per RECIST terminology (the reference script renames it
  `progression_or_mixed_logit`). We still expose it as
  `prob_progression`/`flag_progression` per the spec's column names.
- Model output order is `any_cancer, response, progression, brain, bone,
  adrenal, liver, lung, node, peritoneum` — **not** the order listed in the
  spec section 4. `LABEL_ORDER` in `src/mimic_cancer/dfci_imaging_model.py`
  is the authoritative mapping; do not bind by position anywhere else.
- `peritoneal_head` is the state-dict key; do not rename to `peritoneum_head`.
- ICD codes in MIMIC are space-padded, e.g., `'140  '`. Always `.strip()`.
- Addendum handling uses the `parent_note_id` / `addendum_note_id` rows in
  `radiology_detail`. Child flags are OR'd into the parent; probs take the
  element-wise max; orphan addenda (parent not in candidate set) are kept as
  standalone rows with `is_orphan_addendum=1`.
- Dataset B filters `event_time <= dischtime` per admission **before**
  picking `hadm_last`, which prevents a leakage circularity when notes are
  finalized after discharge.
- `t_prog1 == t_resp1` patients are excluded from Dataset A — this covers
  the real case where the same radiology note has both flags set.
- Splits are patient-level, stratified, seeded (2026), 70/15/15. Dataset A
  and Dataset B have separate strata, so their splits differ.
- The spec's M2 memory budget does not apply on a 125 GiB box, but the
  pipeline still streams radiology.csv.gz in 50k-row chunks and uses
  `pyarrow.parquet.ParquetWriter` to stream row groups into output files.
