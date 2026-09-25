# MIMIC-IV cancer trajectory preprocessing spec
## Real-data pipeline for the order-vs-semantics question

This document is the preprocessing contract for building real-world cancer-trajectory datasets from **MIMIC-IV + MIMIC-IV-Note** so we can quantify the gap between:

- **semantic / bag-like understanding** of the input, and
- **order-aware / trajectory-aware understanding** of the input.

The design is intentionally centered on **radiology-derived oncology trajectories**, because MIMIC-IV-Note contains large-scale radiology reports linked to MIMIC-IV, and there is already a public/shareable MIMIC-compatible cancer outcome extraction resource for **any cancer, progression, response, and metastases**.

This spec is written so the pipeline can turn raw MIMIC files into analysis-ready `X` and `y` artifacts.

---

## 1. Goal

Build **two datasets**.

### Dataset A: strict order-only cancer trajectory task
This is the primary dataset for the order-only question.

We want patients who have both:
- at least one **progression** note
- at least one **response** note

Then define:

- `y_order = 1` if the **first progression** happens **before** the **first response**
- `y_order = 0` if the **first response** happens **before** the **first progression**

To make this as close as possible to an **order-only** task, the **primary analytic subset** must satisfy:

- before the cutoff time `t_cut = max(t_first_progression, t_first_response)`,
- there is **exactly one progression-positive note** and **exactly one response-positive note**,
- and these two notes do **not** share the same timestamp.

That gives two classes with the same core semantic ingredients (`{progression, response}`), but opposite order.

### Dataset B: practical downstream task (1-year mortality)
This is the pragmatic complement.

For each patient trajectory, define:
- `t0 = discharge time of the admission containing the last included cancer-related radiology report`
- `y_death_365 = 1` if `dod <= t0 + 365 days`
- `y_death_365 = 0` otherwise

This answers the practical question: once we already have meaningful oncology events, how much incremental signal is gained by modeling their order?

---

## 2. Data sources and versions

### Required data
Use:

- **MIMIC-IV Clinical Database** (prefer latest available to you; currently v3.1 is the current PhysioNet release)
- **MIMIC-IV-Note** v2.2
- optional but recommended: the public PhysioNet release of **DFCI cancer outcome student models**

### Required raw files

#### From MIMIC-IV (`hosp/`)
- `patients.csv.gz`
- `admissions.csv.gz`
- `diagnoses_icd.csv.gz`
- `d_icd_diagnoses.csv.gz`

#### From MIMIC-IV-Note (`note/`)
- `radiology.csv.gz`
- `radiology_detail.csv.gz`
- optional only:
  - `discharge.csv.gz`
  - `discharge_detail.csv.gz`

### Recommended external model resource
Use the public PhysioNet release:
- **DFCI-imaging-student**
- optional: **DFCI-medonc-student**

Default to **radiology-only** for the main paper experiment.

---

## 3. Compliance, privacy, and repo hygiene

### Hard rules
- **Never send MIMIC text or rows to any third-party API or online LLM service.**
- All extraction and modeling must run **locally**.
- Do **not** commit raw notes, note snippets, or note-derived examples to git.
- Derived artifacts that may be committed should contain only:
  - IDs
  - timestamps
  - structured labels / scores
  - engineered features
  - no raw text

### Safe repo practice
- keep raw MIMIC data outside the repo
- put file paths in `.env` or config files ignored by git
- derived tables with note-level predictions should remain local unless policy permits

---

## 4. High-level design choices

### Why radiology first
Use **radiology reports** as the primary trajectory source.

Reasons:
- closer domain match to the public oncology extraction model
- more uniform note style than discharge summaries
- lower leakage risk than discharge summaries for prognosis tasks
- easier to anchor repeated cancer status updates over time

### Do not use discharge summaries as default predictors
Discharge summaries are optional for:
- cohort auditing
- sensitivity analysis
- secondary experiments

But **do not** use them as default predictors for Dataset B mortality, because they are retrospective summaries and can leak future information.

### Canonical time axis
For each note:
- primary event time = `charttime`
- fallback = `storetime`
- final tie-breaker = `note_id`

### Canonical sequence unit
The canonical sequence unit is a **note-timepoint**, not an exploded token.

Each note-timepoint carries a **multi-hot oncology event vector**:
- `any_cancer`
- `progression`
- `response`
- `met_brain`
- `met_bone`
- `met_liver`
- `met_adrenal`
- `met_lung`
- `met_lymph`
- `met_peritoneum`

You may also build an exploded long table later, but the primary sequence representation should remain **one row per note**.

---

## 5. Directory contract

Recommended local project layout:

```text
project_root/
  README.md
  docs/
    mimic_iv_cancer_preprocessing.md
  configs/
    paths.yaml
    cohort.yaml
    thresholds.yaml
  data_raw/                # gitignored
    mimiciv/
    mimiciv_note/
    dfci_models/
  data_intermediate/       # gitignored
    cohort/
    notes/
    predictions/
    sequences/
    splits/
  data_processed/          # gitignored
    dataset_order_only/
    dataset_mortality365/
  scripts/
    00_validate_inputs.py
    01_build_base_tables.py
    02_prefilter_candidate_notes.py
    03_run_radiology_model.py
    04_build_note_sequences.py
    05_make_order_task.py
    06_make_mortality_task.py
    07_make_baselines_features.py
    08_qc_reports.py
```

---

## 6. Step-by-step preprocessing

## Step 0. Validate inputs
The pipeline should first assert that all required files exist.

At minimum, validate:
- MIMIC-IV hosp files exist
- MIMIC-IV-Note radiology files exist
- the DFCI imaging model weights or inference script exist

Fail fast if any are missing.

---

## Step 1. Load and normalize core tables

### 1.1 Load `patients`
Keep at least:
- `subject_id`
- `gender`
- `anchor_age`
- `anchor_year`
- `anchor_year_group`
- `dod`

### 1.2 Load `admissions`
Keep at least:
- `subject_id`
- `hadm_id`
- `admittime`
- `dischtime`
- `deathtime`
- `admission_type`
- `admission_location`
- `discharge_location`
- `insurance`
- `language`
- `marital_status`
- `race`

### 1.3 Load `diagnoses_icd`
Keep at least:
- `subject_id`
- `hadm_id`
- `seq_num`
- `icd_code`
- `icd_version`

### 1.4 Load `d_icd_diagnoses`
Keep at least:
- `icd_code`
- `icd_version`
- `long_title`

### 1.5 Load `radiology`
Keep at least:
- `note_id`
- `subject_id`
- `hadm_id`
- `note_type` if present
- `note_seq` if present
- `charttime`
- `storetime`
- `text`

### 1.6 Load `radiology_detail`
Keep all columns initially.
Later retain at least:
- `note_id`
- exam/procedure metadata
- addendum / parent-link fields if present

### 1.7 Datetime normalization
Parse all timestamps to timezone-naive pandas datetimes.

For each note, define:
- `event_time = charttime if not null else storetime`

Drop notes with both missing.

---

## Step 2. Build the adult inpatient base cohort

### Inclusion
Keep patients with:
- `anchor_age >= 18`
- radiology notes with non-null `hadm_id`

Rationale:
- adult-only makes the clinical story cleaner
- non-null `hadm_id` makes linkage to admissions, diagnoses, and discharge-based mortality much safer

### Exclusion
Drop:
- notes without `hadm_id`
- notes with missing `event_time`
- admissions with missing `dischtime` for tasks that require discharge-based endpoints

Output:
- `data_intermediate/cohort/base_adult_inpatient_notes.parquet`

---

## Step 3. Build a high-recall cancer candidate cohort

The point of this stage is **not** to be perfectly specific.
It is just to reduce unnecessary inference and create an oncology-relevant working set.

Use the union of the following seeds.

### 3.1 ICD seed (high recall)
Mark an admission as `icd_cancer_seed = 1` if **any** diagnosis code suggests neoplasm.

Recommended rule:
- ICD-10: codes starting with `C`
- ICD-10: codes starting with `D0`
- ICD-9: numeric codes in the broad neoplasm range `140`–`239`

Important:
- this is a **candidate seed only**
- it is allowed to include benign / uncertain / historical neoplasm codes
- final analytic inclusion will be driven by note-level cancer extraction

### 3.2 Radiology text seed (high recall)
Create `text_cancer_seed = 1` if the raw radiology text matches a broad oncology filter.

Default filter, mirroring the public cancer-outcome paper:
- text contains one of:
  - `cancer`
  - `restaging`
  - regex equivalent of `malignan*`

and also one modality-related cue from:
- `ct`
- `mr`
- `pet`
- `nm`
- `mammo`

Implementation notes:
- lowercase before matching
- use word-boundary-aware matching where possible
- keep the prefilter intentionally broad

### 3.3 Candidate patient / note set
Keep radiology notes where at least one of the following is true:
- the note itself satisfies `text_cancer_seed = 1`
- the linked admission has `icd_cancer_seed = 1`
- the patient has at least one admission with `icd_cancer_seed = 1`

Output:
- `candidate_radiology_notes.parquet`
- `candidate_patients.parquet`

---

## Step 4. Run the local cancer outcome extractor on radiology notes

### Default model
Use **DFCI-imaging-student** locally.

Expected outputs per note:
- `logit_any_cancer`
- `logit_progression`
- `logit_response`
- `logit_met_brain`
- `logit_met_bone`
- `logit_met_liver`
- `logit_met_adrenal`
- `logit_met_lung`
- `logit_met_lymph`
- `logit_met_peritoneum`

Also store:
- sigmoid probabilities for all 10 outputs

### Thresholding policy
Do **not** discard raw scores.

Store both:
- continuous model outputs
- binarized flags

Default binarization:
- `flag_label = 1` if `prob_label >= 0.5`

If the public model repo provides a preferred thresholding convention, preserve it as a separate column:
- `flag_label_repo_default`

### Required note-level output table
Create:

`data_intermediate/predictions/radiology_cancer_predictions.parquet`

with at least:

- `note_id`
- `subject_id`
- `hadm_id`
- `event_time`
- `charttime`
- `storetime`
- `prob_any_cancer`
- `prob_progression`
- `prob_response`
- `prob_met_brain`
- `prob_met_bone`
- `prob_met_liver`
- `prob_met_adrenal`
- `prob_met_lung`
- `prob_met_lymph`
- `prob_met_peritoneum`
- all corresponding `flag_*`
- optional note metadata such as modality / exam name
- **no raw text in the processed artifact**

---

## Step 5. Resolve note duplicates and addenda

Use `radiology_detail` to detect addenda / parent-child note relationships if available.

### Default rule
- if a note is clearly an addendum linked to a parent study:
  - merge it into the parent study record if the parent is present
  - otherwise keep it as its own record and mark `is_orphan_addendum = 1`

### Duplicate rule
If two notes share:
- same `subject_id`
- same `hadm_id`
- same `event_time`
- same normalized text hash

keep only one.

Store a QC table of removed duplicates.

---

## Step 6. Create the note-level oncology trajectory table

Create one row per **note-timepoint**.

### Keep only cancer-relevant notes
Default analytic filter:
- keep notes with `flag_any_cancer = 1`
- OR, for sensitivity analyses, keep notes with any positive oncology label among progression/response/metastasis flags

### Derived columns
For each retained note create:

- `subject_id`
- `hadm_id`
- `note_id`
- `event_time`
- `seq_time_rank` within patient
- `delta_days_from_prev_note`
- `admission_index` within patient
- `modality_group` if derivable
- multi-hot flags for all 10 labels
- raw probabilities for all 10 labels
- `n_positive_labels`
- `source = "radiology"`

### Ordering
Sort within patient by:
1. `event_time`
2. `storetime`
3. `note_id`

### Minimum trajectory requirement
For downstream datasets, require each patient to have:
- at least **2 retained cancer-positive radiology notes**

Output:
- `data_intermediate/sequences/note_level_trajectories.parquet`

---

## Step 7. Build Dataset A: strict order-only task

This is the key dataset.

### 7.1 Identify first progression and first response
For each patient:
- `t_prog1 = earliest event_time with flag_progression = 1`
- `t_resp1 = earliest event_time with flag_response = 1`

Exclude patients with:
- no progression-positive note
- no response-positive note
- `t_prog1 == t_resp1`

### 7.2 Define cutoff
Set:

- `t_cut = max(t_prog1, t_resp1)`

Then restrict the usable sequence to notes with:
- `event_time <= t_cut`

### 7.3 Primary strict subset
On the restricted sequence up to `t_cut`, require:

- exactly **one** progression-positive note
- exactly **one** response-positive note

If either count is not exactly one, exclude from the **primary strict subset**.

### 7.4 Label
Define:

- `y_order = 1` if `t_prog1 < t_resp1`
- `y_order = 0` if `t_resp1 < t_prog1`

### 7.5 Output representations
Create two versions.

#### A. Pure order-core representation
Use only the two note-timepoints corresponding to the first progression and first response.
This is the cleanest matched-semantics version.

Store:
- note-level vectors for those two notes
- `delta_days_between_them`

#### B. Order-with-context representation
Use the **entire note sequence up to `t_cut`**.
This is clinically richer but less purely matched.

### 7.6 Outputs
Create:

#### `dataset_order_only/labels.parquet`
Columns:
- `subject_id`
- `y_order`
- `t_prog1`
- `t_resp1`
- `t_cut`
- `n_notes_used_context`
- `n_progression_notes_up_to_cut`
- `n_response_notes_up_to_cut`
- `strict_primary_subset = 1`
- `split`

#### `dataset_order_only/X_seq_context.parquet`
One row per note in the retained context sequence:
- `subject_id`
- `seq_idx`
- `event_time`
- `delta_days_from_prev_note`
- 10 multi-hot labels
- optional modality covariates

#### `dataset_order_only/X_seq_core.parquet`
Exactly two rows per patient:
- first progression note
- first response note

#### `dataset_order_only/X_bag_context.parquet`
One row per patient with counts over the context window:
- count of each oncology label
- total retained notes
- mean / max probabilities per label
- last-note-only indicators if desired

Important:
`X_bag_context` is the baseline semantic/bag representation later used to compare against the sequence model.

---

## Step 8. Build Dataset B: 1-year mortality task

### 8.1 Observation window
For each patient trajectory, use all retained cancer-related radiology notes up to the chosen endpoint time.

Default endpoint construction:
- identify the **last retained cancer-related radiology note**
- let `hadm_last` be its linked admission
- let `t0 = dischtime(hadm_last)`

Exclude if:
- `hadm_last` missing
- `dischtime` missing

### 8.2 Label
Using `patients.dod`:

- `y_death_365 = 1` if `dod` is not null and `dod <= t0 + 365 days`
- `y_death_365 = 0` otherwise

Also store:
- `days_to_death_or_censor = min(365, (dod - t0).days)` if `dod` exists, else `365`

### 8.3 Required outputs

#### `dataset_mortality365/labels.parquet`
- `subject_id`
- `t0`
- `hadm_last`
- `y_death_365`
- `days_to_death_or_censor`
- `n_notes_used`
- `split`

#### `dataset_mortality365/X_seq.parquet`
One row per retained note up to `t0`:
- `subject_id`
- `seq_idx`
- `event_time`
- `delta_days_from_prev_note`
- 10 multi-hot labels
- optional modality covariates

#### `dataset_mortality365/X_bag.parquet`
Aggregated patient-level bag features:
- counts of each label
- total note count
- first/last note label indicators
- mean / max model probabilities per label
- number of unique admissions contributing notes
- observation window length in days

---

## Step 9. Patient-level splits

### Rule
All splits must be at the **patient (`subject_id`) level**.

No patient can appear in more than one split.

### Default split
Use:
- 70% train
- 15% validation
- 15% test

### Stratification
For Dataset A:
- stratify by `y_order`

For Dataset B:
- stratify by `y_death_365`

### Reproducibility
Use a fixed split seed, e.g.:
- `split_seed = 2026`

Persist the split mapping as:

`data_intermediate/splits/patient_splits.parquet`

Columns:
- `subject_id`
- `split_order_task`
- `split_mortality_task`

---

## Step 10. Baseline feature materialization

The preprocessing pipeline should materialize the baselines needed to quantify the bag-vs-order gap.

### 10.1 Bag / semantic baseline
For each dataset, create patient-level bag features:
- counts of each label
- total number of notes
- number of admissions with cancer-positive notes
- first-note label vector
- last-note label vector
- max probability per label
- mean probability per label

### 10.2 Last-event-only baseline
Create a patient-level table using only the last retained note:
- 10 binary flags
- 10 probabilities
- modality info
- time since previous note

### 10.3 Sequence representation
Create the ordered note table plus JSONL serialization.

Recommended JSONL format (illustrative placeholder values, not a real record):
```json
{
  "subject_id": 123,
  "times": ["2154-03-01 12:30:00", "2154-04-17 08:00:00"],
  "delta_days": [0.0, 46.8],
  "note_ids": ["123-RD-1", "123-RD-2"],
  "hadm_ids": [456, 456],
  "labels": [
    {"any_cancer": 1, "progression": 1, "response": 0, "met_liver": 0, "...": 0},
    {"any_cancer": 1, "progression": 0, "response": 1, "met_liver": 1, "...": 0}
  ],
  "y": 1
}
```

### 10.4 Optional shuffled-order control
Do not overwrite the true sequence.
Instead create a separate table containing a deterministic within-patient permutation index using a fixed seed.
This allows later creation of shuffled-order baselines without mutating the canonical data.

---

## Step 11. Quality-control checks

The pipeline must emit a QC report.

### Mandatory counts
Report:
- number of raw radiology notes
- number after adult/inpatient restriction
- number in candidate cancer note set
- number with model predictions
- number of retained cancer-positive notes
- number of patients with >=2 retained notes
- number entering Dataset A
- number entering strict Dataset A
- number entering Dataset B

### Mandatory sanity checks
Assert:
- no duplicate `note_id` in the final note-level sequence table
- no patient appears in more than one split
- note times are monotonic within patient after sorting
- all Dataset A labels are binary
- all Dataset B labels are binary
- no missing `subject_id`
- no missing `event_time` in sequence tables
- no missing `t0` in Dataset B labels

### Mandatory leakage checks
- Dataset A sequences must stop at `t_cut`
- Dataset B sequences must stop at `t0`
- discharge summaries must not appear in Dataset B default predictors
- no post-discharge notes may be included in Dataset B default predictors

---

## 12. Exact output files the pipeline should produce

### Shared intermediate files
- `data_intermediate/cohort/base_adult_inpatient_notes.parquet`
- `data_intermediate/cohort/candidate_patients.parquet`
- `data_intermediate/notes/candidate_radiology_notes.parquet`
- `data_intermediate/predictions/radiology_cancer_predictions.parquet`
- `data_intermediate/sequences/note_level_trajectories.parquet`
- `data_intermediate/splits/patient_splits.parquet`
- `data_intermediate/qc/preprocessing_report.json`

### Dataset A
- `data_processed/dataset_order_only/labels.parquet`
- `data_processed/dataset_order_only/X_seq_context.parquet`
- `data_processed/dataset_order_only/X_seq_core.parquet`
- `data_processed/dataset_order_only/X_bag_context.parquet`
- `data_processed/dataset_order_only/X_last_event.parquet`
- `data_processed/dataset_order_only/X_seq_context.jsonl`

### Dataset B
- `data_processed/dataset_mortality365/labels.parquet`
- `data_processed/dataset_mortality365/X_seq.parquet`
- `data_processed/dataset_mortality365/X_bag.parquet`
- `data_processed/dataset_mortality365/X_last_event.parquet`
- `data_processed/dataset_mortality365/X_seq.jsonl`

---

## 13. Important implementation notes

### 13.1 Keep raw model scores
Do not collapse everything to binary too early.
Later analyses may want:
- raw logits
- probabilities
- threshold sweeps
- confidence filtering

### 13.2 Preserve patient chronology
All MIMIC dates are shifted, but chronology within a patient is preserved.
Do not attempt to align different patients on absolute calendar time.
This is a within-patient trajectory analysis.

### 13.3 Prefer admission-linked notes
By default, keep only note rows with valid `hadm_id`.
This makes mortality anchoring and structured joins substantially cleaner.

### 13.4 Do not use raw note text in final modeling tables
Processed tables should contain:
- IDs
- times
- extracted labels / scores
- derived features

If text embeddings are needed later, compute them locally and store only the embedding vectors.

---

## 13.5 Memory-efficient processing (Apple Silicon M2 constraint)

This pipeline is designed to run on a **MacBook Air M2** (8–16 GB unified memory). Every script must respect tight RAM and VRAM budgets. The following rules apply throughout.

#### General memory rules
- **Never load an entire `.csv.gz` into memory at once** if it exceeds ~500 MB uncompressed. Use `pandas.read_csv(..., chunksize=50_000)` or `pyarrow` streaming reads instead.
- **Drop the `text` column as early as possible.** After extracting any text-based features (e.g., `text_cancer_seed`), immediately drop the raw text column before joining or persisting. The radiology text column alone can consume several GB.
- **Use categorical dtypes** for low-cardinality string columns (`gender`, `race`, `insurance`, `admission_type`, `note_type`). This typically reduces memory by 5–10×.
- **Downcast numeric columns** where safe: `int64` → `int32` or `int16` for IDs and counts; `float64` → `float32` for model probabilities and logits.
- **Write intermediate outputs to parquet eagerly** and release DataFrames with `del df; gc.collect()` before loading the next stage. Do not keep multiple large tables alive simultaneously.
- **Process patients in groups** if any patient-level operation requires holding the full note table. Use `groupby` with lazy iteration rather than `apply` on the full frame.

#### DFCI model inference rules
- Use **`device = "mps"`** (Apple Silicon GPU) if available via PyTorch ≥ 2.0. Fall back to `device = "cpu"` if MPS raises errors for the model architecture.
- Set **batch size = 8** as the default for the imaging model (BERT-base, 512 tokens). If running on 8 GB RAM, reduce to **batch size = 4**. Do not exceed batch size 16 even on 16 GB machines.
- **Stream inference**: load candidate notes in chunks (e.g., 5,000 notes at a time), run inference on each chunk, append predictions to a parquet file, then release the chunk. Never hold the full candidate note set + model + predictions in memory simultaneously.
- **Do not load the raw text into the inference loop.** Tokenize each batch, run forward pass, extract logits, discard input tensors immediately.
- After inference, **unload the model** (`del model; torch.mps.empty_cache()` or `gc.collect()`) before proceeding to downstream pandas work.

#### Disk I/O strategy
- Prefer **parquet** over CSV for all intermediate and output files. Parquet uses columnar compression and supports partial column reads, which is critical when RAM is limited.
- When reading parquet files downstream, use `columns=` parameter to load only the columns needed for the current step.
- Keep `data_intermediate/` and `data_processed/` on the **internal SSD** (not an external drive) for faster I/O.

#### Monitoring
- Each script should log peak memory usage (use `tracemalloc` or `resource.getrusage`) and warn if it exceeds 6 GB on an 8 GB machine or 12 GB on a 16 GB machine.
- If any single step exceeds the memory budget, the script should fail with a clear message indicating which table or operation caused the spike, rather than silently swapping to disk.

---

## 14. Minimal implementation task list

Implement exactly this sequence:

1. validate local paths and file presence
2. load and normalize MIMIC-IV + MIMIC-IV-Note tables
3. build adult inpatient base cohort
4. build ICD/text candidate cancer note set
5. run local DFCI-imaging-student inference on candidate radiology notes
6. build note-level oncology trajectory table
7. build strict Dataset A (`y_order`)
8. build Dataset B (`y_death_365`)
9. materialize:
   - sequence tables
   - bag features
   - last-event features
10. write QC report with exclusion counts and split checks

---

## 15. Nice-to-have sensitivity analyses (not required for first pass)

These are optional, but useful later.

### Sensitivity A
Repeat Dataset A with a **relaxed** cohort:
- allow multiple progression / response notes before `t_cut`
- label still based on first occurrence order

### Sensitivity B
Add discharge-summary-based oncology signals as context only, never as default prognosis predictors.

### Sensitivity C
Restrict radiology modalities more tightly using `radiology_detail` metadata instead of text heuristics alone.

### Sensitivity D
Restrict the cohort to admissions with both:
- neoplasm ICD seed
- at least one `any_cancer=1` radiology note

---

## 16. References used for this preprocessing design

1. MIMIC-IV Clinical Database (PhysioNet): https://physionet.org/content/mimiciv/3.1/
2. MIMIC-IV-Note (PhysioNet): https://physionet.org/content/mimic-iv-note/2.2/
3. DFCI cancer outcome extraction models on PhysioNet: https://physionet.org/content/dfci-cancer-outcomes-ehr/1.0.0/
4. Kehl et al., *Nature Communications* 2024: https://www.nature.com/articles/s41467-024-54071-x
5. PhysioNet guidance on MIMIC data with LLMs / online services: https://physionet.org/news/post/llm-responsible-use/

Key design choices borrowed from those sources:
- use of MIMIC-IV + MIMIC-IV-Note linkage
- local-only handling of credentialed data
- radiology-first cancer outcome extraction
- broad oncology text prefilter using terms like `cancer`, `restaging`, `malignan*` and modality cues such as `ct`, `mr`, `pet`, `nm`, `mammo`

---

## 17. One-line summary

If this document is followed, the result is:
- a **strict real-data order-only cancer trajectory dataset** for the order-only claim,
- and a **practical mortality dataset** to measure whether order helps once semantic oncology signals are already present.
