# MIMIC-IV AKI trajectory audit

This extension is a retrospective, serum-creatinine-only audit of whether a
model uses temporal order. It is not a bedside AKI detector and is not a fully
ADQI- or KDIGO-compliant clinical phenotype: urine output, dialysis, relative
creatinine criteria, and pre-admission longitudinal baselines are deliberately
outside the locked primary estimand.

## Locked analyses

- MIMIC-IV `labevents` item 50912 is validated against `d_labitems`; `charttime`
  is treated as specimen time and values are converted to mg/dL.
- Episodes cannot cross a hospital admission. AKI onset uses only an absolute
  rise of at least 0.3 mg/dL within 48 hours.
- Recovery is defined relative to the configured episode baseline. The locked
  primary timing is recovery within 48 hours; recovery must remain sustained
  for 48 hours.
- `transient` means early sustained recovery without recurrence through the
  configured follow-up; `persistent` means no early sustained recovery; and
  `relapsing` means sustained recovery followed by another qualifying rise in
  the relapse window. Relapse has precedence.
- Censored and ambiguous trajectories are retained in audit tables but never
  silently enter a model. Administrative discharge/death attribution is stored
  separately from the state-machine censor reason.
- The confirmatory analysis is matched transient versus persistent. Matching
  is coarsened-exact, performed within the fixed patient split, and must include
  baseline, peak, duration, and observation count (with any additional
  protocol-selected invariant properties). A failed pair-count or balance criterion
  marks the analysis non-estimable and stops model training.
- The three-class transient/persistent/relapsing task is secondary. Its class
  counts and exclusions are headline outputs, which is particularly important
  when relapsing episodes are rare.
- One patient split is reused everywhere. The split seed is fixed; exactly
  three model seeds are paired with exactly three permutation seeds.

The implementation calls this an “ADQI-inspired SCr-only phenotype,” not an
ADQI definition. ADQI 16 motivates treating persistence beyond the acute event
explicitly. The Heung recovery analysis motivates distinguishing recovery
within two days from later recovery, but does not validate the relapsing label.
Accordingly, a separate peak-anchored sensitivity table reports fast,
intermediate, no-recovery-by-horizon, or censored recovery timing using its own
fully explicit configuration; it never replaces the primary labels.

## Order controls

Each sequence row contains creatinine plus the explicitly configured time
channels. Shuffling permutes values within an episode while retaining the exact
timestamp grid, value multiset, duplicate multiplicities, and observation
count. Reversal also moves only the values. Sequence truncation is applied once
before either perturbation. Invariant, count-only, and positional controls use
the complete configured episode window, so count means the full-window count,
not a neural padding/truncation cap. Every value-based control receives the
same configured raw/baseline-relative/log transform as the sequence models.

Episode duplicate resolution is canonicalized once by the state machine; the
same resolved measurements feed recovery sensitivity, matching, and models.
This prevents labels and model inputs from seeing different simultaneous
values.

The experiment reports:

- ordered-trained LSTM/Transformer on ordered test sequences;
- the same fitted model on shuffled and time-reversed test values;
- an independently trained-and-tested shuffled sequence model;
- logistic regression and XGBoost on invariant summaries;
- count-only logistic regression;
- first value, last value, first-to-last change, slope, and combined positional
  logistic baselines; and
- the deterministic episode state machine as an oracle ceiling.

The oracle is rerun on the configured model window. If that window omits the
baseline or follow-up evidence needed to reconstruct a label, it is skipped as
an explicit non-estimable condition in `condition_estimability`, while label
disagreement and other integrity failures still abort the run. Ordered-model
contrasts are patient-paired against shuffled, reversed, logistic invariant,
count-only, and XGBoost invariant controls.

Evidence for an order effect requires ordered performance to exceed shuffled
performance in the matched cohort. Shuffled performance alone is interpreted
beside invariant-summary and count-only performance: high shuffled performance
can reflect label-separable value multisets rather than learned chronology.

## Protocol configuration

Copy [`configs/mimic_aki.example.yaml`](../configs/mimic_aki.example.yaml) and
replace every `null`. The template is deliberately invalid as committed. The
loader rejects it before accessing MIMIC, ensuring the code does not invent any
of the choices that remain study-specific:

- adult threshold and boundary;
- eligible admission types, stay boundary, and admission linking;
- discharge-versus-death administrative endpoint handling and malformed death
  timestamps;
- measurement-count basis and missing specimen handling;
- numeric source, accepted unit aliases/conversions, missing-unit handling,
  valid physiologic range, and every duplicate policy;
- baseline estimator/evidence, onset and peak tie handling, peak/follow-up
  boundaries, relapse anchor/comparator, observation-density and censor
  thresholds;
- representation window boundaries, truncation, time channels, transforms,
  and invariant summaries;
- exact matching bins, balance limit, minimum pairs, split fractions, and all
  seeds; and
- preprocessing, model hyperparameters, class weighting, thresholding,
missing-class behavior, and bootstrap behavior.

Unknown keys at every fixed-schema nesting level are rejected, and a validated
configuration is recursively immutable so its hash cannot become stale during
a run. Explicit class-weight maps must cover all three labels; the binary task
selects its two entries from that shared map.

`metrics.binary.labels` should be `[transient, persistent]`, while
`metrics.multiclass.labels` should be
`[transient, persistent, relapsing]`, in the intended probability-column order.

Validate a completed protocol with:

```bash
python -m src.mimic.aki validate-config --config path/to/protocol.yaml
```

## Running

The raw directory must contain `labevents.csv[.gz]`, `d_labitems.csv[.gz]`,
`admissions.csv[.gz]`, and `patients.csv[.gz]`.

Run the complete pipeline:

```bash
python -m src.mimic.aki run \
  --config path/to/protocol.yaml \
  --raw-dir data/raw/mimic-iv \
  --output-dir results/mimic/aki_audit/run_name
```

Or run stages independently:

```bash
python -m src.mimic.aki extract \
  --config path/to/protocol.yaml \
  --raw-dir data/raw/mimic-iv \
  --output-dir data/processed/mimic_aki/cohort

python -m src.mimic.aki prepare \
  --config path/to/protocol.yaml \
  --measurements data/processed/mimic_aki/cohort/aki_creatinine_measurements.parquet \
  --admissions data/processed/mimic_aki/cohort/aki_admission_audit.parquet \
  --output-dir data/processed/mimic_aki/prepared \
  --workers 16 \
  --patient-chunk-size 128

python -m src.mimic.aki train \
  --config path/to/protocol.yaml \
  --prepared-dir data/processed/mimic_aki/prepared \
  --output-dir results/mimic/aki_audit/run_name/experiment
```

`prepare --workers` and `--patient-chunk-size` are runtime-only execution
controls. They do not alter the protocol or its configuration hash. The
single-worker default preserves the original serial path; parallel chunks are
collected in sorted patient order so the persisted tables remain deterministic.

Extraction scans the large CSV once in chunks and stages the selected rows as
Parquet. Full rejected-measurement, admission-flow, unit, episode-decision,
censor, matching, split, prediction, per-seed metric, aggregate metric, and
order-contrast artifacts are retained. Existing artifacts are not overwritten
unless `extract --overwrite` is explicitly requested.

The repository's existing visit-sequence LSTM/Transformer code is organized as
categorical-vocabulary experiment scripts (including script-local classes in
`src/mimic/visit_shuffle_test.py`) and cannot be imported safely for continuous
creatinine channels. `src/mimic/aki/models.py` therefore adapts those existing
encoder, padding/masking, deterministic-training, and ordered/shuffled patterns
to numeric `[batch, time, feature]` inputs rather than duplicating the
categorical token interface.

## References

- [MIMIC-IV `labevents` documentation](https://mimic.mit.edu/docs/IV/modules/hosp/labevents.html)
- [ADQI 16 consensus report](https://www.nature.com/articles/nrneph.2017.2)
- [Heung et al., recovery pattern and subsequent CKD risk](https://pmc.ncbi.nlm.nih.gov/articles/PMC6837804/)
- [Dahlem et al., predictability bounds of electronic health records](https://www.nature.com/articles/srep11865)
- [ICHI 2016 temporal versus atemporal EHR sequence study](https://doi.org/10.1109/ICHI.2016.64)
