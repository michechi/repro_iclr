# Attention appendix replacement, 25 September 2026

Prepared the attention appendix released as `paper/attention_appendix.tex`
(wrapped for standalone compilation by `paper/attention_preview.tex`) as a
drop-in replacement for the legacy attention section in the earlier manuscript
draft, released as `paper/manuscript_source.tex`. That draft file has not been
overwritten. The replacement extends the scope from legacy
OC attention to the completed OC, KIP and CKD audits. It retains the appendix and
figure labels and requires only graphicx for its new figure.

## Figure and reproducibility

Run `python -B scripts/build_attention_appendix_figure.py`.
The generator reads completed OC summaries and checks identical sample rows and
completed status. It averages `mean_enrichment` equally over heads, within each
physical layer, using condition `all`. This is the mean of per-example
mass/baseline ratios, not a ratio of population means. Original zero-based
physical layers are displayed with one-based numbering. Category support differs;
465 rows contribute to the lag category. Source summaries and inference outputs
are never changed. No training or inference is performed.

Outputs under `paper/figures/attention_oc_profiles_20260925`:

- PDF, SVG and PNG figure.
- CSV with every plotted value, original layer index and eligible sample count.
- JSON with source paths and SHA256 hashes and the generator hash.

The appendix also embeds the MIMIC-IV CKD ordered/shuffled comparison as
`figures/attention_ckd_profiles_20260925.pdf`. Reproduce it with
`python -B scripts/build_ckd_attention_appendix_figure.py`.
PNG/SVG previews, plotted aggregate CSV, and source-hash JSON accompany the PDF.
The generator verifies completed audits, identical 1,000-patient sample IDs and
77 positive labels across all eight conditions. Each plotted layer averages
heads within each patient, then patients. Top panels show complete-event token
mass; bottom panels show normalized token entropy across all visible tokens,
including prompt/special tokens. No patient-level values are exported. This
figure was generated from the existing summaries and visually inspected; no
additional model inference was needed.

## Evidence for OC

Read `metrics.json`, `manifest.json`, `attention_summary.csv`, and
`counterfactuals.csv` in these completed directories:

- `results/encoder_oc_ckd_attention_20260923/attention/bert_oc_ordered`
- `results/encoder_oc_ckd_attention_20260923/attention/roberta_oc_ordered`
- `results/llama8_oc_ckd_attention_20260924/attention/llama8_oc_ordered`
- `results/qwen_oc_clinical_attention_20260922_seed9950/attention/ocdet`
- `results/qwen38_27b_oc_noisy_attention_20260924_seed9950/attention/ocnoise`

Not distributed in this release: the per-example `counterfactuals.csv` files.
Each directory above retains `metrics.json`, `manifest.json`,
`attention_summary.csv` and `layer_profiles.png`.

All five use the same 1,000 rows, with 292 clean positives and seed 9950.
All four clean fits have zero classification errors in this sample at threshold
0.5. Therefore the legacy assertion about correct-versus-missed compliant
attention is unsupported by the new sample and has been removed. Likewise,
the old absence-of-lag-enrichment claim is replaced by the observed localized
peaks. Peak selection is explicitly descriptive. These are corrected OC data,
not exact reruns of the historical different-key-set figures.

The full test split has 50,000 rows, distinct from the attention sample.
Noisy Qwen full-test observed-label AUROC is 0.6653328762; its sampled clean-label
AUROC is 0.9998839099. These quantities must not be substituted for each other.
Input controls use clean labels and threshold 0.5, whereas the noisy fit's
reported full-test F1 uses validation-selected threshold 0.24.

The final queue-selected OC checkpoint metrics establish training sizes and
epoch selection; planning documents predate user caps. Both Qwen fits completed
six epochs; selected clean/noisy checkpoints are epochs 5/6. Original learning
rate schedules were preserved.

## Evidence for KIP

The replacement TeX now embeds a combined KIP-M4 enrichment figure at
`paper/figures/attention_kip_m4_profiles_20260925.pdf` (also PNG/SVG). Reproduce
with `python -B scripts/build_kip_attention_appendix_figure.py`.
Its CSV records plotted values and category support, and its JSON hashes all
source summaries/manifests/metrics. It shows all four audited M4 checkpoints
with full-test AUROC 1.0, an explicitly disclosed selection, rather than claiming
to represent all 22 checkpoints. The generator verifies identical sampled rows,
completed status and the AUROC criterion. It uses condition `all`, equal-head
means of per-example enrichment, and one-based original block indices. Key-key
and distractor summaries have 1,000 eligible examples; inversion/non-inversion
have 971/961. Curves overlap and are not additive. The plot was visually checked;
all three appendix figure paths (OC, KIP, CKD) resolve.

The 22 completed jobs are evidenced by
`results/kip_checkpoint_attention_20260922/jobs/*/metrics.json` (22 present).
Not distributed in this release: the operational `queue_status.json`. Each
`jobs/<name>/` contains the same core evidence files.
The successful checkpoints are `llama1b_full_kip_m4`,
`llama8b_lora_m4_0.50`, `qwen14b_qlora_m4_0.50`, and
`qwen27b_qlora_m4_0.50`; each has full-test AUROC 1 and 200/200 key-swap
prediction flips, with 0/200 distractor flips. The remaining 18 checkpoint
AUROCs range from 0.49388 to 0.50487. The larger decoder training-size sweep
retains its previously disclosed test-informed stop at perfect 200K test AUROC;
this appendix does not introduce an independent training sweep.

Do not claim all 22 used the repaired capture procedure: six retained completed
audits used eager singleton inference (all Qwen27 fractions and Qwen14 40K/200K).
Sixteen used native attention with zero recorded capture delta. Across all jobs,
maximum saved-reference probability discrepancy is 0.015624, within the original
0.02 tolerance. Source-repair records are preserved under `repair_20260923/` (not distributed
in this release).

Key preference does not uniquely distinguish parity solutions. For example,
chance BERT-LoRA M4 layer 0/head 2 has key enrichment 1.6116 versus distractor
0.8937. This is a post-hoc qualitative example, not a prespecified comparison.

## Evidence for CKD

- Qwen: `results/qwen_oc_clinical_attention_20260922_seed9950/clinical_attention/metrics.json`
- Llama: `results/llama8_oc_ckd_attention_20260924/ckd_paired_comparison/metrics.json`
- Encoders: `results/encoder_oc_ckd_attention_20260923/ckd_paired_comparison/metrics.json`

The corresponding condition directories contain per-patient summaries and
manifests. Identical sample lists across all models contain 1,000 test patients,
77 positive. Full test size is 2,963. Code overlap excludes prompt mass, averages
attention over heads/layers before normalization, and compares separately trained
ordered and shuffled checkpoints. Encoder truncation counts in this sampled
cohort are 191 BERT and 188 RoBERTa; the decoder histories are complete.

## Validation

OC and CKD prose were independently checked against completed artifacts.
The figure was generated and visually inspected. Braces and environments were
checked and its image reference resolves. A TeX compiler is not installed on this
VM, so the replacement has not been compiled as a complete manuscript.
