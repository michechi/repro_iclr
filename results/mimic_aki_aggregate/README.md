# AKI cohort-preparation aggregates

These CSVs are aggregate cohort and preprocessing audit outputs for three protocol stages, retained under neutral aliases `preparation_v1`, `preparation_v2`, and `preparation_v3`. Each stage contains 11 small tables: cohort flow and unit audit, matching flow/balance/estimability, primary/secondary class distributions, recovery-timing sensitivity, censor reasons, episode flow, and protocol metadata. The original machine-specific source-directory suffixes are omitted. Scientific values and configuration hashes are unchanged.

These tables document preparation and matching, **not completed model-training or test-performance results**. The revisions are protocol stages, not independent cohort replications or training seeds. Their estimability/balance flags must remain attached to the relevant version. Do not substitute a later stage's passing flag for an earlier failed check.

No admissions, lab measurements, patient identifiers, individual episodes, patient splits, or sequences are included. Counts such as `n_subjects` are group totals. Raw-data and derived patient-level files remain unavailable in the release and require separately authorized MIMIC access.
