# Do Large Language Models Learn Order-Dependent Rules?

Implementation of the synthetic benchmarks, training pipelines, ablations, clinical evaluations, and attention analyses for *"Do Large Language Models Learn Order-Dependent Rules?"*.

Synthetic tasks (OC, KIP, GPC) are parameterized by a rule in `paper_dataset_manifest.json` (or an explicit override). Published key sets live only in `PAPER_INSTANCES` for regenerating the reported datasets.

## Overview

1. **`src/experiments/`** — OC, KIP, GPC data; pretrained and scratch baselines; Sections 3–5 and Appendices A–E.
2. **`src/mimic/`** — MIMIC-IV CKD→ESRD cohort and ordered-vs-shuffled evaluation; Section 4.4 and Appendix F.
3. **`src/analysis/`** — Attention diagnostics; Appendix G.

Entrypoints are under `scripts/`.
