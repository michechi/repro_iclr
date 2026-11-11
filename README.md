# Do Large Language Models Learn Order-Dependent Rules?

Implementation of the synthetic benchmarks, training pipelines, ablations, clinical evaluations, and attention analyses for *"Do Large Language Models Learn Order-Dependent Rules?"*.

## Overview

1. **Synthetic Data and Main Experiments** — Ordered Compliance (OC), Key-Inversion Parity (KIP), and Generalized Parity Check (GPC) dataset generation, pretrained encoder and decoder fine-tuning, and scratch sequence, tree, and $k$-gram baselines (Sections 3–5 and Appendices A–E).
2. **Clinical Cohort and Ordered-vs.-Shuffled Evaluation** — MIMIC-IV CKD to ESRD longitudinal cohort construction, shortcut controls, and neural sequence evaluation on ordered and shuffled histories (Section 4.4 and Appendix F).
3. **Exploratory Attention Diagnostics** — Visibility-normalized attention enrichment and count-preserving permutation controls across OC, KIP, and MIMIC-IV CKD (Appendix G).
