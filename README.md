# Do Large Language Models Learn Order-Dependent Rules?

This repository contains the implementation of the synthetic benchmarks, model training pipelines, controlled ablations, clinical evaluations, and attention analyses for the paper *"Do Large Language Models Learn Order-Dependent Rules?"*.

## Overview

The codebase is organized into three packages under `src/` and corresponding entrypoints under `scripts/`:

1. **Synthetic Data and Main Experiments (`src/experiments/`, Sections 3, 4.1–4.3, and Appendices A–B):**
   Generates the Ordered Compliance (OC-Deterministic and OC-Noisy), Key-Inversion Parity (KIP-4 and KIP-6), and Generalized Parity Check (GPC) benchmarks, and trains all pretrained encoder and decoder models, randomly initialized sequence baselines, gradient-boosted trees, and $k$-gram classifiers across the reported training sample sizes and random seeds.

2. **Controlled Ablations and Feature Audits (`src/experiments/`, Section 5 and Appendices D–E):**
   Implements the $(m, \lambda)$ sensitivity grid, the explicit count and pair feature ladders, the all-lag sparse recovery experiment, the held-out-rule generalization evaluation, the GPC intermediate helper ladder, and the parity ablations over dataset reuse, alphabet size, key set size, sequence length, and curriculum learning.

3. **Clinical Cohort and Ordered-vs.-Shuffled Evaluation (`src/mimic/`, Section 4.4 and Appendix F):**
   Constructs the MIMIC-IV chronic kidney disease to end-stage renal disease (CKD to ESRD) longitudinal diagnosis cohort, computes cohort summary statistics, and evaluates $k$-gram, bag-of-codes, and neural sequence models on paired temporally ordered and patient-shuffled histories.

4. **Exploratory Attention Diagnostics (`src/analysis/`, Appendix G):**
   Extracts visibility-normalized attention profiles and count-preserving permutation controls on held-out examples for OC, KIP, and MIMIC-IV CKD, and generates the corresponding figures.
