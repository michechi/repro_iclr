# Reproduction protocol

This file maps every claim in the paper to the script that produced it. All scripts run from this directory (`repro/`, the nested package root) via `python -m src.<...>`; see [README.md](README.md) for the environment variables that locate data and outputs (`REPRO_ROOT`, `RELEASE_ROOT`, `DATA_DIR`, `RESULTS_DIR`).

## Conventions

- `<TAG>` is one of: `6` (Tricky Det.), `9` (Tricky Rnd.), `test_just_pair` (Parity), `alph` (Naive sanity check).
- Results land in `${RESULTS_DIR:-./results}/` as one JSON per run; the fields are those written by each experiment module. No separate schema document is distributed with this release.
- All scripts honor `--seed`. Paper figures use seeds 9550, 9551, 9552 (averaged).

## Main results (Figure 1, Table 2)

| Paper element | Script | Command (single seed) |
|---|---|---|
| LLM (Llama, Qwen) on all 3 datasets | `src.experiments.LLM_fraction_experiment` | `python -m src.experiments.LLM_fraction_experiment --number_to_use <TAG> --model_name meta-llama/Llama-3.1-8B --peft --epochs 20 --seed 9550` |
| Random-init small Llama-style decoder (paper "Llama 1M") | same script with `--tiny --tiny_type 1M --cold_start` | `python -m src.experiments.LLM_fraction_experiment --number_to_use <TAG> --tiny --tiny_type 1M --cold_start --seed 9550` |
| BERT-base / RoBERTa-large | `src.experiments.BERT_fraction_experiment` | `python -m src.experiments.BERT_fraction_experiment --number_to_use <TAG> --model_name bert-base-uncased --peft --seed 9550` |
| LSTM / Transformer / RNN-Transformer (paper Table 2 DL row) | `src.experiments.DL_TR_baselines` | `python -m src.experiments.DL_TR_baselines --number_to_use <TAG> --models LSTM,Transformer,RNNTransformer --seed 9550` |
| LSTM / BiLSTM / GRU (Tricky only — `--number_to_use` is int) | `src.experiments.DL_baselines` | `python -m src.experiments.DL_baselines --number_to_use 6 --models BiLSTM --seed 9550` |
| XGBoost (basic / TF-IDF / LLM-emb / categorical) | `src.experiments.XGBoost_fraction_experiment` | `python -m src.experiments.XGBoost_fraction_experiment --csv_to_use <TAG> --run_basic --seed 9550` |
| Figure 1 plot | `src.analysis.plot_main_figure` | `python -m src.analysis.plot_main_figure --results_dir results --output figures/auc_combined_main_1row.png` |

For full Table 2 coverage, sweep `--model_name` over: `bert-base-uncased`, `roberta-large`, `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-70B` (parity only, App. results), `Qwen/Qwen3-4B-Think`, `Qwen/Qwen2.5-14B`. Paper figures average 3 seeds: `9550`, `9551`, `9552`.

Important flag notes:
- LLM/BERT/DL_TR_baselines: `--number_to_use` is `str` (accepts `6`, `9`, `test_just_pair`, `alph`).
- DL_baselines: `--number_to_use` is `int` (only `6` and `9`); use `DL_TR_baselines` for Parity.
- DL scripts: data fractions are hardcoded to `[0.01, 0.10, 0.30, 0.50, 0.75, 1.0]`. Epochs flag is `--final_epochs`, not `--epochs`.
- BERT/LLM: data fractions controlled by `--fractions "0.01,0.10,0.30,0.50,0.75,1.0"`. Epochs flag is `--epochs`.
- Models in `--models` are case-sensitive: `LSTM`, `BiLSTM`, `GRU`, `Transformer`, `RNNTransformer`, `MLP`, `CNN1D`.

## Mechanism-ID audit (Section 4)

Each phase is one CLI script in `src/analysis/mechanism_id/scripts/`. Run sequentially:

```bash
python -m src.analysis.mechanism_id.scripts.phase1_label_audit
python -m src.analysis.mechanism_id.scripts.phase1b_rule_from_logreg
python -m src.analysis.mechanism_id.scripts.phase2_baseline_ladder
python -m src.analysis.mechanism_id.scripts.phase2c_lag_agnostic
python -m src.analysis.mechanism_id.scripts.phase3_matched_histogram
python -m src.analysis.mechanism_id.scripts.phase4_heldout_rule
python -m src.analysis.mechanism_id.scripts.phase5_parity
python -m src.analysis.mechanism_id.scripts.phase6_oracle
python -m src.analysis.mechanism_id.scripts.make_summary_csv
python -m src.analysis.mechanism_id.scripts.make_plots
```

Outputs:
- CSV per phase under `src/analysis/mechanism_id/results/`
- Figures `fig_ladder_standard.png`, `fig_heldout_rule.png`, `fig_parity_decomp.png` under `src/analysis/mechanism_id/plots/`

For an HPC SLURM run: `PHASE=2 EXTRA="--rows_train 200000" sbatch -J mechid_p2 src/analysis/mechanism_id/scripts/slurm/run_phase.sh` (set `CONTAINER` env var to your apptainer image). **Not distributed in this release:** `src/analysis/mechanism_id/scripts/slurm/run_phase.sh`; run the phase modules directly as above. Aggregate outputs of the historical audit are retained in `src/analysis/mechanism_id/results/` (`kip_audit.csv`, `kip_oracle_reveal.csv`, `phase1_label_audit.csv`) and, with a report and plots, in `../results/mechanism_id_reference/`.

## Appendix experiments

| Appendix section | Script | Command |
|---|---|---|
| Naive sanity check (App. A.2) | as main with `--csv_to_use alph` | as above |
| k-gram analysis (App. C, Tables: ngram_appendix, signal_decomposition) | included in `phase2_baseline_ladder` |  |
| Sensitivity sweep over m, lambda (App. H) | `scripts/slurm/sensitivity_*.slurm` + `src.ablations.experiment_C.summarize_regime` | `bash scripts/slurm/sensitivity_BERT.slurm` then `python -m src.ablations.experiment_C.summarize_regime` |
| Parity vocabulary sweep (App. I) | `src.ablations.experiment_A.plot_auc_vs_ell` | `python -m src.ablations.experiment_A.plot_auc_vs_ell --results_dir results/parity_decomp` |
| Parity subset-size sweep (App. I) | `src.ablations.experiment_B.plot_auc_vs_k` | `python -m src.ablations.experiment_B.plot_auc_vs_k --results_dir results/parity_decomp` |
| Length-curriculum stages 1+2 (App. J) | `scripts/slurm/hpc_templates/experiment_F_*` + `src.ablations.experiment_F.summarize_curriculum` | `bash scripts/slurm/hpc_templates/experiment_F_launch.sh` |
| Length sweep (App. K) | `src.ablations.experiment_G.summarize_subset_length` | `bash scripts/slurm/hpc_templates/experiment_G_launch.sh` |
| Parity decomposition: membership-bit MLP (App. I) | `src.experiments.parity_decomposition_dl`, `parity_decomposition_bert`, `parity_decomposition_llm` | DL: `python -m src.experiments.parity_decomposition_dl --variant masked --model Transformer --seed 9550`. BERT: `... --variant masked --seed 9550`. LLM: `... --variant masked --seed 9550 --peft`. Each accepts `--variant {raw,masked,bitonly}`. |
| Few-shot LLM in-context (App. L) | `src.experiments.fewshot_llm_eval` + `src.ablations.summarize_fewshot` | `python -m src.experiments.fewshot_llm_eval --task tricky_det --n_shots 16 --model_name meta-llama/Llama-3.1-8B --output_dir results/fewshot_llm`. `--task` accepts `{tricky_det, tricky_rnd, parity}`. |
| Attention inspection (App. M) | `src.analysis.attention_analysis` | `python -m src.analysis.attention_analysis --model_type bert --number_to_use 6`. `--model_type` accepts `{bert, llama}`. This module imports `simulation.do_check_lag` from the outer release root, so it needs that root on `PYTHONPATH`: from `repro/`, run `PYTHONPATH=..:. python -m src.analysis.attention_analysis ...`. |

Scheduler templates cited above (`scripts/slurm/*.slurm`, `scripts/slurm/hpc_templates/*`) are institution-neutral: fill in `--account=<YOUR_ACCOUNT>`, export `CONTAINER`, `REPRO_ROOT` and `SCRATCH`, and create the `logs/` directory their `--output`/`--error` lines write to before submitting (see [SLURM templates](#slurm-templates)). Aggregate outputs of the historical sensitivity sweep (App. H) are retained in `../results/sensitivity/` (`sensitivity_BERT.md`, `sensitivity_DL.md`, `theoretical_bounds.csv`). No `results/parity_decomp` run outputs are distributed (the outer `../results/parity_decomp_dryrun/` holds only a dry-run `summary_table.tex`), so the App. I plot commands require your own reruns.

## Data regeneration (optional)

The 24 canonical CSVs for tags `6`, `9`, `test_just_pair` and `alph` (`X_`/`y_` x `train`/`val`/`test`) are the same files used for every paper number. In this release they live in the outer tree at `../data/simulation/tested/`, which also holds six `_big` files; the nested package reads them from `${DATA_DIR}/simulation/tested/`, so set `DATA_DIR="$(cd .. && pwd)/data"` as in [README.md](README.md). To regenerate from seed:

```bash
bash scripts/regenerate_data.sh
```

This invokes `src.generators.test_simulation_det` (Tricky Det. + Parity, with `random.seed(959693)`) and `src.generators.properties_9` (Tricky Random). Treat regeneration as best-effort: minor RNG ordering differences may produce slightly different shuffles; the canonical files are authoritative. Note that in this snapshot `test_simulation_det` writes its CSVs to `data/simulation/<name>.csv` under the current directory, not to `data/simulation/tested/` as the script's closing message states, and the export call in `properties_9` is commented out, so the Tricky Random files are not rewritten; move outputs by hand before comparing them with the canonical files.

## SLURM templates

Institution-neutral SLURM templates live in `scripts/slurm/` (root-level for main experiments) and `scripts/slurm/hpc_templates/` (for ablations and length-curriculum staging). They carry no site-specific account; partition names and resource lines may need adapting to your scheduler. Each requires:
- `--account=<YOUR_ACCOUNT>` filled in
- `CONTAINER=/path/to/your/pytorch.sif` exported (any image with Python 3.10+, torch 2.x, transformers, peft, scikit-learn, xgboost works)
- `REPRO_ROOT` and `SCRATCH` env vars set to the repo path and a writable cache location
- a `logs/` directory in the submission directory for the `--output`/`--error` files

The mechanism-audit wrapper `src/analysis/mechanism_id/scripts/slurm/run_phase.sh` is not distributed in this release.

## MIMIC-IV CKD->ESRD audit (Section 5, Appendix F.x)

Raw MIMIC-IV is not shipped (PhysioNet credentialing). Drop `diagnoses_icd.csv`, `admissions.csv`, `patients.csv` into `${DATA_DIR:-./data}/mimic/raw/hosp/` (create the directory; it is not shipped) or override the location with `MIMIC_RAW=...` before running the pipeline.

| Paper element | Script | Command |
|---|---|---|
| Cohort construction (App. F.x) | `src.mimic.build_ckd_cohort_ccs` | `python -m src.mimic.build_ckd_cohort_ccs` |
| Discriminative codes table (`tab:mimic_ckd_discriminative_codes`) and lag analysis | `src.mimic.descriptive_analysis` | `python -m src.mimic.descriptive_analysis` |
| K-gram diagnostic, k=1..7 (`tab:mimic_ckd_audit`, k-gram rows) | `src.mimic.kgram_analysis` | `python -m src.mimic.kgram_analysis` |
| Shuffle test: BoC LogReg / XGBoost + k-gram (`tab:mimic_ckd_audit`, BoC rows) | `src.mimic.shuffle_test` | `python -m src.mimic.shuffle_test` |
| 60/20/20 splits for neural models (ordered + shuffled) | `src.mimic.prepare_training_data` | `python -m src.mimic.prepare_training_data` |
| Transformer / BiLSTM / BERT ordered vs shuffled (`tab:mimic_ckd_audit`, neural rows) | `src.mimic.train_mimic` | `python -m src.mimic.train_mimic --model {transformer,bilstm,bert} --data {ordered,shuffled}` |
| All 12 neural jobs at once (HPC) | `scripts/slurm/submit_all_mimic.sh` | `bash scripts/slurm/submit_all_mimic.sh` (SLURM template; set `--account` in `scripts/slurm/mimic_experiment.slurm` and export `REPRO_ROOT` and `CONTAINER` first) |
| End-to-end CPU pipeline (cohort -> kgram -> shuffle -> splits) | `scripts/reproduce_mimic.sh` | `bash scripts/reproduce_mimic.sh` |
| Visit-level shuffle test (robustness) | `src.mimic.build_visit_level_data` + `src.mimic.visit_shuffle_test` | `python -m src.mimic.build_visit_level_data && python -m src.mimic.visit_shuffle_test --mode all` |

Outputs land in `${RESULTS_DIR:-./results}/mimic/`:
- `kgram_analysis.csv`, `kgram_analysis_raw.csv`
- `shuffle_test.csv`
- `model_results.csv` (one row per `train_mimic` run, appended)
- `visit_shuffle_test.csv`

Cohort artefacts land in `${MIMIC_PROCESSED:-./data/mimic/processed}/`. Path overrides: `MIMIC_DIR`, `MIMIC_RAW`, `MIMIC_PROCESSED`, `MIMIC_TRAINING`, `MIMIC_CODES`, `MIMIC_RESULTS`; their defaults and precedence are defined in [`src/common.py`](src/common.py) (see also [README.md](README.md)).
