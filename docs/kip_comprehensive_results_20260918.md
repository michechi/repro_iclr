# KIP: comprehensive experimental results and interpretation

Evidence review dated **18 September 2026**. This document is a self-contained handoff for writing a paper or giving another language model the complete experimental context. Scope: **KIP M4 and M6** discussed in this project. Results from other repository tasks, such as Tricky, count-parity, order completion, and MIMIC, are separate experiments.

## 1. Main findings

Observed learning success varies strongly with input representation and training recipe. Fully fine-tuned **Llama 3.2 1B solves the recorded M4 test set**, with AUROC, accuracy, precision, recall, and F1 all equal to 1.0; all 50,000 saved test predictions were independently checked. Historical **LSTM** runs also reach AUROC/F1 1.0000 at the available four-decimal log precision: two of three training seeds with early-stopping patience 3, and all three with patience 5. Historical LSTM predictions are unavailable, so those rounded results must not be translated into a verified zero-error count.

Neither architecture succeeds on the recorded M6 experiment under its completed training recipe. The fully fine-tuned Llama 1B M6 AUROC is **0.500215**. The tested scratch Transformer, BERT, and RoBERTa configurations also show little or no discrimination on either task. These are outcomes under particular finite recipes, not proofs of architectural incapacity.

The classical baselines clarify the role of representation. **XGBoost with letter counts is near chance**; with pair counts aggregated across all gaps, it reaches AUROC **1.0** (maximum aggregation) or **0.9999999952** (sum aggregation) on M4. The corresponding M6 results are **0.501550** and **0.502690**. Fixed-gap XGBoost has modest M4 signal at gaps 1 and 2, approximately **0.524**, and stacked all-gap features reach **0.590287**. Thus neither “XGBoost solves KIP” nor “all local-feature models are at chance” is an adequate description.

A separate feature-reveal experiment shows that a **64-unit MLP supplied with the true key-letter precedence bits reaches AUROC/F1 1.0 on both M4 and M6**. That experiment supplies the key identities and an explicit relational representation. It is not equivalent to learning from raw sequences. Revealing the scalar inversion count yields only intermediate MLP performance, AUROC **0.790750 / 0.832924** for M4/M6, despite that scalar containing sufficient information to determine the label.

Order perturbations support the importance of sequence order. The successful Llama 1B M4 model drops from AUROC **1.0** to **0.505268** on paired shuffled test examples, with accuracy **0.50338**. Training a fresh model on shuffled examples produces AUROC **0.498458**. These findings demonstrate dependence on ordering information for this task; they do not identify the learned internal algorithm.

**Llama 3.1 8B full tuning on M6 is interrupted, with no final test result.** Three completed epochs have validation AUROC 0.498131, 0.497944, and 0.497318. Its stale status file must not be treated as evidence that training is still running or that a completed scaling comparison exists.

## 2. Task, datasets, and evaluation conditions

Each example is a length-20 sequence over A–Z. A fixed set of m key letters occurs exactly once each; the other positions contain distractor letters. Read the key letters in sequence order, map them to their fixed reference ranks, and count inversions. The binary label is **1 for even inversion count and 0 for odd inversion count**.

| Dataset | m | Keys in increasing reference-rank order | Generator seed | Possible key orders |
| --- | --- | --- | --- | --- |
| M4 | 4 | C, O, T, A | 959697 | 24 |
| M6 | 6 | G, H, L, Y, I, Z | 959699 | 720 |

| Task | Split | Rows | Positive labels | Positive fraction |
| --- | --- | --- | --- | --- |
| M4 | train | 400,000 | 199,840 | 0.499600 |
| M4 | val | 50,000 | 24,910 | 0.498200 |
| M4 | test | 50,000 | 24,903 | 0.498060 |
| M6 | train | 400,000 | 200,098 | 0.500245 |
| M6 | val | 50,000 | 24,991 | 0.499820 |
| M6 | test | 50,000 | 25,045 | 0.500900 |

For each dataset there is one fixed rule and one fixed train/validation/test split. Training seeds change optimization randomness; they do not generate new tasks. The audit checked every recovered sequence: there are **no duplicate full sequences within splits, no shared full sequences across splits, no incorrect sequence lengths, no incorrect key multiplicities, and no labels inconsistent with the rule**. This rules out direct full-sequence duplication in these splits, but does not establish extrapolation to new rules, lengths, or key-order patterns.

An additional support audit found all **24 M4 and 720 M6 key-order patterns in every split**, and also in the first 100,000 and 200,000 training examples. Thus **0% of test rows contain a key order absent from training**, even compared with the smallest control-training prefix. Each order appears 4,070–4,326 times in the M4 100,000-row prefix and 107–170 times in the M6 prefix. The test examples are new full sequences, but their projected key orders are familiar. This distinction is especially important for interpreting the oracle-feature MLP.

The historical ordered CSVs were recovered by inverting the deterministic event shuffle with seed 777. The recovery manifest records exact forward-shuffle checks, and a retained independent generator audit records byte-identical regeneration of all 12 feature/label CSVs. Current training manifests hash the recovered files. Historical scratch-model logs explicitly record 400,000/50,000/50,000 examples; historical Hugging Face logs record fraction 1.0, with these split sizes inferred from the shared dataset context rather than printed directly.

Three conditions must be distinguished:

1. **Ordered:** fit on original sequences and evaluate on original sequences.
2. **Shuffled training:** fit a fresh model on sequences whose events have been permuted within each example; validation and test inputs are shuffled too. Original labels are retained.
3. **Paired shuffled evaluation:** take the selected ordered-trained model and evaluate it on the corresponding shuffled test examples. No additional model is fitted.

Shuffling events is different from changing training-example order. Labels are not recomputed from the shuffled sequence. The test therefore asks whether the original label remains predictable after the order information that generated it has been removed.

## 3. Evidence inventory and comparability

The completed neural evidence contains **55 distinct fits**: 35 historical fits and 20 recent fits. There are **79 neural evaluation records**, including 24 paired shuffled evaluations of existing fits. The feature/oracle CSVs add **64 evaluation records**: 42 generic-feature results and 22 reveal/oracle results, including two non-fitted closed-form oracles. The consolidated table therefore has **143 completed test-evaluation rows**, not 143 independent neural models.

| Cohort | Evidence | Replication | Important comparison limits |
| --- | --- | --- | --- |
| Historical scratch LSTM / Transformer | Completed logs, settings and metadata | Seeds 9550, 9551, 9552 | Four-decimal metrics; validation-F1 selection; patience 3 and separate M4 patience-5 addendum |
| Historical BERT LoRA / Llama 1B LoRA | Completed logs and metadata | Seed 9550 only | Different batches/objectives from recent runs; some driver jobs were actually skipped |
| Recent BERT / RoBERTa LoRA and full; Llama 1B full | Manifests, histories, metrics, saved predictions and selected models | Seed 9950 only | Shared broad recipe, but architectures/tokenizers differ; loss-reporting caveat for encoders |
| Generic-feature LR, XGBoost, k-gram estimators | Retained CSV and historical script bytecode | One result per setting; no run manifest | Different feature dimensions and training counts; incomplete historical helper source |
| Reveal LR / MLP / oracle | Retained CSV and historical script bytecode | One result per setting; no run manifest | Inputs supply different amounts of rule knowledge; F1 threshold-grid limitation |
| Llama 8B full M6 | Manifest and partial validation history | One incomplete run | No final test metrics or surviving trained checkpoint |

Do not pool recipes into a single model average. In particular, the same three scratch-model seeds were reused for patience 3 and patience 5, and paired shuffled evaluations are not new training replicates. Missing results mean “not completed/available,” not zero performance.

## 4. Recent pretrained neural models: all 20 fits

All runs below completed and used seed **9950** with 400,000 training, 50,000 validation, and 50,000 test examples. AUROC is the primary discrimination measure. F1 uses the validation-selected threshold, while the accuracy column uses the conventional threshold 0.5; these are intentionally different columns.

| Model / tuning | Task | Training data | Test AUROC | Accuracy at .5 | F1 selected | Selected threshold | Best / completed epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-base full | M4 | ordered | 0.500052 | 0.49806 | 0.66494 | 0.10 | 2/5 |
| BERT-base full | M4 | shuffled_train | 0.498708 | 0.49806 | 0.66494 | 0.10 | 2/5 |
| BERT-base full | M6 | ordered | 0.499877 | 0.49910 | 0.66747 | 0.10 | 8/11 |
| BERT-base full | M6 | shuffled_train | 0.499924 | 0.49926 | 0.66747 | 0.10 | 3/6 |
| RoBERTa-base full | M4 | ordered | 0.499979 | 0.50194 | 0.66494 | 0.10 | 4/7 |
| RoBERTa-base full | M4 | shuffled_train | 0.499885 | 0.50194 | 0.66494 | 0.10 | 4/7 |
| RoBERTa-base full | M6 | ordered | 0.500000 | 0.49910 | 0.66747 | 0.10 | 8/11 |
| RoBERTa-base full | M6 | shuffled_train | 0.500000 | 0.49910 | 0.66747 | 0.10 | 3/6 |
| BERT-base lora | M4 | ordered | 0.500887 | 0.50010 | 0.66494 | 0.10 | 3/6 |
| BERT-base lora | M4 | shuffled_train | 0.501632 | 0.50186 | 0.66494 | 0.10 | 3/6 |
| BERT-base lora | M6 | ordered | 0.498280 | 0.49922 | 0.66747 | 0.10 | 2/5 |
| BERT-base lora | M6 | shuffled_train | 0.501182 | 0.50146 | 0.66747 | 0.10 | 2/5 |
| RoBERTa-base lora | M4 | ordered | 0.503707 | 0.50212 | 0.66494 | 0.10 | 20/20 |
| RoBERTa-base lora | M4 | shuffled_train | 0.502599 | 0.49806 | 0.66494 | 0.10 | 3/6 |
| RoBERTa-base lora | M6 | ordered | 0.500712 | 0.50090 | 0.66747 | 0.10 | 5/8 |
| RoBERTa-base lora | M6 | shuffled_train | 0.501729 | 0.49910 | 0.66747 | 0.10 | 6/9 |
| Llama 3.2 1B full | M4 | ordered | 1.000000 | 1.00000 | 1.00000 | 0.10 | 15/18 |
| Llama 3.2 1B full | M4 | shuffled_train | 0.498458 | 0.49816 | 0.66489 | 0.50 | 3/6 |
| Llama 3.2 1B full | M6 | ordered | 0.500215 | 0.49678 | 0.66747 | 0.10 | 3/6 |
| Llama 3.2 1B full | M6 | shuffled_train | 0.503938 | 0.49884 | 0.66747 | 0.10 | 3/6 |

### Order perturbation: all 10 recent ordered-trained models

| Model / tuning | Task | Ordered-test AUROC | Paired shuffled-test AUROC | Shuffled accuracy at .5 |
| --- | --- | --- | --- | --- |
| BERT-base full | M4 | 0.500052 | 0.501998 | 0.49806 |
| BERT-base full | M6 | 0.499877 | 0.500157 | 0.49910 |
| RoBERTa-base full | M4 | 0.499979 | 0.499919 | 0.50194 |
| RoBERTa-base full | M6 | 0.500000 | 0.500000 | 0.49910 |
| BERT-base lora | M4 | 0.500887 | 0.501664 | 0.50240 |
| BERT-base lora | M6 | 0.498280 | 0.501772 | 0.50000 |
| RoBERTa-base lora | M4 | 0.503707 | 0.505272 | 0.50420 |
| RoBERTa-base lora | M6 | 0.500712 | 0.499398 | 0.50090 |
| Llama 3.2 1B full | M4 | 1.000000 | 0.505268 | 0.50338 |
| Llama 3.2 1B full | M6 | 0.500215 | 0.501002 | 0.49892 |

### Training recipe and trainable parameters

Inputs use the prompt `Sequential events: A B ...` with the 20 event letters separated by spaces. Maximum token length is 64, and observed tokenized lengths are 25 for BERT, 26 for RoBERTa, and 24 for Llama; no examples are truncated. Models start independently from pretrained base weights. Full fine-tuning does not continue from a previous LoRA checkpoint.

The recent recipe uses AdamW, learning rate 2e-5, weight decay 0.01, batch size 256, and a linear learning-rate schedule with warmup over 6% of the planned 20-epoch schedule. Checkpoint selection minimizes validation loss; early-stopping patience is 3. The threshold for F1 is selected on validation from 0.10 to 0.89 in increments of 0.01. Llama uses evaluation batch 256; encoders use 512.

| Model / tuning | Total parameters | Trainable parameters | Tokenized length |
| --- | --- | --- | --- |
| BERT-base full | 109,483,778 | 109,483,778 | 25 |
| RoBERTa-base full | 124,647,170 | 124,647,170 | 26 |
| BERT-base lora | 109,927,684 | 443,906 | 25 |
| RoBERTa-base lora | 125,681,668 | 1,034,498 | 26 |
| Llama 3.2 1B full | 1,235,818,496 | 1,235,818,496 | 24 |

Encoder LoRA uses rank 8, alpha 16, dropout 0.1, query/key/value adapters, no adapted bias, and a **trained classification head**. The larger RoBERTa LoRA trainable count includes its classification head. Saved parameter tensors are FP32 and network computation uses BF16 autocast; “full” describes which parameters are updated, not pure FP32 computation throughout. Llama computes classification cross-entropy explicitly in FP32. Recorded execution environment: H100 80GB, PyTorch 2.13.0+cu130, Transformers 5.14.1.

Pinned pretrained snapshots are BERT `86b5e0934494bd15c9632b12f734a8a67f723594`, RoBERTa `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, and Llama 3.2 1B `4e20de362430cd3b72f300e6b0f18e50e7166e08`. Source-file hashes are preserved in the audit JSON and original manifests.

### What the learning histories show

Llama 1B on ordered M4 increases from validation AUROC **0.534721 at epoch 1 to 1.0 at epoch 2**. Validation AUROC remains 1.0 thereafter; the minimum-loss checkpoint is epoch 15, and training stops at epoch 18. Thus validation AUROC saturates much earlier than loss-based checkpoint selection. The final test prediction file confirms 50,000/50,000 correct classifications at threshold 0.5. On shuffled test inputs the model remains highly confident while accuracy falls to 0.50338, consistent with reliance on the original order-sensitive target.

Llama 1B on ordered M6 selects epoch 3 and stops after epoch 6, with validation AUROC **0.498372** and test AUROC **0.500215**. It completed its early-stopping protocol; it did not run all 20 planned epochs. Its accuracy at 0.5 is **0.49678**. At the selected threshold 0.1 it predicts every sample positive, giving F1 **0.667466** and accuracy **0.5009**.

All 16 recent encoder models select threshold 0.1 and predict every test example positive at that threshold. Full RoBERTa on M6 produces exactly one unique probability across the entire test set in both ordered and shuffled-training conditions, giving AUROC exactly 0.5. Other encoder outputs are often nearly constant. RoBERTa LoRA on ordered M4 reaches the full 20-epoch cap but still has AUROC only 0.503707; the remaining encoder runs stop after 5–11 epochs.

These observations describe the selected models. They do not establish that alternative optimization, initialization, precision handling, or training duration would give the same results.

## 5. Historical neural experiments: all aggregate conditions

The table reports mean ± **sample standard deviation across training seeds**, followed by [minimum, maximum]. These are not confidence intervals. Historical source values have four-decimal precision. For n=1, variability is unknown, not zero.

| Model | Dataset | Recipe | Evaluation | n | Test AUC mean ± sample SD [min,max] | Test F1 mean ± sample SD [min,max] |
| --- | --- | --- | --- | ---: | --- | --- |
| BERT | kip_m4 | historical_lora | ordered | 1 | 0.5002 ± — [0.5002,0.5002] | 0.6649 ± — [0.6649,0.6649] |
| BERT | kip_m4 | historical_lora | shuffled_eval | 1 | 0.5004 ± — [0.5004,0.5004] | 0.6649 ± — [0.6649,0.6649] |
| BERT | kip_m4 | historical_lora | shuffled_train | 1 | 0.4983 ± — [0.4983,0.4983] | 0.6649 ± — [0.6649,0.6649] |
| BERT | kip_m6 | historical_lora | ordered | 1 | 0.5019 ± — [0.5019,0.5019] | 0.6675 ± — [0.6675,0.6675] |
| BERT | kip_m6 | historical_lora | shuffled_eval | 1 | 0.5032 ± — [0.5032,0.5032] | 0.6675 ± — [0.6675,0.6675] |
| BERT | kip_m6 | historical_lora | shuffled_train | 1 | 0.5018 ± — [0.5018,0.5018] | 0.6675 ± — [0.6675,0.6675] |
| LSTM | kip_m4 | patience3 | ordered | 3 | 0.8326 ± 0.2899 [0.4978,1.0000] | 0.8850 ± 0.1992 [0.6549,1.0000] |
| LSTM | kip_m4 | patience3 | shuffled_eval | 3 | 0.5015 ± 0.0058 [0.4948,0.5052] | 0.5528 ± 0.0889 [0.5015,0.6555] |
| LSTM | kip_m4 | patience3 | shuffled_train | 3 | 0.5000 ± 0.0022 [0.4977,0.5020] | 0.6401 ± 0.0221 [0.6147,0.6551] |
| LSTM | kip_m4 | patience5 | ordered | 3 | 1.0000 ± 0.0000 [1.0000,1.0000] | 1.0000 ± 0.0000 [1.0000,1.0000] |
| LSTM | kip_m6 | patience3 | ordered | 3 | 0.4996 ± 0.0016 [0.4986,0.5015] | 0.6444 ± 0.0218 [0.6242,0.6675] |
| LSTM | kip_m6 | patience3 | shuffled_eval | 3 | 0.5002 ± 0.0051 [0.4953,0.5054] | 0.6457 ± 0.0199 [0.6284,0.6675] |
| LSTM | kip_m6 | patience3 | shuffled_train | 3 | 0.5031 ± 0.0010 [0.5020,0.5040] | 0.6645 ± 0.0052 [0.6585,0.6675] |
| Llama1B | kip_m4 | historical_lora | ordered | 1 | 0.5054 ± — [0.5054,0.5054] | 0.6649 ± — [0.6649,0.6649] |
| Transformer | kip_m4 | patience3 | ordered | 3 | 0.5000 ± 0.0027 [0.4981,0.5031] | 0.6649 ± 0.0000 [0.6649,0.6649] |
| Transformer | kip_m4 | patience3 | shuffled_eval | 3 | 0.4987 ± 0.0031 [0.4959,0.5020] | 0.6649 ± 0.0000 [0.6649,0.6649] |
| Transformer | kip_m4 | patience3 | shuffled_train | 3 | 0.5002 ± 0.0029 [0.4970,0.5026] | 0.6649 ± 0.0000 [0.6649,0.6649] |
| Transformer | kip_m4 | patience5 | ordered | 3 | 0.5000 ± 0.0027 [0.4981,0.5031] | 0.6649 ± 0.0000 [0.6649,0.6649] |
| Transformer | kip_m6 | patience3 | ordered | 3 | 0.5022 ± 0.0032 [0.4986,0.5049] | 0.6675 ± 0.0000 [0.6675,0.6675] |
| Transformer | kip_m6 | patience3 | shuffled_eval | 3 | 0.5013 ± 0.0027 [0.4982,0.5030] | 0.6675 ± 0.0000 [0.6675,0.6675] |
| Transformer | kip_m6 | patience3 | shuffled_train | 3 | 0.5025 ± 0.0030 [0.5001,0.5059] | 0.6675 ± 0.0000 [0.6675,0.6675] |

The M4 LSTM patience-3 AUROCs by seed 9550/9551/9552 are **1.0000, 1.0000, 0.4978**. Their mean 0.8326 conceals a mixed outcome: two successful fits and one near-chance fit. Patience 5 gives **1.0000, 1.0000, 1.0000** at logged precision. The successful patience-3 fits fall to shuffled-test AUROC **0.5045 / 0.5052**. The historical shuffled evaluations load the patience-3 checkpoints; **no recorded shuffled evaluation tests the patience-5 checkpoints**.

Scratch LSTM uses embedding dimension 64, hidden dimension 128, two layers, dropout 0.5, Adam learning rate 5e-4, and batch 64. The scratch Transformer uses embedding dimension 64, eight heads, two layers, feed-forward dimension 512, dropout 0.3, Adam learning rate 1e-3, and batch 64. Both have a 30-epoch cap, validation-F1 checkpoint selection, and patience 3, except for the separate M4 patience-5 runs. Longer patience improves the observed M4 LSTM success rate for these reused seeds; it does not improve the scratch Transformer’s recorded AUROC.

Historical BERT LoRA uses rank 8, alpha 16, query/key/value targets, dropout 0.1, batch 16, AdamW learning rate 2e-5, a 20-epoch cap, and validation-loss selection with patience 3. M4 ordered has maximum length 512; other historical BERT training conditions use 64. Only seed 9550 has completed evidence. Six M4 seed-9551/9552 jobs were sentinel skips even though the outer driver reported successful exit.

The historical **Llama 3.2 1B LoRA** result is M4 ordered only: AUROC **0.5054**, F1 **0.6649**, seed 9550, with 15 completed epochs. It uses rank 8, alpha 16, dropout 0.1, query/key/value/output attention targets, batch 64, maximum length 50, AdamW learning rate 2e-5, and a **combined causal-language-model plus classification objective**. The recent full-tuning result uses classification-only loss, a different seed, batch, and implementation. Therefore the observed improvement from historical LoRA to recent full tuning cannot be attributed solely to updating all parameters. There is no completed historical Llama 1B LoRA M6 result in this evidence set.

A source-linked appendix supplies every historical per-seed result, epoch count, setting, exclusion, and log reference: [historical audit](../results/kip_results_review_20260918/evidence/kip_historical_audit.md).

## 6. Classical models and feature-reveal experiments: all 64 results

The following tables reproduce all retained feature/oracle results. They are separate experiments from the 400,000-example neural comparisons. Training counts are recorded in the CSVs. Historical script bytecode has seed 9550 and evaluation default 50,000, but the CSVs omit actual run seeds, evaluation counts, versions, data hashes, and invocation manifests. Those defaults must not be promoted to independently verified execution settings.

## Generic-feature audit (all 42 rows)

| Feature family | Model | Dimensions | Training rows | M4 AUC | M4 F1 | M6 AUC | M6 F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| unigram_counts | logreg_L2 | 26 | 200,000 | 0.502743 | 0.481849 | 0.498818 | 0.556212 |
| unigram_counts | xgboost | 26 | 200,000 | 0.503043 | 0.503251 | 0.500339 | 0.519563 |
| kgram_k2 | kgram_estimator | 26^2 keys (sparse dict) | 400,000 | 0.500758 | — | 0.499302 | — |
| kgram_k3 | kgram_estimator | 26^3 keys (sparse dict) | 400,000 | 0.500647 | — | 0.501409 | — |
| kgram_k4 | kgram_estimator | 26^4 keys (sparse dict) | 400,000 | 0.500362 | — | 0.503270 | — |
| lag_pair_l1 | logreg_L2 | 676 | 200,000 | 0.502322 | 0.497842 | 0.500698 | 0.512565 |
| lag_pair_l1 | xgboost | 676 | 200,000 | 0.523819 | 0.510840 | 0.499716 | 0.518845 |
| lag_pair_l2 | logreg_L2 | 676 | 200,000 | 0.501603 | 0.494346 | 0.498906 | 0.512400 |
| lag_pair_l2 | xgboost | 676 | 200,000 | 0.523386 | 0.505482 | 0.497847 | 0.513962 |
| lag_pair_l3 | logreg_L2 | 676 | 200,000 | 0.502249 | 0.493310 | 0.500151 | 0.514775 |
| lag_pair_l3 | xgboost | 676 | 200,000 | 0.502717 | 0.489218 | 0.502620 | 0.521512 |
| lag_pair_l5 | logreg_L2 | 676 | 200,000 | 0.497382 | 0.493588 | 0.497620 | 0.505744 |
| lag_pair_l5 | xgboost | 676 | 200,000 | 0.498464 | 0.492395 | 0.499241 | 0.511430 |
| lag_pair_l7 | logreg_L2 | 676 | 200,000 | 0.499435 | 0.495352 | 0.500297 | 0.513053 |
| lag_pair_l7 | xgboost | 676 | 200,000 | 0.507606 | 0.491053 | 0.497938 | 0.521837 |
| max_over_lag | logreg_L2 | 676 | 200,000 | 0.500909 | 0.495339 | 0.500573 | 0.514062 |
| max_over_lag | xgboost | 676 | 200,000 | 1.000000 | 1.000000 | 0.501550 | 0.513685 |
| sum_over_lag | logreg_L2 | 676 | 200,000 | 0.500987 | 0.495589 | 0.499713 | 0.514046 |
| sum_over_lag | xgboost | 676 | 200,000 | 1.000000 | 0.999960 | 0.502690 | 0.520460 |
| stacked_all_lag | logreg_L2 | 12844 | 100,000 | 0.500196 | 0.498545 | 0.502155 | 0.502440 |
| stacked_all_lag | xgboost | 12844 | 100,000 | 0.590287 | 0.555116 | 0.501181 | 0.509547 |

M4 sum-over-lag XGBoost is near-perfect rather than mathematically exact: AUC **0.9999999951999277**, F1 **0.9999598441954785**. M4 max-over-lag XGBoost records AUC/F1 **1.0/1.0**.

K-gram coverage:
- M4 kgram_k2: 1.
- M4 kgram_k3: 1.
- M4 kgram_k4: 0.9999976470588235.
- M6 kgram_k2: 1.
- M6 kgram_k3: 1.
- M6 kgram_k4: 0.9999529411764706.

## Oracle/reveal ladder (all 22 rows)

| Input / rung | Model | Dimensions M4/M6 | Training rows | M4 AUC | M4 F1 | M6 AUC | M6 F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| oracle_closed_form | closed_form | 6/15 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| a_raw_tokens | logreg | 520/520 | 100,000 | 0.502022 | 0.664940 | 0.502191 | 0.667395 |
| a_raw_tokens | mlp_64 | 520/520 | 100,000 | 0.498876 | 0.664940 | 0.499614 | 0.667422 |
| b_membership_bits | logreg | 20/20 | 100,000 | 0.498794 | 0.664940 | 0.500376 | 0.667475 |
| b_membership_bits | mlp_64 | 20/20 | 100,000 | 0.498631 | 0.664940 | 0.500389 | 0.667413 |
| c_rank_sequence | logreg | 20/20 | 100,000 | 0.499141 | 0.664940 | 0.501126 | 0.667475 |
| c_rank_sequence | mlp_64 | 20/20 | 100,000 | 0.507713 | 0.664940 | 0.500290 | 0.667484 |
| d_precedence_bits | logreg | 6/15 | 100,000 | 0.497548 | 0.684092 | 0.496414 | 0.670695 |
| d_precedence_bits | mlp_64 | 6/15 | 100,000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| e_inversion_count | logreg | 1/1 | 100,000 | 0.499515 | 0.685601 | 0.495666 | 0.669600 |
| e_inversion_count | mlp_64 | 1/1 | 100,000 | 0.790750 | 0.797406 | 0.832924 | 0.808032 |

Both oracle rows have exact label agreement **1.0**; trained models do not have agreement recorded.

### Exactly what XGBoost receives

All generic features use the complete 26-letter alphabet. They are numerical features derived from sequences; they are not pretrained embeddings and do not supply the hidden key set or reference ranking.

For a directed letter pair (a,b) and gap ell, define

`G_ell(a,b) = sum over t=1,...,20-ell of 1[x_t=a and x_(t+ell)=b]`.

Gap 1 means adjacent positions. The tested representations are:

| Feature family | Definition | Dimension | Training rows |
| --- | --- | --- | --- |
| Letter counts | Number of occurrences of each letter | 26 | 200,000 |
| Fixed-gap pairs | G_ell for all directed pairs; separate fits for ell = 1, 2, 3, 5, 7 | 676 | 200,000 |
| Maximum across gaps | max of G_ell over ell = 1,...,19 for each pair | 676 | 200,000 |
| Sum across gaps | sum of G_ell over ell = 1,...,19; number of (a before b) occurrence pairs | 676 | 200,000 |
| Stacked gaps | All 19 gap-specific pair-count matrices concatenated, sparse storage | 12,844 | 100,000 |

These are raw counts, without scaling or binarization in the retained feature construction. For two distinct true key letters, each appearing once, max and sum reduce to a binary indicator that one key precedes the other. Other entries involving repeated distractors can exceed one. Thus the 676-dimensional aggregates contain a relevant precedence submatrix without explicitly selecting it; they are not literally the oracle’s 6/15-bit input.

Retained XGBoost code specifies 200 trees, depth 6, learning rate 0.1, subsample 0.9, column subsample 0.9, histogram tree construction, log-loss evaluation, random state 9550, and all CPU workers. The fit call has no evaluation set or early-stopping argument. Logistic regression uses L2 regularization with C=1, lbfgs, and maximum 2,000 iterations. These are recovered script settings rather than a complete historical environment manifest.

The k=2,3,4 results come from a separate contiguous k-gram estimator with 400,000 training examples. They are not XGBoost results. Recorded test k-gram coverage is essentially complete, but the scoring helper is no longer available, limiting exact reconstruction of that estimator.

### What the feature results support

Letter counts and the recorded k-gram estimators give near-chance discrimination. Logistic regression remains near chance across the generic feature families. XGBoost’s M4 results improve markedly when pair-order information is aggregated across gaps. However, the stacked representation uses only 100,000 training examples versus 200,000 for max/sum; its AUROC 0.590287 cannot be attributed to feature organization alone under a matched-data comparison.

The M6 max/sum representation still contains the true-key order information. Near-chance XGBoost performance therefore does not imply that its input lacks the label-determining information. It indicates that this learner and its recorded settings do not successfully use that information in this experiment.

### What is supplied by the reveal ladder

Raw positional one-hot input has 20 × 26 = 520 dimensions and supplies no key knowledge. Membership bits identify the true key positions, giving 20 binary features. The rank sequence supplies the true rank at key positions and zero at distractor positions, also 20 features. Precedence features explicitly compare all unordered true-key pairs, giving **6 bits for M4 and 15 for M6**. The inversion-count feature supplies the resulting scalar statistic. The closed-form oracle additionally applies the exact known parity rule and has no fitted parameters.

Reveal learners use 100,000 training rows. The MLP has one hidden layer of 64 units, maximum 200 iterations, internal early stopping, and random state 9550; unspecified options inherit historical library defaults. Logistic regression uses lbfgs and maximum 2,000 iterations.

The MLP reaches AUROC/F1 1.0 with true precedence features on both tasks, but stays near chance with raw one-hot, membership, or rank-sequence inputs; M4 rank-sequence AUROC is 0.507713. The inversion-count MLP is intermediate, not perfect. These results show that the accessibility of a representation to a finite learner matters alongside its informational sufficiency.

The 6/15 precedence coordinates satisfy transitivity: they encode only **24/720 valid key orders**, not arbitrary independent 6/15-bit strings. Perfect performance on this fixed support is compatible with learning a finite mapping. It does not prove that a model learned a general inversion-parity algorithm.

The audited retained dataset contains every valid precedence pattern in its first 100,000 training rows, matching the prefix convention of the retained reveal script. Thus the retained dataset has no held-out precedence patterns at that prefix size. The historical result CSV does not preserve input hashes, preventing independent verification of its exact input files.

## 7. Llama 3.1 8B full tuning on M6: incomplete experiment

The run was interrupted during epoch 4 at batch **14,384/25,000 (57.536%)**, with 5,588 optimizer steps. Its historical `training` status is stale; it has no completed final test evaluation. The precise reason for interruption is not recorded.

| Completed epoch | Training loss | Validation loss | Validation AUROC | Epoch hours |
| --- | --- | --- | --- | --- |
| 1 | 0.742222 | 0.694386 | 0.498131 | 4.67 |
| 2 | 0.697805 | 0.693686 | 0.497944 | 4.73 |
| 3 | 0.711997 | 0.703948 | 0.497318 | 4.70 |

The lowest completed-epoch validation loss occurs at epoch 2. Epoch 4 has only a running training-loss estimate of 0.697689, not a completed validation evaluation. **No final test AUROC, test accuracy, test F1, test predictions, selected-model export, or resumable training checkpoint exists for this production run.** Its selected model was kept in memory until final export, so the recorded run cannot resume from its lost training state.

This was genuinely full parameter fine-tuning: all **7,504,932,864 parameters** of the sequence-classification model were trainable. The nominal “8B” base was `meta-llama/Llama-3.1-8B`, snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; the classification model does not retain the original language-model output head. Weights and optimizer states were FP32, network execution used BF16 autocast, and classification loss was explicitly FP32. CPU optimizer offload, gradient checkpointing, microbatch 16 and accumulation 16 gave effective batch 256. The remaining broad recipe matched the recent Llama 1B setup. Completed epochs took approximately 4.7 hours each on the recorded H100 80GB environment.

The defensible conclusion is **no useful validation discrimination in the first three completed epochs under this recipe**. There is no completed result establishing whether this run would eventually learn M6, and no basis for a final 1B-versus-8B scaling claim. The models also differ in Llama release, not just parameter count.

## 8. Metric and methodology cautions for the paper

**AUROC versus F1.** For positive prevalence p, predicting every example positive gives F1 = 2p/(1+p). This equals **0.664940 on M4** and **0.667466 on M6**. Those are exactly the recent encoder F1 scores, confirmed by their saved predictions. F1 near two-thirds is therefore not evidence of learning here. Some historical/reveal scores have the same warning, but without their prediction files every thresholded pattern cannot be verified.

**Encoder loss discrepancy.** Some saved encoder losses differ from cross-entropy recomputed from saved probabilities. For example, full RoBERTa M6 ordered predicts the constant probability 0.49609384: reported test loss is **0.691420313**, while probability-derived log loss is **0.693191760**. The cause is unconfirmed. Do not interpret slightly sub-ln(2) reported encoder losses as evidence of learning or compare those small differences directly with Llama’s explicit FP32 loss. Because encoder checkpoints were selected by their reported validation loss, a harmonized FP32 loss calculation would also be appropriate for a future selection-sensitivity check. Verified AUROC, accuracy, F1, precision and recall remain valid for the selected saved models. Exact loss reconstruction for extremely saturated probabilities can additionally be affected by saved probability precision.

**Threshold protocols differ.** The generic-feature audit uses fixed threshold 0.5. The reveal ladder maximizes validation F1 over 201 candidates, but those candidates span the minimum/maximum **test prediction scores** plus a small margin. That uses test-score range, though not test labels, in constructing the candidate grid. This is a methodological limitation for threshold-dependent metrics. AUROC uses the raw test scores. Recent neural runs select their threshold on validation from a fixed grid. F1 is therefore not harmonized across these cohorts.

**“At chance” is descriptive.** The control CSV’s `at_chance` flag means only that AUROC lies in [0.48, 0.52]. This is not a significance test, equivalence test, confidence interval, or multiple-comparison correction. Small deviations from 0.5 should be reported numerically without declaring meaningful differences between models. No bootstrap or other uncertainty interval was computed in this audit. Historical sample SD describes optimization variation on fixed data, not uncertainty across new rules or datasets.

**Task size is confounded with the sampled rule.** M4 and M6 change the number of keys, but also key identities, the reference ranking, generation seed, key density, distractor count/alphabet, and the finite key-order support. There is one rule for each m. The results suggest greater difficulty in the recorded M6 setting but do not isolate a causal effect of m alone.

**Recipe and resource differences.** Historical and recent runs differ in optimizer, objective, batch size, selection criterion, precision handling, and stopping duration. Feature controls use 100,000, 200,000, or 400,000 training examples. This collection is not a compute-matched architecture benchmark. Near-chance early-stopped outcomes are not universal convergence or capacity results. The patience-5 addendum is an exploratory protocol sensitivity comparison, not a new independent seed sample.

**Generalization scope.** The evidence concerns unseen full sequences from fixed rules at length 20. It does not evaluate new key sets, unseen reference rankings, longer sequences, larger m, or a general algorithmic extrapolation task. A perfect test result cannot establish a model’s internal mechanism. Shuffled-test collapse alone does not exclude every possible order-sensitive shortcut.

**Historical reproducibility.** Old checkpoint weights and unrounded predictions are unavailable; historical neural metrics were recovered from completed logs. The feature-control CSVs survive, but original audit/reveal Python sources and some imported helpers are missing. Their retained bytecode was disassembled, not executed during this review. Exact rerunning requires reconstructing and validating those missing pieces. The recent neural cohorts have substantially stronger provenance, including saved predictions that were independently checked.

## 9. Suggested paper text

The following paragraphs are a cautious starting point for a Results/Discussion section. They deliberately separate observed performance from mechanistic interpretation.

> We evaluated sequence classifiers and feature-based baselines on two fixed-rule KIP tasks with four or six key symbols embedded in length-20 sequences. Labels encode the parity of inversions in the key-symbol order. Each task contains 400,000 training, 50,000 validation, and 50,000 test examples with approximately balanced classes. An audit confirmed label consistency and found no duplicate full sequences within or across splits. Performance varied substantially across representations and training configurations. Fully fine-tuned Llama 3.2 1B achieved AUROC, accuracy, and F1 of 1.0 on M4, with all 50,000 test classifications correct. Historical LSTM runs reached AUROC/F1 1.0000 at logged precision in two of three seeds with early-stopping patience 3 and in all three seeds with patience 5. The scratch Transformer and the tested BERT and RoBERTa configurations remained near chance under their respective recipes. On M6, no completed raw-sequence model tested here showed useful discrimination; full Llama 1B obtained test AUROC 0.5002.
>
> Perturbing event order removed the successful M4 discrimination. The selected Llama 1B model’s AUROC decreased from 1.0 on ordered test examples to 0.5053 on their shuffled counterparts, while a separate model trained on shuffled sequences obtained AUROC 0.4985. Original labels were retained after shuffling. These controls support reliance on ordering information, without establishing which internal computation the model learned. Historical successful patience-3 LSTM fits showed a similar collapse under shuffled evaluation; corresponding evaluations of the patience-5 fits were not available.
>
> Feature-based experiments revealed substantial sensitivity to how ordering information was represented. XGBoost with unigram counts remained near chance, whereas aggregation of directed pair counts across all gaps yielded M4 AUROC 1.0 for maximum aggregation and approximately 1.0 for sum aggregation. Fixed-gap features at distances one and two supported modest discrimination, with AUROC approximately 0.524, while stacked gap-specific features yielded AUROC 0.5903. These comparisons are partly confounded by training-set size: stacked features used 100,000 examples and the other XGBoost representations used 200,000. The same maximum/sum representations produced near-chance M6 performance despite containing the true-key precedence information.
>
> In a separate feature-reveal experiment, a 64-unit MLP supplied with the true key-letter precedence relations achieved AUROC/F1 1.0 on both tasks. In contrast, raw positional one-hot inputs, true key-membership indicators, and true rank sequences did not yield comparable performance under the tested MLP recipe. Providing the inversion count alone produced intermediate AUROC values of 0.7908 and 0.8329 on M4 and M6. These observations distinguish the availability of sufficient information from its successful use by a finite learner. The precedence experiment supplies substantial generator knowledge and operates over only 24 or 720 valid key-order patterns, all of which appear in the audited training prefix; it should not be interpreted as evidence of a general parity algorithm or extrapolation to unseen rules or key orders.
>
> AUROC was treated as the primary discrimination metric because validation-selected F1 could favor trivial predictions. All recent encoder models predicted every test example positive at their selected threshold, yielding F1 approximately 0.665–0.667 despite near-chance AUROC. The recent pretrained comparisons used a single optimization seed, and historical/feature cohorts differed in training objective, sample size, and checkpoint-selection protocol. Each task also used one fixed rule. Accordingly, the findings describe empirical sensitivity to representation and training configuration rather than inherent architectural limitations or an isolated causal effect of the number of keys. A full-tuning Llama 3.1 8B M6 experiment was interrupted after three completed epochs and part of the fourth; its validation AUROC remained near 0.5, and no final test result was available.

For a final methods section, also disclose the encoder loss discrepancy, reveal-threshold grid limitation, and historical provenance limitations above. Do not silently turn rounded historical 1.0000 values into exact error-free claims.

## 10. Highest-priority follow-up work before stronger claims

1. Replicate the recent neural configurations with additional training seeds under one prespecified protocol; add multiple independently generated key sets/rank rules for each m.
2. Complete the M6 8B experiment with persistent resumable checkpoints before making scaling statements. A controlled size comparison should also account for the different Llama releases.
3. Harmonize encoder loss evaluation in FP32 and assess whether checkpoint selection changes. Compare LoRA/full tuning with matched objective, data, seed set, and optimization settings.
4. Repeat XGBoost max/sum/stack comparisons at equal training sizes and document full environments. Reconstruct missing feature helpers, preserve predictions, and use a validation-only threshold grid.
5. For claims about algorithmic generalization, evaluate new rules, larger m, length shifts, or deliberately held-out key-order patterns. The present IID sequence split does not establish those properties.

These are proposed experiments, not additional results produced by this audit.

## 11. Files, traceability, and exclusions

- [All 143 completed evaluations, CSV](../results/kip_results_review_20260918/all_completed_evaluations.csv): distinguishes cohorts, fit IDs, training versus paired evaluation, metric precision, sample-count evidence, and sources. Blank metrics denote unavailable information.
- [Recent neural evaluations, CSV](../results/kip_results_review_20260918/latest_neural_evaluations.csv): 20 fits plus 10 paired shuffled evaluations, including thresholded metrics and probability diagnostics.
- [Historical evaluations, CSV](../results/kip_results_review_20260918/historical_neural_evaluations.csv) and [aggregates, CSV](../results/kip_results_review_20260918/historical_aggregates.csv): all 49 historical evaluations and 21 aggregate conditions.
- [Feature/oracle evaluations, CSV](../results/kip_results_review_20260918/feature_and_oracle_evaluations.csv): all 64 retained control results.
- [8B partial validation, CSV](../results/kip_results_review_20260918/llama8b_partial_validation.csv): deliberately separate from completed test evaluations.
- [Historical audit](../results/kip_results_review_20260918/evidence/kip_historical_audit.md), [recent audit](../results/kip_results_review_20260918/evidence/kip_latest_audit.md), and [feature audit](../results/kip_results_review_20260918/evidence/kip_controls_audit.md): detailed methods and source references. Matching JSON files retain full audited structures and precision.
- [Data integrity audit](../results/kip_results_review_20260918/evidence/kip_data_audit.json), [recovery manifest](../results/kip_results_review_20260918/evidence/recovery_manifest.json), and [independent generator audit](../results/kip_results_review_20260918/evidence/independent_generator_audit.json).
- [Key-order support audit](../results/kip_results_review_20260918/evidence/kip_order_support_audit.json): all observed permutations and counts by split/training prefix.
- [Comparison figure, PDF](../results/kip_results_review_20260918/figures/kip_model_comparison.pdf), [PNG](../results/kip_results_review_20260918/figures/kip_model_comparison.png), and [SVG](../results/kip_results_review_20260918/figures/kip_model_comparison.svg), with [caption and plotted data](../results/kip_results_review_20260918/figures/README.md). Panels separate historical, recent, and feature-control settings.

Not distributed in this release: `llama8b_partial_validation.csv`, the `evidence/` audit notes and JSON files, and the `figures/` comparison plot under `results/kip_results_review_20260918/`. The five evaluation CSVs above are retained; the recovery manifest and independent generator audit are retained at `data/simulation/kip_ordered_recovered/`.

Primary result roots are `results/kip_encoders_20260916_seed9950/`, `results/kip_llama1b_full_20260916_seed9950/`, and the mechanism-audit aggregates `repro/src/analysis/mechanism_id/results/{kip_audit,kip_oracle_reveal}.csv` (also copied, together with `phase1_label_audit.csv`, a report and plots, to `results/mechanism_id_reference/`). Not distributed in this release: `logs/kip/` and the incomplete 8B source root `results/kip_llama8b_full_m6_20260917_seed9950/`. Evidence snapshots retain hashes/source references; original experiment artifacts were not edited by this review.

Excluded from completed scientific results: all smoke/throughput checks; seven historical smoke-CSV entries, including nominal Llama entries that actually used SmolLM2-135M; six BERT sentinel skips; the old Llama full-tuning attempt on a 30% training fraction with no completed test evaluation; and the interrupted 8B run. Outer-driver success messages are not counted as completed training. No model was retrained for this review.
