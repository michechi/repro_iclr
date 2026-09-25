# Llama 3.1 8B LoRA on OC and paired CKD, 24 September 2026

The requested extension adds training and attention analysis on OC and CKD to
the existing Llama 8B KIP experiments. The cached 8B checkpoint is
`meta-llama/Llama-3.1-8B`, revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. The user's message called it
Llama 3.2-8B; a clarification was requested, with the existing 3.1-8B checkpoint
stated as the default interpretation. All outputs explicitly identify the checkpoint actually
used. No download or Hugging Face token is needed.

## Three fresh fits

Each fit starts from the same verified pretrained base, independently of KIP
and the other tasks. Use BF16 frozen backbone weights, FP32 trainable LoRA
parameters and a two-class linear head on the last unmasked token. The existing
KIP factory supplies rank 8, alpha 16, dropout 0.1 on Q/K/V/output projections.
This is LoRA without quantization. Retain every selected adapter/head (~26 MiB),
row indices, validation/test predictions, metrics and provenance.

Deterministic OC uses the identical outcome-stratified 200K training subset
as the completed BERT/RoBERTa extension, seed 9950, from the canonical corrected
400K source. Keep all 50K validation and 50K test rows. Prompt:
`Sequential events: <space-separated events>`. Maximum 20 epochs, validation
patience 3, effective batch 256, initial microbatch/evaluation batch 128,
LR 2e-5, weight decay .01 and 6% warmup followed by linear decay. All 500K
available rows tokenize to 24 tokens; no truncation.

CKD runs separately on ordered and shuffled histories with the existing paired,
patient-disjoint 8887/2963/2963 splits, seed 42. The prompt adds
`Outcome (0 or 1):`, matching Qwen CKD. Preserve full histories without
truncation: maximum Llama lengths are 5594/3394/7137 tokens. Maximum context
16384, maximum 30 epochs, minimum 5, validation patience 5, effective batch 32,
initial microbatch/evaluation batch 4. The optimizer recipe matches OC.
Unlike the BERT/RoBERTa CKD runs, these runs preserve the entire history.

Both tasks minimize unweighted cross-entropy. Minimum validation loss selects
the saved checkpoint; the classification threshold is selected on validation.
Test data never determine stopping or checkpoint selection. The old Qwen OC
six-epoch cap and KIP perfect-test stopping rule do not apply to this new run.

## Attention

Use the same seed-9950, outcome-stratified 1000 held-out rows as the existing
OC/CKD attention studies. Restore the exact saved model and original evaluation
batch membership, padding and native SDPA computation. Attention capture must
preserve native output probabilities within 1e-6; reference prediction
reproduction retains the existing .02 tolerance and reports measured errors.

OC reports attention mass and uniform-visible-token baselines for lag-related
keys, other keys, distractors, self and prompt/readout positions by physical
layer and head. On 200 fixed rows, test count-preserving perturbations with the
canonical OC oracle. Preserve batch geometry for each control. Save numerical
summaries, compressed per-example data, layer plots and fixed example heatmaps.

CKD captures only each selected patient's last unmasked query, preserving
linear diagnostic storage in sequence length. Report event/prompt mass, token
entropy, input-position quartiles, CCS-code mass/enrichment and support, fixed
heatmaps and paired ordered/shuffled code-distribution overlap. Bulk tables are
compressed. No patient identifiers are exported; local row indices identify
the fixed split. Attention and input perturbations are descriptive analyses,
not evidence of causal clinical risk factors or a unique learned algorithm.

## Detached execution and validation

Run isolated OC and CKD short training/attention checks before production.
Smoke fits use 64/32/32 rows and one epoch, preserving class counts and including
the longest sequence in each split. Their test subset is used only for pipeline
verification and never for production selection. Smoke attention uses 16 rows
and four OC controls. Production follows OC, ordered CKD, then shuffled CKD,
each immediately followed by its attention audit, then a paired CKD summary.

The supervisor and workers inherit a dedicated queue lock and the shared GPU
lock. Workers verify frozen sources and data before each job. A fit that runs
out of memory restarts in a new directory with halved microbatch/evaluation
batch and fixed effective batch. Successful smoke batch sizes carry into
production. All selected weights, failed attempts and completed analyses remain
available; completed jobs are skipped on resume. The supervisor starts in a
new session and survives user disconnection while the VM remains powered on.
All work is isolated under `results/llama8_oc_ckd_attention_20260924`.
