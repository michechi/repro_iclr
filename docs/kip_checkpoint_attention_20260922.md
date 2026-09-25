# Existing-checkpoint KIP attention inspection

This audit analyzes only combinations with saved weights. No additional
fine-tuning is part of the attention analysis.

Order: finish both existing CKD fits, then Qwen27 KIP-M4 120K/200K/4K/40K,
Qwen14 and Llama8 LoRA at the same fractions, followed by saved Llama1B, BERT,
and RoBERTa ordered-trained M4/M6 classifiers (full and LoRA where available).
Smoke runs and shuffled-trained runs are excluded. GPC and Qwen27 M6 have no
locally available task-fine-tuned checkpoints. Llama8 full selected weights
were not retained and therefore cannot be inspected. This is an inventory of
local production artifacts, not a promise to train missing cells.

Local scheduling and process-control records are omitted from this release.

## Analysis design

- One fixed proportionally label-stratified sample of 1,000 original held-out
  rows per task; same row identities for every model/fraction. Seed 20260922.
- Original narrative prompt, tokenizer, classifier head, and saved weights.
  Existing reference test predictions must reproduce within 0.02 absolute
  probability tolerance (singleton eager vs batched SDPA BF16 numerics).
  Maximum and mean deviations are reported. No optimization or threshold fitting.
- Exact character-offset event alignment; target subtoken mass is summed and
  query subtokens averaged. Fail on unmapped or ambiguously merged events.
- Actual physical layer numbers. Qwen hybrid recurrent/linear-attention blocks
  are excluded explicitly, not labeled as empty standard-attention layers.
- Per-head attention to keys, inversion/non-inversion key pairs, distractors,
  the source event itself, and prompt/special tokens. Also inspect the readout
  query (CLS for encoders, last unmasked token for decoders).
- Causal future positions excluded from statistics AND their uniform-visible-
  token baselines. Report mass and enrichment; absolute encoder/decoder masses
  do not establish an architecture comparison.
- All sampled examples form the primary result. Correct/incorrect at threshold
  0.5 and label strata are secondary. Empty groups are omitted, not set to zero.
- On a fixed random 200-example subset: transpose the first two keys in sequence
  order (true KIP label flips) and two distinct distractor letters (label stays).
  Save paired input strings, labels, probabilities, and aggregate flip/accuracy
  measures. These probe input dependence, not causal internal circuitry.
- Save per-example/head statistics, CSV summaries, PNG/PDF/SVG layer profiles,
  and heatmaps for the first three predetermined sample rows. Heatmap examples
  are not selected using model outcomes. Training fractions are independent
  fits, so 120K vs 200K is not a within-run learning trajectory.

All outputs are under results/kip_checkpoint_attention_20260922. Existing
training checkpoints, metrics, and runtime source files are read only.
