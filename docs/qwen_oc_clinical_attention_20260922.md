# Qwen27 OC fine-tuning and clinical attention extension

This protocol covers Qwen27 fine-tuning and attention on OC, and attention
analysis of separately trained CKD classifiers. The release includes both
deterministic and noisy OC aggregates. Local scheduling records are omitted.

## OC

Use the canonical corrected OC exports already used for Qwen few-shot evaluation:
400K training, 50K validation, 50K test. Keys W,D,Q,J,X,U; lag 7; canonical maximal
lag-chain rule with nondecreasing key rank and minimum chain length two.
The data preflight recomputes all 500K clean labels from the canonical function,
verifies export hashes, split uniqueness/disjointness, and complete tokenization.
The legacy attention script hard-codes a different key set and its tag-9 default
is an older dataset. This experiment is not an exact replication of those figures.

Fresh pinned Qwen/Qwen3.8-27B initialization, NF4 double-quantized QLoRA,
BF16 compute, rank 8/alpha 16/dropout .1, same hybrid text projections and
last-unmasked-token linear classifier as the KIP fits. CE only. Effective batch
256 (initial microbatch 128, accumulation 2), AdamW 2e-5, weight decay .01,
6% warmup, maximum 20 epochs, patience 3 on minimum validation CE. F1 threshold
is selected on validation; test evaluated after selecting the checkpoint.
No test-informed stopping. Selected adapter/head retained. OOM retries are new
attempts with halved microbatch and unchanged effective batch.

OC attention uses 1,000 fixed clean-label-stratified test examples, seed 9950,
with all rows primary and correct/missed compliant groups secondary. Report
key-to-lag-key, other-key, distractor, self, prompt and readout attention by
physical standard-attention layer/head, with uniform-visible-token baselines.
Hybrid recurrent blocks are excluded explicitly. Input audits on 200 fixed
examples preserve counts: random shuffling, oracle-verified label-flipping
permutations when found within 64 proposals, and distractor-only swaps.
Changed labels are computed, never assumed. The label-flipping subset is a
conditional diagnostic and its support is reported. It is not an unbiased
performance estimate. If noisy OC is added, perturbation labels refer to the
latent clean rule, not an invented noise draw.

## Clinical CKD -> ESRD

Compare completed ordered and shuffled Qwen27 QLoRA checkpoints on the same
1,000 proportionally outcome-stratified test patients (seed 9950). The preflight
checks all 2,963 test rows for paired labels/event multisets and exact tokenizer
alignment. Training sources, outcome definition, prompts, full sequence lengths,
and selected checkpoint weights are preserved. No patient names or identifiers
are exported; row indices reference the existing local split.

For each standard-attention block, recompute only the final readout query's
softmax from the module's actual normalized, rotary-transformed Q and K.
The native SDPA result is returned unchanged. Additional audit storage is linear
in sequence length; no full square attention map or truncation is necessary.
Tests compare this row against eager attention and verify unchanged predictions.
Uniform baselines account for each CCS event's token count. Aggregate repeated
CCS codes within patient, then across patients, with support counts. Report
event/prompt mass, normalized token entropy, input-position quartiles, code-level
mass/enrichment and paired event-conditioned code-distribution overlap.
Position refers to input position, not elapsed clinical time. Heatmaps show all
history in up to 80 consecutive event bins for three predetermined sample rows.

Both audits check restored predictions against saved test probabilities with
0.02 absolute tolerance for BF16 singleton/batch numerical differences and
report the measured maximum/mean differences. The clinical capture itself must
leave the same singleton prediction unchanged within 1e-6.

Attention measures describe model computations, not causal clinical risk factors
or proof that a particular internal algorithm has been learned. Comparing two
condition-trained clinical models also changes weights, not only event order.
Artifacts remain local under results/qwen_oc_clinical_attention_20260922_seed9950.
