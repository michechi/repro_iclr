# BERT/RoBERTa OC and CKD extension, 23 September 2026

This protocol applies attention analysis to BERT and RoBERTa on OC and CKD,
including fresh task-specific fine-tuning. Local scheduling records are omitted.

## Models, fits and selection

Use fresh pinned bert-base-uncased and FacebookAI/roberta-base snapshots already
verified in the original encoder experiments. Full fine-tuning, not initialization
from KIP-trained models. Six production fits: each encoder on deterministic OC,
ordered CKD, and shuffled CKD. Selected full weights and predictions are retained.
Unfinished fits restart from the pretrained base; no persistent optimizer states.

OC uses a fixed, outcome-stratified 200K subset of Qwen's canonical corrected
400K training split (seed 9950), identical for BERT and RoBERTa. The original 50K
validation and 50K test splits are retained in full. This is deterministic OC
(`ocdet`, clean labels), with keys WDQJXU, lag 7, and the maximal nondecreasing
lag-chain oracle. The source CSVs remain unchanged; each fit saves its selected
row indices and their hashes. All labels and file hashes are
verified before launch. Prompt: `Sequential events: <space-separated events>`.
Maximum 20 epochs, patience 3, effective batch 256, LR 2e-5, weight decay .01,
6% linear warmup followed by linear decay. Initial microbatch 128, evaluation 256.

CKD uses the existing 8887/2963/2963 patient-disjoint train/validation/test rows,
seed 42, paired ordered/shuffled labels and event multisets. The frozen Qwen input
hashes preserve that split. Prompt adds `Outcome (0 or 1):`, matching Qwen CKD.
Maximum 30 epochs, patience 5, minimum 5 epochs, effective batch 32, LR 2e-5,
weight decay .01, 6% linear warmup/decay. Initial microbatch/evaluation batch 32.
Loss is unweighted two-class cross-entropy via the classification head,
the same classification objective as Qwen. This is a new full encoder baseline,
not an exact rerun of the historical clinical experiments.

Both tasks use FP32 trainable weights/optimizer states and BF16 autocast, fresh
pretrained initialization, minimum validation loss for checkpoint selection,
validation-only F1 threshold selection, and one final test evaluation. Test scores
never stop training. OOM retries start a new attempt at a smaller microbatch while
preserving effective batch size. Saved failed attempts are not overwritten.

## Context and interpretation

OC fits without truncation. CKD uses native 512-token right truncation as in the
repository's BERT recipe. Full and retained token lengths and truncation rates
are recorded for every split/model/condition. These encoder results are not an
equal-context comparison with Qwen's full histories. Shuffling before truncation
can change which events are visible; ordered/shuffled differences therefore do
not isolate order alone.

## Attention

Use the same 1,000 outcome-stratified test rows and sampling seed 9950 as the
Qwen OC/CKD analyses. Restore exact saved weights and original test evaluation
batch membership, padding and SDPA backend. Attention capture must leave native
probabilities unchanged within 1e-6; saved-test probability reproduction retains
the existing .02 guard, with actual errors reported. Full model checkpoints and
data are hashed. CLS at token 0 is the encoder readout query. Physical layer/head
indices are retained. Bidirectional encoder baselines include all visible tokens;
absolute attention masses do not directly compare with causal decoder attention.

OC: standard per-layer/head key-to-lag-key, other-key, distractor, self and
prompt/readout masses and uniform baselines; primary all-row summaries and
secondary outcome/correctness strata. On 200 fixed rows, evaluate random shuffles,
oracle-verified label-changing permutations, and distractor-only swaps, retaining
original batch geometry. Plot layer profiles and three predetermined examples.

CKD: materialize only the CLS query's softmax, avoiding square attention storage.
Report visible-event/prompt mass, token entropy, quartiles of the visible event
prefix, CCS code mass/enrichment and support, and three predetermined heatmaps.
Only complete visible CCS events get code attribution; tokens from a partially
truncated event are included in other-token mass. Export visible/full event
coverage. Paired code-distribution overlap explicitly includes the limitation
that truncated visible code sets can differ. No patient names or identifiers
are exported; local split row indices are used. Attention is descriptive, not
clinical feature importance or a causal mechanism claim.

Each model/task gets an isolated short GPU training and attention smoke test
after Qwen releases the GPU and before production. Smoke model weights remain in
RAM-backed storage, with small verification artifacts copied to persistent disk.
The six production selected models and all analysis artifacts remain persistent.
