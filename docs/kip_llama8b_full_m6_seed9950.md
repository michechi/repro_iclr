# Llama 3.1 8B full fine-tuning on KIP M6, seed 9950

This historical production run was interrupted before final held-out evaluation.
Its early training measurements do not establish completed test performance.

Not distributed in this release: the run root
`results/kip_llama8b_full_m6_20260917_seed9950/` (including the
`offload_validation_1b/` and `smoke/run/` directories cited below) and
`logs/kip/`. No partial-validation CSV was retained either; the results
inventory records this run as having no completed production test metric.

The requested run uses the base `meta-llama/Llama-3.1-8B` model, pinned to
revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, on ordered M6. The selected
model is evaluated on both the original and paired shuffled test sequences.
This is one full-parameter classification run; every model parameter is
trainable. The language-model output head is replaced by a binary classifier,
giving **7,504,932,864 total and trainable parameters**. The production manifest
confirms `is_smoke=false` and the complete 400,000/50,000/50,000 data splits.

## Model access

Download the pinned base checkpoint using your own authorized Hugging Face
access. Model weights and local storage-management records are not distributed.

## Training settings

| Setting | Value |
| --- | --- |
| Data | 400,000 training / 50,000 validation / 50,000 test |
| Seed | 9950 |
| Training order | Ordered |
| Trainable parameters | 7,504,932,864, equal to total parameters |
| Input | `Sequential events: ` followed by all 20 events; truncation forbidden |
| Tokenized length | 24 tokens in every production split |
| Weights and gradients | FP32 on GPU, all parameters trainable |
| AdamW moment states | FP32 on CPU |
| Network compute / loss | BF16 autocast / FP32 binary cross-entropy |
| Microbatch / accumulation | 16 / 16, effective batch 256 |
| Evaluation batch | 16 |
| Learning rate / weight decay | 2e-5 / 0.01 |
| Schedule | 6% warmup, linear decay |
| Epoch limit / patience | 20 / 3, minimum validation loss |
| Gradient checkpointing | Enabled, non-reentrant |
| Checkpoint | One selected full FP32 model; no optimizer or epoch resume |

`CPUOffloadAdamW` uses PyTorch's single-tensor AdamW function on CPU chunks.
Two CPU moment tensors persist per parameter; parameter and gradient chunks
are copied to CPU and updated parameters are copied back to GPU. There is no
full CPU master-weight copy and no quantized optimizer state. This requires
approximately eight bytes of CPU moment storage and twelve bytes of PCIe
traffic per parameter per update. It will be slower than an optimizer that
fits entirely on GPU. The best CPU model copy is updated in place to avoid
retaining two complete selected checkpoints.

The host has an H100 80 GB and 167 GiB system RAM. The actual 8B smoke check
completed forward/backward passes, an optimizer update, validation, selection
of a CPU model copy and test evaluation on this hardware. Production throughput
and loss trends must be assessed from the full run; the short smoke check is
not an estimate of final accuracy or total runtime. An interrupted unfinished
production run must restart because optimizer and epoch checkpoints are not saved.

## Validation completed

- 27 focused tests passed for optimizer parity, gradient accumulation,
  checkpoint selection/reload, CPU offload integration and launcher storage
  gates. CPU optimizer results exactly match standard FP32 AdamW in the
  tested cases, including chunk boundaries and changing learning rates.
- A cached Llama 3.2 1B GPU smoke run completed with full FP32 parameters,
  CPU optimizer offload, gradient checkpointing and batch 16 x accumulation 16.
  It used M6 subsets of 256 training and 128 examples per evaluation split.
  This validates the training path, not 8B resource fit or model performance.
  Artifacts are in
  `results/kip_llama8b_full_m6_20260917_seed9950/offload_validation_1b/`.
- The actual pinned Llama 3.1 8B GPU smoke run completed at **08:22:24 UTC**,
  with all 7,504,932,864 classifier parameters trainable in FP32, CPU AdamW
  moment states, gradient checkpointing and batch 16 x accumulation 16.
  It completed one epoch using 256 training and 128 examples per evaluation
  split in 60.2 seconds overall. Its manifest and completed status are in
  `results/kip_llama8b_full_m6_20260917_seed9950/smoke/run/`.
  The smoke run intentionally omitted the large selected-model disk export.

