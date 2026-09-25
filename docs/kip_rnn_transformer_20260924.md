# RNNTransformer on ordered KIP-M4 and KIP-M6

This batch fills the RNNTransformer rows of the KIP classification tables.
It trains three independent seeds (9550, 9551, 9552) on each task, using all
400,000 training examples, 50,000 validation examples and 50,000 test examples
from `data/simulation/kip_ordered_recovered`. Each fit starts from random weights.

## Architecture and protocol

The architecture is the repository's code-optimal `RNNTransformerClassifier`
from `src/experiments/DL_TR_baselines_experiment.py`: embedding width 64;
one bidirectional LSTM layer with hidden width 32 per direction; projection to
64; learned position embeddings; one Transformer encoder layer with two heads,
feedforward width 256 and GELU; masked mean pooling, LayerNorm and a two-logit
classifier. Dropout is 0.1. The actual trainable parameter count is **82,498**.
The larger architecture described in the old paper appendix is a different
configuration and must not be used to label this run.

Letters A through Z use IDs 1 through 26; only ID 0 is padding. This corrects
the legacy dataset's A=0/padding=0 collision. A is one of the M4 rule keys.
All sequences have exactly 20 events, so no real padding or truncation occurs.
Rule information is used solely to validate the input labels; only the event
token IDs are given to the model.

Optimization uses Adam, learning rate 0.001, no weight decay, batch size 64,
FP32, and at most 30 epochs. Checkpoints strictly maximize validation F1 at
argmax, with patience 5. This follows the existing RNNTransformer OC recipe.
It differs from the original KIP LSTM/Transformer patience-3 condition and must
be identified as patience 5. No hyperparameter is selected using these tests.
Final test AUC, F1, precision, recall, accuracy and loss are computed once after
fitting, from the selected checkpoint. Argmax ties predict class 0. This batch
does not use the separate Qwen/Llama data-efficiency test-AUC stopping rule.

Three-seed means and sample standard deviations are emitted only when all three
fits for a task have completed. The six production fits exclude every smoke
and dry run. Predictions retain full precision; the generated table rounds to
three decimals. AUC uses positive-class probabilities, with a numerically stable
CPU softmax from the saved FP32 logits. Threshold metrics use argmax.

## Durability and scheduling

`scripts/run_kip_rnn_transformer.py` provides a detached supervisor. It uses
the same exclusive GPU lock as the existing large-model queues, preserving
their active jobs. Queue order at the next available GPU slot is determined
by lock acquisition; no live task is interrupted. While waiting, the queue
has no GPU allocation. Two GPU smoke runs execute before the six production
fits (all M4 seeds, followed by all M6 seeds).

The supervisor and children inherit queue/GPU locks to prevent duplicate
execution even if the supervisor exits. PyTorch allocations are capped at
2 GiB per worker and CPU thread pools at four threads. Disk headroom must
remain at least 400 MiB before each job. Sources and inputs are hashed and
verified against the immutable launch plan.

Every completed epoch atomically saves a resume checkpoint containing model,
optimizer, random states, minibatch-generator state, selected weights and
training history. Rerunning the detached launcher can resume an interrupted
epoch from the previous complete epoch. Completed jobs are verified and skipped.
Checkpoints and compressed test predictions are retained. The VM must stay on;
disconnecting SSH or the client does not stop the detached supervisor.

```bash
python -B scripts/run_kip_rnn_transformer.py --prepare
python -B scripts/run_kip_rnn_transformer.py --background
python -B scripts/run_kip_rnn_transformer.py --status
```

Preparation runs CPU checks on both full datasets and isolated CPU training
smokes, then freezes `launch_plan.json`. GPU checks happen automatically after
the GPU lock becomes available. Logs, manifests, per-fit metrics, `per_seed.csv`,
`summary.csv`, and `table.tex` are saved under
`results/kip_rnn_transformer_20260924/`.

## GPU preflight repair

The first queued GPU smoke failed before model transfer, forward/backward, or
any production fit. PyTorch's allocator API requires an indexed CUDA device;
the unindexed `cuda` argument was rejected. The repaired worker uses `cuda:0`.
CPU regression tests check this API requirement. Smoke evaluation now uses
512 validation and test examples so it exercises the full production evaluation
batch size as well as the training batch of 64. All old launch plans, manifests,
logs and failed-smoke artifacts are retained under `repair_20260924/before/`.
The repaired plan is freshly prepared from the same data and unchanged recipe.
