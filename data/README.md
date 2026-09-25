# Included datasets

| Location | Meaning |
| --- | --- |
| `simulation/kip_ordered_recovered/` | Canonical KIP-M4/M6, each with 400,000 train / 50,000 validation / 50,000 test rows; recovery hashes and oracle rules included |
| `simulation/kip_shuffled/` | Corresponding shuffled-training experiment inputs |
| `simulation/oc_completion/` | Canonical corrected OC parquet splits, noise variants and generator metadata |
| `simulation/oc_hole_audit/` | Synthetic ordered-completion audit data |
| `simulation/tested/` | Historical datasets used by earlier experiments; suffixes and labels must be interpreted with the source recipes |
| `../results/qwen25_14b_fewshot_20260918/data/oc_corrected/` | Corrected OC CSV exports used by the recent classifier and attention runs |
| `mimic/codes/` and `../codes/` | Public CCS diagnosis-category mappings, not patient records |

KIP-M4 uses ordered keys `COTA`; KIP-M6 uses `GHLYIZ`. Every key occurs once in a 20-letter sequence. The outcome is one for an even number of key-order inversions. CSV events are separated by U+001F inside the `Sequences` column; labels are in `Outcome`.

Corrected OC uses keys `WDQJXU`, lag 7 and the canonical monotone-chain rule. Deterministic and noisy inputs have identical row order; noisy labels use independent flips at probability 0.3. The historical `_6` and `_9` files implement earlier labeling logic and must not be relabeled as corrected OC results. GPC's historical generator uses `WDQJXN`, including `N` rather than OC's `U`.

All sequence IDs and row indices in these synthetic datasets refer to generated examples. No clinical patient histories, identifiers, split membership, labels, or per-patient predictions are included. Clinical workflows require separately prepared inputs.
