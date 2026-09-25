# BERT Sensitivity Experiment Results

## Setup
- **Model**: BERT
- **Data**:  simulated sequences
- **Training samples**: 320,000 (80/10/10 split → Train: 320k, Val: 40k, Test: 40k)
- **Seed**: 9550

## Dataset Key
- Datasets 100–103 vary the **sequence length**: 10, 15, 20, 30
- Datasets 110–144 all have sequence length 20 but vary other simulation parameters

## Results

| Dataset | Seq len | Test AUC | Test F1  | Test Precision | Test Recall | Epochs | Training time (s) |
|---------|---------|----------|----------|----------------|-------------|--------|--------------------|
| 100     | 10      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 8      | 3756               |
| 101     | 15      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 20     | 9389               |
| 102     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 11     | 5163               |
| 103     | 30      | 1.0000   | 0.9999   | 0.9999         | 1.0000      | 11     | 5267               |
| 110     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 6      | 2818               |
| 112     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 20     | 9326               |
| 120     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 4      | 1887               |
| 121     | 20      | 1.0000   | 0.9999   | 0.9998         | 0.9999      | 9      | 4275               |
| 122     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 8      | 3734               |
| 123     | 20      | 1.0000   | 1.0000   | 0.9999         | 1.0000      | 8      | 3752               |
| 124     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 6      | 2844               |
| 130     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 10     | 4734               |
| 131     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 20     | 9357               |
| 132     | 20      | 1.0000   | 0.9999   | 1.0000         | 0.9998      | 18     | 8334               |
| 133     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 11     | 5167               |
| 134     | 20      | 1.0000   | 0.9999   | 0.9997         | 1.0000      | 8      | 3754               |
| 140     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 8      | 3769               |
| 141     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 11     | 5146               |
| 142     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 13     | 6111               |
| 143     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 20     | 9335               |
| 144     | 20      | 1.0000   | 1.0000   | 1.0000         | 1.0000      | 20     | 9413               |

## Notable observations
- **BERT achieves near-perfect scores on every dataset**: AUC = 1.0000 across the board, F1 >= 0.9999 everywhere
- **Only 4 datasets have any imperfection at all**: 103 (F1=0.9999), 121 (F1=0.9999), 132 (F1=0.9999, Recall=0.9998), 134 (F1=0.9999, Precision=0.9997) — all still essentially perfect
- **Training time varies a lot**: from ~1887s (dataset 120, 4 epochs) to ~9413s (dataset 144, 20 epochs)
- **Compared to Transformer/LSTM/RNNTransformer**: BERT is substantially more robust — it does not suffer the performance drops seen with those models (e.g., Transformer on dataset 122, RNNTransformer on dataset 103)
