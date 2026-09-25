# Sensitivity Experiment Results

## Setup
- **Data**: simulated sequences
- **Training samples**: 320,000 (80/10/10 split → Train: 320k, Val: 40k, Test: 40k)
- **Models**: Transformer, LSTM, RNNTransformer
- **Training**: up to 30 epochs, patience=3, batch_size=64, optimal hyperparameters
- **Seed**: 9550

## Dataset Key
- The `number_to_use` parameter controls the dataset configuration
- Datasets 100–103 vary the **sequence length**: 10, 15, 20, 30
- Datasets 110–144 all have sequence length 20 but vary other simulation parameters

## Results

| Dataset | Seq len | Model          | AUC    | F1     | Precision | Recall |
|---------|---------|----------------|--------|--------|-----------|--------|
| 100     | 10      | Transformer    | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 100     | 10      | LSTM           | 1.0000 | 0.9986 | 0.9978    | 0.9994 |
| 100     | 10      | RNNTransformer | 1.0000 | 0.9999 | 1.0000    | 0.9997 |
| 101     | 15      | Transformer    | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 101     | 15      | LSTM           | 0.9999 | 0.9979 | 0.9980    | 0.9979 |
| 101     | 15      | RNNTransformer | 1.0000 | 0.9998 | 0.9998    | 0.9998 |
| 102     | 20      | Transformer    | 1.0000 | 0.9998 | 0.9997    | 0.9999 |
| 102     | 20      | LSTM           | 0.9997 | 0.9894 | 0.9899    | 0.9890 |
| 102     | 20      | RNNTransformer | 0.9999 | 0.9947 | 0.9951    | 0.9943 |
| 103     | 30      | Transformer    | 1.0000 | 0.9997 | 1.0000    | 0.9993 |
| 103     | 30      | LSTM           | 0.9998 | 0.9948 | 0.9925    | 0.9971 |
| 103     | 30      | RNNTransformer | 0.9858 | 0.9310 | 0.9202    | 0.9419 |
| 110     | 20      | Transformer    | 1.0000 | 0.9915 | 0.9990    | 0.9842 |
| 110     | 20      | LSTM           | 1.0000 | 0.9937 | 0.9926    | 0.9948 |
| 110     | 20      | RNNTransformer | 1.0000 | 0.9956 | 0.9963    | 0.9948 |
| 112     | 20      | Transformer    | 0.9991 | 0.9984 | 0.9995    | 0.9972 |
| 112     | 20      | LSTM           | 0.9991 | 0.9875 | 0.9880    | 0.9870 |
| 112     | 20      | RNNTransformer | 0.9997 | 0.9915 | 0.9969    | 0.9861 |
| 120     | 20      | Transformer    | 1.0000 | 0.9990 | 0.9991    | 0.9988 |
| 120     | 20      | LSTM           | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 120     | 20      | RNNTransformer | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 121     | 20      | Transformer    | 0.9999 | 0.9926 | 0.9992    | 0.9861 |
| 121     | 20      | LSTM           | 1.0000 | 0.9992 | 0.9993    | 0.9991 |
| 121     | 20      | RNNTransformer | 1.0000 | 0.9962 | 0.9956    | 0.9967 |
| 122     | 20      | Transformer    | 0.9980 | 0.9585 | 0.9779    | 0.9398 |
| 122     | 20      | LSTM           | 1.0000 | 0.9955 | 0.9936    | 0.9974 |
| 122     | 20      | RNNTransformer | 0.9999 | 0.9954 | 0.9976    | 0.9931 |
| 123     | 20      | Transformer    | 1.0000 | 0.9998 | 0.9995    | 1.0000 |
| 123     | 20      | LSTM           | 0.9998 | 0.9903 | 0.9913    | 0.9894 |
| 123     | 20      | RNNTransformer | 1.0000 | 0.9989 | 0.9992    | 0.9986 |
| 124     | 20      | Transformer    | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 124     | 20      | LSTM           | 0.9997 | 0.9865 | 0.9844    | 0.9887 |
| 124     | 20      | RNNTransformer | 1.0000 | 0.9994 | 0.9997    | 0.9991 |
| 130     | 20      | Transformer    | 0.9997 | 0.9741 | 0.9840    | 0.9644 |
| 130     | 20      | LSTM           | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 130     | 20      | RNNTransformer | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 131     | 20      | Transformer    | 0.9998 | 0.9774 | 0.9894    | 0.9657 |
| 131     | 20      | LSTM           | 1.0000 | 0.9992 | 0.9998    | 0.9986 |
| 131     | 20      | RNNTransformer | 1.0000 | 0.9990 | 0.9990    | 0.9990 |
| 132     | 20      | Transformer    | 0.9999 | 0.9858 | 0.9913    | 0.9805 |
| 132     | 20      | LSTM           | 1.0000 | 0.9966 | 0.9969    | 0.9962 |
| 132     | 20      | RNNTransformer | 0.9998 | 0.9844 | 0.9819    | 0.9869 |
| 133     | 20      | Transformer    | 1.0000 | 0.9993 | 0.9994    | 0.9992 |
| 133     | 20      | LSTM           | 0.9999 | 0.9882 | 0.9933    | 0.9831 |
| 133     | 20      | RNNTransformer | 1.0000 | 0.9963 | 0.9986    | 0.9939 |
| 134     | 20      | Transformer    | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 134     | 20      | LSTM           | 0.9999 | 0.9835 | 0.9911    | 0.9761 |
| 134     | 20      | RNNTransformer | 1.0000 | 0.9980 | 0.9997    | 0.9962 |
| 140     | 20      | Transformer    | 0.9994 | 0.9933 | 0.9953    | 0.9913 |
| 140     | 20      | LSTM           | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 140     | 20      | RNNTransformer | 1.0000 | 1.0000 | 1.0000    | 1.0000 |
| 141     | 20      | Transformer    | 0.9996 | 0.9980 | 0.9998    | 0.9963 |
| 141     | 20      | LSTM           | 1.0000 | 0.9995 | 0.9997    | 0.9994 |
| 141     | 20      | RNNTransformer | 0.9994 | 0.9923 | 0.9926    | 0.9920 |
| 142     | 20      | Transformer    | 1.0000 | 0.9995 | 0.9997    | 0.9993 |
| 142     | 20      | LSTM           | 0.9998 | 0.9953 | 0.9958    | 0.9949 |
| 142     | 20      | RNNTransformer | 0.9999 | 0.9973 | 0.9969    | 0.9977 |
| 143     | 20      | Transformer    | 0.9951 | 0.9945 | 0.9998    | 0.9892 |
| 143     | 20      | LSTM           | 0.9989 | 0.9864 | 0.9884    | 0.9843 |
| 143     | 20      | RNNTransformer | 0.9999 | 0.9962 | 0.9991    | 0.9933 |
| 144     | 20      | Transformer    | 0.9941 | 0.9938 | 1.0000    | 0.9876 |
| 144     | 20      | LSTM           | 0.9993 | 0.9887 | 0.9876    | 0.9899 |
| 144     | 20      | RNNTransformer | 0.9996 | 0.9943 | 0.9989    | 0.9897 |

## Notable observations
- **RNNTransformer on dataset 103 (seq=30)**: significant drop (F1=0.9310, AUC=0.9858), worst result across all experiments
- **Transformer on dataset 122**: also notable drop (F1=0.9585), while LSTM and RNNTransformer stay above 0.99
- **Perfect scores (F1=1.0)**: Transformer on datasets 100, 101, 124, 134; LSTM on 120, 130, 140; RNNTransformer on 120, 130, 140
- **Datasets 120, 130, 140**: LSTM and RNNTransformer both achieve perfect 1.0000 across all metrics, while Transformer slightly underperforms
- All models trained on 320k samples with early stopping (patience=3)
