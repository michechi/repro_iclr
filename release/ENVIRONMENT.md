# Validation environment

Packaging checks used Python 3.12.3 on Linux. The available scientific environment reported Torch 2.13.0, NumPy 2.5.1, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0, XGBoost 3.3.0, Matplotlib 3.11.1, Transformers 5.14.1, PEFT 0.19.1, Accelerate 1.14.0, huggingface-hub 1.24.0, safetensors 0.8.0 and Pillow 12.3.0. Bitsandbytes 0.50.2 was available for the original GPU recipes. CPU pytest was also run in an environment with Torch 2.12.1+cu130 and pytest 9.1.1.

These are observations of the available VM, not a lockfile or a promise that every historical training job used identical dependency versions. Per-run metrics/protocols carry experiment-specific provenance where available. Requirements specify supported entry points and dependency families; the installed Transformers build must recognize the requested model architecture.

The release was assembled without reinstalling dependencies, fetching weights or starting full GPU training. No LaTeX compiler was available. Exact checks and their outcomes are in VALIDATION.md.
