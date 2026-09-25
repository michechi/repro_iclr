# Synthetic sequence generators

The Python files in this directory preserve the historical synthetic generators
and diagnostics. Run legacy scripts from the outer `ICLR` directory unless a
script specifies otherwise; inputs and outputs use relative `data/` paths.
No generated datasets are included in this directory. Several legacy scripts
execute generation or analysis immediately when imported: inspect their main
block and arguments before running them.

The current ordered-composition oracle and configurable generators are in
`../repro/src/oc_completion/`. From `ICLR/repro`, use
`python -m src.oc_completion.gen_datasets --smoke --n_workers 1` for a small
CPU generation run. This creates synthetic inputs without model training.
