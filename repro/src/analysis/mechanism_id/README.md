# Mechanism identification audit

The Python scripts in `scripts/` audit legacy labels, feature-family baselines,
histogram-matched evaluation, held-out rules, parity decomposition, and oracles.
Institution-specific scheduler wrappers and generated reports are not included.

Run from `ICLR/repro`; `DATA_DIR` points at the outer `ICLR/data` directory.
Inspect each script with `--help` before invoking a full experiment. For example:

```bash
python src/analysis/mechanism_id/scripts/phase1_label_audit.py --help
python src/analysis/mechanism_id/scripts/phase2_baseline_ladder.py --help
python src/analysis/mechanism_id/scripts/phase6_oracle.py --help
```

`common.py` defines the rule and feature helpers shared by these audits. The
canonical ordered-composition implementation is also exposed by
`src.oc_completion.oracle`; top-level attention audits read the retained
mechanism source to recover the exact historical rule.
