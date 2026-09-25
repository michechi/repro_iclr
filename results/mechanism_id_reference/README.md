# Historical mechanism audit aggregates

These three CSVs retain aggregate results from the mechanism-identification audit. They contain no individual clinical or synthetic examples.

- `kip_audit.csv`: 42 feature-family/model/task evaluation rows.
- `kip_oracle_reveal.csv`: 22 feature-revelation/control evaluation rows.
- `phase1_label_audit.csv`: 66 dataset/rule agreement summaries from the historical label audit. Candidate rule agreement is a diagnostic, not a new classifier score or a replacement for the canonical corrected-OC definition.

The 42 and 22 KIP rows are also represented in the consolidated KIP review's 64 feature/oracle evaluations. Do not count the copies as additional independent experiments. The individual synthetic counterexample CSV from the same source directory is omitted from this aggregate package.
