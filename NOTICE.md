# Notices and provenance

The `repro/` source package includes its pre-existing MIT notice. Other project source has no root-wide license declaration in the supplied project. Anonymization replaces identifying author strings with anonymous placeholders without asserting new ownership.

CCS diagnosis code mappings are derived from the public AHRQ/HCUP classification resources. These mappings contain code/category definitions only. The clinical pipelines operate on MIMIC-IV data obtained separately; this release contains no clinical row-level inputs or outputs.

Model names and tokenizer/checkpoint identifiers identify third-party models used in the experiments. Their weights are not distributed. Consult each provider for access and model terms.

`release/source_transformations.json` records the initial source-to-release copy and sanitation operations. Later packaging repairs and metadata sanitation may change the copied bytes; `MANIFEST.sha256` (generated at packaging time as the final step) records final release bytes and takes precedence for integrity verification. Removed runtime paths and artifact hashes are not a substitute for the released dataset hashes and scientific metrics.
