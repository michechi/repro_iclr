# Anonymous release review

This package is an anonymous snapshot, with no version-control history. Original
research files outside the release were left unchanged. The review concerns the
released artifacts; it does not certify every manuscript claim. See
[MANUSCRIPT_GAPS.md](MANUSCRIPT_GAPS.md) for known scientific and build gaps.

## Decisions

- Author headers use anonymous attribution. Personal email addresses, private
  repository references, absolute workstation paths, process identifiers, and
  local scheduling notes were excluded or replaced with portable instructions.
- Model identifiers, public upstream citations, package versions, scientific
  dates, seeds, and pinned model revisions remain available for reproducibility.
  Upstream legal attribution must not be erased to achieve anonymity. Public
  CCS mapping provenance is retained separately from the project code license.
- No raw MIMIC records, patient-level derived sequences, patient identifiers,
  split membership, predictions, example heatmaps, clinical model weights, logs,
  credentials, environment directories, or machine caches are distributed.
  Clinical pipelines require the reader's own credentialed inputs.
- Clinical result JSON files were reviewed as aggregate metric dictionaries.
  Their permitted arrays describe model layer numbers, not patient identities.
  The CKD attention figure CSV contains 144 model/condition/layer summaries with
  patient counts, mean event-attention mass, and normalized token entropy.
  The corresponding figure was visually checked: it shows aggregate curves,
  with no individual histories, labels, identifiers, or patient examples.
- Synthetic attention manifests can retain synthetic sample indices. These do
  not refer to patients. Synthetic datasets and public code mappings are distinct
  from the excluded clinical data.
- PNG text metadata was inspected; only plotting-software information and
  anonymous creator metadata are retained. The three regenerated attention PDFs
  have empty author fields. No original manuscript PDFs are distributed.
- Symbolic links are excluded. Source transformation hashes record provenance;
  final release integrity is established by `MANIFEST.sha256`.

## Second review pass

A second, independent review of the assembled snapshot used nine reading
lenses (personal and institutional identity, submission history, operational
metadata in results, metadata inside images/PDF/SVG/parquet, non-English text,
completeness against the manuscript, documentation consistency, hard-coded
environment in code, and clinical row-level content), followed by a fix pass.
Findings and actions:

- Cloud-VM hostnames survived in two result CSV columns and in the trainers
  that wrote them; the columns now read `redacted` and the trainers record a
  neutral label (see CODE_PORTABILITY.md).
- References to earlier submission venues, a reviewer identifier, review-cycle
  wording, coding-assistant process notes and a cluster name in one filename
  were removed or renamed, including inside the bibliography header, the
  provenance log, docstrings, one clinical pipeline specification and one
  scheduler template.
- Non-English (Italian) comments, docstrings and user-facing strings in 47
  legacy modules were translated; program structure was verified unchanged.
- The private project name was removed from an environment-variable name, and
  private virtual-environment names from docstrings.
- Image, PDF, SVG and parquet metadata were parsed again: only plotting-software
  and anonymous-creator fields remain. Clinical aggregates were re-read: no
  patient-level rows, identifiers, predictions or split lists are present. Two
  illustrative examples are now labelled as synthetic placeholders.
- Retained on purpose: GPU model names and package versions in audit
  manifests (hardware provenance), relative `data/model_cache/...` model source
  paths (the cache itself is not distributed), and singleton measurement-count
  cells in the AKI cohort-flow audits (counts of rejected laboratory rows, not
  patient counts).

## Repeatable checks

From the release root, run:

```bash
python -B scripts/verify_release.py --require-manifest
```

The verifier uses only the Python standard library. It checks all manifest
hashes and inventory coverage, links and forbidden artifacts, generic credential
and private-path patterns, author headers, text and common image metadata, and
patient-level field names in clinical aggregates. Findings report relative
filenames and issue classes without exposing matching values. During assembly,
omit `--require-manifest` until the final manifest exists.

An optional `--identity-file` accepts a private list of additional identifying
strings. Such a list must stay outside this package; identities are never
embedded in the public verification script or findings.

These automated checks are deliberately limited. They do not establish legal
permission for third-party material, detect every possible identifier in opaque
files, verify empirical claims, or replace visual and scientific review. Newly
added clinical artifacts require a fresh manual review even if the scanner passes.
