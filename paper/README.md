# Paper assets

`attention_appendix.tex` is the current attention appendix. All three referenced figures are supplied as PDF, PNG and SVG, together with the aggregate CSVs used to generate them. Run `python scripts/reproduce_figures.py` from the repository root to replot without training or clinical data.

`attention_preview.tex` wraps the appendix in a standard LaTeX article and uses only `geometry`, `graphicx` and `amsmath`. Compile from this directory. The release environment has no TeX installation, so source/figure references were checked but PDF compilation was not run.

`manuscript_source.tex` is an anonymized working manuscript snapshot. It is included for context, not as a verified final paper. It has simulated/unsupported entries, unresolved citations, absent ICLR style files and two absent main figure assets. These are listed in [MANUSCRIPT_GAPS.md](../docs/MANUSCRIPT_GAPS.md). The measured results inventory is the authority for the exported experimental evidence.

The supplied `references.bib` is the bibliography available with the project. No missing reference or figure was invented to make the working manuscript compile.
