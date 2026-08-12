# Headline result data

`probe_truthfulness.csv` is a compact extract of the clean probe-direction
layer sweep used in the repository README. The source summaries are the local
`results/probe/layer_<layer>_fraction_<fraction>/summary.csv` files.

The five conditions use the same 100 ordered sycophancy samples. Before this
extract was prepared, the manifest-selected Inspect logs were checked by sample
ID: every first assistant response was byte-identical across all coefficients,
with 100/100 matches at layers 9, 18, and 27.

Regenerate the SVG from this table with:

```bash
python docs/plot_publication_figure.py
```
