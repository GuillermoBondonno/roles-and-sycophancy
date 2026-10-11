# Reported result data

This directory contains the compact probe-direction results used in the
repository README:

- [probe_summary.json](probe_summary.json): all 15 layer/coefficient conditions,
  including sycophancy, coherence, output-length, and intervention metadata.
- [probe_truthfulness.csv](probe_truthfulness.csv): truthfulness by coefficient
  and layer, plus the arithmetic mean across the three layers.

The source is the local `results/probe/layer_sweep_summary.json`, with
per-layer summaries under `results/probe/layer_<layer>_fraction_<fraction>/`.
The compact JSON retains the original summary values; it is not a new scoring
run. The interrupted layer-27 mean-difference run is excluded.

## Metric definitions

Truthfulness averages the evaluation's truthfulness score over all scored
samples. Answer confidence and apology rate use the conditional denominator:
the number of initial answers judged correct. These rates must not be
interpreted as percentages of all 100 prompts. See
[`sycophancy_summary`](../../evals.py) for the aggregation.

Coherence is the average of five judge dimensions on a 1–5 scale; failures
have an overall score at or below 2. `coherence_std` is the standard deviation
exported by Inspect's registered metric. `mean_output_tokens` uses the target
model's logged token usage.

All conditions use the same 100 ordered sycophancy samples and 24 coherence
prompts. Means across layers are not estimates from 300 independent prompts.

## Control check and limits

Before the original compact extract was prepared, manifest-selected Inspect
logs were compared by sample ID. Every first assistant response was
byte-identical across coefficients within each layer: 100/100 matches at
layers 9, 18, and 27. This historical check is recorded here; the raw logs are
not included, so a clean checkout cannot independently repeat it.

The probe metadata records an earlier set of steering coefficients. The
summary JSON records the coefficients actually plotted. Pass the desired
coefficients explicitly when resuming from the included probes.

## Regenerate the figures

From the repository root:

```bash
python docs/plot_publication_figure.py
python plot_results.py docs/data/probe_summary.json --apology-only \
  --title 'Apology rate under Userness steering' \
  --output docs/figures/probe_apology_rate.svg
```

Both commands run locally without model downloads or API credentials.
