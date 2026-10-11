# Public experiment artifacts

This directory contains the compact, reusable artifacts from the reported
August 7 Qwen3-8B probe run.

```text
layer_sweep_manifest.json
layer_<index>_fraction_<fraction>/
  probe.npz
  probe_metadata.json
  probe_diagnostics.json
  userness_steering_diagnostic.csv
  userness_steering_diagnostic.json
```

Each `probe.npz` contains both the logistic-probe direction and the
mean-difference direction, including unit-normalized vectors and held-out
projection scales. It can be loaded with `run.py --resume-generation` or
`run.py --resume-all` to generate new coefficient sweeps without extracting
activations and fitting the probes again.

The public bundle intentionally excludes:

- `activation_metadata.jsonl`: roughly 19 MB per layer of reproducible
  token-position bookkeeping, not activation vectors;
- `base_sequences.jsonl`: copied source text sampled from C4; and
- raw Inspect generation and judge logs, which are much larger than the compact
  result tables used by the README.

Fresh end-to-end runs regenerate the omitted activation metadata and sample
their own C4 passages. The compact headline result data and figure live under
`docs/`.
