# Running and resuming experiments

## Configuration

Edit `config.py` before a fresh run. The defaults match the reported study:
Qwen3-8B, 250 C4 base passages, layers at fractions 0.25/0.5/0.75,
coefficients −5/−2.5/0/+2.5/+5, 100 sycophancy samples, 24 coherence prompts,
and seed 42. Qwen3-8B has 36 decoder layers, so those fractions select
zero-indexed layers 9, 18, and 27.

Layer selection uses `min(int(fraction * n_layers), n_layers - 1)`.
Duplicate resolved layers are rejected. Generation is greedy, with thinking
disabled through `CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`.
Changing this option may require a larger token budget.

A compatible target model must load with `AutoModelForCausalLM`, provide a
fast tokenizer with offset mappings and a native chat template, support the
configured roles, and pass a `[batch, sequence, hidden]` tensor as the first
positional input to its feed-forward module. Common decoder layouts are
discovered automatically. For another layout, set `STEERING_MODULE_PATH`,
for example `"transformer.blocks.{layer}.feed_forward"`.
Tokenizer, model, and chat-template keyword arguments are configurable.

## Generate and score

Follow [the installation instructions](../README.md#quick-start), then run:

```bash
python run.py
```

A fresh run extracts activations, fits a probe at each layer, and generates
unscored Inspect logs. It does not call the hosted judge. Use a new artifact
and result location in `config.py` when training a new experiment so that the
included probes remain available.

Copy the result directory from the model machine to a local checkout with
the dependencies installed. Set the judge credential in the shell or in
`.env`; existing shell values take precedence. The default judge is
`openrouter/openai/gpt-4.1-mini` and requires `OPENROUTER_API_KEY`.

```bash
python score_results.py results
```

Scoring does not load the target model. Judge calls are serialized, use
temperature zero, and retry transient provider failures. Each successful
sample is checkpointed under `scored_logs/`; rerunning the scorer resumes
missing scores. `JUDGE_MAX_TOKENS` and `MODEL_API_MAX_RETRIES` control its
budget and retry policy.

Optional run notifications go to ntfy.sh only when you set `NTFY_TOPIC` to
your own topic. Leave it blank to disable them.

## Resume from saved probes

Generate a fresh sweep with the included probes and explicit coefficients:

```bash
python run.py --resume-all \
  --artifacts-dir artifacts \
  --results-dir results_probe_new \
  --steering-coefficients -5 -2.5 0 2.5 5
```

The explicit coefficients matter: saved probe metadata can record an earlier
sweep's coefficients. A resume otherwise uses those saved values rather than
the fresh-run defaults in `config.py`.

To resume only layer 27 in the same result directory:

```bash
python run.py --resume-generation 0.75 \
  --artifacts-dir artifacts \
  --results-dir results_probe_new \
  --steering-coefficients -5 -2.5 0 2.5 5
```

Complete logs are reused and interrupted logs use Inspect's retry support.
An `evaluation_sample_contract.json` fixes the ordered sample IDs, source
hashes, and intervention scope. It prevents mixing different datasets or
protocols during resume, but does not freeze the first assistant output.
Check first-answer identity across coefficients before interpreting results.
See [the known control and scoring issues](findings.md).

To add coefficients, pass the additional values with `--resume-all`.
Completed entries remain in the manifest; coefficients must be finite, unique,
and map to distinct output-directory names.

For a mean-difference direction, use another result root:

```bash
python run.py --resume-all \
  --artifacts-dir artifacts \
  --results-dir results_mean_difference_new \
  --steering-direction mean-difference \
  --steering-coefficients -5 -2.5 0 2.5 5

python score_results.py results_mean_difference_new
```

Each direction uses its own held-out projection standard deviation. Logs
record direction and steering scope; incompatible existing logs are rejected.

## Plot

Plotting a sweep produces one dashboard per layer:

```bash
python plot_results.py results_probe_new
python plot_results.py results_probe_new/layer_27_fraction_0p75 \
  --output /tmp/layer_27.svg
```

For a single apology-rate comparison across layers:

```bash
python plot_results.py docs/data/probe_summary.json --apology-only \
  --title 'Apology rate under Userness steering' \
  --output docs/figures/probe_apology_rate.svg
```

PNG, PDF, and SVG outputs are supported.

## Output layout

```text
artifacts/
  base_sequences.jsonl
  layer_sweep_manifest.json
  layer_<index>_fraction_<fraction>/
    activation_metadata.jsonl
    probe.npz
    probe_metadata.json
    probe_diagnostics.json
    userness_steering_diagnostic.{csv,json}
results/
  layer_sweep_manifest.json
  layer_<index>_fraction_<fraction>/
    evaluation_sample_contract.json
    generation_manifest.json
    inspect_logs/alpha_*/
    scored_logs/alpha_*/
    per_sample/*.jsonl
    summary.{csv,json}
    random_direction_control/       # only when enabled
  layer_sweep_summary.{csv,json}
```

Fresh runs produce the full layout. The included artifact bundle omits copied
C4 passages, activation bookkeeping, and raw logs; see
[artifacts/README.md](../artifacts/README.md). The summary metrics are
documented in [docs/data/README.md](data/README.md).
