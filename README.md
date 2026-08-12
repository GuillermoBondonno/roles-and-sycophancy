# Steering a "Userness" direction in Qwen3-8B

> **Result:** the intervention strongly moved the probe's internal readout and
> changed many generations, but it did **not** produce a consistent directional
> change in sycophancy.

This is an exploratory activation-steering follow-up to [*Prompt Injection as
Role Confusion*](https://arxiv.org/abs/2603.12277) and its [reference
implementation](https://github.com/role-confusion/prompt-injection-as-role-confusion).
That paper uses role probes diagnostically: it asks whether a model internally
represents some text as coming from a user, assistant, system, or chain of
thought. This repository asks a separate causal question:

**If we move a model's hidden state along a direction that a role probe calls
more or less "user-like," does the model become more or less sycophantic?**

I tested Qwen3-8B at layers 9, 18, and 27. The short answer is no—not reliably,
and not with a stable sign across layers. That null result is still useful: a
feature can be linearly readable, and a related intervention can change text,
without the probe direction being a clean causal control for the behavior we
care about.

![Truthfulness under Userness probe steering](docs/figures/probe_truthfulness.svg)

## At a glance

- **Model:** `Qwen/Qwen3-8B`
- **Direction:** `w_user - mean(w_assistant, w_system)` from a multinomial
  logistic role probe
- **Intervention site:** feed-forward input at layers 9, 18, and 27
- **Steering strengths:** alpha = -5, -2.5, 0, 2.5, 5, scaled in held-out
  projection standard deviations
- **Behavioral evaluation:** 100 fixed
  [`inspect_evals.sycophancy`](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/sycophancy)
  samples per condition
- **Quality control:** 24 fixed coherence prompts per condition, judged on a
  1–5 rubric

Mean truthfulness across the three layer-level sweeps was nearly flat. These
are means of three percentages computed on the same fixed 100 prompts, not 300
independent samples:

| Steering coefficient | Mean truthfulness |
|---:|---:|
| -5.0 | 85.3% |
| -2.5 | 85.3% |
| 0.0 | 87.3% |
| +2.5 | 85.0% |
| +5.0 | 85.7% |

The null was not caused by an inert intervention or obvious model collapse:

- held-out role-classification accuracy was 64.2–68.7% across the three layers;
- subtracting the direction at alpha = -5 reduced mean probe-predicted
  Userness from about 0.34 to at most 0.000003;
- at absolute alpha = 5, roughly 50–82% of exact sycophancy outputs changed;
- coherence failures were zero in every reported probe condition, with mean
  scores between 4.76 and 4.87 out of 5; and
- repeated judging of identical alpha-zero outputs disagreed on 3 of 100
  truthfulness labels, comparable to most observed one-to-three-point changes.

The appropriate conclusion is narrow: **this experiment does not support the
claim that this linear Userness direction controls sycophancy in Qwen3-8B.** It
does not show that role representations are behaviorally irrelevant, that no
better Userness direction exists, or that another layer, model, steering rule,
or evaluation would also yield a null result.

## Why this repository may be useful

The code is designed as a scaffold for follow-ups rather than as a polished
one-off notebook. It supports:

- any compatible Hugging Face decoder-only causal language model;
- native chat templates and strict token alignment across roles;
- layer sweeps and arbitrary steering coefficients;
- probe-weight, mean-difference, and equal-norm random directions;
- separate generation and judge-scoring phases;
- resumable Inspect evaluations and per-sample exports; and
- coherence checks that help distinguish semantic steering from generic damage.

Good next experiments include trying other models, training or selecting a
direction with causal rather than diagnostic criteria, changing the sycophancy
task and judge, comparing residual-stream and MLP sites, and evaluating stronger
random or matched-norm controls.

## Important validity note

An interrupted August 7 mean-difference run at layer 27 exposed a control bug:
the supposedly fixed first assistant answer was regenerated separately for each
coefficient, and the pre- and post-resume processes did not reproduce that
answer exactly. The resulting comparison confounded coefficient sign with two
generation regimes. That run is excluded from the headline result above.

The probe-direction runs used above were checked directly: for every layer and
all five coefficients, all 100 first assistant responses were byte-identical.
The implementation should still be hardened by generating each first response
once and replaying it for every coefficient. See
[`findings.md`](findings.md) for the forensic analysis and recommended fix.

## Repository map

| Path | Purpose |
|---|---|
| `run.py` | train probes and generate steered responses |
| `probe.py` | role construction, activation capture, and probe fitting |
| `steered_model.py` | scoped activation intervention and Inspect adapter |
| `score_results.py` | judge generated logs and export summaries |
| `plot_results.py` | plot layer or sweep results |
| `evals.py` | local coherence task and scoring helpers |
| `artifacts/` | reusable probes and compact diagnostics from the reported run |
| `docs/data/` | compact data behind the headline figure |

## Method

The reported run sampled 250 distinct, neutral, non-instructional C4 passages,
truncated them to at most 1,024 tokens, and split base-text IDs before making
any role copies. The sample count is configurable. Each text is rendered
through the tokenizer's real chat template using the roles configured in
`ROLES`; no plain-string role simulation is used. A fast tokenizer's character
offsets locate the one exact base-content span.
Boundary-crossing tokens, template tokens, and role tags are excluded, and
retained tokens must decode back to their exact covered content substring. If a
template merges final content with a suffix, the mapper drops that mixed token
from every role copy and retains the longest identical content-token prefix
across all roles.

The role wrappers can have unavoidable differences in native structural-token
count. This implementation uses each model's native role rendering rather than
inserting role-specific filler, which would introduce a different activation
confound. Content token IDs and retained token ordinals are asserted equal
across roles.

The activation site is exactly the input of each selected decoder layer's
feed-forward module. `TARGET_LAYER_FRACTIONS` configures a sweep by relative
model depth. A fraction `f` maps to `min(int(f * n_layers), n_layers - 1)`, so
`0.5` selects layer 14 in a 28-layer model and `1.0` selects layer 27. Fractions
that collapse to the same actual layer are rejected rather than silently
repeating work. Automatic discovery recognizes common decoder layouts and
feed-forward names. `STEERING_MODULE_PATH` provides an explicit dotted-path
override for other architectures. A small forward pre-hook extracts this tensor;
the model's forward pass is not reimplemented.
The probe is unstandardized multinomial logistic regression:

```text
P(role | h) = softmax(W h + b)
Userness(h) = P(user | h)
```

L2 regularization strength lambda is selected from `1e-4` through `1e3`
(`sklearn C = 1 / lambda`) using held-out base sequences. Diagnostics include
validation accuracy, per-role recall, a confusion matrix, and per-role
validation Userness distributions. To keep the CPU fit manageable, at most 128
deterministically spaced content tokens per base sequence and role are retained.
This explicit token subsampling is a deviation from the paper.

The primary candidate direction is not the raw user row. It is

```text
v_raw = w_user - mean(w_r for r != user)
v_hat = v_raw / ||v_raw||_2
scale = sample_std(h_validation @ v_hat)
h' = h + alpha * scale * v_hat
```

Positive `alpha` adds candidate Userness; negative `alpha` subtracts it; zero
runs through the same hook with a zero shift. Before behavioral evaluation, the
probe artifact records how every configured `alpha` changes probe-predicted
Userness on held-out activations. A mean-difference alternative is also saved:

```text
v_mean = mean(h_train | user) - mean(h_train | non-user)
v_mean_hat = v_mean / ||v_mean||_2
mean_scale = sample_std(h_validation @ v_mean_hat)
```

It is stored in raw, unit-normalized, and held-out-scaled forms. An optional
fixed-seed, equal-norm random control can be evaluated by setting
`RUN_RANDOM_DIRECTION_CONTROL = True`.

During generation the feed-forward-input pre-hook changes only the final
position in the initial prefill and the current position in each cached
autoregressive step. In the sycophancy task, the initial assistant answer is
always generated with an applied coefficient of zero; the configured
coefficient is applied only while generating the second assistant response
after “Are you sure?”. Coherence responses remain steered so they continue to
measure intervention quality. Separate Inspect model providers encode these
two scopes, and the scope is saved in log metadata and generation manifests so
incompatible old logs cannot be reused during resume. The hook never changes
weights or shared tensors, keeps dtype/device unchanged, and is removed in
`finally`. The same loaded model instance, base passages, role variants, and
train/validation split are used for all layers and coefficients. Activations
and probes are computed separately at every resolved layer.

## Evaluation

The local model is passed directly to Inspect through an in-process `ModelAPI`.
It preserves Inspect message roles and full multi-turn history, applies the
model's native chat template, and calls the same `SteeredModel` instance. This
supports the sycophancy task's second-turn “Are you sure?” interaction. Evaluations use
greedy cached generation, fixed prompts, fixed order, fixed seed, cache disabled,
and concurrency one.

Generation and judging are separate phases. `run.py` uses the GPU only to fit
the probes and generate both turns of every sycophancy conversation plus every
coherence response for every layer/coefficient pair. It calls Inspect with
`score=False`, writes one portable generation manifest per layer, and never
contacts the judge. After the logs are downloaded, `score_results.py`
recursively discovers and scores every layer with the independent judge; it
does not load or reconstruct the target model.

The existing sycophancy scorer provides answer confidence, apologize rate, and
truthfulness. The first two aggregates are recomputed from complete per-sample
score dictionaries because Inspect Evals 0.16's registered conditional metrics
incorrectly produce zero with current Inspect's named scalar reductions. A
fixed 24-prompt local Inspect task covers explanation,
supplied-text summarization, comparison, planning, argument, procedure, and
passage continuation. An independent, unsteered judge scores grammar, local and
global consistency, relevance, and non-degeneracy from 1–5. Its rubric excludes
agreement, politeness, ideological position, and factual correctness except for
internal contradiction. The summary reports mean score, population standard
deviation, fraction at or below 2, and mean target-tokenizer output length.

## Installation

Use Python 3.11 or newer (required by current Inspect Evals). Install PyTorch
for the machine first; it is intentionally absent from `requirements.txt` so a
CPU environment does not accidentally install CUDA wheels, and neither path
requires torchvision.

```bash
python -m venv .venv
source .venv/bin/activate

# CPU-only example:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For a GPU, instead use the matching command from pytorch.org/get-started.
pip install -r requirements.txt
```

For the local scoring phase, set any credential needed by `JUDGE_MODEL`. The
default uses GPT-4.1 Mini through OpenRouter and requires
`OPENROUTER_API_KEY`:

```dotenv
OPENROUTER_API_KEY=your-openrouter-api-key
```

Save that entry in a `.env` file at the repository root. The application loads
it automatically; `.env` is ignored by Git. An environment variable already
set in the shell takes precedence over the value in the file.

Model availability and pricing can vary. If needed, replace `JUDGE_MODEL` with
another OpenRouter model identifier.

Local judge requests are serialized and Inspect retries HTTP 429 and other
retryable provider failures with exponential backoff. Scored samples are
checkpointed under `scored_logs/` after every successful judge result;
rerunning `score_results.py` skips them and resumes at the first missing sample.
A judge failure therefore never requires another paid GPU generation run.
Judge responses use deterministic temperature zero and are capped separately by
`JUDGE_MAX_TOKENS`; target-model responses use `MAX_NEW_TOKENS`. Increase or
decrease `MODEL_API_MAX_RETRIES` in `config.py` to change retry policy.

Edit the plainly named constants near the top of `config.py`. The exact
headline-run settings were:

```python
MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER_FRACTIONS = (0.25, 0.5, 0.75)
N_BASE_SEQUENCES = 250
STEERING_COEFFICIENTS = (-5.0, -2.5, 0.0, 2.5, 5.0)
SYCOPHANCY_SAMPLE_LIMIT = 100
COHERENCE_SAMPLE_LIMIT = 24
SEED = 42
```

Then run on the GPU machine:

```bash
python run.py
```

If a run is interrupted after a layer's probe artifacts were saved, resume only
that layer's response generation with its fraction. This loads `probe.npz`,
skips activation extraction and probe fitting, reuses complete task logs, and
uses Inspect's retry support to preserve completed samples from an interrupted
task log:

```bash
python run.py --resume-generation 0.75
```

Each layer result directory also contains
`evaluation_sample_contract.json`, which fixes the ordered sample IDs, a hash
of every source sample, and each task's steering scope before generation
begins. Fresh, reused, and retried logs are checked against this contract before
they can enter the generation manifest. Resume also clears Inspect 0.3.251's
task-level `sample_shuffle=False` override before `eval_retry`: that version
otherwise treats the false value as integer seed zero and shuffles the retry
dataset. New evaluations rely on each task's explicit `shuffle=False` setting
and leave Inspect's separate retry-sensitive shuffle override unset.

Use explicit roots when the downloaded checkpoint directories have different
names, as in the August 2 copy in this checkout:

```bash
python run.py --resume-generation 0.75 \
  --artifacts-dir artifacts_aug_2_v2 \
  --results-dir results_aug_2_v2
```

To add coefficients to every saved layer in an existing sweep, pass only the
new coefficients with `--resume-all`. The model is loaded once, complete logs
are reused, and new task entries are merged into each existing generation
manifest. Neither the saved probe metadata nor earlier results are rewritten:

```bash
python run.py --resume-all \
  --artifacts-dir artifacts_aug_2_v2_combined \
  --results-dir results_aug_2_v2_combined \
  --steering-coefficients -8 -6 -4 -2
```

This command is idempotent: rerunning it resumes partial logs and reuses complete
ones. The result directory remains equivalent to one sweep containing both the
old and new coefficients. Coefficients must be finite and unique; values that
would produce the same output-directory slug are rejected.

To generate from the saved mean-difference direction instead of the logistic
probe direction, select it during resume. Use a new result root: generation
logs and manifests record the direction identity and deliberately reject a
directory containing a different steering direction.

```bash
python run.py --resume-all \
  --artifacts-dir artifacts_aug_2_v2_combined \
  --results-dir results_mean_difference \
  --steering-direction mean-difference \
  --steering-coefficients -3 -1.5 0 1.5 3

python score_results.py results_mean_difference
```

The mean-difference direction uses its own saved held-out projection scale, so
the coefficient retains the same standard-deviation interpretation as in the
probe-direction sweep. `probe` remains the default direction for backward
compatibility with existing runs.

Download the resulting `results/` directory to a local checkout with the same
dependencies and run:

```bash
python score_results.py results
```

Plot either the complete layer sweep or any individual layer as a four-panel
dashboard. Plotting a sweep creates one dashboard in each layer directory. The
default output is `<layer-results>/summary_plot.png`; `.pdf` and `.svg` outputs
are also supported.

```bash
python plot_results.py results
python plot_results.py results/layer_14_fraction_0p5
python plot_results.py results/layer_14_fraction_0p5 --output figures/experiment.svg
```

For Qwen models that expose a thinking-mode chat-template option, disable it in
`CHAT_TEMPLATE_KWARGS` or provide a generation budget large enough for both the
reasoning and final answer. A log of truncated `<think>` content is not a valid
behavioral observation.

```python
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
```

Set `MODEL_NAME` and edit the layer sweep for the substantive experiment:

```python
TARGET_LAYER_FRACTIONS = (0.25, 0.5, 0.75)
```

The target model must:

- load through `AutoModelForCausalLM`;
- have a fast tokenizer with offset mappings and a native chat template;
- support every role listed in `ROLES`; and
- pass hidden states shaped `[batch, sequence, hidden]` as the first positional
  input to the selected feed-forward module.

Most standard decoder-only Transformers models are discovered automatically.
For a custom layout, set for example:

```python
STEERING_MODULE_PATH = "transformer.blocks.{layer}.feed_forward"
```

`TOKENIZER_LOAD_KWARGS`, `MODEL_LOAD_KWARGS`, and `CHAT_TEMPLATE_KWARGS` are
passed through to Transformers. They can enable remote model code, quantization,
or model-specific template options without adding architecture-specific logic.

## Outputs

```text
artifacts/
  base_sequences.jsonl                  # shared sweep inputs and split
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
    generation_manifest.json             # portable GPU-to-local handoff
    inspect_logs/alpha_*/                 # unscored generation logs
    scored_logs/alpha_*/                  # locally judged Inspect logs
    per_sample/*.jsonl                    # outputs and judge metadata
    summary.{csv,json}
    random_direction_control/             # only when enabled
  layer_sweep_summary.{csv,json}          # produced by score_results.py
```

Each layer summary is indexed by coefficient and also records `target_layer`
and `target_layer_fraction`. The sweep summary combines all layer summaries and
contains
`sycophancy_answer_confidence`, `sycophancy_apologize_rate`,
`sycophancy_truthfulness`, `coherence_mean`, `coherence_std`,
`coherence_failure_rate`, and `mean_output_tokens`.

## Caveats and deviations

The original paper uses role probes diagnostically and does **not** perform this
steering experiment. Successful steering would be stronger causal evidence,
but would not prove that this probe identifies the complete role mechanism.
Probe weights are discriminative directions and can fail as causal steering
vectors. Reductions in sycophancy are not useful if caused by incoherence,
refusal, truncation, verbosity changes, or broad model damage.

The detailed paper appendix specifies 250 base sequences, while an earlier
overview footnote reports 1,250; this repository follows the appendix and the
experiment specification here. It uses C4 only (the paper uses C4 and Dolma3),
configured layer fractions rather than necessarily every layer, and
deterministic balanced token subsampling. The paper also requires zero-shot
conversational validity for a
probe used as a diagnostic instrument; this steering study reports its own
held-out synthetic validity and behavioral/coherence controls, but does not
claim to reproduce every paper validity experiment.

The current implementation was written against Inspect AI's current
`ModelAPI.generate(input, tools, tool_choice, config)` interface and the current
dictionary-valued sycophancy scorer. Inspect and Inspect Evals are fast-moving;
rerun the import tests after dependency upgrades. At implementation time this
was Inspect AI 0.3.249 / Inspect Evals 0.16.0 / Transformers 5.14.1; the
methodology repository was inspected at commit
`ec333c40fd43fe991e1ebf66765051b6d7e35784`. Probe capture and intervention now
share the same resolved feed-forward input module, so architecture support does
not depend on a separately named normalization layer.
