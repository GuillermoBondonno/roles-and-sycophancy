# Userness probe steering experiment

This repository tests an exploratory causal hypothesis: a linear probe that
diagnostically separates native chat roles may define a *candidate steering
direction*. Adding or subtracting that direction might change sycophancy, but
only counts as useful evidence if output remains coherent and does not collapse
into refusal, truncation, or generic degradation.

The methodological foundation is [Prompt Injection as Role
Confusion](https://arxiv.org/abs/2603.12277) and its [reference
implementation](https://github.com/role-confusion/prompt-injection-as-role-confusion).
Evaluation uses [Inspect AI](https://inspect.aisi.org.uk/) and the existing
[`inspect_evals.sycophancy`](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/sycophancy)
task. The target can be any compatible Hugging Face decoder-only causal LM.

## Method

`run.py` samples 250 distinct, neutral, non-instructional C4 passages, truncates
them to at most 1,024 tokens, and splits base-text IDs before making any role
copies. Each text is rendered through the tokenizer's real chat template using
the roles configured in `ROLES`; no plain-string role simulation is used. A fast
tokenizer's character offsets locate the one exact base-content span.
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

The activation site is exactly the input of the selected decoder layer's
feed-forward module. Automatic discovery recognizes common decoder layouts and
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
Userness on held-out activations. An optional mean-difference direction is saved
for comparison in raw, unit-normalized, and held-out-scaled forms, and an
optional fixed-seed, equal-norm random control can be evaluated by setting
`RUN_RANDOM_DIRECTION_CONTROL = True`.

During generation the feed-forward-input pre-hook changes only the final
position in the initial prefill and the current position in each cached
autoregressive step. It
never changes weights or shared tensors, keeps dtype/device unchanged, and is
removed in `finally`. The same loaded model instance is used for all
coefficients.

## Evaluation

The local model is passed directly to Inspect through an in-process `ModelAPI`.
It preserves Inspect message roles and full multi-turn history, applies the
model's native chat template, and calls the same `SteeredModel` instance. This
supports the sycophancy task's second-turn “Are you sure?” interaction. Evaluations use
greedy cached generation, fixed prompts, fixed order, fixed seed, cache disabled,
and concurrency one.

The existing sycophancy task is used unchanged with
`scorer_model=JUDGE_MODEL`. Its summary columns are answer confidence, apologize
rate, and truthfulness. A fixed 24-prompt local Inspect task covers explanation,
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

Set any credential needed by `JUDGE_MODEL`. The default uses the free OpenRouter
variant of NVIDIA Nemotron Nano 9B V2 and requires `OPENROUTER_API_KEY`:

```dotenv
OPENROUTER_API_KEY=your-openrouter-api-key
```

Save that entry in a `.env` file at the repository root. The application loads
it automatically; `.env` is ignored by Git. An environment variable already
set in the shell takes precedence over the value in the file.

Free-model availability and rate limits can vary. If this model is retired,
replace `JUDGE_MODEL` with another `openrouter/<model-id>:free` model from the
OpenRouter catalog.

Transient judge failures do not become zero-valued observations. Inspect
retries HTTP 429 and other retryable provider failures with exponential
backoff. `MODEL_API_MAX_RETRIES` defaults to 5 retries after the initial
request. If those retries are exhausted, that sample is recorded as an error
in the raw Inspect log and evaluation continues; summary metrics use only the
successfully scored samples. Evaluation remains serial (`max_samples=1`) to
avoid creating a burst of judge requests. The run logs the completed/requested
sample count for partial evaluations and refuses to produce a summary only when
an evaluation has no usable samples at all. Increase or decrease
`MODEL_API_MAX_RETRIES` in `config.py` to change this policy.

Edit the plainly named constants near the top of `config.py`, then run:

```bash
python run.py
```

The default is a small, openly available instruct model so the pipeline is easy
to exercise. Set `MODEL_NAME` and choose a valid zero-indexed `TARGET_LAYER` for
the substantive experiment. The target model must:

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
  base_sequences.jsonl
  activation_metadata.jsonl
  probe.npz
  probe_metadata.json
  probe_diagnostics.json
  userness_steering_diagnostic.{csv,json}
results/
  inspect_logs/alpha_*/       # complete raw Inspect JSON logs
  per_sample/*.jsonl          # prompts, outputs, judge scores/metadata
  summary.csv
  summary.json
  random_direction_control/   # only when enabled
```

The summary is indexed by coefficient and contains
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
one chosen layer rather than all layers, and deterministic balanced token
subsampling. The paper also requires zero-shot conversational validity for a
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
