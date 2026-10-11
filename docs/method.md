# Method and evaluation

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
supports the sycophancy task's second-turn “Are you sure?” interaction.
Evaluations use greedy generation with a KV cache, fixed prompts, fixed order,
fixed seed, Inspect's response cache disabled, and concurrency one.

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
internal contradiction. The summary reports mean score, the standard deviation
exported by Inspect, fraction at or below 2, and mean target-tokenizer output
length.

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
