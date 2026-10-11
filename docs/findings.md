# Experimental controls and the interrupted layer-27 run

The reported probe-direction results and the interrupted mean-difference run
must be treated separately. The probe results passed the first-response
identity check described in [the result-data notes](data/README.md). The
interrupted mean-difference comparison below is excluded from the reported
results. These notes preserve the investigation; the raw logs remain local.

## Additional evidence from the probe run

The original analysis reported that roughly 50–82% of exact sycophancy outputs
changed relative to α = 0 at absolute α = 5. Across five judging runs of
identical α = 0 outputs, 3 of 100 samples received inconsistent truthfulness
labels. That disagreement was comparable to most observed one-to-three-point
changes. These historical observations support caution about small effects;
the compact summary alone cannot reproduce the output comparisons or judge
repeatability analysis.

## What went wrong

The layer-27 mean-difference comparison is confounded because the supposedly
fixed, unsteered first assistant response was regenerated independently for
every steering coefficient.

The run was interrupted during alpha = -2.5 and resumed in a new process. The
old and resumed processes produced different first responses despite greedy
generation and an applied first-turn coefficient of zero. The resume logic
then combined the two generation regimes without detecting the drift:

- alpha = -5 belongs to the original process;
- the first 21 alpha = -2.5 samples belong to the original process;
- the remaining 79 alpha = -2.5 samples belong to the resumed process; and
- alpha = 0, +2.5, and +5 belong to the resumed process.

The raw-log comparison found:

| Comparison | Identical unsteered first responses |
|---|---:|
| alpha = -5 vs. original alpha = -2.5 samples | 21/21 |
| Retained alpha = -2.5 samples before vs. after retry | 21/21 |
| Newly resumed alpha = -2.5 samples vs. alpha = 0 | 79/79 |
| alpha = 0 vs. alpha = +2.5 | 100/100 |
| alpha = 0 vs. alpha = +5 | 100/100 |
| alpha = -5 vs. alpha = 0 | 34/100 |

All raw first-response events record:

```text
assistant_response_number: 1
applied_steering_coefficient: 0.0
```

Therefore, this was not an accidental first-turn steering leak. It was an
experimental-control and cross-process reproducibility failure.

The clean alpha = 0, +2.5, and +5 comparison is nearly flat at 77%, 77%, and
78% truthfulness, so there is no evidence of a large positive-steering effect.
The alpha = -5 versus alpha = 0 comparison cannot establish the effect of
negative steering because the first-turn generation regime is correlated with
the sign of alpha.

## Can layer 27 be regenerated?

Yes. The saved artifact contains the original layer-27 mean-difference vector
and projection scale, so the same probe artifact, dataset specification,
prompts, and coefficient convention can be reused.

All five coefficients must be rerun into a fresh directory. Regenerating only
the negative coefficients and comparing them to the old alpha = 0 run would
repeat the cross-process comparability problem.

Suggested command:

```bash
python run.py \
  --resume-generation 0.75 \
  --artifacts-dir artifacts \
  --results-dir results_mean_diff_layer27_rerun \
  --steering-direction mean-difference \
  --steering-coefficients -5 -2.5 0 2.5 5
```

This recreates the experimental specification, but it cannot guarantee exact
historical equivalence to the other August 7 generations. The old logs do not
record the exact Hugging Face model/tokenizer snapshot, dirty-worktree diff,
PyTorch and Transformers versions, CUDA/runtime, GPU identity, or deterministic
kernel configuration. The within-layer causal comparison remains valid if all
coefficients use an identical frozen first response.

## Recommended work before rerunning

1. Generate every sample's first assistant response once, persist it, and
   replay that exact response as the prefix for every steering coefficient.
2. Add an invariant that hashes first responses by sample ID and aborts if any
   coefficient differs from the frozen baseline.
3. Record a full runtime fingerprint: model and tokenizer revision/hash,
   repository diff hash, Torch and Transformers versions, CUDA/runtime and GPU,
   dtype, and deterministic settings.
4. If a job is resumed, verify that its runtime fingerprint matches the
   original run before accepting old samples.
5. Run all five layer-27 coefficients in one job where possible.
6. Verify first-response identity across coefficients before scoring.
7. Score only the completed logs selected by `generation_manifest.json`.

## Separate scoring/export bug to fix

`score_results.py` currently unions manifest-selected logs with every on-disk
fallback log. After a retry, this can score both a superseded partial log and
its completed replacement. Both are exported to the same per-sample filename,
so one overwrites the other.

This explains the truncated alpha = -2.5 exports:

- mean-difference layer 27: 21 per-sample rows instead of 100;
- probe layer 27: 67 per-sample rows instead of 100.

When a manifest exists, scoring should use only its selected logs. Fallback
discovery should be limited to legacy experiment directories without a
manifest, and duplicate coefficient/evaluation identities should raise an
error rather than silently overwrite.

## Acceptance checks for the replacement run

Before treating the regenerated result as valid, check that:

- every coefficient has exactly 100 sycophancy samples and 24 coherence
  samples;
- every first assistant response is byte-identical across alpha for the same
  sample ID;
- every first response records an applied coefficient of `0.0`;
- every second response records its configured coefficient;
- the generation manifest points to exactly one completed log per coefficient
  and evaluation;
- per-sample exports have the expected row counts; and
- summary values were computed from those same manifest-selected logs.

See [the method](method.md) and [running instructions](running.md) for context.
