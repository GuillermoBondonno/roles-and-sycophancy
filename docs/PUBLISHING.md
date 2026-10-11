# Maintenance checklist

Before publishing another result:

- Run `python -m pytest -q` from a clean checkout.
- Scan all Git history and the proposed files for secrets; see
  [SECURITY.md](../SECURITY.md).
- Include compact summary data, figures, and reusable probe artifacts.
  Keep credentials, copied source passages, and large raw logs local.
- Regenerate figures using only included data and check README links.
- Record model and tokenizer revisions, dependency versions, GPU/runtime,
  dtype, seed, and repository commit or diff hash for new runs.
- Verify the expected sample counts and byte-identical first assistant answers
  across coefficients before scoring or interpreting a sweep.
- Check manifest selection and per-sample exports against
  [the known issues and acceptance checks](findings.md).
- Describe results as evidence about the tested model, direction, site, and
  evaluation. The original role-confusion paper did not perform this steering
  experiment.
- Choose a code license before inviting others to reuse the implementation.
  No license is currently declared; that choice remains with the author.

The included dependency ranges and historical metadata are enough to describe
the study and reproduce its figures, but not to guarantee identical generations
in a new environment. Do not label a future environment snapshot as the exact
historical environment unless it was recorded during that run.
