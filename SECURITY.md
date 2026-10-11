# Credentials and secret checks

The default hosted judge reads `OPENROUTER_API_KEY` from the environment or a
local `.env` file. Copy [.env.example](.env.example) and fill it in locally.
Never put real credentials in examples, source code, logs, issue reports, or
Git commits.

Git ignores `.env`, `.env.*` (except the empty example), private-key files,
common credential JSON files, virtual environments, and generated results.
An ignore rule does not remove a file already tracked by Git. Review
`git diff --cached` before publishing changes.

Optional notifications require your own `NTFY_TOPIC`. Scoring sends evaluation
prompts and generated answers to the configured hosted judge, so review your
data before using a hosted provider.

## Scan locally

With Gitleaks installed, scan every fetched branch and tag, including history:

```bash
gitleaks git --redact --log-opts='--all --full-history' .
```

To scan the proposed tracked files, including staged additions, without exposing
your local environment file:

```bash
audit_dir=$(mktemp -d)
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$audit_dir"
gitleaks dir --redact "$audit_dir"
```

The audit directory is a disposable copy of tracked files. These checks look
for known secret patterns; passing does not prove that every possible
credential is absent. CI repeats the history scan on pushes and pull requests.

If a real credential is ever committed, revoke or rotate it with the provider
first. Removing it from the latest file alone leaves it in Git history.
History rewriting requires coordination with everyone using the repository.
