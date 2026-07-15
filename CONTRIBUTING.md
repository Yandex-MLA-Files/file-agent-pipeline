# Contributing

This document summarizes the collaboration rules for `file-agent-pipeline`.

## Workflow

Use one issue, one branch, and one pull request for each change.

1. Create an issue from the task or bug template and apply the relevant area label (`parser`, `retrieval`, `generation`, `evaluation`, `backend`, `frontend`, or `devops`).
2. Create a branch from the latest `main`.
3. Make small, focused commits.
4. Open a pull request. Add `Closes #N` when the PR completes the issue, or `Part of #N` when it is only one part of a larger task.
5. Merge after CI passes and the PR has at least one approval.

Do not push directly to `main`.

## Branches

Use `<type>/<short-description>`, for example:

- `feature/pdf-parser`
- `fix/xlsx-empty-sheet`
- `chore/setup-ci`

## Commits

Prefer Conventional Commits:

- `feat:` for new functionality;
- `fix:` for bug fixes;
- `test:` for test changes;
- `docs:` for documentation;
- `ci:` for CI changes;
- `chore:` for infrastructure, dependencies, and configuration.

Keep commits small and focused on one purpose.

## Pull requests and issues

Put the issue reference in the pull request description:

- `Closes #N` closes the issue when the PR is merged. `Fixes` and `Resolves` are equivalent.
- `Part of #N` links the issue without closing it.

Do not duplicate these keywords in commit messages.

## Reviews

- Authors must not merge their own PR without approval.
- Keep PRs small enough to review effectively.
- Keep review feedback and discussion in the PR or linked issue.

## Secrets

- Never commit `.env`, API keys, or folder identifiers.
- Commit only `.env.example` with placeholder values.
