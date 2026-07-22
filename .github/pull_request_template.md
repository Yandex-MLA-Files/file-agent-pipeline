<!--
Use a human-readable PR title following Conventional Commits when possible:
feat: ..., fix: ..., test: ..., docs: ..., chore: ...
-->

## Summary
<!-- Briefly explain what changes and why. -->


## Related issue
<!-- Keep the relevant line and remove the other one. -->
Closes #      <!-- Closes the issue when the PR is merged. -->
Part of #     <!-- Links a larger issue without closing it. -->

## Verification
<!-- Add commands, steps, or screenshots that let a reviewer verify the change. -->
- [ ]

## Checklist
- [ ] A related issue is linked with `Closes #N` or `Part of #N`
- [ ] Work was completed on a separate branch, not directly on `main`
- [ ] `ruff check` and `ruff format --check` pass locally
- [ ] `pytest` passes locally
- [ ] No `.env`, API keys, or folder identifiers are committed
- [ ] README or other documentation is updated when behavior changes
