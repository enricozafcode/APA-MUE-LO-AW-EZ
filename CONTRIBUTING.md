# Contributing Guide

Welcome to the team. This project uses a simple, safe Git workflow so everyone can collaborate without breaking the main branch.

## The Golden Rule

Never write code directly on `main`.

Treat `main` as production-ready code. All changes must go through a feature branch and a Pull Request (PR).

## Our Daily 5-Step Workflow

### 1) Sync your local `main`

Before coding, always update your local copy:

```bash
git checkout main
git pull origin main
```

### 2) Create a feature branch

Create a branch with a descriptive name:

```bash
git checkout -b your-branch-name
```

Examples:
- `setup-agent-loop`
- `add-ollama-client`
- `improve-birdclef-evaluator`

### 3) Commit your progress

Stage and commit your changes:

```bash
git add .
git commit -m "Briefly describe what you added or fixed"
```

Commit early and often. Small commits are easier to review and safer to merge.

### 4) Push your branch

Upload your branch to GitHub:

```bash
git push origin your-branch-name
```

### 5) Open a Pull Request

On GitHub:
1. Click **Compare & pull request**.
2. Add a clear title and short description.
3. Request at least one teammate review.
4. Merge only after checks pass.

After merge, restart from Step 1.

## Project-Specific Workflow Rules

- One branch = one focused change (do not mix unrelated tasks).
- Keep PRs small enough to review in 10-15 minutes.
- PR title format: `type: short description`  
  Examples: `feat: add experiment logger`, `fix: handle failed training runs`.
- Before opening a PR:
  - run the project (or relevant script) locally,
  - check that no secrets are committed,
  - update docs if behavior changed.
- Never commit large raw datasets, trained model binaries, or `.env` files.

## Recommended Branch Naming

- `feat/<short-topic>` for new features
- `fix/<short-topic>` for bug fixes
- `docs/<short-topic>` for documentation
- `exp/<short-topic>` for experiment-specific work

Examples:
- `feat/llm-prompt-router`
- `fix/csv-submission-format`
- `docs/readme-setup`
- `exp/birdclef-melspec-v1`

## Emergency Git Help

### "I want to discard all local uncommitted changes"

```bash
git checkout -- .
```

### "I started coding on `main` by mistake and did not commit yet"

Create a branch now. Your uncommitted changes will move with you:

```bash
git checkout -b my-new-branch
```

### "I cannot push because of merge conflicts"

From your feature branch:

```bash
git pull origin main
```

Then:
1. Open conflicted files in your editor.
2. Resolve conflict markers.
3. Stage and commit:

```bash
git add .
git commit -m "resolve merge conflicts"
git push origin your-branch-name
```

If stuck, ask the team immediately.

## Team Quality Checklist (Before Merge)

- [ ] Code runs locally.
- [ ] No secrets or `.env` committed.
- [ ] No large data/model files committed.
- [ ] README/config docs updated if needed.
- [ ] PR reviewed by at least one teammate.
