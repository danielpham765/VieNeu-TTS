---
name: sync-up-author-code
description: Use when syncing this repo's author branches so local `main-author` is rebased onto `origin/main`, pushed to `daniel/main-author`, then local `main` is rebased onto `main-author` and pushed to `daniel/main` after resolving conflicts while prioritizing newer code from `main-author`.
---

# Sync Up Author Code

Use this skill for the repo workflow:

1. Update `local:main-author` from `origin:main`
2. Push `main-author` to `daniel:main-author`
3. Rebase `local:main` onto `local:main-author`
4. Resolve conflicts with priority on newer code from `main-author`
5. Build and fix all errors if any
6. Push `local:main` to `daniel:main`

## Branch Mapping

- Source upstream: `origin/main`
- Author integration branch: `main-author` <-> `daniel/main-author`
- Publish branch: `main` <-> `daniel/main`

## Workflow

Run from the repo root.

### 1. Refresh remotes

```bash
git fetch origin main
git fetch daniel
```

### 2. Sync `main-author` with `origin/main`

```bash
git switch main-author
git rebase origin/main
git push --force-with-lease daniel main-author
```

If conflicts happen here, preserve the intent of `origin/main` first, then re-apply author-only changes when they are still needed.

### 3. Rebase `main` onto `main-author`

```bash
git switch main
git rebase main-author
```

Goal: `main` should end up containing the newer author code from `main-author`, with any still-valid `main`-only changes replayed on top.

## Conflict Resolution Rules

When rebase stops:

1. Inspect status and the current patch:

```bash
git status --short
git rebase --show-current-patch
```

2. Open the conflicted files and resolve them manually.

3. During `git rebase main-author` while on `main`:
- Prefer the code already present on `main-author`
- Re-apply only the parts from the replayed `main` commit that are still correct
- Do not blindly keep both sides

4. Be careful with `ours` and `theirs` during rebase:
- `--ours` means the rebased-onto branch state, which is usually the `main-author` side here
- `--theirs` means the replayed commit from `main`

5. After resolving:

```bash
git add <resolved-files>
GIT_EDITOR=true git rebase --continue
```

Repeat until the rebase finishes.

## Build Validation

After all conflicts are resolved and the rebase finishes, do not push yet.

Use the repo's documented workflow:

```bash
uv sync
uv run vieneu-web
uv run apps/gradio_main.py
```

Why this flow:

- `uv sync` is the correct dependency/bootstrap step for this repo
- `uv run vieneu-web` verifies the packaged entrypoint still works
- `uv run apps/gradio_main.py` verifies the direct app file still works without a compatibility wrapper

Treat launch failures, import errors, dependency issues, and obvious runtime startup errors as blockers.

If each app starts successfully, stop it after confirming startup.

If `uv sync`, `uv run vieneu-web`, or `uv run apps/gradio_main.py` fails:

1. Read the error carefully
2. Fix the problem in code or project config
3. Re-run `uv sync` if dependencies changed
4. Re-run both launch commands:

```bash
uv run vieneu-web
uv run apps/gradio_main.py
```

5. Repeat until both commands launch cleanly

Do not push `daniel/main` until this passes.

## Validation

Before pushing:

```bash
git status --short --branch
git branch -vv
```

If Python files were edited during conflict resolution, run a lightweight syntax check when practical:

```bash
python -m py_compile <changed-python-files>
```

## Push `main`

Once `main` is clean, rebased successfully, and build validation passes:

```bash
git push --force-with-lease daniel main
```

## Notes

- Use `--force-with-lease`, not `--force`
- Never use destructive resets to escape a conflict unless the user explicitly asks
- If upstream deleted or moved a file, prefer the current repo structure and port only the still-relevant logic
- If dependency lockfiles conflict heavily, prefer regenerating them from the chosen manifest only when needed and when the tooling is available
