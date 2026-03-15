# Contributing

## Setup

```bash
git clone https://github.com/geronimo-iia/nanochat.git
cd nanochat
uv sync
```

## Running Tests

```bash
uv run pytest tests/ -x -q
```

The 10 skipped tests require a Hopper GPU (FA3). Everything else runs on CPU.

## Code Quality

```bash
uv run ruff check src/        # lint
uv run ruff check src/ --fix  # auto-fix
uv run pyright src/           # type check (must stay at 0 errors)
```

CI runs all three on every PR.

## Commits

Use semantic commit format:

```
type(scope): short description

feat, fix, docs, style, refactor, test, ci, chore
```

Examples: `fix(types): resolve reportReturnType errors`, `feat(tasks): add HellaSwag task`

## Pull Requests

- Keep PRs focused — one logical change per PR
- Tests must pass, pyright must stay at 0 errors, ruff must be clean
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes
