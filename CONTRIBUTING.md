# Contributing

Thanks for looking.
Everything you need is in [docs/development.md](docs/development.md); the short version:

```sh
git clone https://github.com/leowei31/buddy.git && cd buddy
uv sync
git config core.hooksPath .githooks    # refuse commits that carry secrets
uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest
```

- **Read [docs/architecture.md](docs/architecture.md) first.** It is how the system works and why each non-obvious decision was made.
- **Reproduce before you fix.** A bug fix starts with a test that fails for the reason a user would see, against real tmux, git and subprocesses where it can.
- **Adding a harness** means running the real CLI before writing a line - see [adding a harness](docs/development.md#adding-a-harness).
- **Never commit a key**, even a test one: build key-shaped values at runtime. The hook, the suite and CI all scan for them.
- Keep the layer rule: only `tmux_runner.py` knows tmux, only `workspace.py` knows git, only an adapter knows its CLI, only a provider knows its API.

Security problems go through [SECURITY.md](SECURITY.md), not public issues.
