# Working on Buddy

```sh
uv sync                                # venv and dev dependencies
uv sync --extra voice --extra overlay  # plus speech and the panel
git config core.hooksPath .githooks    # refuse commits that carry secrets
uv run buddy                           # the CLI from source
uv run pytest                          # the suite
uv run ruff check                      # lint
uv run ruff format                     # format
uv run mypy                            # types
```

`--extra` is **not additive**: `uv sync --extra overlay` on its own uninstalls the voice extra, so name every extra you want in one command.
`--all-extras` is not the answer either - it pulls the Vertex providers, and their `cryptography` dependency has no Intel-macOS wheel.
Tests that need an optional extra skip without it rather than erroring, so a partial install produces an honest run rather than a broken-looking one.

---

## The testing stance

The suite drives **real** tmux, real git, real SQLite, a real HTTP server and real subprocesses.
Hand-written fakes stand in where an external service would cost money or reach off the machine; `unittest.mock` is not used.

That is deliberate.
Nearly every serious bug found in this project lived in a **seam** - between an LLM's output and a shell, between one agent's failure and every other, between a kill and a database write, between a key in a keychain and a client that only read the environment.
Mock-heavy tests pass straight through seams.

| File | Covers | Drives |
|---|---|---|
| `test_manager.py` | Scheduling, dependencies, preemption, health, crash paths | Fake runner and workspace, injected clock |
| `test_reconcile.py` | Recovery after a crash | **Real tmux, real git** - kills the server mid-run |
| `test_tmux_runner.py` | Panes, pipes, process-tree kill | **Real tmux** |
| `test_workspace.py` | Worktrees, checkpoints, merge, conflict | **Real git** |
| `test_shutdown.py` | What survives a shutdown | **Real git** |
| `test_web.py` | The dashboard | **Real uvicorn on a real socket** |
| `test_voice.py` | The speech layer | A real Fish-protocol server on a real socket |
| `test_setup.py` | The twelve steps | This machine, for real; a scripted terminal |
| `test_sandbox.py` | The container boundary | **Real Docker** - skips without a daemon |
| `test_providers.py`, `test_context.py` | Wire formats, the five context layers | Fake SDK clients |
| `test_adapters.py`, `test_run_script.py` | Command lines, parsing, the wrapper | Pure, plus a live `claude` where present |
| `test_harnesses.py` | Codex, OpenCode, Antigravity | Recorded real output; real binaries against `fake_model_api.py` where installed |
| `test_logpipe.py` | Lossless, capped log rotation | Fuzzed across many rotations; one proven against real tmux in `test_tmux_runner.py` |
| `test_brainstorm.py`, `test_conflicts.py` | Brainstorming, and a conflict from report to merged | A real brain with scripted replies; real git |
| `test_leaks.py` | Secrets: recognition, the pre-commit hook, and a scan of this repository | Keys generated at runtime; a real `git commit` against the hook |
| `test_cli.py`, `test_overlay.py` | The commands, the session loop, `buddy config`, the overlay and its control socket | Typer's runner, a real Brain, a real unix socket |
| `test_config.py`, `test_state.py`, `test_logs.py` | Loading and validating config, the SQLite schema and migrations, reading logs | Real files and a real database |
| `test_docs.py` | The documentation describes the CLI that exists: every command, option, config key and link | The CLI's own definition |
| `test_smoke.py` | The package imports and the entry point runs | The installed entry point |

Async tests need no decorator - `asyncio_mode = "auto"`.
`filterwarnings = ["error"]`, so a new deprecation fails the suite.
`timeout = 120` per test, because a hang is a bug and "CI never finished" is the worst way to learn that.

Each tmux-facing test gets a **private socket** via the `tmux_socket` fixture, so a test run can never disturb your own `buddy` session.

### Writing a test that is worth having

Prefer a reproduction to an assertion of intent.
The injection test runs its command line through bash and checks the marker file was never created.
The lookahead test measures when each request arrived.
The sandbox tests run `rm -rf /` inside a container and assert the base repository survived.

Twice, writing the measurement changed the fix:

- Bounding the sentence queue did not bound the TTS lookahead - a fast backend empties a queue faster than a speaker drains it, so all the requests went out at once. The bound belongs on sentences that have not finished *playing*.
- Running a runaway regex in `asyncio.to_thread` did not protect the event loop - `re.search` is C code that never releases the GIL. Measured: two heartbeats in 200 ms. The pattern has to be screened instead.

Both fixes looked right and were wrong.

## Secrets in tests

The suite scans the repository for secrets (`test_the_repository_holds_no_secrets`), and CI scans all of history, so a fixture recorded from a real run must be sanitised before it is committed.
A test that needs something key-shaped builds it at runtime from a seeded random source - see `tests/test_leaks.py` - so the file never contains one.

## Adding a harness

The four viability requirements are a headless mode, a way to feed a prompt, full auto-approve, and a meaningful exit code.
If a CLI cannot do all four, record that it is unsupported and stop.

1. **Install the real CLI and run it** before writing anything. Every adapter here was changed by what that showed: OpenCode hangs forever on a prompt from stdin, `agy -p` takes a value and gives up after five minutes by default, Codex ignores `OPENAI_API_KEY` and retries an unreachable API forever. None of that is in their documentation.
2. Write `buddy/harnesses/<name>.py` with a class extending `BaseAdapter`, and set its class attributes: `binary`, `default_command`, `default_waiting_patterns` (only patterns you have seen the CLI print), `help_args`, `install_hint`, `npm_package`, and `credential_env` - only the variables the CLI demonstrably reads.
3. `preflight()` reads the CLI's own help and returns `await self.report(requirements, notes)`. Notes are for command lines known to break; `auth_status()` answers "signed in?" as cheaply as the CLI allows, or `None` when it cannot be asked.
4. `parse_result()` and `describe_activity()` read its JSON with `harnesses.stream`. Override `is_progress()` if the CLI prints retry lines while going nowhere.
5. Add it to `ADAPTERS` in `harnesses/__init__.py`. That is the only registration: the CLI, setup, doctor and the brain all read that dict.
6. **Record real output as fixtures** in `tests/fixtures/harness_logs/` - a success, and every failure you can provoke - and test the parser against those bytes.
7. If the CLI can be pointed at another model server, add a live test in `test_harnesses.py` against `tests/fake_model_api.py`, so its success path runs for real without an account.

`test_harnesses.py` also runs every adapter's default command line through bash with a brief built to break quoting, and asserts the brief arrives verbatim; a new adapter is covered by adding it to that parametrization.

## Adding a provider

1. Write `buddy/providers/<name>.py` implementing the `ModelProvider` protocol: `stream`, `count_tokens`, `probe`, `aclose`.
2. Translate to and from Buddy's canonical `Turn` / `ToolCall` / `ToolResult` / `ToolDef`. Buddy owns the message list; providers only serialize.
3. Report real `Capabilities` from `probe()`. The context layers are chosen from that probe, never from a table.
4. Register in `KNOWN` and `build()` in `providers/__init__.py`; add to `KEYED` if it authenticates with an API key.
5. Add it to the shared contract in `test_providers.py`, which runs the same suite against every wire format.

## Adding a setup step

Each step is `check → act → verify`, plus `contribute()` for what it wants in `config.toml`.

`contribute()` is separate from `act()` because **already satisfied is the common case** - the CLI is installed, the model is cached, the key is stored.
A step that only wrote its config while installing something would silently drop it on every run after the first.

Record identifying pins in `check()`.
That is what makes `buddy update` able to re-run only what moved.

## Conventions

- `ruff check`, `ruff format --check`, `mypy` and the suite are all clean, and CI keeps them so. Fix a type error by stating the real contract - `NoReturn`, a union, a guard - rather than a `type: ignore`.
- A new command or option is documented in `docs/operating.md` in the same change; `test_docs.py` fails otherwise, and so does a new config key missing from `docs/configuration.md`.
- A test waits on a condition, never on a guessed duration. Timing assertions compare events against each other, not against a clock: a slow CI runner is not a bug.
- Every tmux and git call goes through `asyncio.create_subprocess_exec`, never `subprocess.run`.
- Respect the layer table. A leak across it is a bug even when the tests pass.
- A decision someone would otherwise undo - an order that looks backwards, a check that looks redundant - gets its reason next to it, in the code or in `docs/architecture.md`.
- One logical change per commit, with a message that says why.
