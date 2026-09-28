# Buddy

**Run as many coding agents as you like at once, and talk to the thing that manages them.**

Every agent has a name - one you give it, or one Buddy makes from its task - and runs in its own tmux window, inside its own git worktree, on its own branch.
You say what you want; Buddy writes the brief, starts an agent, watches for stalls, and tells you when something needs you.

```
you>   put an agent called limiter on rate limiting the API, and once that's
       in, update the docs for it

buddy> limiter is on the rate limiting - t-0007, priority 2, on
       buddy/t-0007-add-rate-limiting. The docs task is queued behind it and
       starts when that one merges.
```

Nothing an agent does reaches your real branch until you run `buddy merge`.

---

## Is this for you?

**Probably, if** you already use a coding-agent CLI, you regularly want more than one thing happening at once, and you are comfortable with agents running unattended in a checkout that is not your working copy.

**Probably not, if** you want a hosted product, a GUI-first tool, or a support channel.
This is one person's tool built in the open.
It works, it is tested, and its rough edges are written down rather than hidden - see [known gaps](#known-gaps).

**Before you start, know that:**

- **Agents run with permissions bypassed.** That is what "unattended" means. The boundary is the git worktree and branch, with a container as an optional second one - see [sandboxing](#sandboxing).
- **It costs money.** Every agent is a real coding-agent run, and the orchestrator itself calls an LLM API on top of that. Ten at once is ten bills - `max_concurrent` [caps it](docs/configuration.md#buddy) if you want a ceiling.
- **macOS and Linux only.** Windows is out of scope because tmux is; WSL2 works and counts as Linux.

## What you need

| | |
|---|---|
| **A coding-agent CLI** | Any of [Claude Code](https://docs.claude.com/en/docs/claude-code), [Codex](https://developers.openai.com/codex), [OpenCode](https://opencode.ai) or [Antigravity](https://antigravity.google), installed and signed in. Different tasks can use different ones. See [choosing a harness](docs/operating.md#choosing-a-harness). |
| **An LLM API key** | For the orchestrator's own conversation. [Anthropic](https://console.anthropic.com), OpenAI, or Google Vertex. **This is separate from a Claude Code subscription** - that logs in its own way and does not pay for API calls. |
| **tmux 3.0+ and git 2.20+** | `buddy setup` installs them if they are missing. |
| **A git repository** | The project you want work done in. |

Voice and the desktop panel are optional.
Everything works by typing.

## Install

Buddy is not on PyPI yet, so install it from source.
You need [uv](https://docs.astral.sh/uv/); it supplies the right Python itself.

```sh
git clone https://github.com/leowei31/buddy.git
cd buddy
uv sync                                            # the exact, tested dependency versions
ln -s "$PWD/.venv/bin/buddy" ~/.local/bin/buddy    # or any directory on your PATH
buddy setup
```

The link runs Buddy from this checkout, in the environment `uv sync` built from `uv.lock`, so what you run is what the test suite ran.
(`uv tool install` is not recommended: it ignores the lockfile, and picks the newest Python, where the voice extra has no wheels yet.)

Setup is twelve steps, each `check -> act -> verify`.
A satisfied step is skipped, so re-running the whole thing is always safe, and a failed one stops with the exact step name to re-run.
It asks you for three things: an API key, a git repo to work on, and what to call that repo.

```sh
buddy setup --yes         # accept every default; the API key is still asked for
buddy setup --no-voice    # skip speech entirely
buddy setup --force keys  # re-run one step
```

Secrets go to your OS keychain, never into a config file.
An environment variable of the same name overrides what is stored.

For the optional extras:

```sh
uv sync --extra voice --extra overlay
```

`--extra` is **not additive** - naming one alone uninstalls the others - so list every extra you want in a single command.
Avoid `--all-extras`: it pulls the Google Vertex providers, whose `cryptography` dependency has no wheel on some platforms.

## First run

```sh
buddy doctor    # tmux, git, the API, each harness - what is wrong and how to fix it
buddy
```

Then, in order:

1. `what have you got?` - proves the orchestrator can reach its API.
2. `spawn something on <project> to add a hello world file` - it reads the brief back to you before starting.
3. Open `http://127.0.0.1:4321` and watch it work.
4. `buddy diff t-0001`, then `buddy merge t-0001`.

That round trip - spawn, watch, review, merge - is the whole product.
Run one agent before you run ten.

## Think first, then hand it over

Not every idea is ready to hand to an agent.
Type `/brainstorm`, or say *"let's brainstorm this"*, and Buddy becomes a thinking partner: it reads your code, weighs options with you, and writes the work up as **drafts** - while nothing starts.

That is enforced, not requested: the tools that start or change work are not offered to the model at all while brainstorming.
When you are ready, `/drafts` shows what was captured and `/go` starts exactly those, dependencies first.
See [brainstorming first](docs/operating.md#brainstorming-first).

## Using it

```sh
buddy                         # the interactive session, plus the dashboard on :4321
buddy --no-voice --no-web     # typed only, no dashboard
buddy doctor                  # every dependency, with the fix for each failure
buddy status                  # running agents by priority, then the queue
buddy spawn <project> "<brief>" [--name scout] [--priority N] [--after t-0141]
buddy watch <agent>           # attach to that agent's tmux window, read-only
buddy logs <agent|task_id> -f
buddy kill <agent>            # stop it; its work is checkpointed, the task is not retried
buddy diff <task_id>          # against the base branch
buddy merge <task_id>         # --no-ff, deliberate, never automatic
buddy resolve <task_id>       # a merge conflicted: an agent resolves it on top of that work
buddy shutdown                # stop every agent, clear the worktrees, keep the work
buddy overlay                 # a small always-on-top panel; --toggle to show or hide
buddy config [edit]           # where config.toml is, whether it loads, and editing it safely
```

Exiting a session does **not** stop your agents.
They keep running in tmux, which is the point of putting them there.
`buddy shutdown` is the one that stops everything, and it is deliberately not destructive:

| Gone | Kept |
|---|---|
| The running agent processes | Every branch, checkpointed first, so no work is lost |
| Every worktree | Every attempt's log |
| The tmux session | Pinned memory, the conversation and its search index, every task and run |

A task that was mid-flight is checkpointed and **requeued**, so it resumes from its last commit next time you start.
`buddy diff` and `buddy merge` keep working afterwards - neither ever needed the worktree, only the branch.

## The dashboard and the overlay

`http://127.0.0.1:4321` shows a card for every running agent, the queue with its dependency badges, each task's live log with its colours intact, and the conversation.
It is **read-only by design**: opening, closing or refreshing the tab cannot affect a running task.

`buddy overlay` is a small always-on-top panel for the corner of your screen - each running agent, what it is doing right now, and an amber header the moment something needs you.
Type in the box at the bottom to talk to Buddy without leaving what you are doing.
`buddy overlay --toggle` shows and hides it; point a keyboard shortcut at that.

## Voice

Optional.
`buddy` listens on push-to-talk by default: press the key, speak, press it again.
Set `hands_free = true` under `[voice]` and it listens for a pause instead.

Speed depends almost entirely on the speech model, and the default is sized for Apple Silicon.
Measured on an Intel CPU against a 2.1-second clip:

| Model | Time | vs realtime |
|---|---|---|
| `faster-whisper:tiny.en` | 361 ms | 0.17x |
| `faster-whisper:base.en` | 661 ms | 0.31x |
| `faster-whisper:small` (default) | 4363 ms | 2.05x |

On anything that is not Apple Silicon, set `stt_model = "faster-whisper:base.en"` under `[voice]`.
`buddy doctor` measures your own machine's time-to-first-byte rather than quoting a number from a document.

Speaking aloud needs a [Fish Audio](https://fish.audio) key, or an NVIDIA GPU to run their open-source server locally.
With neither, Buddy types and does not speak.

## Sandboxing

Every task already runs in its own git worktree on its own branch, so nothing an agent does reaches your base branch without `buddy merge`.
Because permissions are bypassed inside that worktree, a container is the recommended second boundary:

```sh
docker build -f buddy/setup/assets/Dockerfile.harness \
    --build-arg HARNESS_INSTALL=@anthropic-ai/claude-code -t buddy-harness:latest .
```

Then set `sandbox = "docker"` and a `sandbox_command` under `[harness.<name>]` -
[the reference](docs/configuration.md#harnessnamesandbox_command) has the exact command to paste and what each part of it is for.

The container gets the worktree read-write, the brief read-only, and the repository's git directory - which it needs, because a worktree's `.git` is a file pointing outside the worktree and without it the agent cannot commit, or even run `git log`, at all.
Inside that git directory, `.git/config` and `.git/hooks` are mounted read-only: both are read by your git, on your machine, the next time you use the repository, so an agent able to write them could run code on the host later.
Nothing else is mounted - not your working checkout, not the rest of `~/.buddy`, not your home directory.

## What stays on your machine

- **API keys live in your OS keychain**, never in `config.toml`. A task's run script holds them only while it runs, mode `0600`, and is deleted afterwards.
- **`~/.buddy` is private to you** (mode `0700`): the conversation, briefs and agent output.
- **Buddy never commits a secret itself.** When it checkpoints an agent's leftover work, `.env` files, key files, and anything containing one of your keys or a provider-shaped key stay in the worktree and out of the commit - and it tells you which.
- **Nothing reaches your branch carrying one.** `buddy merge` refuses a branch that adds a secret. For a key-shaped lookalike you have checked, `--allow-secret-patterns` lets it through; for one of your own keys, nothing does.
- **This repository refuses them too**: `git config core.hooksPath .githooks` turns on a pre-commit hook, and CI scans every file ever committed.

What does leave: your conversation and the files Buddy reads go to your LLM provider, and each agent talks to its own. The dashboard binds to `127.0.0.1` only.
See [secrets](docs/operating.md#secrets) for the details.

## Known gaps

Written down rather than hidden.
The full list is in [docs/operating.md](docs/operating.md#known-gaps).

- **Antigravity's success path is unverified.** Its every failure path was run for real; a successful run needs a Google account, and `agy` cannot be pointed at a scripted server the way Codex and OpenCode were.
- **The container sandbox is only tested where a Docker daemon runs.**
- **An outside tool holding Buddy's database write lock stalls the session** for as long as it holds it. Buddy's own processes do not block each other.

## Documentation

| | |
|---|---|
| [docs/operating.md](docs/operating.md) | Day to day: the commands, naming agents, the dashboard, troubleshooting |
| [docs/configuration.md](docs/configuration.md) | Every `config.toml` key, its real default, and what it changes |
| [docs/architecture.md](docs/architecture.md) | How it actually works, and the decisions behind it |
| [docs/development.md](docs/development.md) | Testing, adding a harness or a provider, the conventions |
| [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) | Working on Buddy, and reporting a vulnerability privately |

## Development

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

The suite drives real tmux, real git, real SQLite, a real HTTP server and real subprocesses rather than mocks.
Tests needing an optional extra, a Docker daemon or a harness binary skip without it rather than failing.
Each tmux-facing test gets a private socket, so a test run can never disturb your own session.

Read [docs/development.md](docs/development.md) before adding a harness or a provider.

## Licence

MIT. See [LICENSE](LICENSE).

Buddy vendors [xterm.js](https://xtermjs.org) (MIT) for the dashboard's log view; its licence sits beside it in `buddy/web/static/vendor/`.
