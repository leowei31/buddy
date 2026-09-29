# Operating Buddy

Day to day: what the commands do, what the screen is telling you, and what to do when something is wrong.

---

## The shape of a session

`buddy` with no arguments is the interactive session.
It reconciles whatever was left over, starts the dashboard, connects to your brain provider, and gives you a prompt.

Underneath, a tick runs once a second: it notices panes that have died, finalizes those runs, checks the health of the rest, and starts whatever in the queue is ready.

**Leaving the session does not stop your agents.**
They are in tmux, which is the entire point of putting them there.
`buddy shutdown` is the thing that stops everything.

## Commands

### Watching

| Command | What it does |
|---|---|
| `buddy status` | Every running agent, most urgent first, then the queue. Catches up on anything that finished while nothing was watching. |
| `buddy watch <agent>` | Attach to that agent's tmux window, read-only. `Ctrl-b d` to detach. |
| `buddy attach` | Attach to the whole session, read-only. `Ctrl-b w` switches between agents' windows. |
| `buddy logs <agent\|task_id> [--follow] [--attempt N] [--lines N]` | Print the last 200 lines of a log, or `--follow` it as it grows - through a [rotation](#logs) too. An agent's name means its latest task; `--attempt` picks an earlier attempt. |
| `buddy history [--project P] [--limit N]` | Recent tasks and how they ended. |
| `buddy doctor` | Preflight: tmux, git, the database, projects, the brain provider, voice, each harness. |

### Working

| Command | What it does |
|---|---|
| `buddy spawn <project> "<brief>"` | Queue a task without talking to the brain. Options: `--name N` (what to call its [agent](#agents); default: made from the title), `--harness H`, `--model M` (the harness's own name for it), `--priority N`, `--title T`, `--after t-0141` (repeatable), `--merge-required` (dependents wait for the merge, not just done), `--wait` (block until it finishes). |
| `buddy reprioritize <task_id> <N>` | Reorder the queue. Not confirmed - it is reversible. |
| `buddy kill <agent> [--yes]` | Stop an agent. Its work is checkpointed, the task is marked killed, and it does **not** come back. |
| `buddy diff <task_id> [--into BRANCH]` | What that task's branch changed. |
| `buddy merge <task_id> [--into BRANCH] [--yes] [--force] [--allow-secret-patterns]` | `--no-ff` into your checkout. Refused while the task's agent is still running, while your checkout has uncommitted changes, while a [conflict fix](#when-a-merge-conflicts) is pending (`--force` overrides), and when the branch adds a [secret](#secrets) (`--allow-secret-patterns` lets checked lookalikes through, never your own keys). A conflict is aborted and reported. |
| `buddy resolve <task_id> [--harness H] [--wait]` | Spawn an agent to resolve that task's merge conflict. |
| `buddy discard <task_id> [--yes]` | Remove the worktree. The branch is kept for [the grace period](#discarding). Refused while the task's agent is still running. |

### Memory

| Command | What it does |
|---|---|
| `buddy memory list` | Pinned cross-session facts. |
| `buddy memory add "<fact>"` | Pin one. |
| `buddy memory rm <id>` | Drop one. |
| `buddy compact` | Force a client-side compaction now. |

### Lifecycle

| Command | What it does |
|---|---|
| `buddy setup` | The twelve-step bootstrap. Safe to re-run; satisfied steps are skipped. Options: `--yes` (accept every default), `--force STEP` (re-run one; repeatable), `--no-voice`, `--cloud-tts` (Fish cloud rather than a local GPU), `--brain PROVIDER`, `--install-harnesses` (opt in to `npm install -g` for the CLIs on npm). |
| `buddy config` | Where `config.toml` is and whether it loads - works even when it does not, which is when you need it. |
| `buddy config edit` | Open it in `$VISUAL` or `$EDITOR`, then check what you saved; a mistake is shown the moment you close the editor. |
| `buddy config path` | Just the path, for scripts: `$EDITOR "$(buddy config path)"`. |
| `buddy --version` | The installed version. |
| `buddy update` | Upgrade the tool, then re-run only the setup steps whose pins moved. Run from a clone - the [install](../README.md#install) the README describes - there is no tool to upgrade, so it says to `git pull && uv sync` in the checkout first. |
| `buddy shutdown [--yes] [--keep-worktrees]` | Stop every agent, clear the worktrees, keep the work. |
| `buddy uninstall [--purge] [--force]` | Remove the container, session and tool. `--purge` also deletes `~/.buddy`, and refuses while any branch holds unmerged work; `--force` purges anyway, unrecoverably. |

### In the session

Typed at the `you>` prompt, or into the overlay. They go straight to Buddy, never to the model.

| Command | What it does |
|---|---|
| `/brainstorm` | Think it through first: nothing starts until `/go`. See [brainstorming](#brainstorming-first). |
| `/brainstorm off` | Stop brainstorming; the drafts are kept. |
| `/drafts` | What has been drafted so far. |
| `/drop <draft>` | Remove a draft, e.g. `/drop d2`. |
| `/go` | Start every draft, dependencies first, and stop brainstorming. |
| `/help` | This list. |

## Choosing a harness

Buddy runs four coding-agent CLIs.
Install any of them, sign in, then `buddy setup --force harnesses`; setup writes each one that passes preflight *and* is signed in, and names the rest with the step that would fix them.

| Harness | Install | Sign in | How far it is verified |
|---|---|---|---|
| `claude_code` | `npm install -g @anthropic-ai/claude-code` | run `claude` once | Real runs, end to end. |
| `codex` | `npm install -g @openai/codex` | `codex login`, or `CODEX_API_KEY` | The real binary running a real tool call and committing in a worktree, through `buddy spawn`, against a scripted model server; rejected-key and unreachable-API runs recorded. |
| `opencode` | `npm install -g opencode-ai` | `opencode auth login` - optional | A real run on OpenCode's own model, and the real binary through `buddy spawn` against a scripted server. |
| `antigravity` | `curl -fsSL https://antigravity.google/cli/install.sh \| bash` | run `agy` once, or its API-key mode | Every failure path run for real. **The success path is not verified**: it needs a Google account, and `agy` cannot be pointed at a scripted server. |

A task runs on the harness you name - *"do it on codex"*, or `buddy spawn --harness codex` - else the project's `default_harness`, else `claude_code` if configured, else the first configured one.
The brain is told which harnesses are configured, a name that is not configured is refused before a task exists, and so is one that is installed but signed out.

What Buddy is careful about, all found by running each CLI:

**Claude Code** runs with `--permission-mode bypassPermissions`.
`acceptEdits` looks unattended but still denies Bash, so the agent can never run the `git commit` its brief asks for.

**Codex** runs `exec` with its own sandbox bypassed.
Inside `--sandbox workspace-write` a write outside the worktree is refused and the network is off, so `npm install`, a Go or Cargo build, or anything with a cache in your home directory fails one tool call at a time - inside a run that can still end "successfully".
It reads `CODEX_API_KEY` and ignores `OPENAI_API_KEY`.
When it cannot reach its API it retries forever; Buddy does not count those retry lines as progress, so the agent goes `stalled` after `stall_timeout` instead of looking busy for two hours.

**OpenCode** gets the brief as an argument, never on stdin: `opencode run < prompt.md` reaches the model and then never produces another byte or exits.
It runs with `--auto`, because without it every permission request is *rejected* and the run still exits 0 with the work undone; Buddy reports any refused tool call as a failure rather than trusting that exit code.
Its model names are `provider/model`, and without any credentials it uses OpenCode's own free default model.

**Antigravity** runs with `--print-timeout 24h`, because print mode otherwise gives up after five minutes and nothing in its headless documentation says so.
The brief is attached to `-p` - it takes a value and does not read stdin - and `--disable-slash-commands` stops a brief that starts with `/` being run as a command.
Signed out, it does not fail: it prints a sign-in URL and waits for a pasted code, which marks the agent `waiting_input`.

## Agents

Every task runs as an **agent**: a tmux window of its own, named for the agent, made when the task starts and closed when it ends.
There is no fixed number of them.
Everything that is ready starts, unless you set [`max_concurrent`](configuration.md#buddy) to put a ceiling on it.

### Naming them

Name an agent when you create it - `buddy spawn webapp "..." --name scout`, or *"put an agent called scout on it"* - or leave it out, and Buddy names it from the task's title: `Add rate limiting to the API` becomes `add-rate-limiting`, and `add-rate-limiting-2` if that is taken.

- A name is letters, digits, `-` and `_`, up to 32 characters, starting with a letter or a digit. Spaces become `-`, so a spoken *"code reviewer"* is `code-reviewer`.
- Case does not matter when you use one: `buddy kill Scout` and `buddy kill scout` stop the same agent.
- No two queued or running agents share a name. Once one finishes, its name is free again, and `buddy logs scout` shows the latest task that ran under it.
- A name that looks like a task id, such as `t-0007`, is refused: `buddy logs t-0007` has to mean one thing.
- A task keeps its agent's name through retries, preemption and restarts.

### What they are doing

| Status | Means |
|---|---|
| `running` | Working. |
| `waiting_input` | The harness asked a question. Buddy **never answers for it** - `buddy watch <agent>` and answer yourself. |
| `stalled` | No progress for `stall_timeout` - no output, or only a harness's retry chatter. Notified, never killed. |

A task moves `queued → running → done | error | killed`, and then `merged` or `discarded` once you decide.
Its agent exists only while it is `running`; how it ended is on the task, and its log stays.

## What happens when things end

Every attempt is checkpointed onto its branch before anything else happens, so nothing is lost even if the rest fails.

| Outcome | What follows |
|---|---|
| `done`, `error` | Recorded. Its window closes and its name is free. |
| `killed` | Recorded. **Not** requeued - kill means kill. |
| `preempted`, `interrupted`, `timeout` | Checkpointed and requeued at the original priority, with a resume note, as the next attempt. |

A second timeout on the same task is an error rather than an endless requeue.

## Preemption

When a queued task outranks a running one and cannot start until something stops, Buddy **proposes** and never acts:

```
limiter is on t-0004 (Add rate limiting, priority 4), which t-0009 (Fix the outage) outranks.
  Preempt it? Its work is checkpointed and it goes back on the queue
```

Accepting checkpoints the victim, keeps its worktree and branch, requeues it at its original priority under the same name, and starts the waiting task.

With no limit - the default - a ready task simply starts, so only two things can hold one back.
One is `max_concurrent`, once you set it and it is reached.
The other is a project that is not a git repository, which runs one agent at a time: a task waiting on it can only start once *that* project's agent stops, which is the only stop Buddy will propose for it.
A stop is only ever proposed if it would actually let the task start.

A proposal can go stale - the victim finishes before you answer.
Accepting a stale one stops nothing, and says why.

## When a merge conflicts

A merge that conflicts is aborted and reported; your checkout is left exactly as it was.
Then say *"have someone fix it"*, or run `buddy resolve <task_id>`.

An agent is spawned on a new branch that starts at the tip of the conflicting one.
Its brief is to merge your base branch in and resolve every conflict, keeping the intent of both sides, with **merge only** - no rebase, no rewritten history - and to finish only when the base is an ancestor, no conflict markers remain, and the project's tests pass.

While that fix is pending, other merges into the project wait: each one would move the base again under the fix.
`buddy merge --force` overrides that.
Merging the fix lands the original work with it, and both tasks are marked merged.

## Discarding

`buddy discard <task_id>` removes the worktree and keeps the branch for `discard_grace` (7 days).
A sweep runs hourly while Buddy runs and deletes branches whose grace has passed.

It never deletes a branch that moved after the discard.
Discarding is a decision about the task, not a lock on the branch: if you checked it out and kept working, its reflog records that, and the branch stays yours however long ago the discard was.

## Logs

Every attempt's output is written to `~/.buddy/tasks/<id>/attempt-N.log`, escapes and all, by a small writer that tmux feeds.
Past `max_log_mb` (64 MiB) it moves the log to `attempt-N.log.1` and carries on in a fresh one, keeping one previous file.

Nothing is lost at the switch - measured against real tmux, 30,000 lines printed as fast as bash can print them all arrive once and in order.
A line that straddles the cap can be split across the two files; `cat attempt-N.log.1 attempt-N.log` is the whole thing.
The dashboard and `buddy logs -f` both follow onto the new file.

## Secrets

### Where they are

| Secret | Kept |
|---|---|
| API keys you give `buddy setup` | Your OS keychain, under the service `buddy`. An environment variable of the same name overrides one. |
| A key a harness needs | Resolved from `${keychain:NAME}` into `tasks/<id>/run.sh`, mode `0600`, deleted when the attempt ends |
| The conversation, briefs, agent output | `~/.buddy`, mode `0700`; `state.db` and every log `0600` |

`config.toml` never holds a secret: `${keychain:NAME}` references and `api_key_env` names only.

### What stops one reaching git

Agents are given your keys so their CLIs can authenticate - which means an agent can write one into a file.
So Buddy checks at the two points where it commits or merges on your behalf.

**Its checkpoint** - the commit Buddy makes of whatever an agent left uncommitted - unstages anything that looks like a secret before committing: `.env` files and their kin, private keys, service-account JSON, and any file containing one of your keys or a provider-shaped key.
Those files stay in the worktree; the task's result says which were kept out, and why.

**`buddy merge`** refuses a branch that adds one, listing each `path:line` and the kind - never the value:

```
not merging buddy/t-0007-add-limits: it adds what looks like secrets. Nothing was changed.
  config/client.py:12: openai key
If these are lookalikes you have checked, merge with --allow-secret-patterns.
```

A match for one of *your own* keys has no override: remove it from the branch.
The brain cannot override either; it tells you instead.

Your own keys are recognised exactly, in any shape, because Buddy compares against the values in its keychain; a Fish Audio key has no prefix to spot.
Keys from providers are recognised by their shape with a randomness check, so `sk-proj-xxxx` in a README or a test fixture is not mistaken for one.
A line that must keep a lookalike can carry `secret-scan: allow`.

### Scanning a repository yourself

```sh
scripts/check-secrets.py --staged    # what is about to be committed
scripts/check-secrets.py --tree      # everything tracked, and everything `git add` would pick up
scripts/check-secrets.py --history   # every file ever committed
```

`git config core.hooksPath <path to buddy>/.githooks` runs the first one before every commit, in Buddy's repository or any other.
These never read the keychain unless given `--with-keychain`: macOS asks permission when a Python other than Buddy's reads it, and a hook must not stop to wait for a dialog.

**If a real key was ever committed anywhere, rotate it.** Deleting it from the file, or even from history, does not un-publish it.

## Brainstorming first

Sometimes you want to think before anything starts.
Say *"let's brainstorm this"*, or type `/brainstorm`.

While brainstorming, **nothing starts or changes**.
That is not a request the model is asked to honour: the tools that spawn, kill, reprioritize, preempt, merge, discard or resolve are not offered to it at all, and a call to one is refused anyway.
It still reads your code, recalls past conversations, and explores options with you.

As ideas firm up, Buddy records them as **drafts** - the same brief it would spawn, but inert.

```
you (brainstorming)> /drafts
d1  Add per-key rate limiting to the public API  (webapp, codex, p1)
d2  Document rate limits in the API reference  (webapp, opencode, after d1)
```

When you are ready, type `/go`, or say *"go ahead"*.
Saying it makes Buddy show you the list and ask once - even with `trust_mode` on, because choosing to brainstorm is choosing a checkpoint before hand-off.
Exactly the drafts you saw are started, dependencies first, with `after d1` becoming a real dependency on the task `d1` became.

A draft that cannot start - its harness signed out, say - stays a draft, along with anything waiting on it, and brainstorming stays on until every draft is out.
`/brainstorm off` leaves without starting anything and keeps the drafts.
They are saved to `~/.buddy/brainstorm.json`, so a brainstorm survives restarting Buddy.

Set `brainstorm_first = true` under `[brain]` to start every session this way.
The dashboard shows a **Brainstorming** section with the drafts, and the overlay's header says `thinking`.

## The overlay

A small always-on-top panel for the corner of your screen, so you can see what the agents are doing without switching to a browser.

```sh
buddy overlay            # open it
buddy overlay --toggle   # open if closed, close if open
buddy overlay --stop     # close it
buddy overlay --status   # one line in the terminal instead
```

It shows each running agent, what it is doing right now, and a summary line - and turns amber the moment an agent needs you, listing that one first.
Drag it by its header; the `–` button folds it to a single line.
It sizes itself to whatever it is showing, and scrolls once there are more agents than fit.

It needs the `overlay` extra (`uv sync --extra overlay`) and a session running with its dashboard, because it reads the same read-only API.
Only one runs at a time, and a stale pid file from a crash or reboot is cleaned up rather than believed.

**You can type at it.** The box at the bottom sends a turn to the running session and shows the reply, so the panel is a way to talk to Buddy and not only to watch it.
It appears only while a session is actually listening, and disappears when one is not - a text box that silently does nothing is worse than no text box.

That path does **not** go through the dashboard.
The panel's own process holds a unix socket at `~/.buddy/control.sock`, mode 0600, and the page asks that process rather than the socket; a browser cannot open a unix socket at all.
The dashboard's routes stay read-only, and what you type joins the same queue voice uses, so exactly one thing ever drives the conversation.

To bind it to a key, point a macOS Shortcut or an Automator quick action at `buddy overlay --toggle`.

## The dashboard

`http://127.0.0.1:4321`, read-only by design: opening, closing or refreshing the tab cannot affect a running task.

- **Dashboard** - a card for every running agent with its name, status, task, priority, age, time since last output, and a three-line tail of what it is doing; then the queue with dependency badges and each task's agent name, then recent tasks.
- **Task** - the log in a terminal with its colours intact, the brief, the branch, a `diff --stat`, and the parsed result.
- **Conversation** - the transcript, updating live.

Websocket connections are refused unless they come from the dashboard's own page.
A browser tab from any other site cannot read your conversation or your logs.

## Shutting down

```sh
buddy shutdown
```

| Gone | Kept |
|---|---|
| Running agent processes | Every branch, checkpointed first |
| Every worktree | Every attempt's log |
| The tmux session | `state.db` entire: memory, conversation, task history |

A mid-flight task is checkpointed and **requeued**, so it resumes from its last commit next time you start.
That also means shutdown is not a way to make things stop permanently - `buddy discard <id>` first if you want a task gone.

`buddy diff` and `buddy merge` keep working afterwards: neither ever needed the worktree, only the branch.

---

## When something is wrong

Start with `buddy doctor`.
It names what is wrong and what to type.

### "projects: none configured"

Nothing can be spawned until there is one.

```sh
buddy setup --force config
```

Then restart `buddy` - the brain reads the project list at startup.

### The brain will not authenticate

```sh
buddy setup --force keys
```

Paste the key **once**. The prompt hides what you type, so a paste that seems not to have registered usually did; pasting again concatenates. Setup now refuses a key that contains its prefix twice, and echoes back the length and last four characters so you can see what landed. It finishes with a real one-token call, so a green result means the key genuinely works.

Your Claude Code subscription is *not* an API key. The brain calls the API directly and needs its own billing.

### An agent says `running` but the task finished

Only a running session ticks.
If you spawned tasks and then closed the session, nothing noticed the panes die.

```sh
buddy status
```

catches up before it prints.

### An agent shows no output at all

For Claude Code, the command is probably still using `--output-format json`, which emits nothing until the run ends; use `stream-json --verbose`.
For the others, compare the command with the defaults in [configuration](configuration.md#harnessname) - `buddy doctor` names the known mistakes, like OpenCode given its brief on stdin.

If the harness is sandboxed and the log holds nothing but the start line, check whether your `sandbox_command` passes `-i`.
It hands the container the pane's terminal as stdin, and a CLI that reads stdin when it is not a terminal then waits for an end that never arrives.
Buddy's own command has no `-i`, and `buddy doctor` says so if yours does.

### A sandboxed agent's own commits are missing

The branch has only Buddy's `WIP (buddy, ...)` checkpoint and none of the agent's commits.
Check whether your `sandbox_command` passes `{git}`.
A worktree's `.git` is a file pointing at a directory outside the worktree, so without it git does not work in the container at all - `git log`, `git diff` and `git commit` all answer "not a git repository" - and only Buddy's end-of-run checkpoint survives.
`buddy doctor` says so if yours is missing it; [configuration](configuration.md#harnessnamesandbox_command) has the current command.

### doctor says a harness is "not signed in"

Buddy asks each CLI whether it can reach its model before giving it work - `codex login status`, `agy models`.

| Harness | Fix |
|---|---|
| `codex` | `codex login`, or set `CODEX_API_KEY` (not `OPENAI_API_KEY`) in `[harness.codex.env]` |
| `antigravity` | run `agy` once and finish the sign-in |

Then `buddy setup --force harnesses`, or just spawn again: a failed check is re-run after fifteen seconds.

### A Codex agent is `stalled` and its card says `retrying`

Codex cannot reach its API and will retry forever.
Check its login and your network; `buddy kill <agent>` stops it.

### An Antigravity agent is `waiting_input`

It is signed out and waiting for a code.
`buddy watch <agent>` shows the URL - or kill it, run `agy` once to sign in, and spawn again.

### Voice is slow

The default whisper model is sized for Apple Silicon.
On any other CPU, set `stt_model = "faster-whisper:base.en"`.
That is 4363 ms → 661 ms per utterance, measured, with no loss of accuracy on the test clip.

### A task is queued and never starts

`buddy status` shows why.
Usual causes: it waits on a dependency; its project is not a git repo and another of its agents is running; or `max_concurrent` is set, reached, and it does not outrank anything running.

A dependency that ended in `error`, `killed` or `discarded` can never be satisfied - Buddy says so once, and the task needs reprioritizing or discarding.

### Merge refuses

By design.
Be on the target branch, with a clean tree.
A conflict is aborted rather than left half-applied, and reported with the conflicting paths - `buddy resolve <task_id>` hands it to an agent.

If it says a conflict fix is pending, merge that fix first; `--force` overrides.

### doctor says a project is "merging"

A merge was left half-done in that project's own checkout - usually a Buddy merge killed mid-conflict, which is the one case no cleanup code can run for.
Buddy never aborts it for you, because you may already be resolving it.
Finish it (resolve, then `git commit`) or undo it with the `git -C <path> merge --abort` doctor shows.

### `config error` when anything starts

`buddy config` shows the error without needing the file to load, and `buddy config edit` opens it and checks it again when you close the editor.

---

## Known gaps

What is not there yet, or not proven.

| Gap | Consequence |
|---|---|
| Antigravity's success path is unverified | Every failure path was run for real; a successful run's output was not, because it needs a Google account. `parse_result` trusts the exit code over anything unexpected. |
| The container sandbox is tested only where Docker runs | The sandbox tests skip without a Docker daemon - on the macOS CI runners, for one - and those that need git inside the container skip unless `BUDDY_TEST_IMAGE` names an image that has it. |
| An outside tool holding `state.db`'s write lock stalls the session | Measured: Buddy's own processes never block each other for more than a couple of milliseconds, but a tool of yours holding a write transaction open freezes the tick for as long as it holds it. |
| A log line straddling `max_log_mb` is split across two files | Nothing is lost; the two files concatenate back exactly. |
| Voice falls back only at the same sample rate | Playback is opened for the primary backend's rate. |
| Windows | Out of scope because tmux is. WSL2 works. |
