# How Buddy actually works

This describes the system as built, and why the parts that look surprising are the way they are.

---

## The shape

You talk to a **brain** - an LLM with tools.
It writes briefs and drives an **agent manager**, which starts each task as a named **agent**.
An agent is a tmux window running a coding-agent CLI inside its own git **worktree** - made when its task starts, closed when it ends, and as many of them as there is work.

```
  you ──voice/text──▶  brain  ──tools──▶  manager ──▶ tmux windows ──▶ harness CLIs
                         │                   │                              │
                         │                   ├── workspace (git worktrees, branches)
                         │                   └── state.db (tasks, runs, agents)
                         └──────────── dashboard (read-only) ◀──────────────┘
```

Three things carry the truth, and they are deliberately separate:

| Source of truth | For |
|---|---|
| `state.db` | *What* happened - tasks, attempts, outcomes, the conversation |
| The log files | *The output* - raw pane capture, escapes intact |
| The git branches | *The code* - every attempt's work, committed |

Buddy's in-memory state is a cache of all three and is rebuilt on startup.
That is what makes killing Buddy - or tmux, or the machine - survivable.

## The layer rule

One module knows each external thing, and no other module may.

| Module | Owns |
|---|---|
| `tmux_runner.py` | tmux. Nothing else runs a tmux command. |
| `workspace.py` | git. Nothing else runs a git command. |
| `harnesses/*.py` | One coding-agent CLI each: its flags, its output format, its prompts. |
| `providers/*.py` | One LLM API each: its wire format and its capabilities. |
| `voice/tts.py` | Fish Audio's protocol. |
| `voice/stt.py` | The speech-recognition engine and the microphone. |
| `state.py` | SQLite. |
| `web/server.py` | HTTP and websockets. |

A leak across those lines is a bug even when the tests pass.
It is why the manager asks an adapter to `describe_activity()` rather than parsing JSON itself, and why setup's package step asks `tmux_runner` for tmux's version instead of running `tmux -V`.

## The module map

```
buddy/
  cli.py            entry point: commands, the session loop, the 1s ticker
  brain.py          the conversation: tools, briefs, confirmations, context
  ideation.py       brainstorming: the mode, drafts, and what launches on /go
  conflicts.py      a merge that conflicted: the fix task, and the merge hold
  manager.py        agents and their names, priority, dependencies, preemption, health, recovery
  workspace.py      worktrees, branches, checkpoint, merge, discard
  tmux_runner.py    session and windows, respawn, pipe-pane, process-tree kill
  state.py          SQLite (WAL)
  config.py         config.toml, secret resolution, the ~/.buddy layout
  models.py         the vocabularies: AgentStatus, TaskState, RunOutcome, events; agent names
  logs.py           reading pane output: ANSI stripping, tails, sentinels
  logpipe.py        the pipe-pane target: appends a pane's output, rotates it losslessly
  leaks.py          recognising secrets, before a checkpoint, a merge or a commit
  control.py        the unix socket the overlay types through
  providers/        base.py + one module per LLM API
  harnesses/        the registry, base.py, stream.py, and one module per coding-agent CLI
  voice/            stt.py (recognition, microphone), tts.py (Fish), session.py
  web/              server.py (routes), events.py (pub/sub), static/
  setup/            the step runner, platform detection, steps/, assets/
```

---

## The three hard mechanisms

Everything else is arrangement. These are the parts that had to work.

### 1. The worktree is disposable; the branch is the work

Each task gets `git worktree add` on a branch named `buddy/<task_id>-<slug>`, cut from the project's base branch.
The agent works there and can do anything it likes, because nothing it does reaches your base branch without an explicit `buddy merge`.

Before any attempt ends - success, kill, timeout, crash - its uncommitted work is **checkpointed**: `git add -A` and a `WIP (buddy, <outcome>, attempt N)` commit on its branch.

This single property is why several other things are safe:

- `buddy shutdown` can delete every worktree without losing anything.
- `buddy diff` and `buddy merge` work after the worktree is gone - both operate on your own checkout and the branch name.
- A resumed attempt reattaches the existing branch and continues from its last checkpoint.

A project that is not a git repository has no worktree, so its agents run in place, one at a time.

### 2. A dead pane is the completion signal

Windows are created with `remain-on-exit on`, so a finished process leaves a **dead pane** that keeps its exit status.
Polling `#{pane_dead}` and `#{pane_dead_status}` is how Buddy knows a task ended, and the status survives Buddy being dead at the time.

The wrapper script also writes `__BUDDY_START__` and `__BUDDY_DONE__` sentinels carrying the task id and exit code.
Those are a **redundant record, never the signal** - and they are matched on task id, because a stale line in a log must never be read as this task's.

Details that only appeared against real tmux:

- A tmux session ends with its last window, and the server with its last session. So there is no window to keep and nothing to create up front: the first agent brings the session, and closing a pipe on a window that has gone - with the server gone too - is a no-op rather than an error.
- `remain-on-exit` is set on every spawn, not only when Buddy makes the window, because a window made by hand does not have it and its exit status would be lost.
- `#{pane_pipe}` reports that *a* pipe exists, never where it points, so a retry that reuses its window re-opens the pipe unconditionally.
- `display -p -t <session>:<window>` silently answers for the session's *current* window when the target does not resolve, so it can never prove a window exists. Enumerating `list-panes -s` can, and hands over every agent's pane in one call.
- Every target is `session:=name`. Without the `=`, tmux matches a prefix, and `scout` would act on `scout-2`.

### 3. Output is captured with `pipe-pane`, not scraped

`pipe-pane` streams the pane's raw output, for the life of the attempt, into `logpipe.py`, which appends it to the attempt's log.
That gives an append-only log with escapes intact - which is why the dashboard can render it in a real terminal emulator, and why stall detection is a `stat` on a growing file rather than a diff of screen captures.

The writer exists to cap the log without losing anything.
The first attempt renamed the log and pointed `pipe-pane` at a new one; tmux tears the old pipe down with bytes in flight, and against real tmux that lost two to four lines at every switch.
One process holding the stream across the switch cannot lose a byte, so the writer rotates itself: never above the cap, at a line end whenever the bytes crossing the cap contain one, and always immediately - never waiting for a newline, so a progress bar stays live.

## The generated wrapper

Buddy never sends a prompt through `send-keys`.
It writes the brief to `prompt.md`, generates a `run.sh`, and the window runs the script.

The prompt never touches a shell quoting layer; the script is always bash whatever your login shell is; and the log is self-describing.
`run.sh` is mode `0600` because resolved secrets are exported in it, and it is deleted once the attempt's `result.json` is written.

Every value substituted into the harness command line is `shlex.quote`d.
That matters more than it looks: `model` is a free-text field an LLM sets, and the result is bash.

## Scheduling

Ready tasks are those that are queued and unblocked, ordered by priority number then FIFO.
A dependency counts as satisfied when it is `done`, or only when it is `merged` if it was declared `merge_required`.

The tick, once a second:

1. Finalize any agent whose pane has died, and close its window.
2. Check the health of the rest - waiting-input patterns in the tail, progress from the log growing, runtime against `max_runtime`. What grew is first offered to the adapter's `is_progress`, so a harness retrying a dead API forever is not mistaken for one working.
3. Start every ready task there is room for - all of them, unless `max_concurrent` is set. A task whose harness cannot be built, whose checkout cannot be made, or whose run cannot be prepared - a secret nobody stored - fails alone, rather than being retried every tick while the queue behind it waits.
4. Propose a preemption where a queued task outranks a running one *and* stopping it would let the queued task start: room is `max_concurrent`, when set, and a non-git project's lock can only be freed by that project's own agent.
5. Once an hour, delete discarded branches past their grace - unless the branch's reflog shows it moved after the discard.

**Each agent's pass is isolated.** A checkpoint that raises - a full disk, a stray `index.lock`, a repo hook `--no-verify` does not suppress - used to leave the whole tick, so every other agent stopped being monitored until a restart.

## Agents and their names

A task carries the name of its agent from the moment it is saved: the one the user gave, checked and normalized, or one made from its title.
The name is its tmux window's name, the handle every command takes, and what the brain calls it.

- **Unique while it matters.** No two queued or running tasks share a name, enforced by a partial unique index in SQLite rather than by the manager alone, so two processes racing for one cannot both win. A finished agent's name is free.
- **Stable across attempts.** A preempted, timed-out or interrupted task is requeued under the same name, and its next attempt gets a new window called the same thing.
- **Resolved, not guessed.** Anything that takes an agent's name also takes a task id. A name means the live task if there is one, otherwise the latest that ran under it; names that look like task ids are refused so the two never collide.
- **Only live agents are stored.** The `agents` table holds what is running now - its status, last output and when it last made progress. How a task ended lives on the task and its runs, so a finished agent leaves nothing to clean up.

## Recovery

`reconcile()` runs before Buddy accepts any input and looks at every agent the database says is running.
It never creates a window: a window that is missing is evidence, and recreating one would put a fresh shell where the task used to be.

| Found | Done |
|---|---|
| Run **already has an outcome** | A crash came between ending the attempt and retiring its agent. Retire it; if the task still claims to be running, do what the outcome says - requeue, stay killed, or keep its verdict. |
| Pane alive, run unfinished | Resume monitoring; re-open the log pipe if needed |
| Pane dead | Finalize from its exit status, and close the window |
| Window or session gone | Read the log's sentinels: finalize if they say it finished, otherwise checkpoint and requeue |

The writes that end an attempt - the run's outcome, the task's state and the agent's row - go in one transaction, so a crash leaves either all of them or none.

## The brain

An LLM with tools, a system prompt, and five layers of context management.

Tools fall into four groups: manager tools (`spawn_agent`, `kill_agent`, `reprioritize`, `get_output`, `list_agents`, preemption answers), workspace tools (`propose_merge`, `resolve_conflict`, `discard_task`), brainstorming tools (`start_brainstorm`, `draft_brief`, `drop_draft`, `hand_off`), and a **read-only** project toolkit (`list_dir`, `read_file`, `grep`, `git_status`, `git_log`, `recent_tasks`) scoped to configured project roots.

Which tools the model is offered depends on the mode.
While brainstorming, every tool that starts or changes work is withheld, and `call_tool` refuses one anyway; the mode's instructions travel in the per-turn state message rather than the system prompt, so switching modes keeps the cached prefix.
On hand-off Buddy spawns the drafts itself from their stored fields, so what starts is exactly what was reviewed.

Confirmations are tiered: none for reversible things, a read-back for spawning, always-ask for anything that cannot be taken back - unless `trust_mode` is on.

### Untrusted input

The brain reads files, greps repositories and tails agent logs.
Any of that can carry text someone else wrote.
Two boundaries exist because of it:

- **Tool arguments are validated before dispatch.** `model` must look like a model identifier; `priority` must be 1-5; `depends_on` must be a list. Without that, `depends_on="t-0007"` became six single-character dependencies and the task queued forever.
- **Secrets are not readable.** `.env`, `.git/`, `.ssh/`, `*.pem`, `id_rsa` and friends are refused by `read_file`, `grep` and `list_dir` alike. Anything read here goes verbatim to a third-party API, into `state.db`, and back out through `recall` for the life of the project.
- **What others wrote is marked as data.** Every result of a tool that returns an agent's output, a repository's files or history, or a diff is wrapped in `<untrusted source="...">`, at dispatch so no tool can forget, with any closing marker inside the content defused. The system prompt tells the model to report such text and never follow it. A model can still be fooled, which is why confirmations and the worktree boundary stay where they are - but a log line saying "ignore your instructions" no longer arrives looking like one.
- **Unreadable arguments are named.** A provider that receives tool arguments it cannot parse says so, instead of passing `{}` on to fail as a missing argument.

### The five context layers

| Layer | What |
|---|---|
| 0 | Running agents and task state re-read from SQLite into the system-prompt region every turn, so it is never stale and never remembered |
| 1 | Tool results truncated at ingestion, plus server-side tool-result clearing where the provider has it |
| 1b | The client-side equivalent |
| 2 | Server-side compaction, with the summary persisted |
| 2b | The client-side equivalent, which is also what `buddy compact` runs |
| 3 | `recall` over an FTS5 index of the conversation |
| 4 | Pinned memory, carried across sessions |

Whether the server-side layers are used is decided by a **capability probe**, not by a table of which provider supports what.

## Harnesses

`harnesses/__init__.py` holds the one registry: the CLI, `buddy setup`, `buddy doctor` and the brain all look a harness up there, so none of them can drift.
Each adapter carries its own defaults - binary, verified command line, waiting patterns, default model, install hint, the credential variables it actually reads - and answers four questions:

| Method | Answers |
|---|---|
| `preflight()` | Does the CLI run headless, take a long prompt, auto-approve, and exit non-zero on failure - read from its own `--help`? And is it signed in, asked as cheaply as the CLI allows? |
| `invocation()` | The exact command line, placeholders filled and quoted. |
| `parse_result()` | The harness's own verdict and final message, from its JSON events. The exit code still decides; a disagreement is recorded. |
| `describe_activity()`, `is_progress()` | Sentences for an agent's card, and whether new output is work or retry chatter. |

`stream.py` gets JSON objects out of a pane log - escapes, sentinels and stderr lines included - once, for all four.
Everything each adapter knows about its CLI was found by running it; its module docstring says what, and against which version.

## Providers

One canonical message format - `Turn`, `ToolCall`, `ToolResult`, `ToolDef` - that Buddy owns.
Each provider serializes it to its own wire format.
The same tools, briefs and context layers run on all four.

Switching provider mid-session forces a compaction boundary first, so no API ever sees another API's tool-call blocks, and closes the outgoing client.

## The dashboard

FastAPI in the same asyncio loop, reading the manager's live state, the database for history, and tailing the log files.
No second polling loop.

Read-only by design, and asserted structurally: a test enumerates every route and requires GET or HEAD, so adding a kill button is a test failure rather than a code review someone has to catch.

Websocket handshakes check `Origin`, because browsers do not apply the same-origin policy to them - without it, any page open in your browser could read your conversation from localhost.

---

## The overlay

A frameless, always-on-top panel in its own process, reading the dashboard's API.

Its own process for three reasons, and the first is not negotiable: macOS wants a Cocoa event loop on the main thread, and Buddy's main thread already belongs to asyncio.
A window inside the session would mean inverting the whole program around it.
The other two are gains: it inherits the dashboard's read-only guarantee by construction rather than by discipline, and it can be killed and restarted without touching the session.

Reading is over HTTP; typing is not.
A turn typed at the panel goes down a unix socket at `~/.buddy/control.sock` (mode 0600) to the session, which puts it on the same queue voice uses - so there is still exactly one thing calling the brain, whatever asked.

A socket rather than a route is what keeps the dashboard read-only.
Every dashboard route is still GET or HEAD and `test_web.py` still enforces it; a browser cannot open a unix socket, so there is no origin to check, no port to scan and no CSRF, and the filesystem does the authorisation.
The page never touches the socket either - it asks its own Python process, which is the only thing holding it.

## Decisions worth knowing

Each of these looks like it could be simpler, and each was simpler once.

| Area | Decision, and why |
|---|---|
| Harness preflight | `preflight()` is `async`: it runs `<harness> --help`, and nothing may block the loop that supervises every agent |
| Command templates | Only their four placeholders are substituted; `str.format` raised on every other brace, which ordinary shell lines are full of |
| Sandbox mounts | The brief is mounted read-only beside the worktree - it lives outside it, and a container with only the worktree cannot read it |
| Sandbox and git | The git directory the container gets, and the identity it commits as, come from the project's own checkout, never from the worktree's `.git`, which the agent can rewrite to name any directory on your machine |
| Git after an agent | Git on the host in a task's worktree runs pinned to the repository's own `.git`, and not at all if the worktree's pointers no longer lead back to it: config there can name a command git runs |
| Reconcile | A missing window is never recreated before it is read: a fresh shell is indistinguishable from a task still running |
| Log capture | `pipe-pane` feeds a rotating writer rather than `cat >>`, because swapping the writer to rotate loses output |
| Run scripts | `run.sh` is created mode `0600` and deleted the moment an attempt ends, however it ends, because it holds resolved keys |
| Merging and discarding | Both are refused while the task's agent is still running: both remove the worktree it is working in |
| Conflicts | The fix merges the base into the conflicting branch and never rebases, so every commit the run history points at survives |
| Confirmations | `trust_mode` removes every confirmation, not only the spoken read-back |
| Context | Layer 0 is injected as a mid-conversation system message; the top-level system field would invalidate the prompt cache every turn |
| Brainstorming | A mode in which the tools that change work are not offered at all, rather than offered and asked not to be used |
| Control channel | A unix socket rather than an HTTP route, so the dashboard stays read-only |
| The dashboard | Requests are refused unless addressed to `127.0.0.1` or `localhost`, which is what stops a DNS-rebound page reading it |
| Voice | "Push-to-talk" is press-to-start, press-to-stop: a TTY reports key presses and never releases |
| Fish TTS | `/v1/tts` has no `streaming` field; the response is chunked regardless |
| Speech models | Sized separately for Apple Silicon and plain CPU, which are 6.6x apart |
