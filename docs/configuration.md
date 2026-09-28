# Configuration reference

Everything Buddy reads from `~/.buddy/config.toml`, with its real default taken from the code.

`buddy setup` writes this file and, on every later run, **merges in new keys without overwriting anything you changed**.
Editing it by hand is expected.
Set `BUDDY_HOME` to point the whole tree somewhere else.

**No secret is ever stored here.** What appears in the file is the *name* of an environment variable or a `${keychain:NAME}` reference; the value lives in your OS keychain, and an environment variable of the same name overrides it.

A malformed value fails at load with a message naming the key.
Booleans must be unquoted: `trust_mode = "false"` is refused rather than silently read as true.

---

## `[buddy]`

| Key | Type | Default | What it does |
|---|---|---|---|
| `max_concurrent` | int | `7` | How many agents may run at once. Capped at seven, because there are seven slots. |
| `default_priority` | int | `3` | Priority for a task that does not name one. 1 is urgent, 5 is whenever. |
| `stall_timeout` | duration | `"10m"` | No output for this long marks a slot `stalled`. Buddy **notifies and never kills**: a long compile looks exactly like a hang, so the decision is yours. |
| `max_runtime` | duration | `"2h"` | Then the task is killed, checkpointed and requeued once. A second timeout is an error. |
| `web_port` | int | `4321` | The dashboard's port, bound to `127.0.0.1` only. |
| `trust_mode` | bool | `false` | `true` stops Buddy asking before anything - spawn, kill, merge, discard. See the warning below. |
| `discard_grace` | duration | `"7d"` | How long a discarded task's branch is kept before an hourly sweep deletes it. A branch that anyone moved after the discard is never deleted - see [operating](operating.md#discarding). |
| `max_log_mb` | int | `64` | Each attempt's log is rotated past this many MiB, keeping one previous file, so an attempt never uses more than twice this on disk. `0` turns the cap off. |

Durations are `"30s"`, `"10m"`, `"2h"`, `"7d"`, or a bare number of seconds.

> **On `trust_mode`.**
> The design scoped this to the spoken read-back only, and made kill, preempt and merge undisableable, on the grounds that those three cannot be taken back.
> It now turns off every confirmation.
> What still protects you is unchanged and is the part that does the work: every task runs on its own branch in its own worktree, and nothing reaches your base branch without a merge.

## `[brain]`

The conversational layer - the thing you talk to, which writes the briefs and drives the scheduler.

| Key | Type | Default | What it does |
|---|---|---|---|
| `provider` | str | `"anthropic"` | One of `anthropic`, `anthropic_vertex`, `openai`, `vertex_gemini`. |
| `model` | str | provider's default | Blank uses the provider's own default (`claude-opus-5` for Anthropic). |
| `strategy` | str | `"auto"` | `auto` uses server-side context management where the provider has it, else client-side. `server` and `client` force one. |
| `compact_trigger_tokens` | int | `100000` | Compact the conversation past this size. The API minimum is 50000. |
| `keep_recent_turns` | int | `6` | Turns kept verbatim after a compaction. |
| `tool_output_tail_lines` | int | `60` | Tool results are truncated to this many lines *at ingestion*, so an enormous one never enters the context at all. |
| `read_file_max_bytes` | int | `20000` | Cap on what `read_file` returns - and now on what it reads. |
| `brainstorm_first` | bool | `false` | Start every session brainstorming, so nothing is spawned until you type `/go`. See [brainstorming](operating.md#brainstorming-first). |

### `[brain.clear_tool_uses]`

| Key | Type | Default | What it does |
|---|---|---|---|
| `trigger` | int | `40000` | Clear old tool results past this size. |
| `keep` | int | `5` | How many recent tool results to keep. |

### `[brain.<provider>]`

One block per provider, read only when that provider is selected.

```toml
[brain.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[brain.openai]
api_key_env = "OPENAI_API_KEY"
base_url    = ""            # set for Azure OpenAI, vLLM, Ollama, or any compatible endpoint

[brain.anthropic_vertex]
project_id  = "your-gcp-project"
region      = "global"

[brain.vertex_gemini]
project     = "your-gcp-project"
location    = "us-central1"
```

`api_key_env` names *where* the key is, never the key.
It is resolved environment-first, then the OS keychain.
The two Vertex providers use Google Application Default Credentials instead, which are a login rather than a string.

An option this provider does not accept produces a message naming the key, not a traceback.

## `[projects.<name>]`

At least one is required; with none, nothing can be spawned.
The name is what you say out loud: *"spawn something on **webapp** to..."*

| Key | Type | Default | What it does |
|---|---|---|---|
| `path` | path | required | The git repository. `~` and `$VAR` are expanded. |
| `base_branch` | str | `"main"` | What task branches are cut from and merged into. |
| `default_harness` | str | see below | Which harness to use when a task does not name one. Unset, it is `claude_code` if that is configured, and otherwise the first configured harness alphabetically. |
| `worktree_root` | path | `~/.buddy/worktrees/<name>` | Where this project's worktrees live. |

A project that is **not** a git repository still works, but takes a slot exclusively: agents run in place, one at a time, because there is no worktree to isolate them.

## `[harness.<name>]`

A harness is the coding-agent CLI a task runs on.
Four are supported, and `buddy setup` writes a block for each one that is installed, passes preflight, and is signed in.

| Name | Binary | Default command | Default model |
|---|---|---|---|
| `claude_code` | `claude` | `claude -p --output-format stream-json --verbose --permission-mode bypassPermissions {model_flag} < {prompt_path}` | `claude-sonnet-5` |
| `codex` | `codex` | `codex exec --json --color never --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check {model_flag} - < {prompt_path}` | the CLI's own |
| `opencode` | `opencode` | `opencode run --format json --auto {model_flag} -- "$(cat {prompt_path})"` | the CLI's own |
| `antigravity` | `agy` | `agy --output-format stream-json --dangerously-skip-permissions --disable-slash-commands --print-timeout 24h {model_flag} -p="$(cat {prompt_path})"` | the CLI's own |

Each of those command lines was read from the CLI's own `--help` and run for real; [operating](operating.md#choosing-a-harness) says what each one is careful about and why.
If you change one, `buddy doctor` warns about the changes known to break a harness - for example feeding OpenCode its brief on stdin, where it hangs.

| Key | Type | Default | What it does |
|---|---|---|---|
| `command` | str | per harness, above | The command line. Placeholders below. A block with an empty command is *parked*: configured, never given a task. |
| `default_model` | str | per harness, above | Used when a task names no model. Model names are the harness's own: OpenCode's are `provider/model`. |
| `waiting_patterns` | list[str] | per harness | Regexes that mean the harness is asking a question, which marks the slot `waiting_input`. Only Claude Code and Antigravity ship any; Codex and OpenCode never stop to ask. |
| `sandbox` | str | `"none"` | `"docker"` runs the harness in a container. |
| `sandbox_command` | str | - | Required when `sandbox = "docker"`. Placeholders below. |

### Command placeholders

Every substituted value is shell-quoted, so none of them can break out of its argument position.

| Placeholder | Becomes |
|---|---|
| `{prompt_path}` | The brief, at `~/.buddy/tasks/<id>/prompt.md` |
| `{worktree}` | The task's checkout |
| `{model}` | The resolved model, or empty |
| `{model_flag}` | `--model <resolved>`, or empty |

Only those four names are placeholders.
Every other brace in a command is left exactly as written, so Codex's `-c 'key={...}'` overrides, an `awk '{print $1}'`, a JSON argument or `${VAR}` all work.

> For Claude Code, `--output-format json` emits a single object when the run *ends*.
> A nine-minute task left a three-line log, a blank pane throughout, and a stall detector that would have fired on a perfectly healthy agent.
> Every default above streams instead, which is what makes a running agent visible.

### `[harness.<name>.sandbox_command]`

The second isolation boundary, off by default.
`{worktree}`, `{prompt_path}`, `{command}`, `{env}`, `{git}` and `{image}` are substituted.
Each arrives already shell-quoted, so write placeholders bare - never inside quotes of your own.

```toml
sandbox = "docker"
sandbox_command = "docker run --rm -v {worktree}:{worktree} -v {prompt_path}:{prompt_path}:ro -w {worktree} --network host{git}{env} --entrypoint sh buddy-harness:latest -c {command}"
```

`{command}` is substituted as **one shell-quoted argument**, so it must be given to a shell inside the container - `--entrypoint sh ... -c {command}` above.
A harness command is a shell line (flags, a redirect, `$(cat ...)`); passed as bare words there is no shell to read them, and an image whose own entrypoint is a shell keeps only the first word, which runs a harness with none of its flags.
The entrypoint is overridden rather than assumed, so any image works.

There is deliberately no `-i`.
A harness runs in a tmux pane, so its stdin is that pane's terminal, and `docker run -i` forwards it into the container as a pipe;
a CLI that reads stdin when it is not a terminal then waits for an end that never comes.
Measured: with stdin closed OpenCode finished in 7 seconds, and with the pane's terminal it hung indefinitely, printing nothing.
The container never needs stdin - the brief is the mounted `prompt.md`, and a `<` redirect in the command is performed by the shell inside the container.

`{env}` becomes one ` -e NAME` per name in `[harness.<name>.env]`, and nothing else.
A container inherits no environment from the host, so without it a key configured the documented way is simply absent inside the sandbox.
Only the name is passed: docker reads the value from the environment `run.sh` exported, so no secret reaches a command line or `docker inspect`.
Drop `{env}` from the command if you would rather mount the CLI's login directory read-only instead.

`{git}` is what makes the worktree a *working* worktree inside the container, and you want it.

A worktree's `.git` is a file reading `gitdir: <repo>/.git/worktrees/<id>`, which is outside the mount.
Without `{git}` every git command in the container answers "not a git repository" - while the working rules Buddy appends to every brief tell the agent to commit as it goes, and a retry's rules tell it to run `git log` first.
The agent's own commits never happen and only Buddy's end-of-run checkpoint survives.

So `{git}` mounts the repository's git directory, and mounts `.git/config` and `.git/hooks` **read-only** inside it.
That second part is a containment boundary rather than tidiness: both are read by *your* git, on the host, the next time you use that repository, so an agent that could write them could run code on your machine later.
Neither is written by committing, so nothing legitimate is lost.
`{git}` also passes the repository's own `user.name` and `user.email`, because a container has no `~/.gitconfig` and git otherwise refuses to commit at all.
It expands to nothing for a project that is not a git repository, which runs in place and has no branch to commit to.
The git directory and the identity are read from the project's own checkout, never from the worktree.
The worktree's `.git` is a file inside the mount, so an agent can rewrite it to name any directory on your machine - and anything read from it would decide what the next attempt's container is given.

`{image}` expands to `buddy-harness:latest`, the image the build below produces; write your own image name instead if you build a different one.

The brief is mounted **read-only and separately**: it lives outside the worktree, so a container with only the worktree mounted cannot read it.
Build the image with:

```sh
docker build -f buddy/setup/assets/Dockerfile.harness -t buddy-harness:latest \
    --build-arg HARNESS_INSTALL="@anthropic-ai/claude-code @openai/codex opencode-ai" \
    --build-arg ANTIGRAVITY=1 .
```

Each `--build-arg` is optional: name the npm CLIs you want, and set `ANTIGRAVITY=1` for `agy`, which is not on npm.
The image carries no credentials; give the container a key through `[harness.<name>.env]`, which `{env}` forwards, or mount the CLI's login directory read-only in `sandbox_command`.

### `[harness.<name>.env]`

Environment for the generated `run.sh`, which is mode `0600` and deleted once the attempt's result is written.
`buddy doctor` and preflight's sign-in checks run with it too, so a key that lives only here still counts as signed in.

```toml
[harness.codex.env]
CODEX_API_KEY = "${keychain:OPENAI_API_KEY}"
```

`${keychain:NAME}` is resolved in this block and nowhere else.
Anywhere else it is refused when the config loads, with the key named - used as literal text it would fail far from its cause.
Other settings take the *name* of a variable instead, like `api_key_env`.

> **Use the name the CLI reads.**
> Codex in `exec` mode reads `CODEX_API_KEY` and ignores `OPENAI_API_KEY` entirely - exported under the wrong name, the key looks configured and every run fails with "Missing bearer".
> OpenCode reads each provider's standard name (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
> Claude Code and Antigravity sign in with their own login and are given no key.
> `buddy setup` wires your brain's key only into a harness that reads it, under the name it reads.

## `[voice]`

| Key | Type | Default | What it does |
|---|---|---|---|
| `stt_backend` | str | `"whisper"` | Speech recognition engine. |
| `stt_model` | str | `"faster-whisper:small"` | See the sizing note below - this default is wrong for a non-Apple-Silicon CPU. |
| `push_to_talk` | str | `"ctrl+space"` | The key that starts and stops listening. |
| `hands_free` | bool | `false` | `true` listens continuously and ends a turn on a pause. |
| `silence_ms` | int | `900` | How long a pause ends a turn. Shorter interrupts you mid-thought; longer adds a dead second to every exchange. |
| `tts_backend` | str | `"fish_local"` | `fish_local` or `fish_cloud`. |
| `tts_fallback` | str | `"fish_cloud"` | Spoken through when `tts_backend` is not answering - at startup, or mid-session. Only at the same sample rate, and never mid-sentence. Empty turns it off. |
| `input_device` | str | first found | Microphone, by name. |
| `output_device` | str | first found | Speaker, by name. |

> **Model sizing.**
> The default is sized for Apple Silicon.
> Measured on an Intel CPU against a 2.1-second clip: `small` takes 4363 ms (2.05x realtime), `base.en` takes 661 ms (0.31x) and transcribed identically, `tiny.en` takes 361 ms but dropped a word.
> On a non-Apple-Silicon CPU, set `stt_model = "faster-whisper:base.en"`.

### `[voice.fish_local]` and `[voice.fish_cloud]`

```toml
[voice.fish_local]
base_url   = "http://127.0.0.1:8080"
format     = "pcm"        # pcm | wav | mp3 | opus - pcm needs no decoding before playback
streaming  = "auto"       # auto | websocket | http; auto probes once at startup

[voice.fish_cloud]
base_url     = "https://api.fish.audio"
api_key_env  = "FISH_API_KEY"
model        = "s2-pro"   # sent as a header; the server reports what actually answered
reference_id = ""         # a voice from your Fish account
```

Local TTS needs an NVIDIA GPU with roughly 8 GB of VRAM and a running Docker daemon.
Without one, setup configures cloud and says why.
Without a `FISH_API_KEY` either, Buddy simply does not speak - everything else works.

---

## A complete example

```toml
[buddy]
max_concurrent   = 7
default_priority = 3
stall_timeout    = "10m"
max_runtime      = "2h"
web_port         = 4321
trust_mode       = false
max_log_mb       = 64

[brain]
provider               = "anthropic"
model                  = "claude-opus-5"
strategy               = "auto"
compact_trigger_tokens = 100000
keep_recent_turns      = 6

[brain.clear_tool_uses]
trigger = 40000
keep    = 5

[brain.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[projects.webapp]
path            = "~/code/webapp"
base_branch     = "main"
default_harness = "claude_code"

[harness.claude_code]
command          = "claude -p --output-format stream-json --verbose --permission-mode bypassPermissions {model_flag} < {prompt_path}"
default_model    = "claude-sonnet-5"
waiting_patterns = ['\(y/n\)', 'Do you want to proceed\?']

[harness.codex]
command = "codex exec --json --color never --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check {model_flag} - < {prompt_path}"

[harness.codex.env]
CODEX_API_KEY = "${keychain:OPENAI_API_KEY}"

[voice]
stt_model   = "faster-whisper:base.en"
tts_backend = "fish_cloud"
hands_free  = false
```

## The rest of `~/.buddy`

The directory itself is mode `0700`, and an older install is tightened on the next run: everything in it is yours alone.

| Path | What it is |
|---|---|
| `config.toml` | This file, mode `0600`. |
| `state.db` | SQLite (WAL), mode `0600`: tasks, runs, slots, the conversation and its search index, pinned memory. |
| `brainstorm.json` | Whether a brainstorm is on, and its drafts. Survives a restart. |
| `control.sock` | The unix socket the overlay types through, mode `0600`, while a session runs. |
| `tasks/<id>/prompt.md` | The brief that was sent. |
| `tasks/<id>/attempt-N.log` | Raw pane output, escapes and all, mode `0600`. |
| `tasks/<id>/attempt-N.log.1` | The previous part of that log, once it passed `max_log_mb`. |
| `tasks/<id>/result.json` | The parsed outcome of the last attempt. |
| `tasks/<id>/run.sh` | The generated wrapper, mode `0600`, deleted once the result is written. |
| `worktrees/<project>/<id>/` | The task's checkout. Disposable: the branch is the work. |
| `setup.lock` | What each setup step settled on, so `buddy update` knows what moved. |
| `platform.json` | What setup detected about this machine. |
| `keys/` | Service-account JSON, mode `0700`. |
