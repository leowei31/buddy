# Security

## Reporting a vulnerability

Please report it privately, not in a public issue: use **Report a vulnerability** on this repository's **Security** tab, which opens a private advisory only the maintainer can see.

Say what you found, how to reproduce it, and what it lets someone do.
You will get an acknowledgement, and a fix or an explanation, before anything about it is made public.

## What Buddy is trusted with

Knowing the boundaries makes a report easier to judge.

- **Coding agents run unattended, with permissions bypassed**, inside a git worktree of your project. The worktree and its branch are the isolation boundary; nothing reaches your own branch without `buddy merge`. A container is the optional second boundary. An agent escaping its worktree *without* the container is expected, not a vulnerability; escaping the container is one. The container is given the repository's git directory, because committing is the whole point of the branch, with `.git/config` and `.git/hooks` read-only inside it - a way to write either of those from the container is an escape, since your own git reads them on the host, and so is any way to make git on the host read configuration the agent wrote.
- **API keys** are kept in the OS keychain, written into a run's `run.sh` (mode `0600`) only while it runs, and never committed by Buddy: its checkpoint and `buddy merge` both refuse secrets. A way to get a key into git, a log that leaves the machine, or another user's view is a vulnerability.
- **The dashboard** binds to `127.0.0.1` and is read-only; its websockets refuse other origins. A way for a web page to read the conversation or a log, or for the dashboard to change anything, is a vulnerability.
- **The overlay's control socket** is `~/.buddy/control.sock`, mode `0600`, so only your own user can send turns through it.
- **Text from agents and repositories** reaches the model marked as untrusted data. A prompt injection that makes Buddy take an action the user did not confirm is worth reporting; one that only makes the model *say* something odd is a known limit of language models.

See [docs/operating.md](docs/operating.md#secrets) for how secrets are handled, and [docs/architecture.md](docs/architecture.md#untrusted-input) for the boundaries in detail.

## If a key was exposed

Rotate it with its provider first.
Removing it from a file, or from git history, does not un-publish it.
