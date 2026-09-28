#!/bin/sh
# Buddy's installer.
#
#   curl -fsSL https://<host>/install.sh | sh
#
# This is the hosted path, and it needs a published package: set
# BUDDY_PACKAGE to the distribution name once one exists, or to a
# git+https URL to install straight from source. Until then, the
# supported route is the clone-and-`uv sync` one in the README.
#
# It does three things and nothing else: install uv if it is missing, install
# the tool, then hand over to `buddy setup`. Everything that decides anything
# - packages, GPU, Docker, keys, models - is a setup step, where it can be
# checked, re-run, and forced. Nothing in here is idempotency-sensitive
# because nothing in here makes a decision.
#
# POSIX sh on purpose: this runs before Buddy has any say over the machine.

set -eu

PACKAGE="${BUDDY_PACKAGE:-buddy-orchestrator}"

say() { printf '%s\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  Darwin | Linux) ;;
  *)
    die "Buddy runs on macOS and Linux. Windows is out of scope because tmux is;
     WSL2 works and counts as Linux."
    ;;
esac

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  die "needs curl or wget to fetch uv."
fi

if command -v uv >/dev/null 2>&1; then
  say "uv is already installed ($(uv --version))."
else
  say "Installing uv, which supplies its own pinned Python."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi
  # uv installs to ~/.local/bin, which is not on PATH in this shell yet.
  for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -x "$candidate/uv" ]; then
      PATH="$candidate:$PATH"
      export PATH
      break
    fi
  done
  command -v uv >/dev/null 2>&1 || die "uv installed but is not on PATH. Open a new shell and re-run."
fi

say "Installing $PACKAGE."
uv tool install --force "$PACKAGE"

if ! command -v buddy >/dev/null 2>&1; then
  # `uv tool install` prints this itself, but a piped installer scrolls past.
  say ""
  say "buddy is installed but not on PATH yet. Run:"
  say "    uv tool update-shell"
  say "then open a new shell and run: buddy setup"
  exit 0
fi

say ""
# exec, not a call: setup is interactive, and it should own the terminal and
# the exit code from here on.
#
# stdin is reattached to the terminal because the documented way to run this
# is `curl ... | sh`, which leaves stdin as the pipe - and setup has to ask
# for an API key. Without this, every prompt hits EOF immediately.
if [ -r /dev/tty ]; then
  exec buddy setup "$@" < /dev/tty
else
  say "No terminal available, so setup cannot ask for anything. Run it yourself:"
  say "    buddy setup"
  exit 0
fi
