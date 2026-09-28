"""The execution layer: tmux sessions, windows, panes, logs.

The only module that knows tmux exists. Every call shells out to the real
`tmux` binary through `asyncio.create_subprocess_exec`, never `subprocess.run`,
so nothing blocks the event loop.

Three mechanisms carry the whole design and are verified by
`tests/test_tmux_runner.py` against a real tmux rather than assumed:

* `remain-on-exit on` keeps a pane after its command exits, so tmux retains
  the exit status and the final output stays on screen.
* `#{pane_dead}` + `#{pane_dead_status}` is the completion signal.
* `pipe-pane` streams the pane into an append-only log that outlives the
  pane and tmux's scrollback.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from buddy.models import TaskRun
from buddy.processes import release

#: tmux ≥ 3.0 is required for `pane_dead_status`.
MIN_TMUX_VERSION = (3, 0)

#: `[buddy] max_log_mb`'s default, in bytes.
DEFAULT_MAX_LOG_BYTES = 64 * 1024 * 1024

#: The `pipe-pane` target, run by path so it imports nothing of Buddy's.
LOGPIPE = Path(__file__).with_name("logpipe.py")

DEFAULT_HISTORY_LIMIT = 50000
KILL_GRACE_SECONDS = 5.0


class TmuxError(Exception):
    """A tmux command failed in a way the caller did not anticipate."""


@dataclass(frozen=True)
class PaneStatus:
    """What `status()` reports.

    `exists=False` is the case restart reconciliation cares about: the window or the whole
    session is gone (machine rebooted, tmux killed), so the log file's tail is
    the only remaining evidence.
    """

    exists: bool
    dead: bool = False
    exit_code: int | None = None
    pid: int | None = None
    piped: bool = False

    @property
    def alive(self) -> bool:
        return self.exists and not self.dead


class TmuxRunner:
    SESSION = "buddy"

    def __init__(
        self,
        session: str = SESSION,
        *,
        tmux_bin: str = "tmux",
        socket_name: str | None = None,
        max_log_bytes: int = DEFAULT_MAX_LOG_BYTES,
    ) -> None:
        self.session = session
        #: Each attempt's log is rotated past this; 0 turns the cap off.
        self.max_log_bytes = max_log_bytes
        self._bin = tmux_bin
        # Tests run against their own tmux server so they can never disturb
        # the user's real `buddy` session.
        self._socket = socket_name

    # -- process plumbing --------------------------------------------------

    def _argv(self, *args: str) -> list[str]:
        prefix = [self._bin]
        if self._socket:
            prefix += ["-L", self._socket]
        return prefix + list(args)

    async def _call(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *self._argv(*args),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await proc.communicate()
        finally:
            release(proc)
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")

    async def _tmux(self, *args: str) -> str:
        code, out, err = await self._call(*args)
        if code != 0:
            raise TmuxError(f"tmux {' '.join(args)} failed ({code}): {err.strip() or out.strip()}")
        return out

    def target(self, agent: str) -> str:
        """`buddy:=scout`. The `=` forces an exact window-name match so tmux
        never resolves an agent to a prefix of another window's name: without
        it, `scout` would match `scout-2`."""
        return f"{self.session}:={agent}"

    @property
    def _session_exact(self) -> str:
        """`=buddy`, for the commands that accept the exact-match prefix on a
        session target. Used where a wrong answer would matter: a session
        merely *starting* with `buddy` must not satisfy an existence check."""
        return f"={self.session}"

    @property
    def _session_target(self) -> str:
        """Plain `buddy`. `set-option` and `kill-session` reject the `=`
        prefix outright (verified against tmux 3.7), and by the time either
        runs a session of exactly this name exists, which tmux matches before
        it ever considers a prefix."""
        return self.session

    # -- session and windows ---------------------------------------

    async def version(self) -> tuple[int, ...]:
        out = await self._tmux("-V")  # "tmux 3.7c"
        raw = out.split()[-1]
        digits = ""
        for char in raw:
            if char.isdigit() or char == ".":
                digits += char
            else:
                break
        return tuple(int(part) for part in digits.strip(".").split(".") if part)

    async def check_version(self) -> tuple[int, ...]:
        """`buddy doctor` calls this. tmux < 3.0 cannot report pane exit
        status, which is the completion signal, so it is fatal."""
        found = await self.version()
        if found < MIN_TMUX_VERSION:
            raise TmuxError(
                f"tmux {'.'.join(map(str, found))} is too old; "
                f"Buddy needs {'.'.join(map(str, MIN_TMUX_VERSION))} for pane_dead_status"
            )
        return found

    async def session_exists(self) -> bool:
        code, _, _ = await self._call("has-session", "-t", self._session_exact)
        return code == 0

    async def window_exists(self, agent: str) -> bool:
        return agent in await self.windows()

    async def windows(self) -> list[str]:
        if not await self.session_exists():
            return []
        out = await self._tmux("list-windows", "-t", self._session_exact, "-F", "#{window_name}")
        return [line for line in out.splitlines() if line]

    async def open_window(self, agent: str) -> None:
        """A window for one agent, created if it is not there.

        One window per running agent, made when it starts and removed when
        it ends, so there is no fixed set to create up front and no limit
        but the one in config. The session itself is created with the first
        window and ends with the last, as tmux sessions do.

        `remain-on-exit` is set on every call, not only on creation: it is
        what makes a pane survive its command's exit and keep the status,
        and a window someone made by hand would not have it.
        """
        if not await self.window_exists(agent):
            if await self.session_exists():
                await self._tmux("new-window", "-d", "-t", self._session_target, "-n", agent)
            else:
                code, out, err = await self._call(
                    "new-session", "-d", "-s", self.session, "-n", agent
                )
                if code != 0 and "duplicate session" in (err + out):
                    # Created by someone else between the check and now.
                    await self._tmux("new-window", "-d", "-t", self._session_target, "-n", agent)
                elif code != 0:
                    raise TmuxError(f"tmux new-session failed ({code}): {err.strip() or out}")
                else:
                    await self._tmux(
                        "set-option",
                        "-t",
                        self._session_target,
                        "history-limit",
                        str(DEFAULT_HISTORY_LIMIT),
                    )
        await self._tmux("set-option", "-t", self.target(agent), "-w", "remain-on-exit", "on")

    async def close_window(self, agent: str) -> None:
        """Remove an agent's window once its attempt is over.

        The pipe goes first, so nothing the pane prints on its way out is
        appended to a finished attempt's log.
        """
        if not await self.window_exists(agent):
            return
        await self.close_pipe(agent)
        await self._call("kill-window", "-t", self.target(agent))

    async def kill_session(self) -> None:
        """`buddy uninstall` and test teardown."""
        await self._call("kill-session", "-t", self._session_target)

    # -- logging ----------------------------------------------------

    async def _pipe_pane(self, agent: str, command: str | None = None) -> bool:
        """Run `pipe-pane`, tolerating a dead pane.

        A pane that has exited holds no pipe and cannot be given one; tmux
        answers "target pane has exited". That is the ordinary state of a
        window being reused for a retry, not an error, so this returns False
        and lets `spawn` attach the pipe after the respawn instead.
        """
        args = ["pipe-pane", "-t", self.target(agent)]
        if command is not None:
            args.append(command)
        code, out, err = await self._call(*args)
        if code == 0:
            return True
        if "has exited" in (err + out):
            return False
        raise TmuxError(f"tmux {' '.join(args)} failed ({code}): {err.strip() or out.strip()}")

    async def close_pipe(self, agent: str) -> bool:
        """`pipe-pane` with no command closes the pipe.

        A window that is gone has no pipe to close - and when it was the
        session's last, tmux ended the session and the server with it, so
        there is nothing to even ask. Finalizing an agent from its log after
        its window vanished went exactly that way.
        """
        if not await self.window_exists(agent):
            return False
        return await self._pipe_pane(agent)

    async def open_pipe(self, agent: str, log_path: Path) -> bool:
        """Append-only, unbounded by scrollback, ANSI preserved - and
        capped, by `logpipe.py`, which rotates without losing a byte."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = " ".join(
            shlex.quote(part)
            for part in (sys.executable, "-I", str(LOGPIPE), str(log_path), str(self.max_log_bytes))
        )
        return await self._pipe_pane(agent, command)

    async def is_piped(self, agent: str) -> bool:
        return (await self.status(agent)).piped

    # -- the task lifecycle ----------------------------------------

    async def spawn(self, agent: str, run: TaskRun, script: Path) -> None:
        """Start a task in its agent's window, creating the window if needed.

        `script` is the generated wrapper. It is passed in rather than
        derived, because where it lives is layout knowledge that belongs to
        `config.Paths`, not to the module that knows tmux.
        """
        await self.open_window(agent)
        # Both pipe calls are no-ops when the window's previous pane is dead,
        # which is the ordinary case for a window left by an earlier attempt.
        await self.close_pipe(agent)
        await self.open_pipe(agent, run.log_path)
        await self._tmux(
            "respawn-window",
            "-k",
            "-t",
            self.target(agent),
            f"bash {shlex.quote(str(script))}",
        )
        # Re-opened, because of what `#{pane_pipe}` can actually tell
        # us: it reports that *a* pipe exists, never where it points. A window
        # being reused still carries the previous attempt's pipe, so checking
        # the flag and stopping there would append attempt N's output to
        # attempt N-1's log. `pipe-pane` with a command closes any existing
        # pipe and opens the new one, so re-opening unconditionally is both
        # the fix for a dropped pipe and the fix for a stale one. run.sh's
        # 250ms sleep is what makes this window harmless.
        await self.open_pipe(agent, run.log_path)
        if not await self.is_piped(agent):
            raise TmuxError(f"could not attach the log pipe for {agent} -> {run.log_path}")

    async def panes(self) -> dict[str, PaneStatus]:
        """Every agent's pane state in one tmux call, keyed by window name.

        `display -p -t <session>:<window>` cannot be used for this. When the
        window does not resolve, tmux does not fail: it silently answers for
        the session's *current* window. That would report a live pane for an
        agent whose window is gone, which is precisely the case restart
        reconciliation has to detect. Enumerating panes and matching
        the name exactly is the only reliable answer.

        One call for every agent is also what the manager's 1s tick wants.
        """
        if not await self.session_exists():
            return {}
        code, out, _ = await self._call(
            "list-panes",
            "-s",
            "-t",
            self._session_target,
            "-F",
            "#{window_name}\t#{pane_dead}\t#{pane_dead_status}\t"
            "#{pane_pid}\t#{pane_pipe}\t#{pane_active}",
        )
        if code != 0:
            return {}
        found: dict[str, PaneStatus] = {}
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            name, dead_raw, status_raw, pid_raw, pipe_raw, active_raw = parts[:6]
            # Buddy gives a window exactly one pane, but a curious user can
            # split one by hand; the active pane is the task's.
            if name in found and active_raw != "1":
                continue
            dead = dead_raw == "1"
            found[name] = PaneStatus(
                exists=True,
                dead=dead,
                # A live pane has no exit status; a dead one always does.
                exit_code=int(status_raw) if dead and status_raw.strip("-").isdigit() else None,
                pid=int(pid_raw) if pid_raw.isdigit() else None,
                piped=pipe_raw == "1",
            )
        return found

    async def status(self, agent: str) -> PaneStatus:
        """The completion signal. Never scrapes text."""
        return (await self.panes()).get(agent, PaneStatus(exists=False))

    async def capture(self, agent: str, lines: int = 60) -> str:
        """What is on screen right now. Previews only: never completion
        detection, never the log."""
        code, out, _ = await self._call(
            "capture-pane", "-e", "-p", "-t", self.target(agent), "-S", f"-{lines}"
        )
        return out if code == 0 else ""

    async def kill(self, agent: str) -> None:
        """Terminate the task and remove its window.

        The whole process tree goes, so a dev server or watcher the harness
        started does not outlive the task.
        """
        status = await self.status(agent)
        if status.pid and not status.dead:
            await kill_process_tree(status.pid, grace=KILL_GRACE_SECONDS)
        await self.close_window(agent)


# --------------------------------------------------------------------------
# Process-tree termination
# --------------------------------------------------------------------------


#: Linux's process table as files. Present there, absent on macOS. Decided
#: once at import: the answer cannot change while Buddy runs, and asking the
#: filesystem on the loop for every kill would be a blocking call for nothing.
PROC = Path("/proc")
HAS_PROC = PROC.is_dir()


def children_from_proc(pid: int, root: Path = PROC) -> list[int]:
    """Direct children of `pid`, read from `/proc`.

    No external command at all, which is the point: a minimal Linux image -
    which is exactly where an orchestrator ends up - often has no `pgrep`,
    and `kill` used to die on that with a FileNotFoundError. Found by running
    the suite on Debian.
    """
    children: list[int] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue  # it exited while we were looking
        # `comm` can hold spaces and brackets, so the fields after the last
        # `)` are the reliable ones: state, then ppid.
        _, _, rest = stat.rpartition(") ")
        fields = rest.split()
        if len(fields) >= 2 and fields[1] == str(pid):
            children.append(int(entry.name))
    return sorted(children)


async def _children_from_commands(pid: int) -> list[int]:
    """Direct children from `pgrep`, or `ps` where there is no `pgrep`."""
    if shutil.which("pgrep"):
        code, out, _ = await _read("pgrep", "-P", str(pid))
        if code in (0, 1):  # 1 is "no children", not a failure
            return [int(line) for line in out.split() if line.isdigit()]
    if shutil.which("ps"):
        code, out, _ = await _read("ps", "-eo", "pid=,ppid=")
        if code == 0:
            children = []
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == str(pid) and parts[0].isdigit():
                    children.append(int(parts[0]))
            return sorted(children)
    return []


async def _read(*args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        out, err = await proc.communicate()
    finally:
        release(proc)
    return proc.returncode or 0, out.decode(errors="replace"), (err or b"").decode(errors="replace")


async def _children(pid: int) -> list[int]:
    if HAS_PROC:
        return await asyncio.to_thread(children_from_proc, pid)
    return await _children_from_commands(pid)


async def process_tree(pid: int) -> list[int]:
    """`pid` and every descendant, deepest last."""
    found = [pid]
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        for child in await _children(parent):
            if child not in found:
                found.append(child)
                frontier.append(child)
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, sig)


async def kill_process_tree(pid: int, *, grace: float = KILL_GRACE_SECONDS) -> None:
    """TERM the tree, wait out the grace period, then KILL the survivors.

    The pane's process group is signalled too, which catches a child that
    reparented away from the tree we walked. Buddy's own process group is
    never signalled, whatever tmux reports.
    """
    pids = await process_tree(pid)
    group = _process_group(pid)

    for target in pids:
        _signal(target, signal.SIGTERM)
    if group:
        _signal(-group, signal.SIGTERM)

    deadline = asyncio.get_running_loop().time() + grace
    while asyncio.get_running_loop().time() < deadline:
        if not any(_alive(target) for target in pids):
            return
        await asyncio.sleep(0.1)

    for target in pids:
        if _alive(target):
            _signal(target, signal.SIGKILL)
    if group:
        _signal(-group, signal.SIGKILL)


def _process_group(pid: int) -> int | None:
    """The pane's pgid, unless it is Buddy's own - signalling that would kill
    Buddy along with the task."""
    try:
        group = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None
    return None if group == os.getpgid(0) else group
