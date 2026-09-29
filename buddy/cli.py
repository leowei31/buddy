"""Entrypoint: subcommands, the asyncio loop, the voice thread, the dashboard.

`buddy` alone is the interactive session: the brain, the read-only
dashboard and the voice layer all run in the one asyncio loop,
and every other subcommand is for scripting or for inspecting state without
it.

`watch` and `attach` hand the terminal to tmux with `os.execvp`, so the user
is talking to tmux directly rather than through a pipe Buddy owns.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from buddy import __version__, conflicts
from buddy.brain import Brain, Tier
from buddy.config import Config, ConfigError, KeyringSecretResolver, keychain_delete
from buddy.harnesses import ADAPTERS, adapter_class
from buddy.harnesses.base import HarnessError, PreflightReport
from buddy.logs import read_tail
from buddy.manager import AgentManager
from buddy.models import (
    Agent,
    AgentHealthChanged,
    AgentNameError,
    BranchDeleted,
    PreemptionProposal,
    RunOutcome,
    TaskBlocked,
    TaskFinished,
    TaskRequeued,
    TaskSpec,
    TaskStarted,
    TaskState,
    utcnow,
)
from buddy.processes import release
from buddy.providers import KNOWN as KNOWN_PROVIDERS
from buddy.providers import build as build_provider
from buddy.providers.base import ProviderError, ProviderNotInstalled
from buddy.state import Store
from buddy.tmux_runner import MIN_TMUX_VERSION, TmuxError, TmuxRunner
from buddy.web.events import Hub
from buddy.workspace import Workspace, WorkspaceError, branch_name

if TYPE_CHECKING:  # the web stack is imported lazily, in `_start_dashboard`
    from buddy.web.server import Dashboard

app = typer.Typer(
    name="buddy",
    help="A voice-driven, harness-agnostic multi-agent orchestrator.",
    no_args_is_help=False,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"buddy {__version__}")
        raise typer.Exit


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    no_voice: bool = typer.Option(
        False, "--no-voice", help="Type instead of speaking; no microphone."
    ),
    no_web: bool = typer.Option(False, "--no-web", help="Do not start the dashboard."),
) -> None:
    """`buddy` with no subcommand starts an interactive session."""
    if ctx.invoked_subcommand is None:
        asyncio.run(_session(voice=not no_voice, web=not no_web))


async def _session(voice: bool = False, web: bool = True) -> None:
    """Reconcile, then converse while the manager ticks underneath.

    Typed or spoken, both arrive on the same two queues. The dashboard runs
    in this same loop, and is
    started before the brain so a provider that will not load still leaves
    something to watch.
    """
    runtime = Runtime()
    console.print("[bold]Buddy[/] - reconciling...")
    _report_events(runtime, await runtime.manager.reconcile())
    await _apply_accepted(runtime)

    dashboard = await _start_dashboard(runtime) if web else None
    control = await _start_control(runtime)
    try:
        await _converse(runtime, voice=voice, control=control)
    finally:
        if control is not None:
            await control.stop()
        if dashboard is not None:
            await dashboard.stop()


async def _start_dashboard(runtime: Runtime) -> Dashboard | None:
    """The read-only dashboard, in this same loop.

    Imported here rather than at module scope: FastAPI and uvicorn cost about
    a tenth of a second to import, and `buddy status` should not pay it.
    """
    from buddy.web.server import start_dashboard

    dashboard = await start_dashboard(
        config=runtime.config,
        store=runtime.store,
        manager=runtime.manager,
        workspace=runtime.workspace,
        hub=runtime.hub,
        on_error=lambda message: console.print(f"[yellow]{escape(message)}[/]"),
    )
    if dashboard is not None:
        console.print(f"Dashboard on [bold]{dashboard.url}[/] (read-only).")
    return dashboard


async def _start_control(runtime: Runtime):
    """The unix socket other surfaces talk to.

    Never fatal: a session that cannot open it is a session you can still
    type to, and the overlay simply says so.
    """
    from buddy.control import ControlServer

    def busy() -> dict:
        return {"agents": len(runtime.manager.agents())}

    server = ControlServer(runtime.config.home, on_status=busy)
    try:
        await server.start()
    except OSError as exc:
        console.print(f"[yellow]No control channel ({escape(str(exc))}).[/]", soft_wrap=True)
        console.print("[dim]The overlay can still watch; it just cannot talk to this session.[/]")
        return None
    return server


async def _converse(runtime: Runtime, voice: bool = False, control=None) -> None:
    try:
        brain = runtime.brain()
    except (ProviderNotInstalled, ProviderError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]", soft_wrap=True)
        console.print("Falling back to watching only. `buddy doctor` for details.")
        await _watch_loop(runtime)
        return

    console.print(
        f"Talking to [bold]{brain.provider.name}[/] ({brain.provider.model}). "
        "Ctrl-C or 'exit' to stop; tasks keep running in tmux. /help for commands."
    )
    if brain.brainstorm.active:
        console.print(f"[magenta]{_brainstorm_banner(brain)}[/]")
    speech = await _start_voice(runtime) if voice else None
    ticker = asyncio.create_task(_ticker(runtime))
    try:
        async for utterance, asked in _utterances(speech, control, label=lambda: _you(brain)):
            if asked is None and utterance.lower() in ("exit", "quit"):
                break
            if asked is not None:
                # Shown in the terminal too, so the session is never a
                # surface where things happen that you cannot see.
                console.print(f"\n[dim]overlay>[/] {escape(utterance)}")
            if utterance.startswith("/"):
                said = await _command(brain, utterance)
                console.print(escape(said))
                if asked is not None:
                    asked.finish(said)
                _report_events(runtime, brain.pending_events)
                brain.pending_events.clear()
                continue
            console.print("[dim]buddy>[/] ", end="")
            reply = await _exchange(runtime, brain, speech, utterance, asked)
            if reply is None:
                continue
            if not reply:
                console.print("[dim](no reply)[/]")
            _report_events(runtime, brain.pending_events)
            brain.pending_events.clear()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        ticker.cancel()
        if speech is not None:
            await speech.stop()
    console.print(
        "\nStopped. Tasks keep running in tmux; `buddy status` to see where they are, "
        "or `buddy shutdown` to stop everything and clear the worktrees."
    )


COMMANDS = """\
/brainstorm        think it through first - nothing starts until /go
/brainstorm off    stop brainstorming; the drafts are kept
/drafts            what has been drafted so far
/drop <draft>      remove a draft, e.g. /drop d2
/go                start every draft, dependencies first, and stop brainstorming
/help              this list"""


def _you(brain) -> str:
    return "you (brainstorming)" if brain.brainstorm.active else "you"


def _brainstorm_banner(brain) -> str:
    count = len(brain.brainstorm.drafts)
    drafts = f"{count} draft{'s' if count != 1 else ''}" if count else "no drafts yet"
    return f"Brainstorming ({drafts}). Nothing starts until /go."


async def _command(brain, line: str) -> str:
    """A typed command, handled here and never sent to the model.

    `/go` in particular: typing it is the user's explicit yes, so it launches
    the drafts directly - exactly as drafted - rather than asking the model
    to, and the model hears what happened on its next turn.
    """
    name, _, rest = line.strip().partition(" ")
    rest = rest.strip()
    storm = brain.brainstorm
    if name == "/help":
        return COMMANDS
    if name == "/brainstorm" and rest == "off":
        if not storm.active:
            return "Not brainstorming."
        brain.stop_brainstorm()
        kept = f" {len(storm.drafts)} draft(s) kept for next time." if storm.drafts else ""
        return "Stopped brainstorming." + kept
    if name == "/brainstorm":
        brain.start_brainstorm()
        return _brainstorm_banner(brain)
    if name == "/drafts":
        return storm.describe()
    if name == "/drop":
        dropped = storm.drop(rest)
        return f"Dropped {dropped.line()}" if dropped else f"There is no draft {rest!r}."
    if name == "/go":
        if not storm.drafts:
            return "Nothing drafted to start." + (" Still brainstorming." if storm.active else "")
        launched, kept = await brain.launch_drafts()
        lines = [f"started  {line}" for line in launched]
        lines += [f"kept     {line}" for line in kept]
        if not kept:
            lines.append("Brainstorming is off.")
        return "\n".join(lines)
    return f"Unknown command {name}. /help lists them."


async def _exchange(runtime: Runtime, brain, speech, utterance: str, asked=None) -> str | None:
    """One turn, spoken aloud if there is a voice, cancellable by barge-in.

    Returns None when the turn did not complete - a provider that would not
    answer, or the user talking over the reply. Neither is a reason to
    end the session.
    """
    spoken: list[str] = []

    def on_text(chunk: str) -> None:
        console.print(chunk, end="")
        spoken.append(chunk)
        if asked is not None:
            asked.send("delta", chunk)

    try:
        if speech is None:
            reply = await brain.send(utterance, on_text=on_text)
        else:
            reply = await speech.answer(brain, utterance, on_text=on_text)
            console.print()
        if asked is not None:
            asked.finish(reply)
        return reply
    except (ProviderError, ProviderNotInstalled) as exc:
        # A provider that will not answer ends the exchange, never the
        # session: there are tasks running in tmux, a queue to drain,
        # and a dashboard attached to all of it.
        console.print()
        console.print(f"[red]{escape(str(exc))}[/]", soft_wrap=True)
        console.print(
            "[dim]The session is still up. `buddy doctor` checks the "
            "provider; tasks keep running either way.[/]"
        )
        if asked is not None:
            asked.fail(str(exc))
        return None
    finally:
        if speech is None:
            console.print()


async def _utterances(
    speech, control=None, label: Callable[[], str] = lambda: "you"
) -> AsyncIterator[tuple[str, Any]]:
    """What the user said, however they said it.

    Typed and spoken do not interleave - push-to-talk and `input()` both own
    the terminal, so a voice session listens and a typed one reads. The
    control channel is different: it arrives from another process, so it is
    raced against whichever of those is in play.

    Yields `(text, source)`, where source is the `Utterance` to answer when it
    came from the control channel and None when it came from this terminal.
    """
    pending: asyncio.Task[str | None] | None = None
    try:
        while True:
            if pending is None:
                # Kept across iterations: `input()` in a thread cannot be
                # cancelled, so a control message arriving mid-prompt must
                # leave the prompt standing rather than abandon it.
                pending = asyncio.create_task(_next_local(speech, label()))

            waiting = {pending}
            from_control: asyncio.Task | None = None
            if control is not None:
                from_control = asyncio.create_task(control.heard.get())
                waiting.add(from_control)

            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)

            if from_control is not None and from_control in done:
                utterance = from_control.result()
                yield utterance.text, utterance
                continue
            if from_control is not None:
                # The terminal answered first; the queue wait is dropped
                # without losing anything, because nothing was taken from it.
                from_control.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await from_control

            typed = pending.result()
            pending = None
            if typed is None:
                return
            if typed.strip():
                yield typed.strip(), None
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pending


async def _next_local(speech, label: str = "you") -> str | None:
    """One turn from the terminal, spoken or typed. None means it is over."""
    if speech is not None:
        return await speech.next_turn()
    try:
        return await asyncio.to_thread(input, f"\n{label}> ")
    except EOFError:
        return None


async def _start_voice(runtime: Runtime):
    """The voice layer, or a sentence saying why there is not one.

    Imported here so the speech stack costs nothing on `buddy --no-voice` or
    on any of the scripting commands.
    """
    try:
        from buddy.voice.session import Speech
    except ImportError as exc:  # pragma: no cover - the extra is optional
        console.print(f"[yellow]Voice is unavailable: {escape(str(exc))}[/]", soft_wrap=True)
        return None

    speech = Speech(runtime.config, runtime.resolver)
    speech.on_problem = lambda message: console.print(f"[yellow]{escape(message)}[/]")
    try:
        ready, why = await speech.start()
    except Exception as exc:  # noqa: BLE001 - voice never takes the session
        # Voice is decoupled entirely, and this is where that has to be true
        # in practice: a missing device, a driver that throws, a half-written
        # [voice] block. None of it is a reason to stop orchestrating agents.
        console.print(
            f"[yellow]Voice could not start ({escape(str(exc))}). Type instead.[/]", soft_wrap=True
        )
        await speech.stop()
        return None
    if not ready:
        console.print(f"[yellow]Voice is off: {escape(why)}. Type instead.[/]")
        await speech.stop()
        return None
    console.print(f"[dim]{escape(why)}[/]")
    return speech


async def _ticker(runtime: Runtime) -> None:
    """The manager's 1s tick, running under the conversation.

    A tick that raises must not stop the heartbeat. It used to: anything but
    `CancelledError` propagated out of a task nobody awaited, so the loop
    died in silence while the session kept chatting, the dashboard kept
    serving state that would never change again, and no run was ever
    finalized. A single bad tick is worth one line on screen, not the end of
    the orchestrator.
    """
    with contextlib.suppress(asyncio.CancelledError):
        await _tick_forever(runtime, lead="\n")


async def _tick_forever(runtime: Runtime, *, lead: str = "") -> None:
    """Tick once a second until cancelled, surviving any tick that raises."""
    failures = 0
    while True:
        try:
            events = await runtime.manager.tick()
            failures = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, then retried
            failures += 1
            if failures <= 3 or failures % 60 == 0:
                console.print(
                    f"{lead}[yellow]tick failed ({type(exc).__name__}: {escape(str(exc))})[/]"
                    + ("" if failures > 1 else " - still running; it will try again")
                )
            await asyncio.sleep(1)
            continue
        if events:
            if lead:
                console.print()
            _report_events(runtime, events)
            await _apply_accepted(runtime)
        await asyncio.sleep(1)


async def _watch_loop(runtime: Runtime) -> None:
    console.print("Watching every agent. Ctrl-C to stop; tasks keep running in tmux.")
    try:
        await _tick_forever(runtime)
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\nStopped watching. `buddy status` to see where things stand.")


# --------------------------------------------------------------------------
# Shared wiring
# --------------------------------------------------------------------------


class Runtime:
    """Everything a command needs, assembled once."""

    def __init__(self, home: Path | None = None) -> None:
        try:
            self.config = Config.load(home=home)
        except ConfigError as exc:
            console.print(f"[red]config error:[/] {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc
        self.config.paths.ensure()
        self.store = Store(self.config.paths.db)
        #: Feeds the dashboard's websockets. Publishing with no subscribers
        #: costs nothing, so this is wired up whether or not the web server
        #: is running.
        self.hub = Hub()
        self.store.on_turn(self.hub.publish_turn)
        self.runner = TmuxRunner(max_log_bytes=self.config.buddy.max_log_mb * 1024 * 1024)
        self.workspace = Workspace(self.config)
        # Env first, then the OS keychain, which is where `buddy setup`
        # files API keys.
        self.resolver = KeyringSecretResolver()
        self._preflights: dict[str, tuple[float, PreflightReport]] = {}
        self.manager = AgentManager(
            self.config,
            self.store,
            self.runner,
            self.workspace,
            adapter_for=self.build_adapter,
            resolver=self.resolver,
        )

    def brain(self) -> Brain:
        """The conversation, on whichever provider config names."""
        provider = build_provider(self.config.brain, resolver=self.resolver)
        return Brain(
            self.config,
            self.store,
            self.manager,
            self.workspace,
            provider,
            confirm=_confirm,
            preflight=self.preflight,
        )

    #: How long a preflight answer is reused. A usable harness rarely stops
    #: being usable; a failing one is re-checked soon, so logging in to a CLI
    #: takes effect without restarting Buddy.
    PREFLIGHT_OK_SECONDS = 300.0
    PREFLIGHT_FAILED_SECONDS = 15.0

    async def preflight(self, name: str) -> PreflightReport:
        """`adapter.preflight()`, cached, for the brain's spawns.

        Each one runs the CLI three or four times, which is fine per spawn
        and wasteful when the brain queues five tasks in one breath.
        """
        now = time.monotonic()
        cached = self._preflights.get(name)
        if cached is not None:
            at, report = cached
            ttl = self.PREFLIGHT_OK_SECONDS if report.usable else self.PREFLIGHT_FAILED_SECONDS
            if now - at < ttl:
                return report
        report = await self.build_adapter(name).preflight()
        self._preflights[name] = (now, report)
        return report

    def build_adapter(self, name: str):
        """The adapter for a configured harness, or HarnessError.

        What the manager calls, inside the session loop - so it raises
        something the manager can turn into one failed task, and never exits
        the process.
        """
        factory = adapter_class(name)
        try:
            return factory(self.config.harness(name), resolver=self.resolver)
        except ConfigError as exc:
            raise HarnessError(f"{exc}; add a [harness.{name}] block or run `buddy setup`") from exc

    def adapter(self, name: str):
        """`build_adapter` for a CLI command, where a bad name ends the command."""
        try:
            return self.build_adapter(name)
        except HarnessError as exc:
            console.print(f"[red]unusable harness[/] {name!r}: {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from None

    def task_or_exit(self, task_id: str) -> TaskSpec:
        task = self.store.get_task(task_id)
        if task is None:
            console.print(f"[red]no such task:[/] {task_id}")
            raise typer.Exit(2)
        return task

    def named_task_or_exit(self, target: str) -> TaskSpec:
        """A task by id, or by its agent's name - the latest task under it."""
        task = self.store.find_task(target)
        if task is None:
            console.print(f"[red]no such task or agent:[/] {escape(target)}")
            raise typer.Exit(2)
        return task


def _confirm(prompt: str, tier: Tier) -> bool:
    """The tiered confirmation, at the terminal.

    A read-back is a yes/no on a one-line summary; the always-ask tier is the
    same question with no way to turn it off.
    """
    marker = "[yellow]?[/]" if tier is Tier.READ_BACK else "[red]![/]"
    console.print(f"\n{marker} {escape(prompt)}")
    return typer.confirm("  ok?", default=tier is Tier.READ_BACK)


def fail(message: str) -> NoReturn:
    # soft_wrap: errors carry paths and commands, and a hard wrap breaks them
    # mid-token so what you copy is not what Buddy said.
    console.print(f"[red]{escape(message)}[/]", soft_wrap=True)
    raise typer.Exit(1)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


@app.command(
    help="Check tmux, git, projects, the brain, voice and each harness, and how to fix failures."
)
def doctor() -> None:
    """Preflight: tmux, git, and each configured harness."""
    asyncio.run(_doctor())


async def _doctor() -> None:
    if await run_doctor():
        raise typer.Exit(1)


async def run_doctor(home: Path | None = None) -> int:
    """Print the table and return how many things are wrong.

    Setup's step 12 calls this rather than re-implementing it: if setup's last
    word and `buddy doctor` could disagree, the one people can run again
    tomorrow is the one that should be right.
    """
    runtime = Runtime(home)
    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail", overflow="fold")
    problems = 0

    runner = runtime.runner
    try:
        version = await runner.check_version()
        table.add_row("tmux", "[green]ok[/]", ".".join(map(str, version)))
    except (TmuxError, FileNotFoundError) as exc:
        problems += 1
        needed = ".".join(map(str, MIN_TMUX_VERSION))
        table.add_row("tmux", "[red]fail[/]", f"{exc} (need >= {needed})")

    git = shutil.which("git")
    table.add_row("git", "[green]ok[/]" if git else "[red]fail[/]", git or "not on PATH")
    problems += 0 if git else 1

    table.add_row("state.db", "[green]ok[/]", str(runtime.config.paths.db))
    if runtime.config.projects:
        table.add_row("projects", "[green]ok[/]", ", ".join(sorted(runtime.config.projects)))
        for name in sorted(runtime.config.projects):
            # A Buddy merge killed mid-conflict leaves your own checkout that
            # way, and nothing else would ever mention it.
            unfinished = await runtime.workspace.merge_in_progress(name)
            if unfinished:
                problems += 1
                table.add_row(f"project {name}", "[red]merging[/]", escape(unfinished))
    else:
        # Counted, not noted. Buddy with no project cannot spawn anything at
        # all, so "everything checks out" alongside "nothing can be spawned"
        # is doctor contradicting itself - and it is exactly what sends
        # someone into a conversation that dead-ends.
        #
        # Doctor's job is a named list of what is not green and how to fix
        # each. Naming a problem without the remedy is half a job.
        problems += 1
        table.add_row(
            "projects",
            "[red]none[/]",
            "none configured, so nothing can be spawned.\nFix: buddy setup --force config",
        )

    configured = runtime.config.brain.provider
    if configured not in KNOWN_PROVIDERS:
        problems += 1
        table.add_row("brain provider", "[red]fail[/]", f"unknown provider {configured!r}")
    else:
        try:
            provider = build_provider(runtime.config.brain, resolver=runtime.resolver)
        except (ProviderNotInstalled, ProviderError) as exc:
            problems += 1
            # escape(): these messages name extras like [gemini], which rich
            # would otherwise read as a style tag and swallow.
            table.add_row(f"brain {configured}", "[red]fail[/]", escape(str(exc)))
        else:
            result = await provider.probe()
            if result.reachable:
                detail = result.summary() + (f"\n{result.detail}" if result.detail else "")
                table.add_row(f"brain {configured}", "[green]ok[/]", escape(detail))
            else:
                problems += 1
                table.add_row(f"brain {configured}", "[red]fail[/]", escape(result.summary()))

    await _doctor_voice(runtime, table)

    for name in sorted(ADAPTERS):
        factory = adapter_class(name)
        block = runtime.config.harnesses.get(name)
        if block is None or not block.command.strip():
            # Not configured is a choice, not a problem - but a CLI that is
            # installed and unused is worth one line saying how to use it.
            installed = shutil.which(factory.binary) is not None
            why = (
                f"{factory.binary} is installed; `buddy setup --force harnesses` adds it"
                if installed
                else f"not installed ({factory.install_hint})"
            )
            if block is not None:
                why = f"[harness.{name}] has no command, so it is parked"
            # escape(): rich reads a bare [harness] as a style tag and
            # silently drops it.
            table.add_row(f"harness {name}", "[yellow]skip[/]", escape(why))
            continue
        report: PreflightReport = await runtime.adapter(name).preflight()
        if report.usable:
            table.add_row(f"harness {name}", "[green]ok[/]", escape(report.summary()))
        else:
            problems += 1
            detail = report.summary()
            for requirement in report.failures:
                detail += f"\n  - {requirement.description}: {requirement.detail}"
            table.add_row(f"harness {name}", "[red]fail[/]", escape(detail))
        if report.authenticated is None and report.auth_detail:
            table.add_row("", "", escape(f"note: {report.auth_detail}"))
        for note in report.notes:
            table.add_row("", "", escape(f"note: {note}"))

    console.print(table)
    if problems:
        console.print(f"[red]{problems} problem(s).[/]")
    else:
        console.print("[green]Everything checks out.[/]")
    return problems


async def _doctor_voice(runtime: Runtime, table: Table) -> None:
    """The Fish backend's health, model and measured TTFB.

    Never counted as a problem. Buddy schedules agents perfectly well without
    a voice, and a red row for a missing speaker would train you to ignore
    the ones that matter - so this reports, in as much detail as it can get,
    and leaves the exit code alone.
    """
    try:
        from buddy.voice import stt
        from buddy.voice.tts import choose_client
    except ImportError as exc:  # pragma: no cover - the extra is optional
        table.add_row("voice", "[yellow]off[/]", escape(str(exc)))
        return

    model = runtime.config.voice.stt_model.split(":", 1)[-1]
    try:
        cached = stt.is_cached(model, "none")
        table.add_row(
            "voice stt",
            "[green]ok[/]" if cached else "[yellow]off[/]",
            f"faster-whisper:{model}" + ("" if cached else " is not cached"),
        )
    except stt.SttNotInstalled as exc:
        table.add_row("voice stt", "[yellow]off[/]", escape(str(exc)))

    # The client a session would actually speak through: the configured
    # backend, or `tts_fallback` when that one is not answering.
    chooser, speaking = await choose_client(runtime.config.voice, runtime.resolver)
    client = chooser.active
    if chooser.switched_because:
        table.add_row("voice tts", "[yellow]fallback[/]", escape(speaking))
    try:
        health = await client.measure()
    except Exception as exc:  # noqa: BLE001 - a probe never fails a doctor run
        table.add_row("voice tts", "[yellow]off[/]", escape(f"{type(exc).__name__}: {exc}"))
        return
    finally:
        await chooser.aclose()

    if not health.reachable:
        table.add_row("voice tts", "[yellow]off[/]", escape(health.summary()))
        return
    # The two-second budget is for the whole spoken turn, so first byte is only part of it -
    # but a first byte that is already over is a certainty, not a risk.
    over = health.ttfb_seconds is not None and health.ttfb_seconds > 2.0
    table.add_row(
        "voice tts",
        "[yellow]slow[/]" if over else "[green]ok[/]",
        escape(health.summary() + (" - over the 2s budget for a spoken turn" if over else "")),
    )


# --------------------------------------------------------------------------
# spawn and status
# --------------------------------------------------------------------------


@app.command(help="Queue a task on a project without talking to Buddy.")
def spawn(
    project: str,
    brief: str,
    title: str = typer.Option("", "--title", help="Human label; defaults to the first line."),
    name: str = typer.Option(
        "",
        "--name",
        "-N",
        help="What to call its agent. Default: made from the title.",
        show_default=False,
    ),
    harness: str = typer.Option(
        "",
        "--harness",
        "-H",
        help="claude_code, codex, opencode or antigravity. Default: the project's.",
    ),
    priority: int = typer.Option(
        0,
        "--priority",
        "-p",
        help="1 (urgent) .. 5 (whenever). Default: the default_priority setting.",
        show_default=False,
    ),
    model: str = typer.Option(
        "", "--model", "-m", help="The harness's own model name. Default: its own."
    ),
    after: list[str] = typer.Option([], "--after", help="Task ids this one depends on."),
    merge_required: bool = typer.Option(
        False, "--merge-required", help="Dependents wait for this to be merged, not just done."
    ),
    wait: bool = typer.Option(False, "--wait", help="Block until the task finishes."),
) -> None:
    """Queue a brief under a named agent. The scheduler starts it."""
    asyncio.run(
        _spawn(
            project,
            brief,
            title,
            harness,
            priority,
            model,
            list(after),
            merge_required,
            wait,
            name=name,
        )
    )


async def _spawn(
    project: str,
    brief: str,
    title: str,
    harness: str,
    priority: int,
    model: str,
    after: list[str],
    merge_required: bool,
    wait: bool,
    *,
    name: str = "",
) -> None:
    runtime = Runtime()
    config = runtime.config
    try:
        config.project(project)
    except ConfigError as exc:
        fail(str(exc))

    try:
        harness_name = config.choose_harness(project, harness)
    except ConfigError as exc:
        fail(str(exc))
    adapter = runtime.adapter(harness_name)
    report = await adapter.preflight()
    if not report.usable:
        # Refused at spawn, before anything is created - and a CLI that is not signed
        # in is refused too, because every task given to it would fail.
        fail(f"{report.summary()} - run `buddy doctor`")

    for dependency in after:
        if runtime.store.get_task(dependency) is None:
            fail(f"--after names a task that does not exist: {dependency}")

    task = TaskSpec(
        id=runtime.store.next_task_id(),
        title=title or brief.strip().splitlines()[0][:60],
        brief=brief,
        harness=harness_name,
        model=model or None,
        project=project,
        priority=priority or config.buddy.default_priority,
        depends_on=after,
        merge_required=merge_required,
        max_runtime=config.buddy.max_runtime,
        stall_timeout=config.buddy.stall_timeout,
        agent=name,
    )

    try:
        events = await runtime.manager.submit(task)
    except (WorkspaceError, AgentNameError) as exc:
        fail(str(exc))
    except TmuxError as exc:
        # The task is already saved as queued; only starting it failed. It
        # starts on the next tick that can reach tmux - which is worth saying
        # plainly instead of as a traceback.
        fail(
            f"{task.id} ({task.agent}) is queued, but tmux would not start it: {exc}. "
            "It starts once tmux works - `buddy doctor` checks it."
        )

    _report_events(runtime, events)
    started = [
        event for event in events if isinstance(event, TaskStarted) and event.task_id == task.id
    ]
    if started:
        agent = started[0].agent
        console.print(f"  watch:  buddy watch {agent}\n  logs:   buddy logs {agent} -f")
    elif runtime.store.get_task_state(task.id) is TaskState.QUEUED:
        blocked = runtime.manager.dependency_block(task)
        ahead = runtime.manager.queue_position(task.id)
        console.print(
            f"[yellow]{task.agent} ({task.id}) queued[/] "
            + (blocked if blocked else f"behind {ahead} task(s)")
        )

    if wait:
        await _tick_until_done(runtime, task.id)


async def _catch_up(runtime: Runtime) -> None:
    """Finalize anything that ended while nothing was watching.

    Cheap - one `list-panes` and whatever finished - and safe to call from
    any read-only command, because a tick only acts on panes that are
    already dead.
    """
    try:
        events = await runtime.manager.tick()
    except Exception as exc:  # noqa: BLE001 - reporting state must not fail on it
        console.print(
            f"[yellow]could not catch up ({type(exc).__name__}: {escape(str(exc))})[/]",
            soft_wrap=True,
        )
        return
    if events:
        _report_events(runtime, events)


def _report_events(runtime: Runtime, events: list) -> None:
    """One line per event, which is what the brain narrates.

    Also the single point where events reach the dashboard: every
    path that produces them - tick, reconcile, spawn, kill, the brain's own
    tool calls - comes through here.
    """
    runtime.hub.publish_events(events)
    for event in events:
        if isinstance(event, TaskStarted):
            console.print(
                f"[green]{event.agent}[/] started on [bold]{event.task_id}[/]"
                f" (attempt {event.attempt})"
            )
        elif isinstance(event, TaskFinished):
            colour = "green" if event.outcome is RunOutcome.DONE else "red"
            who = f"{event.agent} ({event.task_id})" if event.agent else event.task_id
            console.print(
                f"[{colour}]{who} {event.outcome.value}[/]"
                + (f" (exit {event.exit_code})" if event.exit_code is not None else "")
            )
            if event.summary and event.summary != event.outcome.value:
                # Agent output, verbatim: rich would read `[...]` in it as markup
                # and silently drop it - Google's own error text lost half itself.
                console.print(f"  {escape(event.summary)}")
            # soft_wrap: a hard wrap lands inside the command, and what you
            # copy is then not what Buddy said.
            if event.branch and event.outcome is RunOutcome.DONE:
                console.print(
                    f"  branch {event.branch} - `buddy diff {event.task_id}` to review",
                    soft_wrap=True,
                )
            elif event.branch:
                console.print(
                    f"  branch {event.branch} keeps anything it got done - "
                    f"`buddy diff {event.task_id}` shows what, if anything",
                    soft_wrap=True,
                )
        elif isinstance(event, TaskRequeued):
            console.print(
                f"[yellow]{event.task_id} {event.reason.value}[/]; "
                f"requeued as attempt {event.next_attempt}, nothing lost"
            )
        elif isinstance(event, AgentHealthChanged):
            colour = {"waiting_input": "yellow", "stalled": "yellow"}.get(
                event.status.value, "green"
            )
            console.print(
                f"[{colour}]{event.agent} is {event.status.value}[/]"
                + (f": {escape(event.detail)}" if event.detail else "")
            )
        elif isinstance(event, BranchDeleted):
            console.print(
                f"[dim]deleted branch {escape(event.branch)} - {event.task_id} was discarded "
                "and its grace period is over[/]"
            )
        elif isinstance(event, TaskBlocked):
            console.print(f"[red]{event.task_id} is blocked[/]: {escape(event.reason)}")
        elif isinstance(event, PreemptionProposal):
            _offer_preemption(runtime, event)


def _offer_preemption(runtime: Runtime, proposal: PreemptionProposal) -> None:
    """Always asked, never assumed."""
    victim = runtime.store.get_task(proposal.victim_task_id)
    incoming = runtime.store.get_task(proposal.incoming_task_id)
    console.print(
        f"[yellow]{proposal.victim_agent}[/] is on {proposal.victim_task_id}"
        f" ({victim.title if victim else '?'}, priority {victim.priority if victim else '?'}),"
        f" which {proposal.incoming_task_id}"
        f" ({incoming.title if incoming else '?'}) outranks."
    )
    if typer.confirm("  Preempt it? Its work is checkpointed and it goes back on the queue"):
        # Recorded, not applied here: this runs inside event reporting, and
        # accepting is itself an async operation that schedules more work.
        _pending_accept.append(proposal.proposal_id)
    else:
        runtime.manager.decline_preemption(proposal.proposal_id)
        console.print("  Left running.")


#: Proposals the user said yes to, applied by `_apply_accepted` once the
#: current reporting pass is over.
_pending_accept: list[str] = []


async def _apply_accepted(runtime: Runtime) -> None:
    while _pending_accept:
        events = await runtime.manager.accept_preemption(_pending_accept.pop(0))
        _report_events(runtime, events)


async def _tick_until_done(runtime: Runtime, task_id: str) -> None:
    """Poll the manager until the task reaches a terminal state."""
    terminal = {TaskState.DONE, TaskState.ERROR, TaskState.KILLED, TaskState.MERGED}
    with console.status(f"waiting for {task_id}..."):
        while runtime.store.get_task_state(task_id) not in terminal:
            events = await runtime.manager.tick()
            if events:
                console.print()
                _report_events(runtime, events)
                await _apply_accepted(runtime)
            await asyncio.sleep(1)


@app.command(help="Every running agent, most urgent first, then the queue.")
def status() -> None:
    """Running agents sorted by priority, then the queue."""
    asyncio.run(_status())


async def _status() -> None:
    runtime = Runtime()
    # One tick first. Only a running session ticks, so a task that
    # finished after you closed one stays "running" in every read-only view
    # until something notices the pane died - and `buddy status` is usually
    # the thing you run to find out.
    await _catch_up(runtime)
    panes = await runtime.runner.panes()
    table = Table(show_header=True, header_style="bold")
    for column in ("agent", "status", "task", "harness", "pri", "age", "pane"):
        table.add_column(column)

    # Most urgent first, then oldest first, which is how the store keeps them.
    agents = runtime.store.load_agents()
    for agent in agents:
        pane = panes.get(agent.name)
        if pane is None:
            pane_text = "[red]missing[/]"
        elif pane.dead:
            pane_text = f"dead ({pane.exit_code})"
        else:
            pane_text = "alive"
        age = ""
        if agent.started_at:
            age = f"{int((utcnow() - agent.started_at).total_seconds() // 60)}m"
        table.add_row(
            escape(agent.name),
            agent.status.value,
            agent.task_id,
            agent.harness or "",
            str(agent.priority or ""),
            age,
            pane_text,
        )
    if agents:
        console.print(table)
    else:
        console.print("No agents running.")

    queued = runtime.store.tasks_in_state(TaskState.QUEUED)
    if queued:
        console.print(f"\n[bold]Queue[/] ({len(queued)})")
        for task in queued:
            waits = f" waits for {', '.join(task.depends_on)}" if task.depends_on else ""
            console.print(
                f"  {task.id}  {escape(task.agent)}  p{task.priority}  {escape(task.title)}{waits}"
            )


@app.command(help="Stop a running agent. Its work is checkpointed; it is not retried.")
def kill(
    agent: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Kill a running agent. Always confirmed: it is mid-work."""
    asyncio.run(_kill(agent, yes))


def _running_agent(runtime: Runtime, name: str) -> Agent:
    """The running agent called `name`, or exit saying who is running."""
    agent = runtime.store.get_agent(name)
    if agent is not None:
        return agent
    running = ", ".join(a.name for a in runtime.store.load_agents()) or "none"
    ended = runtime.store.task_for_agent(name)
    if ended is not None:
        state = runtime.store.get_task_state(ended.id)
        fail(
            f"{name} is not running ({ended.id} is {state.value if state else '?'}); "
            f"`buddy logs {name}` shows its output. Running now: {running}"
        )
    fail(f"no agent is called {name}. Running now: {running}")


async def _kill(name: str, yes: bool) -> None:
    runtime = Runtime()
    agent = _running_agent(runtime, name)
    task = runtime.store.get_task(agent.task_id)
    if not yes and not typer.confirm(
        f"Kill {agent.name} ({agent.task_id}: {task.title if task else '?'})? "
        "Its work is checkpointed but the task is not retried"
    ):
        console.print("Left running.")
        raise typer.Exit
    _report_events(runtime, await runtime.manager.kill_agent(agent.name))


@app.command(help="Change a task's priority: 1 is urgent, 5 is whenever.")
def reprioritize(task_id: str, priority: int) -> None:
    """Reorder the queue. No confirmation: it is reversible."""
    runtime = Runtime()
    runtime.task_or_exit(task_id)
    if not 1 <= priority <= 5:
        fail("priority runs from 1 (urgent) to 5 (whenever)")
    runtime.manager.reprioritize(task_id, priority)
    console.print(f"{task_id} is now priority {priority}")
    ahead = runtime.manager.queue_position(task_id)
    if runtime.store.get_task_state(task_id) is TaskState.QUEUED:
        console.print(f"  {ahead} ready task(s) still ahead of it")
    else:
        console.print("  it is already running; this only changes its rank for preemption")


@app.command(help="Recent tasks and how they ended.")
def history(
    project: str = typer.Option("", "--project", "-P", help="Only this project's tasks."),
    limit: int = typer.Option(20, "--limit", "-n", help="How many."),
) -> None:
    """Recent tasks and how they ended."""
    runtime = Runtime()
    table = Table(show_header=True, header_style="bold")
    for column in ("task", "state", "project", "pri", "attempts", "title"):
        table.add_column(column)
    for task in runtime.store.recent_tasks(project or None, limit):
        state = runtime.store.get_task_state(task.id)
        runs = runtime.store.runs_for(task.id)
        table.add_row(
            task.id,
            state.value if state else "?",
            task.project,
            str(task.priority),
            str(len(runs)),
            task.title,
        )
    console.print(table)


@app.command(help="Summarise the conversation now, to free the brain's context.")
def compact() -> None:
    """Force a client-side compaction now (context layer 2b).

    Always client-side: server compaction is token-triggered only, so a
    manual one has nowhere to hook into.
    """
    asyncio.run(_compact())


async def _compact() -> None:
    runtime = Runtime()
    try:
        brain = runtime.brain()
    except (ProviderNotInstalled, ProviderError) as exc:
        fail(str(exc))
    summary = await brain.compact_now()
    console.print(summary)


config_app = typer.Typer(help="Where config.toml is, whether it loads, and editing it safely.")
app.add_typer(config_app, name="config")

#: What a new config.toml starts as when `buddy config edit` finds none.
CONFIG_STARTER = """\
# Buddy's configuration. Every key, its default and what it changes is in
# docs/configuration.md in Buddy's repository.
#
# `buddy setup` writes a complete one for this machine; this is only here so
# there is something to open.
"""


def _config_path() -> Path:
    from buddy.config import DEFAULT_HOME, Paths, expand

    return Paths(expand(os.environ.get("BUDDY_HOME") or DEFAULT_HOME)).config_file


def _check_config(path: Path) -> Config | None:
    """Load it and say so. Deliberately not `Runtime`: that exits on a broken
    config, which is exactly when this command is needed."""
    try:
        config = Config.load(path, home=path.parent)
    except ConfigError as exc:
        # soft_wrap: a path broken across two lines cannot be copied.
        console.print(f"[red]invalid[/] {escape(str(path))}", soft_wrap=True)
        console.print(f"  {escape(str(exc))}", soft_wrap=True)
        return None
    console.print(f"[green]valid[/] {escape(str(path))}", soft_wrap=True)
    projects = ", ".join(sorted(config.projects)) or "none - `buddy setup --force config`"
    harnesses = ", ".join(config.runnable_harnesses()) or "none - `buddy setup --force harnesses`"
    console.print(f"  projects   {escape(projects)}")
    console.print(f"  harnesses  {escape(harnesses)}")
    console.print(f"  brain      {escape(config.brain.provider)}")
    return config


@config_app.callback(invoke_without_command=True)
def config_show(ctx: typer.Context) -> None:
    """Where config.toml is, and whether it loads."""
    if ctx.invoked_subcommand is not None:
        return
    path = _config_path()
    if not path.exists():
        console.print(
            f"No config yet at {escape(str(path))}. `buddy setup` writes one.", soft_wrap=True
        )
        raise typer.Exit(1)
    if _check_config(path) is None:
        console.print("Fix it with `buddy config edit`.")
        raise typer.Exit(2)


@config_app.command("path", help="Print the path, for scripts.")
def config_path() -> None:
    """Print the path, for scripts: `$EDITOR "$(buddy config path)"`."""
    typer.echo(str(_config_path()))


@config_app.command("edit", help="Open it in $VISUAL or $EDITOR, then check what you saved.")
def config_edit() -> None:
    """Open config.toml in $VISUAL or $EDITOR, then check what you saved.

    A mistake is shown the moment you close the editor - and, at a terminal,
    you are offered the file again - rather than surfacing later as a
    session that will not start.
    """
    path = _config_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CONFIG_STARTER)
        path.chmod(0o600)
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    while True:
        try:
            subprocess.run([*shlex.split(editor), str(path)], check=False)
        except FileNotFoundError:
            fail(f"could not run the editor {editor!r}; set $EDITOR")
        if _check_config(path) is not None:
            return
        if not (sys.stdin.isatty() and typer.confirm("Open it again to fix it?", default=True)):
            raise typer.Exit(2)


memory_app = typer.Typer(help="Facts Buddy remembers across every session.")
app.add_typer(memory_app, name="memory")


@memory_app.command("list", help="Every pinned fact, with its id.")
def memory_list() -> None:
    runtime = Runtime()
    facts = runtime.store.memories()
    if not facts:
        console.print("Nothing pinned yet.")
        raise typer.Exit
    for fact in facts:
        console.print(f"[{fact['id']}] {escape(str(fact['fact']))}")


@memory_app.command("add", help="Pin a fact for every future session.")
def memory_add(fact: str) -> None:
    runtime = Runtime()
    console.print(f"remembered as [{runtime.store.remember(fact)}]")


@memory_app.command("rm", help="Forget a pinned fact, by the id `buddy memory list` shows.")
def memory_rm(memory_id: int) -> None:
    runtime = Runtime()
    if runtime.store.forget(memory_id):
        console.print(f"forgot [{memory_id}]")
    else:
        fail(f"no pinned fact [{memory_id}]")


# --------------------------------------------------------------------------
# watching
# --------------------------------------------------------------------------


def _exec(argv: list[str]) -> None:
    """Hand this terminal to `argv`, which `tmux_runner` wrote."""
    found = shutil.which(argv[0])
    if not found:
        fail(f"{argv[0]} is not installed")
    os.execvp(found, [found, *argv[1:]])  # noqa: S606 - handing the terminal to tmux is the point


@app.command(help="Attach to one agent's terminal, read-only. Ctrl-b d detaches.")
def watch(agent: str) -> None:
    """Attach to one agent's window, read-only."""
    runtime = Runtime()
    name = _running_agent(runtime, agent).name
    runtime.store.close()
    _exec(runtime.runner.attach_command(name))


@app.command(help="Attach to every agent's window, read-only. Ctrl-b w switches windows.")
def attach() -> None:
    """Attach to the session, read-only. Ctrl-b w switches windows."""
    _exec(TmuxRunner().attach_command())


@app.command(help="Print or follow a task's output without attaching.")
def logs(
    target: str,
    follow: bool = typer.Option(False, "-f", "--follow", help="Keep printing as it grows."),
    attempt: int = typer.Option(
        0, "-a", "--attempt", help="Which attempt. Default: the latest.", show_default=False
    ),
    lines: int = typer.Option(200, "-n", "--lines", help="How many lines from the end."),
) -> None:
    """Print or tail a task's log without attaching. `target` is an agent's
    name or a task id; a name that has been reused means its latest task."""
    runtime = Runtime()
    task_id = runtime.named_task_or_exit(target).id

    run = runtime.store.get_run(task_id, attempt) if attempt else runtime.store.latest_run(task_id)
    if run is None:
        fail(f"no run recorded for {task_id}")
    path = run.log_path
    if not path.exists():
        fail(f"no log at {path}")

    console.print(read_tail(path, lines))
    if follow:
        _follow(path)


def _follow(path: Path) -> None:
    """`tail -F`, not `tail -f`: when the log is rotated, follow the new one.

    An open handle keeps reading the file it opened under its new name, so
    following by handle alone goes quiet at the first rotation and never
    wakes up.
    """
    from buddy.logs import strip_ansi

    handle = path.open("r", errors="replace")
    handle.seek(0, os.SEEK_END)
    try:
        while True:
            line = handle.readline()
            if line:
                typer.echo(strip_ansi(line).rstrip("\n"))
                continue
            try:
                replaced = os.stat(path).st_ino != os.fstat(handle.fileno()).st_ino
            except OSError:
                replaced = False  # between the rename and the new file
            if replaced:
                handle.close()
                handle = path.open("r", errors="replace")
                continue
            time.sleep(0.2)
    except KeyboardInterrupt:
        return
    finally:
        handle.close()


# --------------------------------------------------------------------------
# merging
# --------------------------------------------------------------------------


@app.command(help="What a task's branch changed, against the branch it will merge into.")
def diff(
    task_id: str,
    into: str = typer.Option("", "--into", help="Compare against this branch. Default: the base."),
) -> None:
    """Diff a task's branch against its base, --stat first."""
    asyncio.run(_diff(task_id, into or None))


async def _diff(task_id: str, into: str | None) -> None:
    runtime = Runtime()
    task = runtime.task_or_exit(task_id)
    try:
        console.print(await runtime.workspace.diff_stat(task, into=into) or "(no changes)")
        body = await runtime.workspace.diff(task, into=into)
    except WorkspaceError as exc:
        fail(str(exc))
    if body:
        console.print(body)


@app.command(help="Merge a task's branch into your checkout. Always confirmed.")
def merge(
    task_id: str,
    into: str = typer.Option("", "--into", help="Merge into this branch. Default: the base."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    force: bool = typer.Option(
        False, "--force", help="Merge even while a conflict fix is pending in the project."
    ),
    allow_secret_patterns: bool = typer.Option(
        False,
        "--allow-secret-patterns",
        help="Merge although the branch adds key-shaped strings you have checked. "
        "Never overrides a match for one of your own keys.",
    ),
) -> None:
    """Merge a task's branch into your real branch. Always confirmed."""
    asyncio.run(_merge(task_id, into or None, yes, force, allow_secret_patterns))


async def _merge(
    task_id: str,
    into: str | None,
    yes: bool,
    force: bool = False,
    allow_secret_patterns: bool = False,
) -> None:
    runtime = Runtime()
    task = runtime.task_or_exit(task_id)
    target = into or runtime.config.project(task.project).base_branch
    if (busy := runtime.manager.still_working(task.id)) is not None:
        # Not overridable by --force: that answers the conflict hold, and
        # nothing makes deleting a working agent's checkout safe.
        fail(f"not merging: {busy}")
    if (hold := conflicts.merge_hold(runtime.store, task)) is not None and not force:
        fail(
            f"{hold.id} is resolving merge conflicts in {task.project}; merging {task.id} now "
            f"would move {target} under it. Merge {hold.id} first, or pass --force."
        )
    if not yes and not typer.confirm(
        f"Merge {branch_name(task)} into {target} in {runtime.config.project(task.project).path}?"
    ):
        console.print("Left alone.")
        raise typer.Exit
    try:
        sha = await runtime.workspace.merge(
            task, into=into, allow_secret_patterns=allow_secret_patterns
        )
    except WorkspaceError as exc:  # MergeConflict and SecretsInBranch say what to do
        fail(str(exc))
    marked = conflicts.record_merge(runtime.store, task)
    also = f" (which also lands {', '.join(marked[1:])})" if len(marked) > 1 else ""
    console.print(f"[green]merged[/] {task.id} into {target} as {sha[:8]}{also}")


@app.command(help="Spawn an agent to resolve a task's merge conflict.")
def resolve(
    task_id: str,
    harness: str = typer.Option("", "--harness", "-H", help="Defaults to the task's own."),
    wait: bool = typer.Option(False, "--wait", help="Block until the fix finishes."),
) -> None:
    """Spawn an agent to resolve a task's merge conflict."""
    asyncio.run(_resolve(task_id, harness or None, wait))


async def _resolve(task_id: str, harness: str | None, wait: bool) -> None:
    runtime = Runtime()
    original = runtime.task_or_exit(task_id)
    if (existing := conflicts.pending_fix(runtime.store, original)) is not None:
        fail(f"{existing.id} is already resolving {task_id}'s conflicts")
    chosen = harness or original.harness
    report = await runtime.adapter(chosen).preflight()
    if not report.usable:
        fail(f"{report.summary()} - run `buddy doctor`")
    fix = conflicts.fix_task(
        runtime.config,
        runtime.store,
        original,
        harness=chosen,
        task_id=runtime.store.next_task_id(),
    )
    events = await runtime.manager.submit(fix)
    _report_events(runtime, events)
    console.print(
        f"{fix.id} resolves {task_id}: it merges "
        f"{runtime.config.project(original.project).base_branch} into "
        f"{branch_name(original)}. Merges into {original.project} wait for it."
    )
    if wait:
        await _tick_until_done(runtime, fix.id)


@app.command(help="Remove a task's worktree. Its branch is kept for the grace period.")
def discard(
    task_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Remove a task's worktree; its branch is kept for the grace period."""
    asyncio.run(_discard(task_id, yes))


async def _discard(task_id: str, yes: bool) -> None:
    runtime = Runtime()
    task = runtime.task_or_exit(task_id)
    grace = runtime.config.buddy.discard_grace
    if (busy := runtime.manager.still_working(task.id)) is not None:
        fail(f"not discarding: {busy}")
    if not yes and not typer.confirm(
        f"Discard {task.id}? The worktree goes; {branch_name(task)} is kept for {grace.days} days."
    ):
        console.print("Left alone.")
        raise typer.Exit
    try:
        await runtime.workspace.discard(task)
    except WorkspaceError as exc:
        fail(str(exc))
    runtime.store.set_task_state(task.id, TaskState.DISCARDED)
    console.print(f"[yellow]discarded[/] {task.id}; branch kept as {branch_name(task)}")


# --------------------------------------------------------------------------
# setup, update, uninstall
# --------------------------------------------------------------------------


class ConsoleUI:
    """The steps' only route to the terminal.

    `--yes` takes every default here rather than in each step, so no step can
    forget it. A secret is the one exception: an API key is still prompted
    for, once, even under `--yes`, because it is never a default.
    """

    def __init__(self, *, assume_yes: bool = False) -> None:
        self.assume_yes = assume_yes
        # Asked once. Without a terminal, getpass falls back to an echoing
        # read and warns about it twice before failing anyway, so the prompt
        # is never attempted rather than attempted and apologised for.
        self.interactive = sys.stdin.isatty()

    def say(self, message: str) -> None:
        console.print(escape(message))

    def detail(self, message: str) -> None:
        console.print(f"[dim]{escape(message)}[/]")

    def warn(self, message: str) -> None:
        console.print(f"[yellow]{escape(message)}[/]")

    def confirm(self, question: str, *, default: bool = True) -> bool:
        if self.assume_yes:
            console.print(f"[dim]{escape(question)} -> {'yes' if default else 'no'} (--yes)[/]")
            return default
        if not self.interactive:
            return self._no_terminal(question, default)
        try:
            return typer.confirm(question, default=default)
        except (EOFError, typer.Abort):
            return self._no_terminal(question, default)

    def ask(self, question: str, *, default: str = "", secret: bool = False) -> str:
        if self.assume_yes and not secret and default:
            console.print(f"[dim]{escape(question)} -> {default} (--yes)[/]")
            return default
        if not self.interactive:
            self._no_terminal(question, bool(default))
            return default
        try:
            return typer.prompt(
                question, default=default, hide_input=secret, show_default=not secret
            )
        except (EOFError, typer.Abort):
            self._no_terminal(question, bool(default))
            return default

    def _no_terminal(self, question: str, default: bool) -> bool:
        """There is nobody to ask.

        `curl ... | sh` leaves stdin as the pipe, and CI has no terminal at
        all. Taking the default and saying so beats aborting the run with a
        traceback about EOF.
        """
        console.print(f"[yellow]No terminal to ask on: {escape(question)}[/]")
        console.print(f"[dim]Taking the default ({'yes' if default else 'no'}).[/]")
        return default


def _setup_table(outcomes: list) -> Table:
    from buddy.setup import Status

    colour = {
        Status.DONE: "green",
        Status.SKIPPED: "cyan",
        Status.NOT_APPLICABLE: "dim",
        Status.FAILED: "red",
    }
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    # The name, not the title: this column is what `--force` takes.
    table.add_column("step")
    table.add_column("result")
    table.add_column("detail", overflow="fold")
    for outcome in outcomes:
        table.add_row(
            outcome.number,
            outcome.step,
            f"[{colour[outcome.status]}]{outcome.status.value}[/]",
            escape(outcome.detail),
        )
    return table


@app.command(help="Install and configure Buddy for this machine. Safe to run again.")
def setup(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Accept every default. API keys are still asked for."
    ),
    no_voice: bool = typer.Option(False, "--no-voice", help="Skip speech entirely."),
    cloud_tts: bool = typer.Option(False, "--cloud-tts", help="Use Fish cloud, not a local GPU."),
    install_harnesses: bool = typer.Option(
        False, "--install-harnesses", help="Opt in to `npm i -g` for the known harness CLIs."
    ),
    brain: str = typer.Option("", "--brain", help="Brain provider, overriding the config."),
    force: list[str] = typer.Option(
        [], "--force", help="Re-run a step even if it is satisfied. Repeatable."
    ),
) -> None:
    """Idempotent, hardware-aware, resumable bootstrap."""
    raise SystemExit(
        asyncio.run(
            _setup(
                yes=yes,
                no_voice=no_voice,
                cloud_tts=cloud_tts,
                install_harnesses=install_harnesses,
                brain=brain,
                force=force,
            )
        )
    )


async def _setup(
    *,
    yes: bool = False,
    no_voice: bool = False,
    cloud_tts: bool = False,
    install_harnesses: bool = False,
    brain: str = "",
    force: list[str] | None = None,
    update: bool = False,
    home: Path | None = None,
) -> int:
    from buddy.setup import (
        SetupContext,
        SetupLock,
        SetupOptions,
        Status,
        all_steps,
        hint_for,
        run_steps,
        step_names,
    )

    known = step_names()
    for name in force or []:
        if name not in known:
            fail(f"unknown step {name!r}. The steps are: {', '.join(known)}")

    resolved = Config.load(home=home).home
    options = SetupOptions(
        yes=yes,
        no_voice=no_voice,
        cloud_tts=cloud_tts,
        install_harnesses=install_harnesses,
        brain=brain,
        force=frozenset(force or []),
        update=update,
    )
    ctx = SetupContext(
        home=resolved,
        options=options,
        ui=ConsoleUI(assume_yes=yes),
        lock=SetupLock.load(Config(home=resolved).paths.setup_lock),
    )

    console.print(
        f"[bold]buddy {'update' if update else 'setup'}[/] - {resolved}\n"
        "Every step is check, act, verify; a satisfied one is skipped, so this is "
        "safe to re-run.\n"
    )
    outcomes = await run_steps(ctx, all_steps())
    console.print(_setup_table(outcomes))

    failed = [outcome for outcome in outcomes if outcome.status is Status.FAILED]
    if failed:
        console.print(f"[red]Stopped at '{failed[0].step}':[/] {escape(failed[0].detail)}")
        console.print(hint_for(failed[0].step))
        return 1
    console.print(f"[dim]Recorded what each step settled on in {ctx.lock.path}[/]")
    console.print("\n[green]Ready.[/] Start it with: [bold]buddy[/]")
    return 0


@app.command(help="Upgrade Buddy, then re-run only the setup steps that changed.")
def update() -> None:
    """Upgrade Buddy, then re-run only the steps whose pins moved."""
    raise SystemExit(asyncio.run(_update()))


def _source_checkout() -> Path | None:
    """The git checkout Buddy is running from, if it is one - the install the
    README describes - or None for an installed tool."""
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file() and (root / ".git").exists():
        return root
    return None


async def _update() -> int:
    checkout = _source_checkout()
    if checkout is not None:
        # `uv tool upgrade` cannot touch a clone. Asked anyway, it failed and
        # told the user to `uv tool install` a package that is not published.
        where = escape(shlex.quote(str(checkout)))
        console.print(
            f"[bold]Running from a checkout[/] - update it with "
            f"[bold]cd {where} && git pull && uv sync[/], then run buddy update again. "
            "Re-checking the setup steps now.",
            soft_wrap=True,
        )
    elif tool := shutil.which("uv"):
        console.print("[bold]Upgrading the tool[/] - uv tool upgrade buddy-orchestrator")
        proc = await asyncio.create_subprocess_exec(
            tool,
            "tool",
            "upgrade",
            "buddy-orchestrator",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await proc.communicate()
        finally:
            release(proc)
        said = out.decode(errors="replace").strip()
        if proc.returncode != 0:
            # Not fatal: re-running the steps is still worth doing.
            last = said.splitlines()[-1] if said else "upgrade failed"
            console.print(f"[yellow]{escape(last)}[/]")
        elif said:
            console.print(f"[dim]{escape(said.splitlines()[-1])}[/]")
    else:
        console.print("[yellow]uv is not on PATH, so the tool itself was not upgraded.[/]")
    return await _setup(update=True)


@app.command(help="Remove Buddy: the tmux session, the container, the tool.")
def uninstall(
    purge: bool = typer.Option(False, "--purge", help="Also delete ~/.buddy."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    force: bool = typer.Option(
        False, "--force", help="Purge even with unmerged work. It is not recoverable."
    ),
) -> None:
    """Stop the container, kill the tmux session, remove the tool."""
    raise SystemExit(asyncio.run(_uninstall(purge=purge, yes=yes, force=force)))


async def _uninstall(*, purge: bool, yes: bool, force: bool) -> int:
    from buddy.setup.platform import run
    from buddy.setup.steps.tts import CONTAINER

    runtime = Runtime()
    home = runtime.config.home

    if purge:
        unmerged = await _unmerged_work(runtime)
        if unmerged and not force:
            # Refused: a branch with work on it is the one thing here
            # that cannot be recreated from anywhere else.
            console.print(f"[red]Refusing to purge:[/] {len(unmerged)} task(s) have unmerged work.")
            for task_id, branch in unmerged:
                console.print(f"  {task_id} on {branch}")
            console.print("`buddy merge <id>` or `buddy discard <id>` each of them, or --force.")
            return 1
        if unmerged:
            console.print(f"[yellow]--force: discarding {len(unmerged)} unmerged branch(es).[/]")

    if not yes and not typer.confirm(
        f"Remove the tmux session and the {CONTAINER} container"
        + (f", and delete {home}" if purge else "")
        + "?"
    ):
        console.print("Left alone.")
        return 1

    await run("docker", "rm", "-f", CONTAINER)
    console.print(f"[dim]removed the {CONTAINER} container if it existed[/]")
    try:
        await runtime.runner.kill_session()
        console.print("[dim]killed the tmux session[/]")
    except TmuxError as exc:
        console.print(f"[dim]tmux: {escape(str(exc))}[/]", soft_wrap=True)

    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FISH_API_KEY"):
        if keychain_delete(name):
            console.print(f"[dim]removed {name} from the keychain[/]")

    if purge:
        runtime.store.close()
        shutil.rmtree(home, ignore_errors=True)
        console.print(f"[yellow]deleted {home}[/]")

    console.print("Uninstall the tool itself with: uv tool uninstall buddy-orchestrator")
    return 0


async def _unmerged_work(runtime: Runtime) -> list[tuple[str, str]]:
    """Tasks whose branch still holds work nothing else has."""
    unmerged: list[tuple[str, str]] = []
    for state in (TaskState.DONE, TaskState.ERROR, TaskState.KILLED, TaskState.QUEUED):
        for task in runtime.store.tasks_in_state(state):
            run = runtime.store.latest_run(task.id)
            if run is None:
                continue
            try:
                diff = await runtime.workspace.diff_stat(task)
            except WorkspaceError:
                continue
            if diff.strip():
                unmerged.append((task.id, run.branch))
    return unmerged


@app.command(help="Stop every agent and clear the worktrees, keeping all work and memory.")
def shutdown(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
    keep_worktrees: bool = typer.Option(
        False, "--keep-worktrees", help="Stop the agents but leave the checkouts on disk."
    ),
) -> None:
    """Stop every agent and clear the worktrees, keeping the work and the memory."""
    raise SystemExit(asyncio.run(_shutdown(yes=yes, keep_worktrees=keep_worktrees)))


async def _shutdown(*, yes: bool, keep_worktrees: bool) -> int:
    runtime = Runtime()
    # A tick first, so anything that finished while nothing was watching is
    # finalized rather than stopped as though it were still running.
    await _catch_up(runtime)

    running = runtime.manager.agents()
    if running:
        console.print(f"[yellow]{len(running)} agent(s) still running:[/]")
        for agent in running:
            console.print(f"  {escape(agent.name)}: {agent.task_id}")
        console.print(
            "[dim]They will be checkpointed onto their branches and requeued, so they "
            "pick up where they left off next time. Nothing is lost.[/]"
        )
    if not yes and not typer.confirm(
        "Stop everything"
        + ("" if keep_worktrees else " and remove the worktrees")
        + "? Branches, logs and memory are kept."
    ):
        console.print("Left alone.")
        return 1

    report = await runtime.manager.shutdown(remove_worktrees=not keep_worktrees)

    for stopped in report.stopped:
        console.print(f"[yellow]stopped[/] {stopped}")
    if report.removed:
        console.print(f"[dim]removed {len(report.removed)} worktree(s)[/]")
    for task_id, why in report.failed:
        console.print(f"[red]{task_id}:[/] {escape(why)}")

    try:
        await runtime.runner.kill_session()
        console.print("[dim]killed the tmux session[/]")
    except (TmuxError, FileNotFoundError) as exc:
        console.print(f"[dim]tmux: {escape(str(exc))}[/]", soft_wrap=True)

    console.print()
    console.print("[green]Down.[/] Kept: every branch, every log, and the database -")
    console.print("[dim]  pinned memory, the conversation, and every task's history.[/]")
    if report.unmerged:
        console.print(f"\n{len(report.unmerged)} branch(es) still hold unmerged work:")
        for task_id, branch in report.unmerged:
            console.print(f"  {task_id}  {branch}")
        console.print("[dim]`buddy merge <id>` still works - it never needed the worktree.[/]")
    return 1 if report.failed else 0


@app.command(help="A small always-on-top panel showing what every agent is doing.")
def overlay(
    toggle: bool = typer.Option(False, "--toggle", help="Close it if it is open, open it if not."),
    stop: bool = typer.Option(False, "--stop", help="Close a running overlay."),
    status: bool = typer.Option(False, "--status", help="Print the summary instead of drawing it."),
) -> None:
    """A small always-on-top panel showing what every agent is doing."""
    from buddy import overlay as panel

    runtime = Runtime()
    home = runtime.config.home
    url = f"http://127.0.0.1:{runtime.config.buddy.web_port}"

    if status:
        console.print(panel.summarise(url))
        raise SystemExit(0)

    already = panel.running_pid(home)
    if stop or (toggle and already):
        raise SystemExit(0 if panel.stop(home) else _say_not_running())
    if already:
        console.print(f"[yellow]The overlay is already open (pid {already}).[/]")
        console.print("[dim]`buddy overlay --toggle` closes it, `--stop` closes it outright.[/]")
        raise SystemExit(0)

    if not panel.dashboard_is_up(url):
        console.print(f"[yellow]Nothing is serving {url}.[/]")
        console.print(
            "The overlay reads the dashboard, so start a session first: [bold]buddy[/]\n"
            "[dim](or `buddy --no-voice` if you would rather type.)[/]"
        )
        raise SystemExit(1)

    try:
        raise SystemExit(panel.show(url, home))
    except panel.OverlayNotInstalled as exc:
        fail(escape(str(exc)))


def _say_not_running() -> int:
    console.print("[dim]No overlay was open.[/]")
    return 0


def main() -> None:
    """Console-script entry point."""
    app()
