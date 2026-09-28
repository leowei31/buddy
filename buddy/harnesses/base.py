"""HarnessAdapter protocol, PreflightReport, ResultSummary.

An adapter is the only thing that knows a specific harness. It answers
three questions and nothing else: can this harness be driven unattended
(`preflight`), what exactly do we run (`invocation`), and what did it say
(`parse_result`, `waiting_patterns`).

Execution is not its business. The wrapper script, the log pipe, the worktree
and the exit code are all handled elsewhere, which is why adding a fifth
harness never touches the core.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from buddy.config import (
    ConfigError,
    EnvSecretResolver,
    HarnessConfig,
    SecretResolver,
    fill_template,
    resolve_secrets,
)
from buddy.models import TaskRun
from buddy.processes import kill_and_reap, release
from buddy.sandbox import sandboxed, wrap_in_sandbox

#: The four requirements for driving a harness unattended. One that fails any is
#: unsupported, full stop.
REQUIREMENTS = {
    "headless": "Non-interactive invocation that exits when the task is done",
    "prompt_input": "Accepts a multi-paragraph prompt from a file, stdin, or one argument",
    "auto_approve": "An auto-approve / non-interactive-permissions flag",
    "exit_codes": "A meaningful exit code (0 on success)",
}


class HarnessError(Exception):
    pass


class PreflightFailed(HarnessError):
    """Raised at spawn when an adapter's preflight does not hold."""


@dataclass(frozen=True)
class Requirement:
    key: str
    satisfied: bool
    detail: str = ""

    @property
    def description(self) -> str:
        return REQUIREMENTS.get(self.key, self.key)


@dataclass(frozen=True)
class PreflightReport:
    """What `buddy doctor` prints per harness."""

    harness: str
    binary: str | None = None
    version: str | None = None
    requirements: tuple[Requirement, ...] = ()
    notes: tuple[str, ...] = ()
    #: Whether the CLI can reach its model: True, False, or None when the
    #: CLI offers no cheap way to ask. Separate from `ok`, because a CLI that
    #: meets the four requirements but is not signed in is installed correctly and still
    #: fails every task - the two need different fixes.
    authenticated: bool | None = None
    auth_detail: str = ""
    #: Where Buddy looked, for the message when it found nothing. A sandboxed
    #: harness lives in an image, and "not on PATH" would send its owner to
    #: install a CLI on a machine that is never going to run it.
    where: str = "PATH"

    @property
    def installed(self) -> bool:
        return self.binary is not None

    @property
    def ok(self) -> bool:
        return self.installed and all(r.satisfied for r in self.requirements)

    @property
    def usable(self) -> bool:
        """Able to run a task now: the requirements hold and nothing says it is signed out."""
        return self.ok and self.authenticated is not False

    @property
    def failures(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if not r.satisfied)

    def summary(self) -> str:
        if not self.installed:
            preposition = "on" if self.where == "PATH" else "in"
            return f"{self.harness}: not {preposition} {self.where}"
        if self.ok and self.authenticated is False:
            return f"{self.harness}: not signed in - {self.auth_detail}"
        if self.ok:
            return f"{self.harness}: ok ({self.version or 'unknown version'})"
        missing = ", ".join(r.key for r in self.failures)
        return f"{self.harness}: unusable, missing {missing}"


@dataclass(frozen=True)
class ResultSummary:
    """The harness's own account of what it did.

    `ok` is the adapter's reading of the log. It is never the completion
    signal: that is the pane's exit status. When the two disagree,
    the exit code wins and the disagreement is worth surfacing.
    """

    ok: bool
    summary: str = ""
    exit_code: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HarnessAdapter(Protocol):
    name: str
    waiting_patterns: list[re.Pattern[str]]
    """Regexes that, matched at the tail of the log, mean the harness is
    blocked waiting for a human. Drives WAITING_INPUT. Buddy
    never answers the prompt itself."""

    async def preflight(self) -> PreflightReport:
        """Is the binary present, and does it satisfy the four requirements?

        Async because it runs `<harness> --help` as a subprocess, and nothing
        may block the event loop for one.
        """
        ...

    def invocation(self, run: TaskRun, prompt_path: Path) -> str:
        """The exact harness command line for this run.

        No `cd`, no logging, no sentinels: the wrapper script adds those.
        Runs with cwd = run.worktree.
        """
        ...

    def parse_result(self, log_text: str, exit_code: int) -> ResultSummary:
        """The harness's final message, so the brain can narrate one sentence
        instead of reading 400 lines."""
        ...


# --------------------------------------------------------------------------
# Shared implementation
# --------------------------------------------------------------------------


#: A container start plus a CLI's own startup, with room to spare. A probe
#: that times out is reported as a broken harness, so it must not be tight.
SANDBOX_PROBE_SECONDS = 60.0


async def run_command(
    *args: str, seconds: float = 20.0, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """Run a short-lived probe, e.g. `<harness> --help`."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        async with asyncio.timeout(seconds):
            try:
                out, err = await proc.communicate()
            finally:
                release(proc)
    except TimeoutError:
        await kill_and_reap(proc)
        return 124, "", f"{args[0]} did not answer within {seconds}s"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


class BaseAdapter:
    """What every adapter shares: finding the binary, formatting the command
    template from config, and proving exit codes.

    The command template lives in config.toml, never in code, so a harness
    release is an edit and not a Buddy release.
    """

    name: str = ""
    binary: str = ""
    #: Placeholders an adapter's `command` template may use.
    placeholders = ("prompt_path", "worktree", "model", "model_flag")

    #: What `buddy setup` writes into `[harness.<name>]` - the command line
    #: this adapter was verified against, never a remembered one.
    default_command: str = ""
    default_waiting_patterns: tuple[str, ...] = ()
    #: None means "whatever the CLI itself is configured to use".
    default_model: str | None = None
    #: The arguments that print the help `preflight` reads. A CLI whose
    #: headless mode is a subcommand documents its flags there, not at the top.
    help_args: tuple[str, ...] = ("--help",)
    #: How a person installs it. Setup prints this; it never runs it.
    install_hint: str = ""
    #: `npm i -g` name, for the opt-in `--install-harnesses`.
    npm_package: str | None = None
    #: Brain-provider key -> the variable this CLI actually reads for it.
    #: Setup wires only these, because a key exported under a name the CLI
    #: ignores looks configured and is not: Codex reads CODEX_API_KEY in exec
    #: mode and ignores OPENAI_API_KEY (verified against 0.154.0).
    credential_env: ClassVar[dict[str, str]] = {}

    def __init__(self, config: HarnessConfig, resolver: SecretResolver | None = None) -> None:
        self.config = config
        self.waiting_patterns = list(config.waiting_patterns)
        #: For `[harness.<name>.env]`'s `${keychain:NAME}` references, so a
        #: login probe sees the credentials a real run would. Environment
        #: only when none is given, which never touches the keychain.
        self.resolver: SecretResolver = resolver or EnvSecretResolver()

    # -- the contract every adapter fills in -----------------------

    async def preflight(self) -> PreflightReport:
        raise NotImplementedError(f"{type(self).__name__} must implement preflight()")

    def parse_result(self, log_text: str, exit_code: int) -> ResultSummary:
        raise NotImplementedError(f"{type(self).__name__} must implement parse_result()")

    # -- probing, where the harness will actually run ----------------------

    @property
    def sandboxed(self) -> bool:
        """Whether runs of this harness happen inside a container."""
        return sandboxed(self.config)

    async def probe(self, *args: str, seconds: float = 20.0) -> tuple[int, str, str]:
        """Run `<binary> <args>` where a task would run it.

        Every preflight question - is the CLI here, which version, does it
        list the flags it needs, is it signed in - is a question about the
        place the harness runs, and for a sandboxed harness that is the
        image, not this machine. Asking the host instead made the whole
        sandbox unusable: the documented setup installs the CLIs *in the
        image*, and `buddy doctor` then called a working harness "not on
        PATH" while `buddy spawn` refused every task given to it.
        """
        if not self.sandboxed:
            return await run_command(self.binary, *args, seconds=seconds, env=self.run_env())
        return await self.probe_shell(shlex.join([self.binary, *args]), seconds=seconds)

    async def probe_shell(self, line: str, *, seconds: float = 20.0) -> tuple[int, str, str]:
        """Run one shell line inside this harness's sandbox."""
        if shutil.which("docker") is None:
            return 127, "", "docker is not installed, and this harness is sandboxed"
        # The sandbox command mounts a worktree and the brief; a probe has
        # neither, so it gets empty stand-ins rather than anything of yours.
        with tempfile.TemporaryDirectory(prefix="buddy-probe-") as scratch:
            prompt = Path(scratch) / "prompt.md"
            prompt.touch()
            wrapped = wrap_in_sandbox(self.config, line, Path(scratch), prompt_path=prompt)
            # Starting an image is slower than running a local binary, and a
            # probe that times out reads as a broken harness.
            return await run_command(
                "bash",
                "-c",
                wrapped,
                seconds=max(seconds, SANDBOX_PROBE_SECONDS),
                env=self.run_env(),
            )

    def which(self) -> str | None:
        """Where the binary is on *this machine*. `locate` is the one
        preflight asks, because a sandboxed harness need not be here at all."""
        return shutil.which(self.binary)

    async def locate(self) -> str | None:
        """Where the binary is, wherever this harness runs.

        `command -v` rather than `--version`: whether a binary exists is not
        the same question as whether it implements a version flag, and a
        shell that answers neither is still perfectly present.
        """
        if not self.sandboxed:
            return self.which()
        code, out, _ = await self.probe_shell(f"command -v {shlex.quote(self.binary)}")
        found = [line for line in out.splitlines() if line.strip()]
        return f"{found[-1].strip()} in {self.image}" if code == 0 and found else None

    @property
    def where(self) -> str:
        """Where a missing binary was looked for, for the message."""
        return self.image if self.sandboxed else "PATH"

    @property
    def image(self) -> str:
        """The image name in `sandbox_command`, for messages. Best effort:
        this is a user-written command line, not a parsed structure."""
        command = self.config.sandbox_command or ""
        match = re.search(r"--entrypoint\s+\S+\s+(\S+)", command)
        return match.group(1) if match else "the sandbox image"

    async def version(self) -> str | None:
        code, out, err = await self.probe("--version")
        return (out or err).strip().splitlines()[0] if code == 0 and (out or err).strip() else None

    # -- the command line -------------------------------------------------

    def resolve_model(self, run_model: str | None) -> str | None:
        """What the user asked for, else the harness's configured default
        ."""
        return run_model or self.config.default_model

    def invocation(self, run: TaskRun, prompt_path: Path, *, model: str | None = None) -> str:
        """The command line, with every substituted value shell-quoted.

        `model` reaches here from `spawn_agent`, which means an LLM chose it,
        and the result is spliced into a `run.sh` that bash executes.
        Unquoted, `sonnet; rm -rf ~; echo` was a working command line - so
        every placeholder is quoted here, the way the surrounding script
        already quoted its own variables.
        """
        resolved = self.resolve_model(model)
        command = fill_template(
            self.config.command,
            prompt_path=shlex.quote(str(prompt_path)),
            worktree=shlex.quote(str(run.worktree)),
            model=shlex.quote(resolved) if resolved else "",
            model_flag=self.model_flag(resolved),
        )
        return " ".join(command.split()) if "  " in command else command

    def model_flag(self, model: str | None) -> str:
        """`--model X`, or nothing. Adapters that name the flag differently
        override this."""
        return f"--model {shlex.quote(model)}" if model else ""

    # -- requirement 4 ----------------------------------------------------

    def describe_activity(self, log_text: str, lines: int = 20) -> str:
        """What this run is doing, in lines a person can read.

        The default is the tail itself, which is right for a harness that
        prints prose. A harness that emits a machine format overrides this,
        because only its adapter may know what that format looks
        like - the manager and the dashboard just get sentences.
        """
        from buddy.logs import tail

        return tail(log_text, lines)

    async def check_exit_codes(self) -> Requirement:
        """Prove the binary distinguishes success from failure, without
        spending an API call: `--help` must succeed and a nonsense flag must
        not."""
        ok_code, _, _ = await self.probe("--help")
        bad_code, _, _ = await self.probe("--buddy-preflight-not-a-flag")
        if ok_code == 0 and bad_code != 0:
            return Requirement("exit_codes", True, f"--help exits 0, bad flag exits {bad_code}")
        return Requirement(
            "exit_codes",
            False,
            f"--help exited {ok_code} and an invalid flag exited {bad_code}",
        )

    async def help_text(self) -> str:
        _, out, err = await self.probe(*self.help_args)
        return out + err

    async def auth_status(self) -> tuple[bool | None, str]:
        """Whether the CLI can reach its model, as cheaply as the CLI allows.

        None when there is no way to ask without spending a model call. An
        adapter that can ask overrides this; nothing here guesses.
        """
        return None, ""

    def run_env(self) -> dict[str, str]:
        """The environment a run of this harness gets: Buddy's own, plus
        `[harness.<name>.env]` with its secrets resolved.

        What `run.sh` exports, rebuilt for probes. A login check that ran
        without it would report "signed out" for a CLI whose key lives only
        in that block - which is exactly how Antigravity's API-key mode was
        misreported before this existed. A reference that cannot be resolved
        is left out rather than raised: the probe then answers honestly that
        the credential is missing.
        """
        env = dict(os.environ)
        for key, value in self.config.env.items():
            try:
                label = f"harness.{self.name}.env.{key}"
                env[key] = resolve_secrets(value, self.resolver, key=label)
            except ConfigError:
                continue
        return env

    def credential_in_env(self, *names: str) -> str | None:
        """The first of `names` a run would have set, non-empty."""
        env = self.run_env()
        for name in names:
            if env.get(name):
                return name
        return None

    def is_progress(self, new_output: str) -> bool:
        """Whether output that just arrived counts as the run moving.

        Stall detection reads log growth as progress. That is right for
        almost everything, and wrong for a harness that has lost its API and
        prints a retry line every few seconds forever: the log grows, the
        agent says RUNNING, and nothing is happening. An adapter that knows
        its own retry lines returns False for them, so the agent goes STALLED
        after `stall_timeout` like any other run that is going nowhere.
        """
        return True

    def sandbox_notes(self) -> list[str]:
        """Warnings about a `sandbox_command` written before Buddy knew better.

        Both of these fail silently, which is why they are worth a line in
        `buddy doctor`: one makes every commit impossible, the other makes the
        run produce no output at all until the stall timeout ends it. A
        command already in someone's `config.toml` is never rewritten for
        them, so it has to be said.
        """
        command = self.config.sandbox_command or ""
        notes: list[str] = []
        if not self.sandboxed:
            return notes
        if "{git}" not in command:
            notes.append(
                "sandbox_command has no {git}: a worktree's .git points outside the worktree, "
                "so git does not work in the container and the agent cannot commit its work"
            )
        if re.search(r"\s-(?:[a-zA-Z]*i)\b", command.split("--entrypoint")[0]):
            notes.append(
                "sandbox_command passes -i: the pane's terminal becomes the container's stdin "
                "and a harness that reads stdin waits for an end that never comes"
            )
        return notes

    async def report(
        self, requirements: tuple[Requirement, ...], notes: list[str]
    ) -> PreflightReport:
        """The common tail of every `preflight`: version and login state."""
        notes = [*notes, *self.sandbox_notes()]
        authenticated, auth_detail = await self.auth_status()
        return PreflightReport(
            harness=self.name,
            where=self.where,
            binary=await self.locate(),
            version=await self.version(),
            requirements=requirements,
            notes=tuple(notes),
            authenticated=authenticated,
            auth_detail=auth_detail,
        )

    def flag_requirement(self, key: str, help_text: str, *flags: str) -> Requirement:
        """Requirements 1 and 3 are answered by what `--help` actually lists,
        never by flags remembered from a previous release."""
        found = [flag for flag in flags if flag in help_text]
        if found:
            return Requirement(key, True, f"found {', '.join(found)}")
        return Requirement(key, False, f"none of {', '.join(flags)} in --help")
