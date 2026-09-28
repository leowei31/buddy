"""`buddy setup` / `update` / `uninstall`: the step runner and setup.lock.

Setup is an ordered list of steps, each `check -> act -> verify`. The
shape is the promise: a satisfied step is skipped, so re-running setup is
always safe; a failed step stops with the exact name to re-run under
`--force STEP`; and because every step is independently re-runnable, a
failure never leaves a half-configured machine behind.

`setup.lock` records what each step settled on - package versions, an image
digest, a whisper model, a harness command template. `buddy update` re-runs
only the steps whose pins have moved, which is the whole reason the
pins are recorded rather than a bare "done".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from buddy.config import Config
from buddy.models import utcnow
from buddy.setup.platform import PlatformReport

LOCK_VERSION = 1


class SetupError(Exception):
    """A step that could not finish, naming itself so it can be re-run.

    Everything a user has to *do* about a failed setup is in here: which step,
    what went wrong, and what to try. `cli.py` prints exactly these three.
    """

    def __init__(self, step: str, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.step = step
        self.message = message
        self.hint = hint


class Status(StrEnum):
    SKIPPED = "skipped"  # already satisfied
    DONE = "done"  # acted, then verified
    NOT_APPLICABLE = "n/a"  # --no-voice, no GPU, wrong OS
    FAILED = "failed"


@dataclass
class CheckResult:
    """What a step found before deciding whether to act.

    `pins` is what makes `buddy update` possible: it is the identity of what
    this step would install or use *now*, so a difference from the lock is
    what makes an already-satisfied step unsatisfied again.
    """

    satisfied: bool
    detail: str = ""
    pins: dict[str, str] = field(default_factory=dict)


@dataclass
class Outcome:
    step: str  # the name `--force` takes
    title: str
    status: Status
    detail: str = ""
    number: str = ""  # its place in the order, as the table shows it


@runtime_checkable
class Step(Protocol):
    name: str  # what `--force` takes
    title: str  # what the table shows
    number: str  # its place in the order, for cross-referencing

    async def applies(self, ctx: SetupContext) -> str: ...
    async def check(self, ctx: SetupContext) -> CheckResult: ...
    async def act(self, ctx: SetupContext) -> None: ...
    async def verify(self, ctx: SetupContext) -> str: ...
    async def contribute(self, ctx: SetupContext) -> None: ...


class BaseStep:
    """Defaults every step shares.

    `verify` re-runs `check` unless a step overrides it, which is the right
    default: a step whose check is honest needs no separate verification, and
    a step whose check is cheap (a file exists) usually wants a real one.
    """

    name = ""
    title = ""
    number = ""

    async def applies(self, ctx: SetupContext) -> str:
        return ""

    async def contribute(self, ctx: SetupContext) -> None:
        """What this step wants in `config.toml`, whether or not it had to act.

        Separate from `act` because being *already* satisfied is the common
        case - the CLI is installed, the model is cached, the key is in the
        keychain - and a step that only writes its config while installing
        something would silently drop it on every run after the first.
        """
        return None

    async def check(self, ctx: SetupContext) -> CheckResult:
        raise NotImplementedError

    async def act(self, ctx: SetupContext) -> None:
        raise NotImplementedError

    async def verify(self, ctx: SetupContext) -> str:
        result = await self.check(ctx)
        if not result.satisfied:
            raise SetupError(self.name, f"{self.title} did not take: {result.detail}")
        return result.detail

    def fail(self, message: str, *, hint: str = "") -> SetupError:
        return SetupError(self.name, message, hint=hint)


# --------------------------------------------------------------------------
# The lock
# --------------------------------------------------------------------------


@dataclass
class SetupLock:
    """`~/.buddy/setup.lock`: what setup settled on, per step."""

    path: Path
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> SetupLock:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls(path=path)
        steps = data.get("steps") if isinstance(data, dict) else None
        return cls(path=path, steps=steps if isinstance(steps, dict) else {})

    def pins(self, step: str) -> dict[str, str]:
        entry = self.steps.get(step) or {}
        recorded = entry.get("pins")
        return recorded if isinstance(recorded, dict) else {}

    def record(self, step: str, *, status: Status, pins: dict[str, str], detail: str = "") -> None:
        self.steps[step] = {
            "status": status.value,
            "pins": pins,
            "detail": detail,
            "at": utcnow().isoformat(),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": LOCK_VERSION, "updated_at": utcnow().isoformat(), "steps": self.steps},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


@runtime_checkable
class SetupUI(Protocol):
    """Everything a step is allowed to do to the terminal.

    Behind a protocol so the suite can drive a whole setup run without a tty,
    and so no step can quietly `input()` its way around `--yes`.

    `confirm` and `ask` block the loop, deliberately. The non-blocking rule
    exists so a running agent is never stalled by Buddy; during setup there
    are no agents, and a prompt that is already waiting on a human has
    nothing to yield to.
    """

    def say(self, message: str) -> None: ...
    def detail(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def confirm(self, question: str, *, default: bool = True) -> bool: ...
    def ask(self, question: str, *, default: str = "", secret: bool = False) -> str: ...


@dataclass
class SetupOptions:
    yes: bool = False
    no_voice: bool = False
    cloud_tts: bool = False
    install_harnesses: bool = False
    brain: str = ""  # override the provider step 5 would ask about
    force: frozenset[str] = frozenset()
    #: `buddy update`: a satisfied step whose pins moved is run again.
    update: bool = False


@dataclass
class SetupContext:
    home: Path
    options: SetupOptions
    ui: SetupUI
    lock: SetupLock
    report: PlatformReport | None = None
    #: Filled in by steps 5-9 and rendered into config.toml by step 10, so the
    #: config is written once from everything that was actually verified
    #: rather than in nine partial edits.
    plan: dict[str, Any] = field(default_factory=dict)
    #: Steps that have already run, by name, for the few that read each other.
    results: dict[str, str] = field(default_factory=dict)
    #: Which tmux server step 11 talks to. None is the user's own, which is
    #: what setup wants; the suite points it at a private socket so a test
    #: run can never create or disturb the real `buddy` session.
    tmux_socket: str | None = None

    @property
    def paths(self):
        return Config(home=self.home).paths

    def platform(self) -> PlatformReport:
        """The step 1 report. Steps 2 onwards may assume it exists."""
        if self.report is None:  # pragma: no cover - the runner fills it first
            raise SetupError("platform", "the platform step has not run yet")
        return self.report

    def section(self, *names: str) -> dict[str, Any]:
        """A nested dict inside `plan`, created on the way down."""
        node = self.plan
        for name in names:
            node = node.setdefault(name, {})
        return node


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


async def run_steps(ctx: SetupContext, steps: list[Step]) -> list[Outcome]:
    """Run the list in order, stopping at the first failure.

    Ordering is fixed, not a dependency graph: the steps are numbered and
    later ones read what earlier ones wrote. A graph would let them be
    reordered, which is exactly what must not happen.
    """
    outcomes: list[Outcome] = []
    for step in steps:
        outcome = await _run_one(ctx, step)
        outcomes.append(outcome)
        ctx.lock.save()
        if outcome.status is Status.FAILED:
            break
    return outcomes


async def _run_one(ctx: SetupContext, step: Step) -> Outcome:
    def outcome(status: Status, detail: str) -> Outcome:
        return Outcome(step.name, step.title, status, detail, step.number)

    why_not = await step.applies(ctx)
    if why_not:
        ctx.lock.record(step.name, status=Status.NOT_APPLICABLE, pins={}, detail=why_not)
        return outcome(Status.NOT_APPLICABLE, why_not)

    forced = step.name in ctx.options.force
    try:
        found = await step.check(ctx)
    except SetupError:
        raise
    except Exception as exc:  # noqa: BLE001 - any probe failure is this step's
        detail = f"{type(exc).__name__}: {exc}"
        ctx.lock.record(step.name, status=Status.FAILED, pins={}, detail=detail)
        return outcome(Status.FAILED, detail)

    moved = ctx.options.update and found.pins != ctx.lock.pins(step.name)
    if found.satisfied and not forced and not moved:
        await step.contribute(ctx)
        ctx.results[step.name] = found.detail
        ctx.lock.record(step.name, status=Status.SKIPPED, pins=found.pins, detail=found.detail)
        return outcome(Status.SKIPPED, found.detail)

    if moved:
        ctx.ui.detail(f"{step.title}: pins changed since the last run, doing it again")

    try:
        await step.act(ctx)
        await step.contribute(ctx)
        detail = await step.verify(ctx)
    except SetupError as exc:
        ctx.lock.record(step.name, status=Status.FAILED, pins={}, detail=exc.message)
        return outcome(Status.FAILED, exc.message)
    except Exception as exc:  # noqa: BLE001 - reported, never a traceback
        detail = f"{type(exc).__name__}: {exc}"
        ctx.lock.record(step.name, status=Status.FAILED, pins={}, detail=detail)
        return outcome(Status.FAILED, detail)

    after = await step.check(ctx)
    ctx.results[step.name] = detail
    ctx.lock.record(step.name, status=Status.DONE, pins=after.pins, detail=detail)
    return outcome(Status.DONE, detail)


def hint_for(step: str) -> str:
    """What to tell the user after a failure."""
    return f"Fix it, then re-run just that step with: buddy setup --force {step}"


def all_steps() -> list[Step]:
    """The twelve steps, in order.

    Imported here rather than at module scope so `buddy setup --help` and the
    rest of the CLI do not drag in every step's dependencies.
    """
    from buddy.setup.steps import (
        audio,
        config,
        detect,
        docker,
        doctor,
        harnesses,
        keys,
        layout,
        packages,
        stt,
        tmux,
        tts,
    )

    return [
        detect.PlatformStep(),
        packages.PackagesStep(),
        docker.DockerStep(),
        layout.LayoutStep(),
        keys.KeysStep(),
        stt.SttStep(),
        tts.TtsStep(),
        harnesses.HarnessesStep(),
        audio.AudioStep(),
        config.ConfigStep(),
        tmux.TmuxStep(),
        doctor.DoctorStep(),
    ]


def step_names() -> list[str]:
    return [step.name for step in all_steps()]


__all__ = [
    "BaseStep",
    "CheckResult",
    "Outcome",
    "SetupContext",
    "SetupError",
    "SetupLock",
    "SetupOptions",
    "SetupUI",
    "Status",
    "Step",
    "all_steps",
    "hint_for",
    "run_steps",
    "step_names",
]
