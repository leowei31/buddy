"""AgentManager: named agents, priority, deps, preemption, health.

Also the parts a single task needs to run: composing the prompt Buddy sends,
and generating the per-task wrapper script.

The wrapper lives here rather than in `tmux_runner` or an adapter because of
the layer rule: the runner knows only tmux, an adapter knows only its
harness, and the script is where a task's worktree, environment, sentinels
and harness command are assembled into one thing to run.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from buddy.config import (
    Config,
    ConfigError,
    EnvSecretResolver,
    HarnessConfig,
    SecretResolver,
    resolve_secrets,
)
from buddy.logs import DONE_SENTINEL, START_SENTINEL, read_sentinels, read_tail, tail
from buddy.models import (
    Agent,
    AgentHealthChanged,
    AgentNameError,
    AgentStatus,
    BranchDeleted,
    Event,
    PreemptionProposal,
    RunOutcome,
    TaskBlocked,
    TaskFinished,
    TaskRequeued,
    TaskRun,
    TaskSpec,
    TaskStarted,
    TaskState,
    check_agent_name,
    default_agent_name,
    utcnow,
)
from buddy.sandbox import wrap_in_sandbox
from buddy.state import Store
from buddy.tmux_runner import PaneStatus, TmuxRunner
from buddy.workspace import Workspace, WorkspaceError, branch_name

#: `cd` into the worktree failed. Distinct from any harness exit code so the
#: log says plainly that the task never started.
WORKTREE_MISSING_EXIT = 97

#: Appended by Buddy, never written by the brain, so it is identical for every
#: task.
WORKING_RULES = (
    "## Working rules\n"
    "- You are on branch {branch} in an isolated worktree."
    " Commit your work with clear messages as you go.\n"
    "- Do not push, do not switch branches, do not touch anything outside this worktree.\n"
    "- Never write an API key, token or password into a file, and never commit `.env` or key"
    " files. Read secrets from the environment; Buddy keeps secret-looking files out of its"
    " commits and will not merge a branch that contains one.\n"
    "- If you are blocked or the task is underspecified,"
    " state exactly what you need and stop."
)

#: The same block for a project that is not a git repo, where there is no
#: branch and the task runs in place, one agent at a time.
WORKING_RULES_NO_GIT = (
    "## Working rules\n"
    "- You are working directly in {worktree}, which is not a git repository,"
    " so there is no branch to fall back on. Be conservative.\n"
    "- Do not touch anything outside that directory.\n"
    "- If you are blocked or the task is underspecified,"
    " state exactly what you need and stop."
)

#: Prefixed on every attempt after the first.
RESUME_NOTE = (
    "- This worktree contains partial work from a previous attempt."
    " Run `git log` and `git diff {base_ref}` first, then continue."
)


def compose_prompt(task: TaskSpec, run: TaskRun) -> str:
    """The exact text written to `prompt.md` and handed to the harness.

    The brain writes the brief; Buddy appends the working rules and, on a
    retry, the resume note.
    """
    parts = [task.brief.rstrip()]
    if run.branch:
        parts.append(WORKING_RULES.format(branch=run.branch))
    else:
        parts.append(WORKING_RULES_NO_GIT.format(worktree=run.worktree))
    if run.attempt > 1:
        parts.append(RESUME_NOTE.format(base_ref=run.base_ref or "the base commit"))
    return "\n\n".join(parts) + "\n"


def write_prompt(path: Path, task: TaskSpec, run: TaskRun) -> Path:
    """`prompt.md`: the brief, exactly as sent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compose_prompt(task, run))
    return path


def resolve_env(harness: HarnessConfig, resolver: SecretResolver) -> dict[str, str]:
    """`[harness.<name>.env]` with its `${keychain:NAME}` references resolved.

    Resolved at generation time, which is why `run.sh` is mode 0600 and
    deleted once the attempt's result is written.
    """
    return {
        key: resolve_secrets(value, resolver, key=f"harness.{harness.name}.env.{key}")
        for key, value in harness.env.items()
    }


#: The most of a log's new bytes shown to `is_progress` in one tick. A tick
#: is seconds apart; anything larger is progress by any definition.
PROGRESS_WINDOW = 64_000


def _made_progress(adapter: object, log_path: Path, seen: int, size: int) -> bool:
    """Whether the bytes between `seen` and `size` are the run moving.

    A shrunk or first-seen log is progress by definition, and so is any
    failure to judge: this only ever *withholds* progress when an adapter
    positively says its own output is retry chatter.
    """
    judge = getattr(adapter, "is_progress", None)
    if judge is None or seen < 0 or size < seen or size - seen > PROGRESS_WINDOW:
        return True
    try:
        with log_path.open("rb") as handle:
            handle.seek(seen)
            grown = handle.read(size - seen).decode(errors="replace")
        return bool(judge(grown))
    except Exception:  # noqa: BLE001 - never let a judgement call cost a stall signal
        return True


def describe_activity(adapter: object, log_text: str, lines: int = 20) -> str:
    """What a run is doing, however much its adapter can say.

    `BaseAdapter` supplies a default, but an adapter written elsewhere may
    predate the method - and a missing niceness must never cost a finished
    run its bookkeeping.
    """
    describe = getattr(adapter, "describe_activity", None)
    if describe is None:
        return tail(log_text, lines)
    try:
        return describe(log_text, lines)
    except Exception:  # noqa: BLE001 - cosmetic; the run still finalizes
        return tail(log_text, lines)


def render_run_script(
    *,
    task: TaskSpec,
    run: TaskRun,
    prompt_path: Path,
    invocation: str,
    env: dict[str, str] | None = None,
) -> str:
    """The generated bash wrapper.

    Why a script at all: the brief never touches a shell quoting layer, the
    run is bash regardless of the user's login shell, the log is
    self-describing through its sentinels, and the file is an exact audit
    record of what ran.
    """
    exports = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in sorted((env or {}).items())
    )
    env_block = (
        f"\n# provider credentials for this harness, from [harness.<name>.env]\n{exports}\n"
        if exports
        else ""
    )

    return f"""\
#!/usr/bin/env bash
# {run.log_path.parent / "run.sh"} - generated by Buddy; regenerated each attempt
set -uo pipefail
TASK_ID={shlex.quote(task.id)}
ATTEMPT={shlex.quote(str(run.attempt))}
WORKTREE={shlex.quote(str(run.worktree))}
PROMPT={shlex.quote(str(prompt_path))}

echo "{START_SENTINEL} ${{TASK_ID}} attempt=${{ATTEMPT}} $(date -u +%FT%TZ)"
sleep 0.25          # lets pipe-pane attach before any real output

cd "${{WORKTREE}}" || {{
  echo "{DONE_SENTINEL} ${{TASK_ID}} {WORKTREE_MISSING_EXIT}"
  exit {WORKTREE_MISSING_EXIT}
}}
{env_block}
{invocation}
code=$?

echo "{DONE_SENTINEL} ${{TASK_ID}} ${{code}} $(date -u +%FT%TZ)"
exit "${{code}}"
"""


def write_run_script(
    path: Path,
    *,
    task: TaskSpec,
    run: TaskRun,
    prompt_path: Path,
    invocation: str,
    env: dict[str, str] | None = None,
) -> Path:
    """Write `run.sh` mode 0600, because resolved secrets are in it.

    Created with that mode rather than chmod-ed to it afterwards: writing
    first and restricting second leaves the secrets readable under the
    umask for as long as the write takes. A script left by an earlier
    attempt is replaced, never appended to, and never trusted for its mode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as handle:
        handle.write(
            render_run_script(
                task=task, run=run, prompt_path=prompt_path, invocation=invocation, env=env
            )
        )
    return path


def remove_run_script(path: Path) -> None:
    """Deleted the moment an attempt ends, however it ends, so resolved
    secrets never sit on disk longer than the run."""
    path.unlink(missing_ok=True)


def prepare_run(
    config: Config,
    task: TaskSpec,
    run: TaskRun,
    *,
    invocation_for: object,
    resolver: SecretResolver,
) -> tuple[Path, Path]:
    """Write `prompt.md` and `run.sh` for one attempt.

    `invocation_for` is a `HarnessAdapter`; it is typed loosely here so the
    manager depends on the protocol rather than on any harness.
    """
    paths = config.paths
    prompt_path = paths.prompt_file(task.id)
    script_path = paths.run_script(task.id)
    harness = config.harness(task.harness)

    write_prompt(prompt_path, task, run)
    command = invocation_for.invocation(run, prompt_path, model=task.model)  # type: ignore[attr-defined]
    write_run_script(
        script_path,
        task=task,
        run=run,
        prompt_path=prompt_path,
        invocation=wrap_in_sandbox(
            harness,
            command,
            run.worktree,
            prompt_path=prompt_path,
            repo=config.project(task.project).path,
        ),
        env=resolve_env(harness, resolver),
    )
    return prompt_path, script_path


def script_is_private(path: Path) -> bool:
    """Mode 0600 and nothing else."""
    return stat.S_IMODE(os.stat(path).st_mode) == 0o600


# --------------------------------------------------------------------------
# Finalizing a run
# --------------------------------------------------------------------------


async def finalize_run(
    *,
    config: Config,
    store: Store,
    workspace: Workspace,
    runner: TmuxRunner,
    adapter: object,
    task: TaskSpec,
    run: TaskRun,
    exit_code: int,
    outcome: RunOutcome | None = None,
) -> TaskFinished:
    """Close out an attempt whose pane has gone dead.

    `tick` calls this the moment `pane_dead_status` appears. The order
    matters: the run script goes first and the checkpoint next, so neither a
    secret nor the work is lost if the rest fails.
    """
    await runner.close_pipe(run.agent)
    # First, not last: the pane is dead, so nothing needs the script, and a
    # checkpoint that raises below must not leave resolved secrets behind.
    remove_run_script(config.paths.run_script(task.id))

    resolved = outcome or (RunOutcome.DONE if exit_code == 0 else RunOutcome.ERROR)
    run.wip_commit = await workspace.checkpoint(run, resolved)

    # What the checkpoint kept out of git because it looked like a secret.
    withheld = getattr(workspace, "withheld", {}).pop((task.id, run.attempt), [])

    log_text = read_tail(run.log_path, lines=2000, clean=False)
    summary = adapter.parse_result(log_text, exit_code)  # type: ignore[attr-defined]
    result_path = config.paths.result_file(task.id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "task_id": task.id,
                "attempt": run.attempt,
                "exit_code": exit_code,
                "outcome": resolved.value,
                "ok": summary.ok,
                "summary": summary.summary,
                "wip_commit": run.wip_commit,
                "withheld_from_commit": withheld,
                "detail": summary.detail,
            },
            indent=2,
        )
        + "\n"
    )

    run.ended_at = utcnow()
    run.exit_code = exit_code
    run.outcome = resolved
    store.save_run(run)
    # Merged and discarded are decisions a person made about a finished task;
    # a late tick noticing the pane died must not undo one. This is reachable
    # whenever `buddy merge` runs from a second terminal while no session is
    # ticking - which is exactly how it was found.
    settled = (TaskState.MERGED, TaskState.DISCARDED)
    if store.get_task_state(task.id) not in settled:
        store.set_task_state(
            task.id, TaskState.DONE if resolved is RunOutcome.DONE else TaskState.ERROR
        )

    return TaskFinished(
        task_id=task.id,
        agent=run.agent,
        attempt=run.attempt,
        outcome=resolved,
        exit_code=exit_code,
        branch=run.branch,
        summary=summary.summary
        + (
            f"\nKept out of the commit, left in the worktree: {'; '.join(withheld)}"
            if withheld
            else ""
        ),
    )


# --------------------------------------------------------------------------
# The scheduler
# --------------------------------------------------------------------------


@dataclass
class ShutdownReport:
    """What a shutdown stopped, removed, and deliberately kept."""

    stopped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unmerged: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


class StalePreemption(RuntimeError):
    """A preemption accepted after the situation it was about had changed."""


@dataclass
class _Pending:
    """A preemption awaiting the user's answer."""

    proposal: PreemptionProposal
    incoming: TaskSpec


class AgentManager:
    """Named agents, priority, dependencies, preemption, health.

    An agent is a task while it runs, under the name the user calls it by:
    its own tmux window, created when the attempt starts and removed when it
    ends. There is no fixed number of them. `max_concurrent` can cap how
    many run at once; unset, anything ready starts.

    Knows the harness *protocol* but never a harness: adapters arrive
    through `adapter_for`, which is why adding a fifth harness does not touch
    this file.

    `now` is injected so stall and timeout behaviour can be tested without
    waiting ten real minutes.
    """

    def __init__(
        self,
        config: Config,
        store: Store,
        runner: TmuxRunner,
        workspace: Workspace,
        adapter_for: Callable[[str], object],
        *,
        resolver: SecretResolver | None = None,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self.config = config
        self.store = store
        self.runner = runner
        self.workspace = workspace
        self.adapter_for = adapter_for
        self.resolver = resolver or EnvSecretResolver()
        self.now = now
        self._pending: dict[str, _Pending] = {}
        #: Dropped proposals and why, so a late yes gets a reason, not a KeyError.
        self._stale: dict[str, str] = {}
        self._last_sweep: datetime | None = None
        #: Last observed log size per agent, which is how `last_output_at` is
        #: maintained: the file growing, not a capture-pane diff.
        self._log_sizes: dict[str, int] = {}
        #: Tasks already reported as blocked, so the brain hears it once.
        self._reported_blocked: set[str] = set()
        #: Which projects are not git repos and so run one agent at a time.
        #: Filled by `refresh_project_locks`.
        self._exclusive: dict[str, bool] = {}

    @property
    def max_concurrent(self) -> int | None:
        """How many agents may run at once, or None for no limit."""
        return self.config.buddy.max_concurrent or None

    def _has_room(self, active: list[Agent]) -> bool:
        limit = self.max_concurrent
        return limit is None or len(active) < limit

    # -- agents ------------------------------------------------------------

    def agents(self) -> list[Agent]:
        """Every running agent, most urgent first."""
        return self.store.load_agents()

    def agent(self, name: str) -> Agent | None:
        """The running agent with this name, whatever its case."""
        return self.store.get_agent(name)

    def claim_name(self, requested: str, title: str) -> str:
        """The name a new task's agent will have.

        What the user asked for, checked and normalized - or, when they did
        not say, one made from the title. Either way it is not the name of
        another queued or running agent: a name is a handle, and "kill
        scout" must mean exactly one thing. A finished agent's name is free.
        """
        taken = self.store.live_agent_names()
        if not requested.strip():
            return default_agent_name(title, taken)
        name = check_agent_name(requested)
        holder = taken.get(name.lower())
        if holder is not None:
            state = self.store.get_task_state(holder)
            raise AgentNameError(
                f"{name} is already {holder}, which is {state.value if state else 'live'}. "
                "Pick another name, or wait for it to finish."
            )
        return name

    async def _retire(self, agent: Agent) -> None:
        """An attempt is over: its window and its row go.

        The window is closed before the row is removed, so a crash between
        the two leaves a row with no window - which reconciliation recovers
        from the run's outcome - rather than a window nothing tracks.
        """
        await self.runner.close_window(agent.name)
        remove_run_script(self.config.paths.run_script(agent.task_id))
        self.store.remove_agent(agent.name)
        self._log_sizes.pop(agent.name, None)

    # -- dependencies ----------------------------------------------

    def dependency_block(self, task: TaskSpec) -> str | None:
        """Why a queued task is not ready, or None if it is.

        A dependency counts as satisfied when it is DONE, or only when it is
        MERGED if that dependency was declared `merge_required`.
        """
        for dep_id in task.depends_on:
            dep = self.store.get_task(dep_id)
            if dep is None:
                return f"depends on {dep_id}, which does not exist"
            state = self.store.get_task_state(dep_id)
            if dep.merge_required:
                if state is not TaskState.MERGED:
                    return (
                        f"waits for {dep_id} to be merged (it is {state.value if state else '?'})"
                    )
            elif state not in (TaskState.DONE, TaskState.MERGED):
                return f"waits for {dep_id} ({state.value if state else '?'})"
        return None

    def _unsatisfiable(self, task: TaskSpec) -> str | None:
        """A dependency that can never be satisfied, so the brain can say so
        rather than leaving the task queued forever."""
        dead = (TaskState.ERROR, TaskState.KILLED, TaskState.DISCARDED)
        for dep_id in task.depends_on:
            state = self.store.get_task_state(dep_id)
            if state in dead:
                return f"{dep_id} ended as {state.value}"
        return None

    def ready_tasks(self) -> list[TaskSpec]:
        """Queued, unblocked, lowest priority number first, FIFO within a
        priority. `tasks_in_state` already orders them."""
        return [
            task
            for task in self.store.tasks_in_state(TaskState.QUEUED)
            if self.dependency_block(task) is None
        ]

    # -- submitting and scheduling ---------------------------------

    async def submit(self, task: TaskSpec) -> list[Event]:
        """Name the task's agent, persist the task as queued, then schedule.

        Raises `AgentNameError`, before anything is saved, for a name that
        cannot be used.
        """
        task.agent = self.claim_name(task.agent, task.title)
        self.store.create_task(task, TaskState.QUEUED)
        return await self.schedule()

    def queue_position(self, task_id: str) -> int:
        """How many ready tasks outrank this one. `spawn_agent` reports
        "queued behind N tasks" with it."""
        for index, task in enumerate(self.ready_tasks()):
            if task.id == task_id:
                return index
        return len(self.ready_tasks())

    async def schedule(self) -> list[Event]:
        """Start every ready task there is room for; propose a preemption if
        the top ready task outranks something running and cannot start
        otherwise. Never preempts on its own."""
        await self.refresh_project_locks()
        events: list[Event] = []

        while True:
            active = self.agents()
            if not self._has_room(active):
                break
            task = self._next_schedulable(self.ready_tasks(), active)
            if task is None:
                break
            events.append(await self._start(task))

        events.extend(self._report_blocked())
        events.extend(await self._maybe_propose_preemption(self.agents()))
        return events

    def _next_schedulable(self, ready: list[TaskSpec], active: list[Agent]) -> TaskSpec | None:
        """The highest-ranked ready task that can actually start now.

        A non-git project runs one agent at a time, so a task for one is
        skipped while another task in that project is running rather than
        blocking everything behind it.
        """
        busy_projects = {
            running.project
            for agent in active
            if (running := self.store.get_task(agent.task_id)) is not None
        }
        for task in ready:
            if task.project in busy_projects and self._exclusive.get(task.project):
                continue
            return task
        return None

    async def refresh_project_locks(self) -> None:
        """Which projects are not git repos, and so run one agent at a time.
        Cached: it cannot change while Buddy runs without someone
        running `git init` underneath it."""
        for name in self.config.projects:
            if name not in self._exclusive:
                self._exclusive[name] = await self.workspace.requires_exclusive_run(name)

    def _refusal(self, task: TaskSpec) -> str | None:
        """Why this task can never start, or None if it can.

        Checked before anything is created for it. The harness is named by
        whoever wrote the task - often the brain - and a name with no adapter,
        or a `[harness.<name>]` block parked with no command, is not a reason
        to retry: an empty command line runs nothing and exits 0, which would
        look exactly like a task that succeeded.
        """
        try:
            adapter = self.adapter_for(task.harness)
        except Exception as exc:  # noqa: BLE001 - any failure here is this task's alone
            return f"harness {task.harness!r} is not usable: {exc}"
        command = getattr(getattr(adapter, "config", None), "command", None)
        if command is not None and not str(command).strip():
            return f"[harness.{task.harness}] has no command, so there is nothing to run"
        return None

    def _refuse(self, task: TaskSpec, reason: str) -> Event:
        self.store.set_task_state(task.id, TaskState.ERROR)
        return TaskFinished(
            at=self.now(),
            task_id=task.id,
            agent=task.agent,
            attempt=task.attempt,
            outcome=RunOutcome.ERROR,
            summary=f"could not start: {reason}",
        )

    async def _start(self, task: TaskSpec) -> Event:
        """Worktree, prompt, wrapper, and a window of its own."""
        refusal = self._refusal(task)
        if refusal is not None:
            return self._refuse(task, refusal)
        try:
            checkout = await self.workspace.create(task)
        except (WorkspaceError, ConfigError) as exc:
            # A project path that is gone, a base branch with no commit, a
            # project removed from config.toml. Raised out of here, it left
            # `schedule` on every tick: the same task stayed first in line
            # and failed again each second, and nothing queued behind it ever
            # started. It fails alone instead, saying why, and can be spawned
            # again once the cause is fixed.
            return self._refuse(task, f"could not create its checkout: {exc}")
        run = TaskRun(
            task_id=task.id,
            attempt=task.attempt,
            agent=task.agent,
            worktree=checkout.path,
            branch=checkout.branch or "",
            base_ref=checkout.base_ref,
            log_path=self.config.paths.log_file(task.id, task.attempt),
            started_at=self.now(),
        )
        adapter = self.adapter_for(task.harness)
        try:
            _, script = prepare_run(
                self.config, task, run, invocation_for=adapter, resolver=self.resolver
            )
        except Exception as exc:  # noqa: BLE001 - deterministic, so retrying cannot help
            # A secret nobody stored, a template that will not render: the
            # same thing will happen on every tick, so this task stops here
            # and the queue behind it keeps moving. The worktree stays; the
            # branch is reattached if the task is ever retried.
            return self._refuse(task, f"could not prepare the run: {exc}")
        if self.store.get_run(task.id, run.attempt) is None:
            self.store.create_run(run)
        else:
            self.store.save_run(run)
        try:
            await self.runner.spawn(task.agent, run, script)
        except BaseException:
            # The task stays queued and is tried again on a later tick; its
            # resolved secrets do not wait on disk for that.
            remove_run_script(script)
            raise

        agent = Agent(
            name=task.agent,
            task_id=task.id,
            run_attempt=run.attempt,
            harness=task.harness,
            priority=task.priority,
            started_at=run.started_at,
            last_output_at=run.started_at,
        )
        with self.store.transaction():
            self.store.save_agent(agent)
            self.store.set_task_state(task.id, TaskState.RUNNING)
        self._log_sizes[agent.name] = 0
        return TaskStarted(at=self.now(), task_id=task.id, agent=agent.name, attempt=run.attempt)

    def _report_blocked(self) -> list[Event]:
        events: list[Event] = []
        for task in self.store.tasks_in_state(TaskState.QUEUED):
            reason = self._unsatisfiable(task)
            if reason and task.id not in self._reported_blocked:
                self._reported_blocked.add(task.id)
                events.append(TaskBlocked(at=self.now(), task_id=task.id, reason=reason))
        return events

    # -- preemption -------------------------------------------------

    async def _maybe_propose_preemption(self, active: list[Agent]) -> list[Event]:
        """Propose stopping a running task for a queued one that outranks it
        - but only a stop that would actually let it start.

        Two things can hold a ready task back, and so two things decide that.
        One is `max_concurrent`, when it is set: stopping any lower-ranked
        agent makes room. The other is a non-git project, which runs one
        agent at a time: a queued task for it can only start once *that*
        project's agent stops, so it is the only victim that helps. A stop
        that frees nothing for the task is never proposed, because the
        scheduler would put the victim straight back.
        """
        self._drop_stale_proposals(active)
        if not active:
            return []
        promised = {p.proposal.victim_agent.lower() for p in self._pending.values()}
        for incoming in self.ready_tasks():
            if any(p.proposal.incoming_task_id == incoming.id for p in self._pending.values()):
                return []  # one question at a time about the same task
            victims = [
                agent
                for agent in active
                if agent.name.lower() not in promised
                and agent.priority is not None
                and incoming.priority < agent.priority
                and self._would_start_without(incoming, agent, active)
            ]
            if not victims:
                continue
            victim = max(victims, key=lambda a: (a.priority, a.started_at or datetime.min))
            proposal = PreemptionProposal(
                at=self.now(),
                proposal_id=f"p-{uuid.uuid4().hex[:8]}",
                victim_agent=victim.name,
                victim_task_id=victim.task_id,
                incoming_task_id=incoming.id,
            )
            self._pending[proposal.proposal_id] = _Pending(proposal, incoming)
            return [proposal]
        return []

    def _would_start_without(self, incoming: TaskSpec, victim: Agent, active: list[Agent]) -> bool:
        """Whether `incoming` could start if `victim` stopped."""
        others = [a for a in active if a.name != victim.name]
        if not self._has_room(others):
            return False
        if self._exclusive.get(incoming.project):
            for agent in others:
                running = self.store.get_task(agent.task_id)
                if running is not None and running.project == incoming.project:
                    return False
        return True

    def _drop_stale_proposals(self, active: list[Agent]) -> None:
        """A proposal whose victim stopped, or whose incoming task is no
        longer waiting, is a question with no subject any more."""
        by_name = {agent.name.lower(): agent for agent in active}
        for key, pending in list(self._pending.items()):
            agent = by_name.get(pending.proposal.victim_agent.lower())
            still_there = agent is not None and agent.task_id == pending.proposal.victim_task_id
            waiting = self.store.get_task_state(pending.incoming.id) is TaskState.QUEUED
            if not still_there:
                self._stale[key] = (
                    f"{pending.proposal.victim_agent} is no longer running "
                    f"{pending.proposal.victim_task_id}, so nothing was stopped"
                )
            elif not waiting:
                self._stale[key] = (
                    f"{pending.incoming.id} is no longer waiting, so nothing was stopped"
                )
            else:
                continue
            self._pending.pop(key, None)

    def stale_reason(self, proposal_id: str) -> str | None:
        """Why a proposal that was dropped was dropped, for the answer to a
        late yes."""
        return self._stale.get(proposal_id)

    @property
    def pending_preemptions(self) -> dict[str, PreemptionProposal]:
        return {key: value.proposal for key, value in self._pending.items()}

    async def accept_preemption(self, proposal_id: str) -> list[Event]:
        """Kill, checkpoint, keep the branch, requeue, then start the task it
        was stopped for. Only ever called after the user says yes."""
        pending = self._pending.pop(proposal_id, None)
        if pending is None:
            if reason := self._stale.pop(proposal_id, None):
                raise StalePreemption(reason)
            raise KeyError(f"no such preemption proposal: {proposal_id}")

        proposal = pending.proposal
        victim_task = self.store.get_task(proposal.victim_task_id)
        victim = self.agent(proposal.victim_agent)
        if victim_task is None or victim is None or victim.task_id != victim_task.id:
            # The victim finished, was killed, or was preempted already. Saying
            # "preempted" here would be a lie about work that was never stopped.
            raise StalePreemption(
                f"{proposal.victim_agent} is no longer running {proposal.victim_task_id}, "
                "so nothing was stopped"
            )
        if self.store.get_task_state(pending.incoming.id) is not TaskState.QUEUED:
            raise StalePreemption(
                f"{pending.incoming.id} is no longer waiting, so nothing was stopped"
            )

        events: list[Event] = []
        events.extend(await self._stop_and_requeue(victim, victim_task, RunOutcome.PREEMPTED))
        # The room goes to the task it was made for - not to whatever a fresh
        # scheduling pass would pick, which could be the victim itself.
        incoming = self.store.get_task(pending.incoming.id)
        if incoming is not None and self.dependency_block(incoming) is None:
            events.append(await self._start(incoming))
        events.extend(await self.schedule())
        return events

    def decline_preemption(self, proposal_id: str) -> None:
        self._pending.pop(proposal_id, None)

    async def _stop_and_requeue(
        self, agent: Agent, task: TaskSpec, outcome: RunOutcome
    ) -> list[Event]:
        """The shared path for preempt, timeout and interrupt.

        Nothing is discarded: the worktree and branch are kept, whatever was
        uncommitted is checkpointed, and the task goes back on the queue at
        its original priority, under the same name, with the resume note.
        """
        run = self.store.get_run(task.id, agent.run_attempt)
        await self._kill(agent.name, task.id)
        if run is not None:
            run.wip_commit = await self.workspace.checkpoint(run, outcome)
            run.ended_at = self.now()
            run.outcome = outcome

        task.attempt += 1
        # One transaction: a crash can leave the attempt stopped, but never
        # recorded as over while the task still claims to be running.
        with self.store.transaction():
            if run is not None:
                self.store.save_run(run)
            self.store.update_task(task)
            self.store.set_task_state(task.id, TaskState.QUEUED)
            self.store.remove_agent(agent.name)
        self._log_sizes.pop(agent.name, None)
        return [
            TaskRequeued(at=self.now(), task_id=task.id, next_attempt=task.attempt, reason=outcome)
        ]

    async def _kill(self, name: str, task_id: str | None) -> None:
        """Stop an agent's process tree and close its window, and with them
        the attempt's `run.sh`.

        Every way an attempt ends that is not its pane dying on its own -
        kill, preempt, timeout, interrupt, shutdown - comes through here, so
        none of them can leave resolved secrets on disk. A requeued
        attempt writes a fresh script when it starts again.
        """
        await self.runner.kill(name)
        if task_id:
            remove_run_script(self.config.paths.run_script(task_id))

    # -- the tick ---------------------------------------------------

    async def tick(self) -> list[Event]:
        """One pass: finish what ended, check health, start what is ready."""
        await self.refresh_project_locks()
        events: list[Event] = []
        panes = await self.runner.panes()

        for agent in self.agents():
            task = self.store.get_task(agent.task_id)
            if task is None:
                await self._retire(agent)  # nothing it could be running
                continue
            try:
                events.extend(await self._tick_agent(agent, task, panes.get(agent.name)))
            except Exception as exc:  # noqa: BLE001 - one agent, not all of them
                # A disk-full worktree, a stray index.lock, a repo hook that
                # `--no-verify` does not suppress: any of them can make one
                # agent's checkpoint raise. Before this, that exception left
                # `tick` entirely, so every other agent stopped being
                # monitored too - no finalize, no stall detection, no
                # scheduling - until someone restarted the process.
                events.append(
                    AgentHealthChanged(
                        at=self.now(),
                        agent=agent.name,
                        task_id=task.id,
                        status=agent.status,
                        detail=f"tick failed for this agent: {type(exc).__name__}: {exc}",
                    )
                )

        events.extend(await self.schedule())
        events.extend(await self.sweep())
        return events

    #: How often discarded branches are checked. Nothing about it is urgent.
    SWEEP_INTERVAL = timedelta(hours=1)

    async def sweep(self) -> list[Event]:
        """Keep `buddy discard`'s promise: branches go after the grace.

        Every piece of this existed and was tested, and nothing called it, so
        discarded branches piled up in your repository forever.
        Runs at most once per `SWEEP_INTERVAL`, and never lets a git problem
        in one project reach the tick.
        """
        now = self.now()
        if self._last_sweep is not None and now - self._last_sweep < self.SWEEP_INTERVAL:
            return []
        self._last_sweep = now
        events: list[Event] = []
        for task, discarded_at in self.store.discarded_tasks():
            try:
                if await self.workspace.prune_discarded(task, discarded_at, now=now):
                    events.append(BranchDeleted(at=now, task_id=task.id, branch=branch_name(task)))
            except Exception:  # noqa: BLE001 - housekeeping, never worth a tick
                continue
        for name in self.config.projects:
            with contextlib.suppress(Exception):
                if not self._exclusive.get(name):
                    await self.workspace.prune_worktrees(name)
        return events

    async def _tick_agent(self, agent: Agent, task: TaskSpec, pane) -> list[Event]:
        """One agent's pass, so a failure in it stays in it."""
        if pane is None or not pane.exists:
            # The window vanished under a running task: the restart case,
            # reachable at runtime too if someone closes the window.
            return await self._handle_missing_window(agent, task)
        if pane.dead:
            return [await self._finalize(agent, task, self._exit_code(agent, task, pane))]
        return await self._check_health(agent, task)

    #: A process killed by signal N exits, by shell convention, with 128 + N.
    SIGNAL_EXIT_BASE = 128

    def _exit_code(self, agent: Agent, task: TaskSpec, pane: PaneStatus) -> int | None:
        """How a dead pane's attempt ended.

        tmux's own status first. It has none for a process killed by a signal
        - measured on Linux, where a finished task was recorded as an error
        with exit code -1 - so next the wrapper's `__BUDDY_DONE__` line, which
        carries the harness's exit code and exists for exactly this, and last
        the signal itself, as a shell would report it.
        """
        if pane.exit_code is not None:
            return pane.exit_code
        run = self.store.get_run(task.id, agent.run_attempt)
        log_path = run.log_path if run else self.config.paths.log_file(task.id, task.attempt)
        found = read_sentinels(log_path, task.id)
        if found.finished and found.exit_code is not None:
            return found.exit_code
        if pane.signal is not None:
            return self.SIGNAL_EXIT_BASE + pane.signal
        return None

    async def _finalize(self, agent: Agent, task: TaskSpec, exit_code: int | None) -> Event:
        run = self.store.get_run(task.id, agent.run_attempt)
        if run is None:
            self.store.set_task_state(task.id, TaskState.ERROR)
            await self._retire(agent)
            return TaskFinished(
                at=self.now(),
                task_id=task.id,
                agent=agent.name,
                outcome=RunOutcome.ERROR,
                exit_code=exit_code,
                summary="no run was recorded for this attempt",
            )
        event = await finalize_run(
            config=self.config,
            store=self.store,
            workspace=self.workspace,
            runner=self.runner,
            adapter=self.adapter_for(task.harness),
            task=task,
            run=run,
            exit_code=exit_code if exit_code is not None else -1,
        )
        await self._retire(agent)
        return event

    # -- health -----------------------------------------------------

    async def _check_health(self, agent: Agent, task: TaskSpec) -> list[Event]:
        now = self.now()
        run = self.store.get_run(task.id, agent.run_attempt)
        log_path = run.log_path if run else self.config.paths.log_file(task.id, task.attempt)

        adapter = self.adapter_for(task.harness)

        # `last_output_at` tracks the log file growing - with what grew
        # offered to the adapter first, because a harness retrying a dead API
        # forever grows its log the whole time and is going nowhere.
        size = log_path.stat().st_size if log_path.exists() else 0
        seen = self._log_sizes.get(agent.name, -1)
        if size != seen:
            self._log_sizes[agent.name] = size
            if _made_progress(adapter, log_path, seen, size):
                agent.last_output_at = now

        previous = agent.status
        # Unstripped: the adapter decides what its own output means, and a
        # harness that emits JSON needs the raw lines to parse.
        tail_text = read_tail(log_path, lines=80)
        waiting = any(
            pattern.search(tail_text) for pattern in getattr(adapter, "waiting_patterns", [])
        )

        started = agent.started_at or (run.started_at if run else None)
        if started and now - started > task.max_runtime:
            return await self._handle_timeout(agent, task)

        if waiting:
            agent.status = AgentStatus.WAITING_INPUT
        elif agent.last_output_at and now - agent.last_output_at > task.stall_timeout:
            # Notified, never killed: a long compile looks exactly like a hang,
            # so the decision is the user's.
            agent.status = AgentStatus.STALLED
        else:
            agent.status = AgentStatus.RUNNING

        agent.last_output = describe_activity(adapter, tail_text, 20)
        self.store.save_agent(agent)

        if agent.status is previous:
            return []
        detail = {
            AgentStatus.WAITING_INPUT: "the harness is asking for input; attach to answer it",
            AgentStatus.STALLED: f"no progress for {task.stall_timeout}",
            AgentStatus.RUNNING: "producing output again",
        }[agent.status]
        return [
            AgentHealthChanged(
                at=now, agent=agent.name, task_id=task.id, status=agent.status, detail=detail
            )
        ]

    async def _handle_timeout(self, agent: Agent, task: TaskSpec) -> list[Event]:
        """Kill, checkpoint, requeue once; a second timeout is an error."""
        already_timed_out = any(
            run.outcome is RunOutcome.TIMEOUT for run in self.store.runs_for(task.id)
        )
        if not already_timed_out:
            return await self._stop_and_requeue(agent, task, RunOutcome.TIMEOUT)
        run = self.store.get_run(task.id, agent.run_attempt)
        await self._kill(agent.name, task.id)
        if run is not None:
            run.wip_commit = await self.workspace.checkpoint(run, RunOutcome.TIMEOUT)
            run.ended_at = self.now()
            run.outcome = RunOutcome.TIMEOUT
        with self.store.transaction():
            if run is not None:
                self.store.save_run(run)
            self.store.set_task_state(task.id, TaskState.ERROR)
            self.store.remove_agent(agent.name)
        self._log_sizes.pop(agent.name, None)
        return [
            TaskFinished(
                at=self.now(),
                task_id=task.id,
                agent=agent.name,
                attempt=task.attempt,
                outcome=RunOutcome.TIMEOUT,
                summary=f"timed out twice after {task.max_runtime}; not retried again",
            )
        ]

    async def _handle_missing_window(self, agent: Agent, task: TaskSpec) -> list[Event]:
        """The window is gone. The log outlives it, so it decides."""
        run = self.store.get_run(task.id, agent.run_attempt)
        log_path = run.log_path if run else self.config.paths.log_file(task.id, task.attempt)
        found = read_sentinels(log_path, task.id)
        if found.finished and found.exit_code is not None:
            # It did finish; Buddy just missed the moment.
            return [await self._finalize(agent, task, found.exit_code)]
        return await self._stop_and_requeue(agent, task, RunOutcome.INTERRUPTED)

    # -- reconciliation --------------------------------------------

    async def reconcile(self) -> list[Event]:
        """Run on every startup, before accepting input.

        Nothing is ever left in an unknown state, because the log file and the
        worktree survive everything short of disk loss.
        """
        panes = await self.runner.panes()
        await self.refresh_project_locks()
        events: list[Event] = []

        for agent in self.agents():
            task = self.store.get_task(agent.task_id)
            if task is None:
                await self._retire(agent)
                continue
            pane = panes.get(agent.name)
            run = self.store.get_run(task.id, agent.run_attempt)

            if run is not None and run.outcome is not None:
                # This attempt already ended, and a crash came before its
                # agent was retired. Whatever the window holds now - a dead
                # pane, or nothing - is not the task.
                events.extend(await self._recover_ended(agent, task, run))
            elif pane is None or not pane.exists:
                events.extend(await self._handle_missing_window(agent, task))
            elif pane.dead:
                events.append(await self._finalize(agent, task, self._exit_code(agent, task, pane)))
            else:
                # Alive: resume monitoring, and make sure the log pipe is open
                # so the next tick's stall detection has something to read.
                if run is not None and not pane.piped:
                    await self.runner.open_pipe(agent.name, run.log_path)
                if run is not None and run.log_path.exists():
                    self._log_sizes[agent.name] = run.log_path.stat().st_size

        # Any run left with no outcome that no agent claims was interrupted
        # between the pane ending and the row being written.
        events.extend(self._orphaned_runs())
        return events

    async def _recover_ended(self, agent: Agent, task: TaskSpec, run: TaskRun) -> list[Event]:
        """Finish the bookkeeping a crash interrupted.

        The work itself is safe either way: whatever the attempt produced was
        already checkpointed onto its branch before the outcome was written.
        What is left is retiring the agent and, if the task still claims to
        be running, deciding what it does next - which is what the outcome
        says: resumable outcomes go back on the queue, a kill stays killed,
        and a finished run keeps the verdict it had.
        """
        await self._retire(agent)
        if self.store.get_task_state(task.id) is not TaskState.RUNNING:
            return []
        outcome = run.outcome or RunOutcome.INTERRUPTED
        if outcome.is_resumable:
            task.attempt = max(task.attempt, run.attempt) + 1
            with self.store.transaction():
                self.store.update_task(task)
                self.store.set_task_state(task.id, TaskState.QUEUED)
            return [
                TaskRequeued(
                    at=self.now(), task_id=task.id, next_attempt=task.attempt, reason=outcome
                )
            ]
        verdict = {
            RunOutcome.DONE: TaskState.DONE,
            RunOutcome.KILLED: TaskState.KILLED,
        }.get(outcome, TaskState.ERROR)
        self.store.set_task_state(task.id, verdict)
        return []

    def _orphaned_runs(self) -> list[Event]:
        claimed = {(agent.task_id, agent.run_attempt) for agent in self.agents()}
        events: list[Event] = []
        for run in self.store.unfinished_runs():
            if (run.task_id, run.attempt) in claimed:
                continue
            run.outcome = RunOutcome.INTERRUPTED
            run.ended_at = self.now()
            self.store.save_run(run)
            state = self.store.get_task_state(run.task_id)
            if state is TaskState.RUNNING:
                self.store.set_task_state(run.task_id, TaskState.QUEUED)
            events.append(
                TaskRequeued(
                    at=self.now(),
                    task_id=run.task_id,
                    next_attempt=run.attempt + 1,
                    reason=RunOutcome.INTERRUPTED,
                )
            )
        return events

    # -- tools exposed to the brain --------------------------------

    async def shutdown(self, *, remove_worktrees: bool = True) -> ShutdownReport:
        """Stop everything and clear the working state, keeping the work.

        The distinction that makes this safe: the **branch** carries
        the work and the **worktree** is a disposable checkout of it. So every
        running agent is checkpointed onto its branch before anything is torn
        down, and what is removed afterwards can be rebuilt from what is kept.

        What survives a shutdown:

        - every branch, so `buddy diff` and `buddy merge` still work - neither
          touches a worktree, both operate on the project's own checkout;
        - `state.db` entire: pinned memory, the conversation and its FTS
          index, every task and every run;
        - every attempt's log.

        What goes: the running processes, their windows, and the worktrees. A
        task that was running is checkpointed and requeued under its name
        rather than killed, so it picks up from its last commit the next time
        Buddy starts - `kill` means kill, and this is not that.
        """
        report = ShutdownReport()

        for agent in self.agents():
            task = self.store.get_task(agent.task_id)
            if task is None:
                await self._retire(agent)
                continue
            try:
                await self._stop_and_requeue(agent, task, RunOutcome.INTERRUPTED)
                report.stopped.append(f"{agent.name} ({task.id})")
            except Exception as exc:  # noqa: BLE001 - one agent, not the shutdown
                report.failed.append((task.id, f"could not stop: {exc}"))

        if not remove_worktrees:
            return report

        for task in self.store.recent_tasks(limit=1000):
            worktree = self.config.worktree_path(task.project, task.id)
            if not worktree.exists():
                continue
            try:
                await self.workspace.remove_worktree(task)
                report.removed.append(task.id)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                report.failed.append((task.id, f"could not remove worktree: {exc}"))

        for project in sorted(self.config.projects):
            with contextlib.suppress(Exception):
                await self.workspace.prune_worktrees(project)

        report.unmerged = await self._unmerged_branches()
        return report

    async def _unmerged_branches(self) -> list[tuple[str, str]]:
        """Branches still holding work nothing else has, so it can be said."""
        held: list[tuple[str, str]] = []
        settled = (TaskState.MERGED, TaskState.DISCARDED)
        for task in self.store.recent_tasks(limit=1000):
            if self.store.get_task_state(task.id) in settled:
                continue
            if self.store.latest_run(task.id) is None:
                continue
            try:
                if (await self.workspace.diff_stat(task)).strip():
                    held.append((task.id, branch_name(task)))
            except Exception:  # noqa: BLE001 - a branch that cannot be diffed is not news
                continue
        return held

    def list_agents(self) -> list[dict]:
        """Every running agent, most urgent first, each with a two-line tail."""
        rows = []
        for agent in self.agents():
            task = self.store.get_task(agent.task_id)
            run = self.store.get_run(agent.task_id, agent.run_attempt)
            rows.append(
                {
                    "agent": agent.name,
                    "status": agent.status.value,
                    "task_id": agent.task_id,
                    "title": task.title if task else None,
                    "harness": agent.harness,
                    "priority": agent.priority,
                    "branch": run.branch if run else None,
                    "age_seconds": (
                        int((self.now() - agent.started_at).total_seconds())
                        if agent.started_at
                        else None
                    ),
                    "tail": tail(agent.last_output, 2),
                }
            )
        return rows

    def get_output(self, target: str, lines: int = 60) -> str:
        """Tail of the log, ANSI stripped, for an agent's name or a task id."""
        task = self.store.find_task(target)
        run = self.store.latest_run(task.id) if task else None
        return read_tail(run.log_path, lines) if run else ""

    def still_working(self, task_id: str) -> str | None:
        """Why a task's branch cannot be merged or discarded yet, or None.

        Both remove the task's worktree. With an agent still running in it,
        that deleted the agent's uncommitted work out from under it - while
        the agent kept running, and billing, in a directory that no longer
        existed - and a merge took whatever half of the work happened to be
        committed at that moment.
        """
        if self.store.get_task_state(task_id) is not TaskState.RUNNING:
            return None
        agent = next((a for a in self.agents() if a.task_id == task_id), None)
        if agent is None:
            return f"{task_id} is still running. Let it finish first."
        return (
            f"{task_id} is still running as {agent.name}, which is working in the worktree. "
            f"Let it finish, or stop it first with `buddy kill {agent.name}`."
        )

    def reprioritize(self, task_id: str, new_priority: int) -> None:
        """Reorders the queue. On a running task it only changes its rank for
        future preemption decisions. No confirmation."""
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(f"no such task: {task_id}")
        task.priority = new_priority
        self.store.update_task(task)
        for agent in self.agents():
            if agent.task_id == task_id:
                agent.priority = new_priority
                self.store.save_agent(agent)

    async def kill_agent(self, name: str) -> list[Event]:
        """Kill and checkpoint. The task is marked killed, not requeued: that
        is what kill means."""
        agent = self.agent(name)
        if agent is None:
            raise KeyError(f"no running agent is called {name}")
        task = self.store.get_task(agent.task_id)
        run = self.store.get_run(agent.task_id, agent.run_attempt)
        await self._kill(agent.name, agent.task_id)
        events: list[Event] = []
        if run is not None:
            run.wip_commit = await self.workspace.checkpoint(run, RunOutcome.KILLED)
            run.ended_at = self.now()
            run.outcome = RunOutcome.KILLED
        with self.store.transaction():
            if run is not None:
                self.store.save_run(run)
            if task is not None:
                self.store.set_task_state(task.id, TaskState.KILLED)
            self.store.remove_agent(agent.name)
        self._log_sizes.pop(agent.name, None)
        if task is not None:
            events.append(
                TaskFinished(
                    at=self.now(),
                    task_id=task.id,
                    agent=agent.name,
                    attempt=run.attempt if run else task.attempt,
                    outcome=RunOutcome.KILLED,
                    branch=run.branch if run else "",
                    summary="killed",
                )
            )
        events.extend(await self.schedule())
        return events
