"""When a merge conflicts: an agent to resolve it, and a hold until it has.

A merge that conflicts is never resolved automatically - `Workspace.merge`
aborts it and leaves your checkout exactly as it was. What this adds is the
next step: say "have someone fix it", and a task is spawned to do exactly
that.

Three decisions shape it:

* **The fix starts on the branch that conflicted**, not on the base. Its job
  is to merge the base *into* that work and resolve what collides, so when
  the fix is merged, the original work comes with it and the original task
  counts as merged too.
* **Merge, never rebase.** Rebasing rewrites the agent's commits, which the
  run history, the checkpoints and any review so far all point at. A merge
  commit keeps every one of them and records the resolution on its own.
* **Nothing else merges into that project while the fix is pending.** Every
  other merge moves the base again, and a fix made against an older base can
  conflict all over again when it lands.
"""

from __future__ import annotations

from buddy.config import Config
from buddy.models import TaskSpec, TaskState
from buddy.state import Store
from buddy.workspace import branch_name

#: States in which a conflict fix still holds its project's merges.
PENDING = (TaskState.QUEUED, TaskState.RUNNING, TaskState.DONE)

BRIEF = """\
# Task: Resolve merge conflicts for {task_id}
## Goal
The branch `{branch}` ({title}) no longer merges cleanly into `{base}`.
Bring `{base}` into this branch and resolve every conflict, so that the work
of {task_id} is preserved *and* works with everything that has landed on
`{base}` since it started.
## Context
You are in a worktree of a new branch that starts at the tip of `{branch}`.
The original brief for {task_id}, for its intent:

{original}
## Constraints
- Run `git merge {base}` in this worktree, then resolve the conflicts.
- Merge only. Do not rebase, amend, reset or force-push: the existing commits
  are referenced by the run history and must survive unchanged.
- Where both sides changed the same thing, keep the intent of both. If they
  genuinely cannot coexist, keep what `{base}` has, re-apply {task_id}'s
  intent on top of it, and say so in the merge commit message.
- Leave no conflict markers anywhere.
## Definition of done
- `git merge-base --is-ancestor {base} HEAD` succeeds.
- `git grep -nE '^(<<<<<<<|>>>>>>>)'` finds nothing.
- The project's own build and tests pass.
- The merge is committed, with a message naming what was resolved and how."""


def pending_fix(store: Store, task: TaskSpec) -> TaskSpec | None:
    """An unmerged conflict fix for this task, if one already exists."""
    for candidate in store.tasks_in_state(*PENDING):
        if candidate.resolves == task.id:
            return candidate
    return None


def merge_hold(store: Store, task: TaskSpec) -> TaskSpec | None:
    """The conflict fix that holds merges into `task`'s project, if any.

    A fix never holds itself, or the task it is resolving - merging either of
    those is how the hold ends.
    """
    for candidate in store.tasks_in_state(*PENDING):
        if (
            candidate.resolves
            and candidate.project == task.project
            and candidate.id != task.id
            and candidate.resolves != task.id
        ):
            return candidate
    return None


def fix_task(
    config: Config,
    store: Store,
    original: TaskSpec,
    *,
    harness: str,
    task_id: str,
    utterance: str = "",
) -> TaskSpec:
    """The task that resolves `original`'s conflict. Not yet persisted."""
    base = config.project(original.project).base_branch
    branch = branch_name(original)
    return TaskSpec(
        id=task_id,
        title=f"Resolve merge conflicts for {original.id}",
        brief=BRIEF.format(
            task_id=original.id,
            branch=branch,
            title=original.title,
            base=base,
            original=original.brief.strip(),
        ),
        harness=harness,
        model=original.model,
        project=original.project,
        priority=original.priority,
        max_runtime=config.buddy.max_runtime,
        stall_timeout=config.buddy.stall_timeout,
        created_from_utterance=utterance,
        resolves=original.id,
        start_from=branch,
    )


def record_merge(store: Store, task: TaskSpec) -> list[str]:
    """Mark a merged task, and the task it resolved, as merged.

    Returns every id marked. A fix's branch contains the original's commits,
    so once the fix lands the original has landed too.
    """
    marked = [task.id]
    store.set_task_state(task.id, TaskState.MERGED)
    if task.resolves and store.get_task(task.resolves) is not None:
        store.set_task_state(task.resolves, TaskState.MERGED)
        marked.append(task.resolves)
    return marked
