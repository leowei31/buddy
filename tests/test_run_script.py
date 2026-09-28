"""The generated wrapper and the prompt it delivers.

The scripts are executed, not just matched against, because the point of a
generated script is that what runs is exactly what the file says.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from buddy.config import Config, EnvSecretResolver, HarnessConfig
from buddy.logs import DONE_SENTINEL, START_SENTINEL, read_sentinels
from buddy.manager import (
    WORKTREE_MISSING_EXIT,
    compose_prompt,
    prepare_run,
    remove_run_script,
    resolve_env,
    script_is_private,
    wrap_in_sandbox,
    write_prompt,
    write_run_script,
)
from buddy.models import TaskRun, TaskSpec


def make_task(**overrides) -> TaskSpec:
    defaults = {
        "id": "t-0142",
        "title": "Fix onboarding flow",
        "brief": "# Task: Fix onboarding flow\n## Goal\nMake it work.",
        "harness": "claude_code",
        "project": "webapp",
        "priority": 2,
    }
    return TaskSpec(**(defaults | overrides))


def make_run(tmp_path: Path, attempt: int = 1, branch: str = "buddy/t-0142-fix") -> TaskRun:
    worktree = tmp_path / "wt"
    worktree.mkdir(exist_ok=True)
    return TaskRun(
        task_id="t-0142",
        attempt=attempt,
        agent="scout",
        worktree=worktree,
        branch=branch,
        base_ref="abc1234",
        log_path=tmp_path / "tasks" / "t-0142" / f"attempt-{attempt}.log",
    )


def run_script(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(path)], capture_output=True, text=True)


# -- the prompt -----------------------------------------------------


def test_the_brief_is_sent_with_buddys_working_rules_appended():
    task, run = make_task(), make_run(Path("/tmp"))
    prompt = compose_prompt(task, run)
    assert prompt.startswith("# Task: Fix onboarding flow")
    assert "## Working rules" in prompt
    assert "buddy/t-0142-fix" in prompt
    assert "Do not push" in prompt


def test_a_first_attempt_carries_no_resume_note():
    assert "previous attempt" not in compose_prompt(make_task(), make_run(Path("/tmp")))


def test_a_retry_is_told_to_read_its_own_partial_work(tmp_path: Path):
    """A retry picks up the partial work rather than starting over."""
    prompt = compose_prompt(make_task(), make_run(tmp_path, attempt=2))
    assert "partial work from a previous attempt" in prompt
    assert "git diff abc1234" in prompt


def test_a_non_git_project_gets_rules_that_do_not_mention_a_branch(tmp_path: Path):
    prompt = compose_prompt(make_task(), make_run(tmp_path, branch=""))
    assert "not a git repository" in prompt
    assert "switch branches" not in prompt


def test_write_prompt_creates_the_task_directory(tmp_path: Path):
    path = write_prompt(
        tmp_path / "tasks" / "t-0142" / "prompt.md", make_task(), make_run(tmp_path)
    )
    assert path.exists()
    assert "Working rules" in path.read_text()


# -- the wrapper ----------------------------------------------------


def test_a_successful_run_reports_its_sentinels_and_exit_code(tmp_path: Path):
    run = make_run(tmp_path)
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=run,
        prompt_path=tmp_path / "prompt.md",
        invocation="echo 'the harness ran'",
    )
    result = run_script(script)

    assert result.returncode == 0
    assert f"{START_SENTINEL} t-0142 attempt=1" in result.stdout
    assert f"{DONE_SENTINEL} t-0142 0" in result.stdout
    assert "the harness ran" in result.stdout


@pytest.mark.parametrize("exit_code", [0, 1, 42])
def test_the_harness_exit_code_is_the_scripts_exit_code(tmp_path: Path, exit_code: int):
    """This is what `pane_dead_status` reports."""
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=make_run(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        invocation=f"bash -c 'exit {exit_code}'",
    )
    result = run_script(script)
    assert result.returncode == exit_code
    assert f"{DONE_SENTINEL} t-0142 {exit_code}" in result.stdout


def test_the_script_runs_in_the_worktree(tmp_path: Path):
    run = make_run(tmp_path)
    (run.worktree / "marker.txt").write_text("here")
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=run,
        prompt_path=tmp_path / "prompt.md",
        invocation="cat marker.txt",
    )
    assert "here" in run_script(script).stdout


def test_a_missing_worktree_fails_loudly_without_running_the_harness(tmp_path: Path):
    run = make_run(tmp_path)
    run.worktree.rmdir()
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=run,
        prompt_path=tmp_path / "prompt.md",
        invocation="echo SHOULD_NOT_RUN",
    )
    result = run_script(script)
    assert result.returncode == WORKTREE_MISSING_EXIT
    assert "SHOULD_NOT_RUN" not in result.stdout
    assert f"{DONE_SENTINEL} t-0142 {WORKTREE_MISSING_EXIT}" in result.stdout


def test_the_prompt_never_passes_through_a_quoting_layer(tmp_path: Path):
    """A4: the brief reaches the harness as a file, whatever is in it."""
    nasty = 'don\'t `rm -rf /` $(whoami) "quoted" \\backslash\n## Goal\nsurvive'
    prompt_path = tmp_path / "prompt.md"
    write_prompt(prompt_path, make_task(brief=nasty), make_run(tmp_path))
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(brief=nasty),
        run=make_run(tmp_path),
        prompt_path=prompt_path,
        invocation='cat "${PROMPT}"',
    )
    result = run_script(script)
    assert result.returncode == 0
    assert "rm -rf /" in result.stdout
    assert "$(whoami)" in result.stdout


def test_a_path_with_spaces_survives(tmp_path: Path):
    odd = tmp_path / "my code" / "wt"
    odd.mkdir(parents=True)
    run = make_run(tmp_path)
    run = TaskRun(**{**run.__dict__, "worktree": odd})
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=run,
        prompt_path=tmp_path / "a prompt.md",
        invocation="pwd",
    )
    assert str(odd) in run_script(script).stdout


def test_the_log_is_self_describing_after_the_fact(tmp_path: Path):
    """The sentinels are the secondary signal, and reconciliation reads them back."""
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=make_run(tmp_path, attempt=3),
        prompt_path=tmp_path / "prompt.md",
        invocation="bash -c 'exit 7'",
    )
    log = tmp_path / "attempt-3.log"
    log.write_text(run_script(script).stdout)

    found = read_sentinels(log, "t-0142")
    assert found.started and found.finished
    assert found.attempt == 3
    assert found.exit_code == 7
    assert read_sentinels(log, "t-0999").started is False


# -- environment and secrets ----------------------------------


def test_env_is_exported_and_secrets_resolved(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    harness = HarnessConfig.from_dict(
        "codex", {"command": "codex", "env": {"OPENAI_API_KEY": "${keychain:OPENAI_API_KEY}"}}
    )
    env = resolve_env(harness, EnvSecretResolver())
    assert env == {"OPENAI_API_KEY": "sk-secret-value"}

    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=make_run(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        invocation='echo "key=${OPENAI_API_KEY}"',
        env=env,
    )
    assert "key=sk-secret-value" in run_script(script).stdout


def test_run_script_is_private_because_it_holds_resolved_secrets(tmp_path: Path):
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=make_run(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        invocation="true",
        env={"SECRET": "value"},
    )
    assert script_is_private(script)


def test_the_run_script_is_never_readable_even_for_a_moment(tmp_path: Path, monkeypatch):
    """Created 0600, not written under the umask and restricted afterwards -
    and a script left by an earlier attempt, whatever its mode, is replaced
    rather than rewritten in place."""
    import os
    import stat

    stale = tmp_path / "run.sh"
    stale.write_text("from an earlier attempt\n")
    stale.chmod(0o644)
    opened: list[int] = []
    real_open = os.open

    def watching(path, flags, mode=0o777, *args, **kwargs):
        if str(path) == str(stale):
            opened.append(mode)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", watching)
    previous = os.umask(0)
    try:
        write_run_script(
            stale,
            task=make_task(),
            run=make_run(tmp_path),
            prompt_path=tmp_path / "prompt.md",
            invocation="true",
            env={"SECRET": "value"},
        )
    finally:
        os.umask(previous)

    assert opened == [stat.S_IRUSR | stat.S_IWUSR]
    assert script_is_private(stale)
    assert "from an earlier attempt" not in stale.read_text()


def test_a_sandbox_mounts_the_projects_git_directory_not_what_the_worktree_names(
    tmp_path: Path,
):
    """A worktree's `.git` is a file inside the mount. Reproduced: one agent
    rewrote it to name `~/.ssh`, and the retry's `docker run` mounted that
    directory read-write. The git directory comes from the project's own
    checkout now, which the sandbox never mounts."""
    from buddy.sandbox import DEFAULT_SANDBOX_COMMAND

    repo = tmp_path / "webapp"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    worktree = tmp_path / "worktrees" / "t-0001"
    worktree.mkdir(parents=True)
    precious = tmp_path / "home" / ".ssh"
    precious.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {precious}\n")
    harness = HarnessConfig.from_dict(
        "claude_code",
        {"command": "claude -p", "sandbox": "docker", "sandbox_command": DEFAULT_SANDBOX_COMMAND},
    )

    wrapped = wrap_in_sandbox(
        harness, "claude -p", worktree, prompt_path=tmp_path / "p.md", repo=repo
    )

    assert str(precious) not in wrapped
    git_dir = repo / ".git"
    assert f"-v {git_dir}:{git_dir} " in wrapped
    assert f"-v {git_dir / 'config'}:{git_dir / 'config'}:ro" in wrapped


def test_a_sandbox_quotes_the_paths_it_mounts(tmp_path: Path):
    """A home directory with a space in it split `-v` in two."""
    from buddy.sandbox import DEFAULT_SANDBOX_COMMAND

    worktree = tmp_path / "Jane Doe" / "worktrees" / "t-0001"
    prompt = tmp_path / "Jane Doe" / "tasks" / "t-0001" / "prompt.md"
    harness = HarnessConfig.from_dict(
        "claude_code",
        {"command": "claude -p", "sandbox": "docker", "sandbox_command": DEFAULT_SANDBOX_COMMAND},
    )

    wrapped = wrap_in_sandbox(harness, "claude -p", worktree, prompt_path=prompt)

    words = shlex.split(wrapped)
    assert f"{worktree}:{worktree}" in words
    assert f"{prompt}:{prompt}:ro" in words


def test_the_script_is_deleted_once_the_result_is_written(tmp_path: Path):
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=make_run(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        invocation="true",
    )
    remove_run_script(script)
    assert not script.exists()
    remove_run_script(script)  # idempotent


def test_an_env_value_with_shell_metacharacters_is_quoted(tmp_path: Path):
    script = write_run_script(
        tmp_path / "run.sh",
        task=make_task(),
        run=make_run(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        invocation='echo "[${TRICKY}]"',
        env={"TRICKY": "a b; echo pwned $(id)"},
    )
    result = run_script(script)
    assert "[a b; echo pwned $(id)]" in result.stdout
    assert "pwned" in result.stdout and "uid=" not in result.stdout


# -- sandbox ---------------------------------------------------


def test_the_sandbox_mounts_the_brief_as_well_as_the_worktree(tmp_path: Path):
    """Mounting only the worktree cannot work.

    The brief lives at `~/.buddy/tasks/<id>/prompt.md`, outside the worktree,
    and every harness invocation reads it - so a container with only the
    worktree mounted fails on the first redirect. It is mounted read-only,
    and it is the only other thing that is.
    """
    from buddy.sandbox import DEFAULT_SANDBOX_COMMAND

    worktree = tmp_path / "worktrees" / "t-0001"
    prompt = tmp_path / "tasks" / "t-0001" / "prompt.md"
    harness = HarnessConfig.from_dict(
        "claude_code",
        {"command": "claude -p", "sandbox": "docker", "sandbox_command": DEFAULT_SANDBOX_COMMAND},
    )

    wrapped = wrap_in_sandbox(harness, "claude -p < prompt", worktree, prompt_path=prompt)

    assert f"-v {worktree}:{worktree}" in wrapped
    assert f"-v {prompt}:{prompt}:ro" in wrapped, "the brief is unreadable without this"
    assert f"-w {worktree}" in wrapped
    # One shell-quoted argument to a shell inside the container: bare words
    # leave nothing there to read a redirect or a flag.
    assert wrapped.endswith("--entrypoint sh buddy-harness:latest -c 'claude -p < prompt'")
    # Nothing else from the host: not $HOME, not the base repo, not ~/.buddy.
    mounts = [part for part in wrapped.split() if part.startswith("-v")]
    assert len(mounts) == 2


def test_sandbox_wraps_the_invocation_when_configured(tmp_path: Path):
    harness = HarnessConfig.from_dict(
        "claude_code",
        {
            "command": "claude -p",
            "sandbox": "docker",
            "sandbox_command": (
                "docker run --rm -v {worktree}:{worktree} -w {worktree} img {command}"
            ),
        },
    )
    wrapped = wrap_in_sandbox(harness, "claude -p", tmp_path)
    assert wrapped.startswith("docker run")
    assert "claude -p" in wrapped


def test_no_sandbox_leaves_the_invocation_alone(tmp_path: Path):
    harness = HarnessConfig.from_dict("claude_code", {"command": "claude -p"})
    assert wrap_in_sandbox(harness, "claude -p", tmp_path) == "claude -p"


# -- prepare_run end to end ------------------------------------------------


def test_prepare_run_writes_both_files(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path}"\n'
        "[harness.claude_code]\n"
        'command = "claude -p {model_flag} < {prompt_path}"\n'
    )
    config = Config.load(home=tmp_path)
    from buddy.harnesses.claude_code import ClaudeCodeAdapter

    adapter = ClaudeCodeAdapter(config.harness("claude_code"))
    task = make_task(model="claude-sonnet-5")
    run = make_run(tmp_path)

    prompt_path, script_path = prepare_run(
        config, task, run, invocation_for=adapter, resolver=EnvSecretResolver()
    )

    assert prompt_path == config.paths.prompt_file("t-0142")
    assert script_path == config.paths.run_script("t-0142")
    assert "Working rules" in prompt_path.read_text()
    body = script_path.read_text()
    assert "--model claude-sonnet-5" in body
    assert str(prompt_path) in body
    assert script_is_private(script_path)


def test_a_sandboxed_run_can_actually_read_its_brief(tmp_path: Path):
    """The bug this caught: `prepare_run` wrapped without the prompt path, so
    every sandboxed run would have died at the redirect it could not open."""
    from buddy.harnesses.claude_code import ClaudeCodeAdapter
    from buddy.sandbox import DEFAULT_SANDBOX_COMMAND

    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path}"\n'
        "[harness.claude_code]\n"
        'command = "claude -p < {prompt_path}"\n'
        'sandbox = "docker"\n'
        f"sandbox_command = '{DEFAULT_SANDBOX_COMMAND}'\n"
    )
    config = Config.load(home=tmp_path)
    task = make_task()
    run = make_run(tmp_path)

    prompt_path, script_path = prepare_run(
        config,
        task,
        run,
        invocation_for=ClaudeCodeAdapter(config.harness("claude_code")),
        resolver=EnvSecretResolver(),
    )

    body = script_path.read_text()
    assert "docker run --rm " in body
    assert f"-v {prompt_path}:{prompt_path}:ro" in body
    assert f"-v {run.worktree}:{run.worktree}" in body


def test_a_model_name_cannot_smuggle_a_shell_command(tmp_path: Path):
    """`model` reaches the adapter from `spawn_agent`, which means an LLM
    chose it, and the result is spliced into a run.sh that bash executes.
    Unquoted, `sonnet; rm -rf ~; echo` was a working command line - so this
    runs the thing through bash and checks the side effect never happens.
    """
    from buddy.harnesses.claude_code import ClaudeCodeAdapter

    marker = tmp_path / "pwned"
    harness = HarnessConfig.from_dict("claude_code", {"command": "echo {model_flag}"})
    hostile = f"sonnet; touch {marker}; echo"
    adapter = ClaudeCodeAdapter(harness)
    line = adapter.invocation(make_run(tmp_path), tmp_path / "p.md", model=hostile)

    result = subprocess.run(["bash", "-c", line], capture_output=True, text=True)

    assert not marker.exists(), f"the injected command ran: {line}"
    # It arrived as one argument, verbatim, which is what quoting is for.
    assert hostile in result.stdout
    assert shlex.split(line)[-1] == hostile


def test_an_ordinary_model_name_is_left_alone(tmp_path: Path):
    from buddy.harnesses.claude_code import ClaudeCodeAdapter

    harness = HarnessConfig.from_dict(
        "claude_code", {"command": "claude -p {model_flag} < {prompt_path}"}
    )
    line = ClaudeCodeAdapter(harness).invocation(
        make_run(tmp_path), tmp_path / "p.md", model="claude-sonnet-5"
    )
    assert "--model claude-sonnet-5" in line


def test_literal_braces_in_a_command_are_left_alone(tmp_path: Path):
    """Found by running Codex for real through `buddy spawn`.

    The template was expanded with `str.format`, so any brace that was not a
    placeholder raised KeyError - and Codex's own `-c key={...}` syntax, an
    `awk '{print $1}'` or a JSON argument are all ordinary command lines. It
    raised inside the scheduler after the worktree existed, so the task
    stayed queued and every tick failed on it.
    """
    from buddy.harnesses.codex import CodexAdapter

    command = (
        'codex exec -c \'model_providers.x={name="x",base_url="http://h/v1"}\' '
        "{model_flag} - < {prompt_path} | awk '{print $1}'"
    )
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path}"\n'
        f"[harness.codex]\ncommand = {json.dumps(command)}\n"
    )
    config = Config.load(home=tmp_path)
    adapter = CodexAdapter(config.harness("codex"))
    rendered = adapter.invocation(make_run(tmp_path), Path("/p/prompt.md"), model="gpt-5")

    assert 'model_providers.x={name="x",base_url="http://h/v1"}' in rendered
    assert "awk '{print $1}'" in rendered
    assert "--model gpt-5 - < /p/prompt.md" in rendered


def test_a_sandbox_command_with_literal_braces_still_wraps(tmp_path: Path):
    from buddy.manager import wrap_in_sandbox

    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path}"\n'
        "[harness.claude_code]\n"
        'command = "claude -p < {prompt_path}"\n'
        'sandbox = "docker"\n'
        'sandbox_command = "docker run --label x=\'{\\"a\\":1}\' {image} {command}"\n'
    )
    config = Config.load(home=tmp_path)
    wrapped = wrap_in_sandbox(config.harness("claude_code"), "claude -p", tmp_path / "w")
    assert wrapped.endswith("buddy-harness:latest 'claude -p'")
    assert '{"a":1}' in wrapped
