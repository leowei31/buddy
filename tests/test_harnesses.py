"""The harness adapters, against what each CLI actually does.

Three layers, from cheapest to most real:

* **Recorded runs.** `fixtures/harness_logs/` holds real output from each
  binary - successes, rejected keys, a signed-out Antigravity, a Codex that
  cannot reach its API. The parsers are tested against those bytes rather
  than against a transcription of the documentation.
* **The command lines, through bash.** Each adapter's default template, with
  a stand-in binary that records its argv, fed a brief built to break
  quoting. What arrives must be the brief, verbatim, and nothing may run.
* **The binaries themselves.** Where a CLI is installed, its real `--help` is
  preflighted, and Codex and OpenCode run a real tool call - a file written
  and committed in a git worktree - against a scripted model server on
  localhost. Antigravity cannot be pointed at another server, so its success
  path is the one thing here not exercised for real.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
from pathlib import Path

import pytest

from buddy.config import HarnessConfig
from buddy.harnesses import ADAPTERS, UnknownHarness, adapter_class
from buddy.harnesses.antigravity import AntigravityAdapter
from buddy.harnesses.codex import CodexAdapter, unwrap_shell
from buddy.harnesses.opencode import OpenCodeAdapter
from buddy.models import TaskRun
from tests.fake_model_api import FakeChatCompletions, FakeResponses, Script, Step

LOGS = Path(__file__).parent / "fixtures" / "harness_logs"


def log(name: str) -> str:
    return (LOGS / name).read_text()


def adapter(kind: type, **block) -> object:
    config = {"command": kind.default_command} | block
    return kind(HarnessConfig.from_dict(kind.name, config))


def run_in(worktree: Path) -> TaskRun:
    return TaskRun(
        task_id="t-0001",
        attempt=1,
        slot="Monday",
        worktree=worktree,
        branch="buddy/t-0001-x",
        base_ref="",
        log_path=worktree / "attempt-1.log",
    )


# -- the registry ----------------------------------------------------------


def test_every_harness_the_design_names_has_an_adapter():
    assert set(ADAPTERS) == {"claude_code", "codex", "opencode", "antigravity"}


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_each_adapter_carries_what_setup_needs(name):
    kind = adapter_class(name)
    assert kind.name == name
    assert kind.binary
    assert "{prompt_path}" in kind.default_command
    assert kind.install_hint


def test_an_unknown_harness_names_the_known_ones():
    with pytest.raises(UnknownHarness, match="claude_code"):
        adapter_class("cursor")


# -- the command lines, through bash ---------------------------------------

#: Every way a brief could be mistaken for shell or for a flag.
HOSTILE_BRIEF = (
    "- starts with a dash, like a flag\n"
    "/slash at the start of a line, \"double quotes\", 'single quotes'\n"
    "$(touch {marker}) and `touch {marker}` and ${{HOME}}\n"
    "a trailing backslash \\\n"
)


@pytest.mark.parametrize("kind", [CodexAdapter, OpenCodeAdapter, AntigravityAdapter])
def test_the_default_command_delivers_the_brief_verbatim(kind, tmp_path):
    """Codex reads stdin; OpenCode and `agy` take the brief as an argument,
    because OpenCode hangs on stdin and `agy -p` takes a value. Both routes
    have to arrive as the exact bytes of the brief, and neither may run it."""
    marker = tmp_path / "pwned"
    brief = HOSTILE_BRIEF.format(marker=marker)
    prompt = tmp_path / "prompt dir" / "prompt.md"
    prompt.parent.mkdir()
    prompt.write_text(brief)

    # A stand-in binary that records exactly what it was given.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorder = bin_dir / kind.binary
    recorder.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}, open({str(tmp_path / 'seen.json')!r}, 'w'))\n"  # noqa: E501
    )
    recorder.chmod(recorder.stat().st_mode | stat.S_IXUSR)

    line = adapter(kind).invocation(run_in(tmp_path), prompt, model="some-model")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    subprocess.run(
        ["bash", "-c", line], env=env, stdin=subprocess.DEVNULL, check=True, cwd=tmp_path
    )

    seen = json.loads((tmp_path / "seen.json").read_text())
    assert not marker.exists(), f"the brief ran as shell: {line}"
    received = seen["stdin"] if kind is CodexAdapter else seen["argv"][-1]
    if kind is AntigravityAdapter:
        assert received.startswith("-p=")
        received = received[len("-p=") :]
    # `$(cat ...)` drops trailing newlines and nothing else.
    assert received.rstrip("\n") == brief.rstrip("\n")
    assert "--model" in seen["argv"]


# -- Codex -----------------------------------------------------------------


def test_codex_reads_its_own_verdict_from_a_real_success():
    result = adapter(CodexAdapter).parse_result(log("codex_success.jsonl"), 0)
    assert result.ok
    assert result.summary == "Created hello.txt and committed it."
    assert result.detail["turn_completed"] is True
    assert result.detail["commands"] == 1
    assert result.detail["agrees_with_exit_code"] is True


def test_codex_reports_a_rejected_key_as_the_failure_it_is():
    result = adapter(CodexAdapter).parse_result(log("codex_auth_failed.jsonl"), 1)
    assert not result.ok
    assert "401 Unauthorized" in result.summary
    assert result.detail["agrees_with_exit_code"] is True


def test_codex_activity_reads_as_sentences():
    lines = adapter(CodexAdapter).describe_activity(log("codex_success.jsonl")).splitlines()
    assert lines[0] == "started"
    assert any(line.startswith("$ printf 'hi from codex") for line in lines)
    assert "Created hello.txt and committed it." in lines
    assert lines[-1] == "done"
    assert not any(line.startswith("{") for line in lines)


def test_codex_retry_chatter_is_not_progress():
    codex = adapter(CodexAdapter)
    retries = log("codex_unreachable.jsonl").splitlines()[-3:]
    assert not codex.is_progress("\n".join(retries) + "\n")
    stderr = "2026-09-14T18:36:45.520031Z ERROR codex_api::endpoint: failed to connect\n"
    assert not codex.is_progress(stderr)
    work = '{"type":"item.started","item":{"type":"command_execution","command":"ls"}}\n'
    assert codex.is_progress(work)
    assert codex.is_progress("\n".join(retries) + "\n" + work)
    assert "retrying" in codex.describe_activity(log("codex_unreachable.jsonl"))


def test_codex_commands_are_shown_without_the_shell_wrapper():
    assert unwrap_shell('/bin/zsh -lc "git status"') == "git status"
    assert unwrap_shell("/bin/bash -c 'ls -la'") == "ls -la"
    assert unwrap_shell("ls") == "ls"


def test_codex_warns_when_its_own_sandbox_is_kept():
    command = "codex exec --json --sandbox workspace-write - < {prompt_path}"
    notes = adapter(CodexAdapter, command=command)._command_notes()
    assert any("network is off" in note for note in notes)


# -- OpenCode --------------------------------------------------------------


def test_opencode_reads_its_verdict_from_a_real_success():
    result = adapter(OpenCodeAdapter).parse_result(log("opencode_success.jsonl"), 0)
    assert result.ok
    assert result.summary == "Created `hello.txt` and committed it to git."
    assert result.detail["tool_calls"] == 2
    assert result.detail["finished"] is True


def test_opencode_reports_a_rejected_key():
    result = adapter(OpenCodeAdapter).parse_result(log("opencode_auth_failed.jsonl"), 1)
    assert not result.ok
    assert result.summary == "API key is invalid."


def test_opencode_work_refused_permission_is_not_success():
    """Without `--auto`, `opencode run` rejects the tool call and exits 0.
    An exit code alone would call that a finished task."""
    rejected = log("opencode_permission_rejected.jsonl")
    result = adapter(OpenCodeAdapter).parse_result(rejected, 0)
    assert not result.ok
    assert result.detail["rejected_tool_calls"] == 1
    assert "refused permission" in result.summary
    activity = adapter(OpenCodeAdapter).describe_activity(rejected)
    assert "bash: echo PERMTEST > out.txt (refused permission)" in activity


def test_opencode_activity_names_each_tool_and_what_it_touched():
    lines = adapter(OpenCodeAdapter).describe_activity(log("opencode_success.jsonl")).splitlines()
    assert lines[0] == "write: /work/repo/hello.txt"
    assert lines[1].startswith("bash: git add hello.txt")
    assert lines[-1] == "done"


def test_opencode_warns_when_the_brief_would_go_to_stdin():
    notes = adapter(OpenCodeAdapter, command="opencode run --auto < {prompt_path}")._command_notes()
    assert any("hangs forever" in note for note in notes)


# -- Antigravity -----------------------------------------------------------


def test_antigravity_reports_a_rejected_key_from_its_result_envelope():
    result = adapter(AntigravityAdapter).parse_result(log("antigravity_auth_failed.jsonl"), 1)
    assert not result.ok
    assert result.detail["status"] == "ERROR"
    assert "API key not valid" in result.summary


def test_antigravity_signed_out_is_waiting_for_a_person():
    """Signed out, `agy -p` does not fail: it prints a URL and waits for a
    pasted code. The slot has to say it needs you."""
    text = log("antigravity_signed_out.txt")
    patterns = list(AntigravityAdapter.default_waiting_patterns)
    agy = adapter(AntigravityAdapter, waiting_patterns=patterns)
    assert any(pattern.search(text) for pattern in agy.waiting_patterns)
    assert "https://" not in agy.describe_activity(text)


def test_antigravity_success_envelope_as_documented():
    """Not a recorded run - see the module docstring - so it is held to the
    documented shape, and the exit code still has the last word."""
    stream = "\n".join(
        [
            json.dumps({"event": "init", "conversation_id": "c1", "init": {"cwd": "/w"}}),
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {"state": "DONE", "step_type": "run_command"},
                }
            ),  # noqa: E501
            json.dumps(
                {
                    "event": "result",
                    "result": {"status": "SUCCESS", "response": "Committed.", "num_turns": 3},
                }
            ),  # noqa: E501
        ]
    )
    agy = adapter(AntigravityAdapter)
    assert agy.parse_result(stream, 0).ok
    assert agy.parse_result(stream, 0).summary == "Committed."
    assert not agy.parse_result(stream, 1).ok
    assert agy.describe_activity(stream).splitlines() == ["started", "run command", "done"]


def test_antigravity_warns_about_the_five_minute_print_timeout():
    command = (
        'agy --output-format stream-json --dangerously-skip-permissions -p="$(cat {prompt_path})"'
    )
    notes = adapter(AntigravityAdapter, command=command)._command_notes("--print-timeout  Timeout")
    assert any("5 minutes" in note for note in notes)
    assert not adapter(AntigravityAdapter)._command_notes("--print-timeout  Timeout")


# -- the real binaries -----------------------------------------------------


@pytest.mark.parametrize("kind", [CodexAdapter, OpenCodeAdapter, AntigravityAdapter])
async def test_preflight_against_the_installed_cli(kind):
    if shutil.which(kind.binary) is None:
        pytest.skip(f"{kind.binary} is not installed")
    report = await adapter(kind).preflight()
    assert report.ok, [(r.key, r.detail) for r in report.failures]
    assert report.notes == ()  # the default command draws no warnings


@pytest.fixture(scope="session")
def harness_home(pytestconfig, tmp_path_factory) -> Path:
    """A HOME for the harness binaries that is not yours.

    Persistent across runs where pytest's cache is on, because OpenCode
    installs its provider packages into it the first time - the only network
    this file ever needs. With the cache disabled it is per-session instead.
    """
    cache = getattr(pytestconfig, "cache", None)
    if cache is None:
        return tmp_path_factory.mktemp("harness-home")
    return Path(cache.mkdir("harness-home"))


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    """A real git worktree, because that is where Buddy runs agents and where
    a sandbox that cannot reach `.git` would fail."""
    base = tmp_path / "base"
    base.mkdir()
    git = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], cwd=base, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], cwd=base, check=True)
    path = tmp_path / "wt"
    subprocess.run(
        [*git, "worktree", "add", "-q", "-b", "buddy/t-0001-x", str(path)], cwd=base, check=True
    )
    return path


def run_harness(line: str, worktree: Path, home: Path, **extra: str) -> tuple[int, str]:
    env = dict(
        os.environ,
        HOME=str(home),
        NO_COLOR="1",
        GIT_AUTHOR_NAME="agent",
        GIT_AUTHOR_EMAIL="agent@example.com",
        GIT_COMMITTER_NAME="agent",
        GIT_COMMITTER_EMAIL="agent@example.com",
        **extra,
    )
    for leaked in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(leaked, None)
    # Its own process group, killed whole on timeout. `subprocess.run`'s
    # timeout kills only bash, and a harness under it lives on: two Codex
    # processes left that way were still retrying a dead API a day later.
    proc = subprocess.Popen(
        ["bash", "-c", line],
        cwd=worktree,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=110)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        raise AssertionError(f"the harness did not finish:\n{output[-2000:]}") from None
    return proc.returncode, output


SCRIPT = [
    Step(run="printf 'hello\\n' > HELLO.md && git add HELLO.md && git commit -qm 'Add HELLO.md'"),
    Step(say="Added HELLO.md and committed it."),
]


def committed(worktree: Path) -> list[str]:
    out = subprocess.run(
        ["git", "log", "--format=%s", "main..HEAD"], cwd=worktree, capture_output=True, text=True
    )
    return out.stdout.split("\n")[:-1]


async def test_codex_runs_a_real_tool_call_and_commits_in_a_worktree(
    worktree, harness_home, tmp_path
):
    if shutil.which("codex") is None:
        pytest.skip("codex is not installed")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Add HELLO.md and commit it.\n")
    with FakeResponses(Script(list(SCRIPT))) as server:
        codex = adapter(CodexAdapter)
        line = codex.invocation(run_in(worktree), prompt).replace(
            " - < ", f" {server.codex_flags()} - < ", 1
        )
        code, output = run_harness(line, worktree, harness_home, FAKE_RESPONSES_KEY="fake")

    assert code == 0, output[-2000:]
    assert committed(worktree) == ["Add HELLO.md"]
    result = codex.parse_result(output, code)
    assert result.ok and result.summary == "Added HELLO.md and committed it."
    assert "Process exited with code 0" in server.script.tool_outputs[0]


async def test_opencode_runs_a_real_tool_call_and_commits_in_a_worktree(
    worktree, harness_home, tmp_path
):
    if shutil.which("opencode") is None:
        pytest.skip("opencode is not installed")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("- Add HELLO.md and commit it.\n")  # a leading dash, on purpose
    with FakeChatCompletions(Script(list(SCRIPT))) as server:
        config = tmp_path / "opencode.json"
        config.write_text(server.opencode_config())
        opencode = adapter(OpenCodeAdapter, default_model="fake/fake-model")
        line = opencode.invocation(run_in(worktree), prompt)
        code, output = run_harness(line, worktree, harness_home, OPENCODE_CONFIG=str(config))

    if code != 0 and "npm" in output.lower() and not server.script.requests:
        pytest.skip("OpenCode could not install its provider package (offline first run)")
    assert code == 0, output[-2000:]
    assert committed(worktree) == ["Add HELLO.md"]
    result = opencode.parse_result(output, code)
    assert result.ok and result.summary == "Added HELLO.md and committed it."
