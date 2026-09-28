"""Install and setup: the step runner, the lock, and every step.

The last test in this file runs all twelve steps against this actual machine,
with a scripted terminal and a stubbed provider - real platform detection,
real package checks, the real Docker daemon, a real database, the real
whisper model, the real `claude` preflight, real audio devices, a real
config.toml and a real tmux server on a private socket. It is as close to a
clean-VM install as a single machine can get.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from buddy.config import Config
from buddy.setup import (
    BaseStep,
    CheckResult,
    SetupContext,
    SetupError,
    SetupLock,
    SetupOptions,
    Status,
    all_steps,
    hint_for,
    run_steps,
    step_names,
)
from buddy.setup.platform import Docker, Gpu, PlatformReport, detect
from buddy.setup.steps.config import ConfigStep, merge_missing
from buddy.setup.steps.layout import LayoutStep
from buddy.state import SCHEMA_VERSION, Store


class ScriptedUI:
    """A terminal with a script instead of a person."""

    def __init__(self, answers: dict[str, str] | None = None, *, confirm: bool = True) -> None:
        self.answers = answers or {}
        self.always = confirm
        self.said: list[str] = []
        self.asked: list[str] = []

    def say(self, message: str) -> None:
        self.said.append(message)

    detail = say
    warn = say

    def confirm(self, question: str, *, default: bool = True) -> bool:
        self.asked.append(question)
        return self.always

    def ask(self, question: str, *, default: str = "", secret: bool = False) -> str:
        self.asked.append(question)
        for key, value in self.answers.items():
            if key.lower() in question.lower():
                return value
        return default


def by_step(outcomes) -> dict:
    return {outcome.step: outcome for outcome in outcomes}


def make_ctx(tmp_path: Path, ui=None, **options) -> SetupContext:
    home = tmp_path / "home"
    ctx = SetupContext(
        home=home,
        options=SetupOptions(**options),
        ui=ui or ScriptedUI(),
        lock=SetupLock.load(home / "setup.lock"),
    )
    ctx.report = PlatformReport(
        system="linux",
        arch="x86_64",
        package_manager="apt",
        gpu=Gpu(),
        docker=Docker(installed=True, running=True, version="1.0"),
        audio_backend="alsa",
        python="3.12.0",
    )
    return ctx


# -- the runner ----------------------------------------------------


class Recording(BaseStep):
    name = "recording"
    title = "Recording"
    number = "0"

    def __init__(self, *, satisfied: bool = False, pins: dict | None = None) -> None:
        self.satisfied = satisfied
        self.pins = pins or {}
        self.acted = 0

    async def check(self, ctx):
        return CheckResult(self.satisfied, "detail", dict(self.pins))

    async def act(self, ctx):
        self.acted += 1
        self.satisfied = True


async def test_a_satisfied_step_is_skipped(tmp_path):
    step = Recording(satisfied=True)
    (outcome,) = await run_steps(make_ctx(tmp_path), [step])
    assert outcome.status is Status.SKIPPED
    assert step.acted == 0


async def test_an_unsatisfied_step_acts_then_verifies(tmp_path):
    step = Recording(satisfied=False)
    (outcome,) = await run_steps(make_ctx(tmp_path), [step])
    assert outcome.status is Status.DONE
    assert step.acted == 1


async def test_force_re_runs_a_step_that_is_already_satisfied(tmp_path):
    step = Recording(satisfied=True)
    ctx = make_ctx(tmp_path, force=frozenset({"recording"}))
    (outcome,) = await run_steps(ctx, [step])
    assert outcome.status is Status.DONE
    assert step.acted == 1


async def test_a_failure_stops_the_run_and_names_the_step(tmp_path):
    class Broken(Recording):
        name = "broken"

        async def act(self, ctx):
            raise SetupError("broken", "the thing did not thing")

    after = Recording()
    outcomes = await run_steps(make_ctx(tmp_path), [Broken(), after])
    assert [o.status for o in outcomes] == [Status.FAILED]
    assert after.acted == 0, "nothing after a failure should run"
    assert "--force broken" in hint_for("broken")


async def test_an_unexpected_exception_is_a_failure_not_a_traceback(tmp_path):
    class Exploding(Recording):
        name = "exploding"

        async def act(self, ctx):
            raise RuntimeError("something nobody predicted")

    (outcome,) = await run_steps(make_ctx(tmp_path), [Exploding()])
    assert outcome.status is Status.FAILED
    assert "RuntimeError: something nobody predicted" in outcome.detail


async def test_a_step_that_does_not_apply_says_why(tmp_path):
    class Voice(Recording):
        name = "voice"

        async def applies(self, ctx):
            return "--no-voice" if ctx.options.no_voice else ""

    step = Voice()
    (outcome,) = await run_steps(make_ctx(tmp_path, no_voice=True), [step])
    assert outcome.status is Status.NOT_APPLICABLE
    assert outcome.detail == "--no-voice"
    assert step.acted == 0


async def test_a_verify_that_fails_fails_the_step(tmp_path):
    class NeverTakes(Recording):
        name = "never"

        async def act(self, ctx):
            self.acted += 1  # acts, but check still says no

    (outcome,) = await run_steps(make_ctx(tmp_path), [NeverTakes()])
    assert outcome.status is Status.FAILED
    assert "did not take" in outcome.detail


# -- the lock ------------------------------------------------------


async def test_the_lock_records_what_each_step_settled_on(tmp_path):
    ctx = make_ctx(tmp_path)
    await run_steps(ctx, [Recording(pins={"version": "1.2.3"})])
    written = json.loads(ctx.lock.path.read_text())
    assert written["version"] == 1
    assert written["steps"]["recording"]["pins"] == {"version": "1.2.3"}
    assert written["steps"]["recording"]["status"] == "done"


async def test_update_re_runs_a_satisfied_step_whose_pins_moved(tmp_path):
    """The whole point of recording pins rather than a bare "done"."""
    ctx = make_ctx(tmp_path)
    await run_steps(ctx, [Recording(pins={"version": "1.0"})])

    same = Recording(satisfied=True, pins={"version": "1.0"})
    ctx = make_ctx(tmp_path, update=True)
    (outcome,) = await run_steps(ctx, [same])
    assert outcome.status is Status.SKIPPED and same.acted == 0

    moved = Recording(satisfied=True, pins={"version": "2.0"})
    ctx = make_ctx(tmp_path, update=True)
    (outcome,) = await run_steps(ctx, [moved])
    assert outcome.status is Status.DONE and moved.acted == 1


def test_an_unreadable_lock_is_an_empty_one_not_a_crash(tmp_path):
    path = tmp_path / "setup.lock"
    path.write_text("{ this is not json")
    assert SetupLock.load(path).steps == {}
    assert SetupLock.load(tmp_path / "absent.lock").steps == {}


# -- pasted keys -------------------------------------------


def test_a_key_pasted_more_than_once_is_refused():
    """How a 319-character "key" got into a real keychain: the prompt hides
    what you type, so a paste that did not seem to register often did, and
    the next Cmd-V concatenated."""
    from buddy.setup.steps.keys import BadKey, check_key

    one = "sk-ant-api03-" + "x" * 95
    with pytest.raises(BadKey) as refused:
        check_key("ANTHROPIC_API_KEY", one * 3)
    assert "pasted 3 times" in str(refused.value)


@pytest.mark.parametrize(
    ("value", "because"),
    [
        ("", "nothing was entered"),
        ("   ", "nothing was entered"),
        ("sk-ant-api03-with a space", "space or a newline"),
        ("hunter2", "should start with"),
    ],
)
def test_a_key_that_cannot_be_one_is_refused(value, because):
    from buddy.setup.steps.keys import BadKey, check_key

    with pytest.raises(BadKey) as refused:
        check_key("ANTHROPIC_API_KEY", value)
    assert because in str(refused.value)


def test_a_good_key_is_accepted_and_described_without_being_shown():
    from buddy.setup.steps.keys import check_key, describe_key

    key = "sk-ant-api03-" + "x" * 95
    assert check_key("ANTHROPIC_API_KEY", f"  {key}\n") == key
    described = describe_key(key)
    assert described == "108 characters, ending xxxx"
    assert key not in described, "the whole point is not to print the secret"


async def test_force_keys_replaces_a_stored_key(tmp_path, monkeypatch):
    """Otherwise there is no way to replace one through setup, which is the
    one thing you need when the stored key is the problem."""
    from buddy.setup.steps import keys as step_module

    stored = {"ANTHROPIC_API_KEY": "sk-ant-api03-" + "x" * 95}
    replacement = "sk-ant-api03-" + "y" * 95
    monkeypatch.setattr(step_module, "keychain_get", lambda name: stored.get(name))

    def remember(name, value):
        stored[name] = value

    monkeypatch.setattr(step_module, "keychain_set", remember)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    ui = ScriptedUI({"ANTHROPIC_API_KEY": replacement})
    ctx = make_ctx(tmp_path, ui=ui, no_voice=True)
    await step_module.KeysStep()._collect_api_key(ctx, "anthropic", True)
    assert stored["ANTHROPIC_API_KEY"] == "sk-ant-api03-" + "x" * 95, "asked without --force"

    forced = make_ctx(
        tmp_path,
        ui=ScriptedUI({"ANTHROPIC_API_KEY": replacement}),
        no_voice=True,
        force=frozenset({"keys"}),
    )
    await step_module.KeysStep()._collect_api_key(forced, "anthropic", True)
    assert stored["ANTHROPIC_API_KEY"] == replacement


async def test_a_malformed_stored_key_is_noticed_without_being_forced(tmp_path, monkeypatch):
    """The 319-character one would otherwise sit there being reported as
    "already in the keychain" forever."""
    from buddy.setup.steps import keys as step_module

    tripled = ("sk-ant-api03-" + "x" * 95) * 3
    stored = {"ANTHROPIC_API_KEY": tripled}
    monkeypatch.setattr(step_module, "keychain_get", lambda name: stored.get(name))

    def remember(name, value):
        stored[name] = value

    monkeypatch.setattr(step_module, "keychain_set", remember)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    good = "sk-ant-api03-" + "z" * 95
    ctx = make_ctx(tmp_path, ui=ScriptedUI({"ANTHROPIC_API_KEY": good}), no_voice=True)
    await step_module.KeysStep()._collect_api_key(ctx, "anthropic", True)

    assert stored["ANTHROPIC_API_KEY"] == good
    assert any("cannot be right" in message for message in ctx.ui.said)


# -- platform ----------------------------------------------


async def test_detection_describes_this_machine():
    report = await detect()
    assert report.system in ("macos", "linux")
    assert report.supported
    assert report.python
    assert dict(report.rows())["os"]


def test_a_report_round_trips_and_tolerates_unknown_fields(tmp_path):
    report = PlatformReport(system="linux", arch="x86_64", gpu=Gpu("cuda", "A100", 40000))
    path = tmp_path / "platform.json"
    report.write(path)
    assert PlatformReport.read(path) == report

    path.write_text('{"system": "linux", "invented_later": 7, "gpu": {"kind": "cuda"}}')
    older = PlatformReport.read(path)
    assert older.system == "linux" and older.gpu.kind == "cuda"
    assert PlatformReport.read(tmp_path / "nope.json") is None


def test_only_a_cuda_gpu_with_room_runs_the_local_tts_server():
    """Never stand up local TTS on hardware that would crawl."""
    assert Gpu("cuda", "4090", 24000).can_run_local_tts
    assert not Gpu("cuda", "1050", 4000).can_run_local_tts
    assert not Gpu("mps", "Apple M3").can_run_local_tts  # the image is CUDA-only
    assert not Gpu().can_run_local_tts


def test_windows_is_out_of_scope_because_tmux_is():
    assert not PlatformReport(system="windows").supported
    assert PlatformReport(system="linux", is_wsl=True).supported


# -- layout -------------------------------------------------


async def test_layout_creates_the_tree_and_migrates_the_schema(tmp_path):
    ctx = make_ctx(tmp_path)
    (outcome,) = await run_steps(ctx, [LayoutStep()])
    assert outcome.status is Status.DONE

    paths = Config(home=ctx.home).paths
    assert paths.tasks.is_dir() and paths.worktrees.is_dir()
    # ~/.buddy/keys holds service-account JSON, so it is not readable
    # by anyone else.
    assert paths.keys.stat().st_mode & 0o777 == 0o700
    with Store(paths.db) as store:
        assert store.schema_version() == SCHEMA_VERSION
        assert "conversation_log_fts" in store.table_names()
        assert "agents" in store.table_names()


async def test_layout_notices_a_missing_fts_index(tmp_path):
    """Losing the FTS index costs the brain its recall, silently."""
    ctx = make_ctx(tmp_path)
    await run_steps(ctx, [LayoutStep()])
    with Store(Config(home=ctx.home).paths.db) as store:
        store._conn.execute("DROP TABLE conversation_log_fts")
    result = await LayoutStep().check(ctx)
    assert not result.satisfied
    assert "FTS" in result.detail


# -- config ------------------------------------------------


def test_merging_adds_what_is_missing_and_overwrites_nothing():
    import tomlkit

    document = tomlkit.parse('[buddy]\nweb_port = 9999  # mine\n\n[brain]\nprovider = "openai"\n')
    added = merge_missing(
        document,
        {
            "buddy": {"web_port": 4321, "trust_mode": False},
            "brain": {"provider": "anthropic"},
            "harness": {"claude_code": {"command": "claude -p"}},
        },
    )
    rendered = tomlkit.dumps(document)

    assert added == ["buddy.trust_mode", "harness"]
    assert "web_port = 9999  # mine" in rendered, "an existing value and its comment survive"
    assert 'provider = "openai"' in rendered, "setup never overwrites a chosen provider"
    assert "trust_mode = false" in rendered
    assert 'command = "claude -p"' in rendered


async def test_config_writes_a_commented_file_that_parses(tmp_path):
    ctx = make_ctx(tmp_path, no_voice=True)
    ctx.home.mkdir(parents=True)
    ctx.section("brain")["provider"] = "openai"
    ctx.section("brain")["model"] = "gpt-5"
    ctx.section("harness", "claude_code")["command"] = "claude -p < {prompt_path}"

    (outcome,) = await run_steps(ctx, [ConfigStep()])
    assert outcome.status is Status.DONE

    path = Config(home=ctx.home).paths.config_file
    text = path.read_text()
    assert text.startswith("# Buddy's configuration")
    assert "# the API minimum is 50000" in text, "the file setup writes is commented"
    # Nothing secret, ever: only the name to look it up by.
    assert "sk-" not in text

    config = Config.load(home=ctx.home)
    assert config.brain.provider == "openai"
    assert config.harnesses["claude_code"].command.startswith("claude -p")
    assert path.stat().st_mode & 0o777 == 0o600


async def test_re_running_config_keeps_a_hand_edited_value(tmp_path):
    ctx = make_ctx(tmp_path, no_voice=True)
    ctx.home.mkdir(parents=True)
    await run_steps(ctx, [ConfigStep()])

    path = Config(home=ctx.home).paths.config_file
    written = path.read_text()
    assert "max_concurrent   = 0" in written, "the edit below would silently do nothing"
    path.write_text(written.replace("max_concurrent   = 0", "max_concurrent   = 3"))

    later = make_ctx(tmp_path, no_voice=True)
    later.section("buddy")["max_concurrent"] = 7
    later.section("voice")["stt_backend"] = "whisper"
    await run_steps(later, [ConfigStep()])

    config = Config.load(home=ctx.home)
    assert config.buddy.max_concurrent == 3, "setup must never overwrite what you chose"
    assert config.voice.stt_backend == "whisper", "but it does add what is new"


async def test_config_asks_for_a_project_once(tmp_path):
    repo = tmp_path / "myrepo"
    repo.mkdir()
    ui = ScriptedUI({"path to a git repo": str(repo), "call it": "myrepo"})
    ctx = make_ctx(tmp_path, ui=ui, no_voice=True)
    ctx.home.mkdir(parents=True)
    await run_steps(ctx, [ConfigStep()])
    assert Config.load(home=ctx.home).projects["myrepo"].path == repo

    later = make_ctx(tmp_path, ui=ScriptedUI(), no_voice=True)
    await run_steps(later, [ConfigStep()])
    assert not [q for q in later.ui.asked if "git repo" in q.lower()], "asked twice"


# -- the whole sequence ----------------------------------------------------


def test_every_step_in_section_13_2_is_present():
    assert step_names() == [
        "platform",
        "packages",
        "docker",
        "layout",
        "keys",
        "stt",
        "tts",
        "harnesses",
        "audio",
        "config",
        "tmux",
        "doctor",
    ]
    assert [step.number for step in all_steps()] == [str(n) for n in range(1, 13)]


def test_every_step_is_forceable_by_name():
    """`--force STEP` is only useful if the names are the ones printed."""
    for step in all_steps():
        assert step.name and step.name == step.name.lower()
        assert " " not in step.name


@pytest.fixture
def bare_repo(tmp_path: Path) -> Path:
    """A repo for step 10 to point a project at. Sync, so the loop is never
    blocked on a subprocess."""
    path = tmp_path / "project"
    path.mkdir()
    subprocess.run(["git", "-C", str(path), "init", "-q", "-b", "main"], check=True)
    return path


@pytest.mark.timeout(300)
@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
@pytest.mark.skipif(
    not any(shutil.which(b) for b in ("claude", "codex", "opencode", "agy")),
    reason="no coding-agent CLI is installed, so setup has nothing to run",
)
async def test_the_whole_of_setup_against_this_machine(
    tmp_path, bare_repo, tmux_socket, monkeypatch
):
    """All twelve steps, for real, with a script instead of a person.

    Only two things are faked, and both because they cost money or reach
    outside the machine: the provider's test call, and the terminal.
    """
    project = bare_repo

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")

    class StubProvider:
        name = "anthropic"
        model = "claude-opus-5"

        async def probe(self):
            from buddy.providers.base import Capabilities, ProbeResult

            return ProbeResult(
                provider=self.name,
                model=self.model,
                reachable=True,
                capabilities=Capabilities(),
                detail="stubbed for the test; a real run makes a one-token call",
            )

    # Every name the two callers bound at import time, not just the module
    # attribute: `cli.py` does `from buddy.providers import build as
    # build_provider`, so patching only `buddy.providers.build` works or not
    # depending on whether `buddy.cli` was imported before this test ran -
    # and when it did not work, step 12 built a real Anthropic client, which
    # then failed a *different* test when its connection pool was collected
    # against a closed loop.
    import buddy.cli

    stub = lambda *args, **kwargs: StubProvider()  # noqa: E731
    monkeypatch.setattr("buddy.providers.build", stub)
    monkeypatch.setattr("buddy.setup.steps.keys.build_provider", stub)
    monkeypatch.setattr(buddy.cli, "build_provider", stub)

    # A machine without the voice extra legitimately runs setup without
    # speech, so the test does what the machine can do rather than failing
    # over an optional dependency - and still exercises all twelve steps,
    # three of them reporting "not applicable".
    speech = importlib.util.find_spec("faster_whisper") is not None
    ui = ScriptedUI({"path to a git repo": str(project), "call it": "project"})
    ctx = make_ctx(tmp_path, ui=ui, no_voice=not speech)
    ctx.report = None  # let step 1 detect this machine for real
    ctx.tmux_socket = tmux_socket

    outcomes = await run_steps(ctx, all_steps())
    by_name = by_step(outcomes)
    failures = [o for o in outcomes if o.status is Status.FAILED]
    assert not failures, f"{failures[0].step}: {failures[0].detail}"
    assert len(outcomes) == 12, "the run stopped early"

    # Step 4 built the tree, step 10 wrote a config that parses, step 11
    # created the session, and step 12 had something to report about.
    config = Config.load(home=ctx.home)
    assert config.paths.db.exists()
    assert "project" in config.projects
    assert config.runnable_harnesses()
    assert by_name["tmux"].status in (Status.DONE, Status.SKIPPED)
    assert "each agent gets its own window" in by_name["tmux"].detail

    # The lock knows what to compare against next time.
    lock = json.loads(config.paths.setup_lock.read_text())
    assert lock["steps"]["platform"]["pins"]["system"] in ("macos", "linux")
    # Whatever this machine can run, pinned and written alike - Claude Code
    # among them wherever it is installed.
    pinned = lock["steps"]["harnesses"]["pins"]["harnesses"].split(",")
    assert pinned and set(pinned) <= set(config.harnesses)
    if shutil.which("claude"):
        assert "claude_code" in pinned

    # A model that exists. Claude Code's own `sonnet` alias resolves to a
    # retired one, so a task naming no model would 404 on its first call -
    # which is the first thing a new user would see.
    harness = config.harnesses["claude_code"]
    assert harness.default_model == "claude-sonnet-5"
    assert "--model claude-sonnet-5" in harness.command.format(
        prompt_path="p", worktree="w", model="claude-sonnet-5", model_flag="--model claude-sonnet-5"
    )

    # And the second run does almost nothing, which is the promise.
    again = make_ctx(tmp_path, ui=ScriptedUI(), no_voice=not speech)
    again.report = None
    again.tmux_socket = tmux_socket
    repeated = await run_steps(again, all_steps())
    acted = [o.step for o in repeated if o.status is Status.DONE]
    assert not [o for o in repeated if o.status is Status.FAILED]
    # Nothing that installs, downloads, writes or creates should run twice.
    # `platform` and `doctor` always run by design, and `tts` keeps reporting
    # itself unconfigured for as long as there is no FISH_API_KEY, which is
    # the truth and worth repeating. So does `docker` on a machine whose
    # daemon is installed but not running: setup cannot start it, only say so.
    may_repeat = {"platform", "doctor", "tts"}
    if again.report is None or not again.report.docker.running:
        may_repeat.add("docker")
    assert set(acted) <= may_repeat, f"re-running setup redid {acted}"

    unchanged = ["packages", "layout", "keys", "harnesses", "config", "tmux"]
    if speech:
        unchanged += ["stt", "audio"]
    for step in unchanged:
        assert by_step(repeated)[step].status is Status.SKIPPED, f"{step} did work twice"


async def test_packages_asks_for_nothing_buddy_does_not_use(tmp_path, monkeypatch):
    """ffmpeg was required, and offered a `sudo` install, while nothing in
    Buddy ever ran it - speech is decoded by PyAV, which brings its own. On a
    machine without it, setup failed; on CI it tried to install it."""
    from buddy.setup.steps import packages

    probed: list[str] = []
    real = packages.run

    async def watching(*args, **kwargs):
        probed.append(args[0])
        return await real(*args, **kwargs)

    monkeypatch.setattr(packages, "run", watching)
    ctx = make_ctx(tmp_path, no_voice=True)

    missing, pins = await packages.PackagesStep()._missing(ctx)

    assert missing == []
    assert set(pins) == {"tmux", "git"}
    assert "ffmpeg" not in probed
    assert all("ffmpeg" not in names for names in packages.PACKAGE_NAMES.values())
