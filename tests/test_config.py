"""config.toml loading, validation, and the on-disk layout."""

from datetime import timedelta
from pathlib import Path

import pytest

from buddy.config import (
    Config,
    ConfigError,
    EnvSecretResolver,
    parse_duration,
    resolve_secrets,
)

SAMPLE = """
[buddy]
max_concurrent = 5
stall_timeout = "90s"
max_runtime = "3h"
trust_mode = true

[brain]
provider = "openai"
model = "some-model"
compact_trigger_tokens = 60000

[brain.clear_tool_uses]
trigger = 30000
keep = 2

[brain.openai]
api_key_env = "OPENAI_API_KEY"
base_url = "http://localhost:8000/v1"

[projects.webapp]
path = "~/code/webapp"
base_branch = "trunk"
default_harness = "claude_code"

[projects.other]
path = "/tmp/other"
worktree_root = "/tmp/elsewhere"

[harness.claude_code]
command = "claude -p"
waiting_patterns = ['\\(y/n\\)']

[harness.claude_code.env]
ANTHROPIC_API_KEY = "${keychain:ANTHROPIC_API_KEY}"

[harness.antigravity]
"""


@pytest.fixture
def config(tmp_path: Path) -> Config:
    (tmp_path / "config.toml").write_text(SAMPLE)
    return Config.load(home=tmp_path)


def test_missing_config_file_yields_defaults(tmp_path: Path):
    cfg = Config.load(home=tmp_path)
    assert cfg.buddy.max_concurrent == 0, "no limit on agents unless one is set"
    assert cfg.buddy.stall_timeout == timedelta(minutes=10)
    assert cfg.brain.provider == "anthropic"
    assert cfg.projects == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [("30s", 30), ("10m", 600), ("2h", 7200), ("1d", 86400), ("45", 45), (90, 90)],
)
def test_duration_parsing(text, expected):
    assert parse_duration(text, key="t") == timedelta(seconds=expected)


def test_duration_rejects_nonsense():
    with pytest.raises(ConfigError, match="not a duration"):
        parse_duration("soon", key="buddy.stall_timeout")


def test_sections_are_read(config: Config):
    assert config.buddy.max_concurrent == 5
    assert config.buddy.stall_timeout == timedelta(seconds=90)
    assert config.buddy.max_runtime == timedelta(hours=3)
    assert config.buddy.trust_mode is True
    assert config.brain.provider == "openai"
    assert config.brain.clear_tool_uses.keep == 2
    assert config.brain.options_for()["base_url"] == "http://localhost:8000/v1"


def test_project_lookup_and_worktree_paths(config: Config, tmp_path: Path):
    assert config.project("webapp").base_branch == "trunk"
    assert config.harness_for("webapp") == "claude_code"
    # Default root is ~/.buddy/worktrees/<project>/<task_id>.
    assert config.worktree_path("webapp", "t-0142") == tmp_path / "worktrees/webapp/t-0142"
    # An explicit worktree_root wins.
    assert config.worktree_path("other", "t-0142") == Path("/tmp/elsewhere/t-0142")


def test_unknown_project_names_the_known_ones(config: Config):
    with pytest.raises(ConfigError, match="other, webapp"):
        config.project("nope")


def test_harness_patterns_compile_and_empty_block_is_unconfigured(config: Config):
    claude = config.harness("claude_code")
    assert claude.is_configured
    assert claude.waiting_patterns[0].search("Overwrite? (y/n)")
    # An empty block parks a harness until its CLI is proven.
    assert not config.harness("antigravity").is_configured


def test_layout_matches_section_3_4(config: Config, tmp_path: Path):
    paths = config.paths
    assert paths.db == tmp_path / "state.db"
    assert paths.prompt_file("t-0142") == tmp_path / "tasks/t-0142/prompt.md"
    assert paths.run_script("t-0142") == tmp_path / "tasks/t-0142/run.sh"
    assert paths.log_file("t-0142", 2) == tmp_path / "tasks/t-0142/attempt-2.log"
    assert paths.result_file("t-0142") == tmp_path / "tasks/t-0142/result.json"


def test_layout_ensure_is_idempotent(config: Config):
    config.paths.ensure()
    config.paths.ensure()
    assert config.paths.tasks.is_dir()
    assert config.paths.worktrees.is_dir()


def test_keychain_references_resolve_from_env(config: Config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    raw = config.harness("claude_code").env["ANTHROPIC_API_KEY"]
    assert resolve_secrets(raw, EnvSecretResolver(), key="k") == "sk-test"


def test_missing_secret_is_an_error(config: Config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    raw = config.harness("claude_code").env["ANTHROPIC_API_KEY"]
    with pytest.raises(ConfigError, match="is not available"):
        resolve_secrets(raw, EnvSecretResolver(), key="k")


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ('[brain]\nprovider = "hal9000"', "not one of"),
        ('[brain]\nstrategy = "vibes"', "not one of"),
        ("[brain]\ncompact_trigger_tokens = 1000", "below the API minimum"),
        ("[projects.p]\nbase_branch = 'main'", "'path' is required"),
        ("[harness.h]\ncommand = 'x'\nwaiting_patterns = ['(']", "is not a regex"),
        ("[harness.h]\ncommand = 'x'\nsandbox = 'docker'", "sandbox_command"),
    ],
)
def test_validation_errors_name_the_key(tmp_path: Path, block: str, message: str):
    (tmp_path / "config.toml").write_text(block)
    with pytest.raises(ConfigError, match=message):
        Config.load(home=tmp_path)


def test_a_keychain_reference_outside_harness_env_is_refused_not_used_literally(tmp_path):
    """`${keychain:NAME}` is resolved only in `[harness.<name>.env]`.
    Anywhere else it used to load fine and be sent as the literal string - a
    base URL of "${keychain:PROXY}", say - which fails far from the cause."""
    (tmp_path / "config.toml").write_text('[brain.openai]\nbase_url = "${keychain:MY_PROXY}"\n')
    with pytest.raises(ConfigError, match=r"brain\.openai\.base_url.*harness\.<name>\.env"):
        Config.load(home=tmp_path)


def test_a_keychain_reference_in_harness_env_is_still_fine(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[harness.codex]\ncommand = "codex exec - < {prompt_path}"\n'
        '[harness.codex.env]\nCODEX_API_KEY = "${keychain:OPENAI_API_KEY}"\n'
    )
    config = Config.load(home=tmp_path)
    assert config.harness("codex").env["CODEX_API_KEY"] == "${keychain:OPENAI_API_KEY}"


def test_a_negative_limit_on_agents_is_refused(tmp_path: Path):
    (tmp_path / "config.toml").write_text("[buddy]\nmax_concurrent = -1\n")
    with pytest.raises(ConfigError, match="buddy.max_concurrent: -1 is less than 0"):
        Config.load(home=tmp_path)
