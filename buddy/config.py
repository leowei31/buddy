"""config.toml loading and validation.

Also owns the on-disk layout, because where a task's worktree lands is
itself configurable (`[projects.*].worktree_root`), so path resolution and
configuration cannot be separated without one of them guessing.

Structural validation happens at load. Whether the paths actually exist, the
binaries are installed, or the keys work is `buddy doctor`'s job.
"""

from __future__ import annotations

import contextlib
import os
import re
import tomllib
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, cast

DEFAULT_HOME = Path("~/.buddy")

#: `${keychain:NAME}`. Resolved in `[harness.<name>.env]` only - see
#: `_misplaced_secret_refs`, which refuses one anywhere else.
_KEYCHAIN_RE = re.compile(r"\$\{keychain:([A-Za-z0-9_.-]+)\}")

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "": 1}

KNOWN_PROVIDERS = ("anthropic", "anthropic_vertex", "openai", "vertex_gemini")
KNOWN_STRATEGIES = ("auto", "server", "client")


class ConfigError(Exception):
    """A config.toml that cannot be trusted. Always names the offending key."""


def parse_duration(value: str | int | float, *, key: str) -> timedelta:
    """`"10m"` / `"2h"` / `"30s"` / `"1d"`, or a bare number of seconds."""
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    match = _DURATION_RE.match(str(value))
    if not match:
        raise ConfigError(f"{key}: {value!r} is not a duration like '10m' or '2h'")
    amount, unit = match.groups()
    return timedelta(seconds=float(amount) * _DURATION_UNITS[unit.lower()])


def as_int(value: object, *, key: str, default: int | None = None) -> int:
    """An integer from config, or a `ConfigError` naming the key.

    Every caller of `Config.load` catches only `ConfigError`, because this
    module validates at load. A bare `int()` on a typo'd value raised
    `ValueError` straight past all of them and out as a traceback.
    """
    if value is None and default is not None:
        return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key}: {value!r} is not a whole number") from exc


def at_least(floor: int, value: int, *, key: str) -> int:
    """`value`, or a `ConfigError` naming the key when it is below `floor`."""
    if value < floor:
        raise ConfigError(f"{key}: {value} is less than {floor}")
    return value


def as_bool(value: object, *, key: str, default: bool = False) -> bool:
    """A TOML boolean, and only a boolean.

    `bool("false")` is `True`, so a quoted boolean - an easy mistake in a file
    that mixes quoted and bare values - silently turned a setting on. Rejected
    outright rather than guessed at.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{key}: {value!r} is not true or false (write it unquoted)")


def expand(path: str | Path) -> Path:
    """`~` and `$VAR` expansion, then absolute."""
    return Path(os.path.expandvars(str(path))).expanduser()


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------


class SecretResolver(Protocol):
    """Resolves `${keychain:NAME}` references."""

    def resolve(self, name: str) -> str | None: ...


class EnvSecretResolver:
    """Environment variables only.

    Kept as the resolver for tests and for anything that must not touch the
    user's keychain. The one Buddy actually runs with is
    `KeyringSecretResolver`, which falls back to this behaviour.
    """

    def resolve(self, name: str) -> str | None:
        return os.environ.get(name)


#: The service name every Buddy secret is filed under in the OS keychain.
KEYCHAIN_SERVICE = "buddy"


def keychain_available() -> tuple[bool, str]:
    """Whether the OS keychain can be used here, and what it is.

    A headless Linux box with no secret-service daemon has no keychain, which
    is a fact to report rather than an error: env vars still work, and take
    precedence anyway.
    """
    try:
        import keyring
        from keyring.backends import fail
    except ImportError as exc:  # pragma: no cover - keyring is a core dependency
        return False, str(exc)
    backend = keyring.get_keyring()
    if isinstance(backend, fail.Keyring):
        return False, "no usable keyring backend on this system"
    return True, type(backend).__module__.rsplit(".", 1)[-1]


def keychain_get(name: str) -> str | None:
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, name)
    except Exception:  # noqa: BLE001 - an unreachable keychain is "no value"
        return None


def keychain_set(name: str, value: str) -> None:
    import keyring

    keyring.set_password(KEYCHAIN_SERVICE, name, value)


def keychain_delete(name: str) -> bool:
    try:
        import keyring

        keyring.delete_password(KEYCHAIN_SERVICE, name)
        return True
    except Exception:  # noqa: BLE001 - deleting what is not there is not a failure
        return False


class KeyringSecretResolver:
    """Environment first, then the OS keychain.

    That precedence is deliberate: setup files secrets in the keychain, and
    an env var set for one session overrides them without editing anything.
    """

    def resolve(self, name: str) -> str | None:
        return os.environ.get(name) or keychain_get(name)


def resolve_secrets(value: str, resolver: SecretResolver, *, key: str) -> str:
    """Substitute every `${keychain:NAME}` in a config string."""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        secret = resolver.resolve(name)
        if secret is None:
            raise ConfigError(f"{key}: secret {name!r} is not available")
        return secret

    return _KEYCHAIN_RE.sub(_sub, value)


def _misplaced_secret_refs(data: object, path: tuple[str, ...] = ()) -> str | None:
    """The dotted key of the first `${keychain:...}` outside a harness env."""
    if isinstance(data, dict):
        for key, value in data.items():
            found = _misplaced_secret_refs(value, (*path, str(key)))
            if found:
                return found
        return None
    if isinstance(data, list):
        for item in data:
            found = _misplaced_secret_refs(item, path)
            if found:
                return found
        return None
    in_harness_env = len(path) == 4 and path[0] == "harness" and path[2] == "env"
    if isinstance(data, str) and _KEYCHAIN_RE.search(data) and not in_harness_env:
        return ".".join(path)
    return None


def has_secret_refs(value: str) -> bool:
    return bool(_KEYCHAIN_RE.search(value))


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BuddySection:
    #: How many agents may run at once. 0, the default, means no limit.
    max_concurrent: int = 0
    default_priority: int = 3
    stall_timeout: timedelta = timedelta(minutes=10)
    max_runtime: timedelta = timedelta(hours=2)
    web_port: int = 4321
    trust_mode: bool = False  # true = no confirmation for anything
    # A discarded task's branch is kept this long before it is pruned.
    discard_grace: timedelta = timedelta(days=7)
    #: An attempt's log is rotated past this many MiB, keeping one previous
    #: file. 0 means never.
    max_log_mb: int = 64

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuddySection:
        return cls(
            max_concurrent=at_least(
                0,
                as_int(data.get("max_concurrent"), key="buddy.max_concurrent", default=0),
                key="buddy.max_concurrent",
            ),
            default_priority=as_int(
                data.get("default_priority"), key="buddy.default_priority", default=3
            ),
            stall_timeout=parse_duration(
                data.get("stall_timeout", "10m"), key="buddy.stall_timeout"
            ),
            max_runtime=parse_duration(data.get("max_runtime", "2h"), key="buddy.max_runtime"),
            web_port=as_int(data.get("web_port"), key="buddy.web_port", default=4321),
            trust_mode=as_bool(data.get("trust_mode"), key="buddy.trust_mode"),
            discard_grace=parse_duration(
                data.get("discard_grace", "7d"), key="buddy.discard_grace"
            ),
            max_log_mb=as_int(data.get("max_log_mb"), key="buddy.max_log_mb", default=64),
        )


@dataclass(frozen=True)
class ClearToolUses:
    """Context layer 1: server-side tool-result clearing, and its budgets."""

    trigger: int = 40000
    keep: int = 5


@dataclass(frozen=True)
class BrainSection:
    """`[brain]`: the provider, its model, and the context budgets."""

    provider: str = "anthropic"
    model: str | None = None
    strategy: str = "auto"
    compact_trigger_tokens: int = 100000  # API minimum 50000
    keep_recent_turns: int = 6
    tool_output_tail_lines: int = 60
    read_file_max_bytes: int = 20000
    clear_tool_uses: ClearToolUses = field(default_factory=ClearToolUses)
    provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Start every session brainstorming, so nothing is spawned until `/go`.
    brainstorm_first: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainSection:
        provider = str(data.get("provider", "anthropic"))
        if provider not in KNOWN_PROVIDERS:
            raise ConfigError(
                f"brain.provider: {provider!r} is not one of {', '.join(KNOWN_PROVIDERS)}"
            )
        strategy = str(data.get("strategy", "auto"))
        if strategy not in KNOWN_STRATEGIES:
            raise ConfigError(
                f"brain.strategy: {strategy!r} is not one of {', '.join(KNOWN_STRATEGIES)}"
            )
        trigger = as_int(
            data.get("compact_trigger_tokens"), key="brain.compact_trigger_tokens", default=100000
        )
        if trigger < 50000:
            raise ConfigError(
                f"brain.compact_trigger_tokens: {trigger} is below the API minimum of 50000"
            )
        clear = data.get("clear_tool_uses", {})
        return cls(
            provider=provider,
            model=data.get("model") or None,
            strategy=strategy,
            compact_trigger_tokens=trigger,
            keep_recent_turns=as_int(
                data.get("keep_recent_turns"), key="brain.keep_recent_turns", default=6
            ),
            tool_output_tail_lines=as_int(
                data.get("tool_output_tail_lines"), key="brain.tool_output_tail_lines", default=60
            ),
            read_file_max_bytes=as_int(
                data.get("read_file_max_bytes"), key="brain.read_file_max_bytes", default=20000
            ),
            clear_tool_uses=ClearToolUses(
                trigger=as_int(
                    clear.get("trigger"), key="brain.clear_tool_uses.trigger", default=40000
                ),
                keep=as_int(clear.get("keep"), key="brain.clear_tool_uses.keep", default=5),
            ),
            provider_options={p: dict(data[p]) for p in KNOWN_PROVIDERS if p in data},
            brainstorm_first=as_bool(data.get("brainstorm_first"), key="brain.brainstorm_first"),
        )

    def options_for(self, provider: str | None = None) -> dict[str, Any]:
        return self.provider_options.get(provider or self.provider, {})


@dataclass(frozen=True)
class ProjectConfig:
    """`[projects.<name>]`."""

    name: str
    path: Path
    base_branch: str = "main"
    default_harness: str | None = None
    worktree_root: Path | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> ProjectConfig:
        if "path" not in data:
            raise ConfigError(f"projects.{name}: 'path' is required")
        root = data.get("worktree_root")
        return cls(
            name=name,
            path=expand(data["path"]),
            base_branch=str(data.get("base_branch", "main")),
            default_harness=data.get("default_harness") or None,
            worktree_root=expand(root) if root else None,
        )


#: `{name}` - and only when `name` is one the caller supplies.
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def fill_template(template: str, **values: str) -> str:
    """Substitute the named placeholders and leave every other brace alone.

    Not `str.format`. A harness command is an ordinary shell line, and shell
    lines are full of braces that are not placeholders: Codex's own `-c
    key={...}` syntax, `awk '{print $1}'`, a JSON argument, `${VAR}`. Under
    `format` each of those raised KeyError, from inside the scheduler.
    """
    return _PLACEHOLDER.sub(
        lambda match: values[match.group(1)] if match.group(1) in values else match.group(0),
        template,
    )


@dataclass(frozen=True)
class HarnessConfig:
    """`[harness.<name>]`.

    `command` is a template, never parsed here: the adapter owns its
    placeholders, and keeping the flags in config is what makes a
    harness release an edit rather than a Buddy release.
    """

    name: str
    command: str
    waiting_patterns: tuple[re.Pattern[str], ...] = ()
    default_model: str | None = None
    sandbox: str = "none"
    sandbox_command: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> HarnessConfig:
        command = data.get("command")
        if not command:
            # An empty block is how a harness is parked while its CLI is
            # unproven, so this is not an error
            # until something tries to spawn it.
            command = ""
        patterns = []
        for raw in data.get("waiting_patterns", []):
            try:
                patterns.append(re.compile(raw))
            except re.error as exc:
                raise ConfigError(
                    f"harness.{name}.waiting_patterns: {raw!r} is not a regex: {exc}"
                ) from exc
        sandbox = str(data.get("sandbox", "none"))
        if sandbox not in ("none", "docker"):
            raise ConfigError(f"harness.{name}.sandbox: {sandbox!r} is not 'none' or 'docker'")
        if sandbox == "docker" and not data.get("sandbox_command"):
            raise ConfigError(f"harness.{name}.sandbox_command: required when sandbox = 'docker'")
        return cls(
            name=name,
            command=str(command),
            waiting_patterns=tuple(patterns),
            default_model=data.get("default_model") or None,
            sandbox=sandbox,
            sandbox_command=data.get("sandbox_command"),
            env={str(k): str(v) for k, v in data.get("env", {}).items()},
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.command)


@dataclass(frozen=True)
class VoiceSection:
    """`[voice]`."""

    stt_backend: str = "whisper"
    stt_model: str = "faster-whisper:small"
    push_to_talk: str = "ctrl+space"
    tts_backend: str = "fish_local"
    tts_fallback: str | None = "fish_cloud"
    #: Chosen by `buddy setup` step 9, which picks defaults and stores their ids.
    input_device: str = ""
    output_device: str = ""
    #: Hands-free listening. Off by default: push-to-talk is the
    #: v1 behaviour, and turning a microphone on permanently is the user's
    #: call to make, not a default to inherit.
    hands_free: bool = False
    silence_ms: int = 900
    fish_local: dict[str, Any] = field(default_factory=dict)
    fish_cloud: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceSection:
        fallback = data.get("tts_fallback", "fish_cloud")
        return cls(
            stt_backend=str(data.get("stt_backend", "whisper")),
            stt_model=str(data.get("stt_model", "faster-whisper:small")),
            push_to_talk=str(data.get("push_to_talk", "ctrl+space")),
            tts_backend=str(data.get("tts_backend", "fish_local")),
            tts_fallback=str(fallback) or None,
            input_device=str(data.get("input_device", "")),
            output_device=str(data.get("output_device", "")),
            hands_free=as_bool(data.get("hands_free"), key="voice.hands_free"),
            silence_ms=as_int(data.get("silence_ms"), key="voice.silence_ms", default=900),
            fish_local=dict(data.get("fish_local", {})),
            fish_cloud=dict(data.get("fish_cloud", {})),
        )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """The `~/.buddy` tree. One place that knows these names."""

    home: Path

    @property
    def config_file(self) -> Path:
        return self.home / "config.toml"

    @property
    def db(self) -> Path:
        return self.home / "state.db"

    @property
    def buddy_log(self) -> Path:
        return self.home / "buddy.log"

    @property
    def tasks(self) -> Path:
        return self.home / "tasks"

    @property
    def worktrees(self) -> Path:
        return self.home / "worktrees"

    @property
    def keys(self) -> Path:
        return self.home / "keys"

    def task_dir(self, task_id: str) -> Path:
        return self.tasks / task_id

    def prompt_file(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "prompt.md"

    def run_script(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "run.sh"

    def log_file(self, task_id: str, attempt: int) -> Path:
        return self.task_dir(task_id) / f"attempt-{attempt}.log"

    def result_file(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "result.json"

    @property
    def setup_lock(self) -> Path:
        """What each setup step settled on, so `buddy update` knows what
        moved."""
        return self.home / "setup.lock"

    @property
    def platform_report(self) -> Path:
        """Where setup step 1 writes the platform report every later step reads."""
        return self.home / "platform.json"

    def ensure(self) -> None:
        """Create the tree, private to you. Idempotent.

        Everything under here is yours alone: the conversation, every brief,
        every agent's output, `run.sh` with its resolved secrets while a run
        lasts. So the home itself is 0700, and an install made before that
        was true is tightened rather than left readable.
        """
        for directory in (self.home, self.tasks, self.worktrees):
            directory.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            self.home.chmod(0o700)


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    home: Path
    buddy: BuddySection = field(default_factory=BuddySection)
    brain: BrainSection = field(default_factory=BrainSection)
    projects: dict[str, ProjectConfig] = field(default_factory=dict)
    harnesses: dict[str, HarnessConfig] = field(default_factory=dict)
    voice: VoiceSection = field(default_factory=VoiceSection)

    @property
    def paths(self) -> Paths:
        return Paths(self.home)

    def project(self, name: str) -> ProjectConfig:
        try:
            return self.projects[name]
        except KeyError:
            known = ", ".join(sorted(self.projects)) or "none configured"
            raise ConfigError(f"unknown project {name!r} (known: {known})") from None

    def harness(self, name: str) -> HarnessConfig:
        try:
            return self.harnesses[name]
        except KeyError:
            known = ", ".join(sorted(self.harnesses)) or "none configured"
            raise ConfigError(f"unknown harness {name!r} (known: {known})") from None

    def worktree_root(self, project: str) -> Path:
        """`~/.buddy/worktrees/<project>` unless the project overrides it."""
        configured = self.project(project).worktree_root
        return configured if configured else self.paths.worktrees / project

    def worktree_path(self, project: str, task_id: str) -> Path:
        return self.worktree_root(project) / task_id

    def harness_for(self, project: str) -> str | None:
        """The harness a task gets when the user did not name one."""
        return self.project(project).default_harness

    def runnable_harnesses(self) -> list[str]:
        """Configured harnesses with a command to run, `claude_code` first.

        A block with no command is how a harness is parked, so it is
        configured and still not something a task can be given.
        """
        names = [name for name, block in self.harnesses.items() if block.command.strip()]
        return sorted(names, key=lambda name: (name != "claude_code", name))

    def choose_harness(self, project: str, requested: str | None = None) -> str:
        """The harness a task gets: the one asked for, else the project's
        default, else the first runnable one.

        Never a hardcoded name. Before this, an unspecified harness meant
        `claude_code` even on a machine where only OpenCode was set up, and
        the task failed at spawn for a reason nobody had chosen.
        """
        if requested:
            return requested
        if default := self.harness_for(project):
            return default
        runnable = self.runnable_harnesses()
        if not runnable:
            raise ConfigError("no harness is configured with a command; run `buddy setup`")
        return runnable[0]

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, home: Path) -> Config:
        return cls(
            home=home,
            buddy=BuddySection.from_dict(data.get("buddy", {})),
            brain=BrainSection.from_dict(data.get("brain", {})),
            projects={
                name: ProjectConfig.from_dict(name, block)
                for name, block in data.get("projects", {}).items()
            },
            harnesses={
                name: HarnessConfig.from_dict(name, block)
                for name, block in data.get("harness", {}).items()
            },
            voice=VoiceSection.from_dict(data.get("voice", {})),
        )

    @classmethod
    def load(cls, path: Path | None = None, *, home: Path | None = None) -> Config:
        """Read config.toml. A missing file yields defaults, so a fresh
        machine works before `buddy setup` has written one."""
        resolved_home = expand(home or os.environ.get("BUDDY_HOME") or DEFAULT_HOME)
        config_file = path or Paths(resolved_home).config_file
        if not config_file.exists():
            return cls(home=resolved_home)
        try:
            data = tomllib.loads(config_file.read_text())
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{config_file}: {exc}") from exc
        if misplaced := _misplaced_secret_refs(data):
            raise ConfigError(
                f"{misplaced}: ${{keychain:...}} is only resolved in [harness.<name>.env]; "
                "anywhere else it would be used as the literal text. Other settings take "
                "the *name* of an environment variable instead (e.g. api_key_env)."
            )
        return cls.from_dict(data, home=resolved_home)
