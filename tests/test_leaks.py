"""Finding secrets before anything commits them (buddy/leaks.py).

Every key-shaped value here is generated at runtime from a seeded random
source, so this file never contains one - it is itself scanned by
`test_the_repository_holds_no_secrets`.
"""

from __future__ import annotations

import random
import string
import subprocess
import sys
from pathlib import Path

import pytest

from buddy import leaks

ALPHABET = string.ascii_letters + string.digits + "-_"
ROOT = Path(__file__).resolve().parent.parent


def random_token(length: int, seed: int, alphabet: str = ALPHABET) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(length))


def realistic() -> dict[str, str]:
    """One convincing credential per kind the scanner knows."""
    return {
        "anthropic key": "sk-" + "ant-api03-" + random_token(93, 1),
        "openai key": "sk-" + "proj-" + random_token(120, 2),
        "google api key": "AI" + "za" + random_token(35, 3),
        "github token": "gh" + "p_" + random_token(36, 4, string.ascii_letters + string.digits),
        "aws access key": "AK" + "IA" + random_token(16, 5, string.ascii_uppercase + string.digits),
        "private key": "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
    }


# -- recognising a secret ----------------------------------------------------


@pytest.mark.parametrize("kind", sorted(realistic()))
def test_each_kind_of_credential_is_found(kind):
    value = realistic()[kind]
    [finding] = leaks.scan_text(f'API_KEY = "{value}"\n', "settings.py")
    assert finding.kind == kind
    assert finding.line == 1 and finding.path == "settings.py"


def test_a_finding_never_repeats_the_secret():
    value = realistic()["anthropic key"]
    [finding] = leaks.scan_text(value, "x.py")
    assert value[8:] not in str(finding) and value not in finding.describe()


def test_your_own_keys_are_found_in_any_shape():
    """Your actual keychain values are matched exactly, whatever they look
    like - a Fish Audio key has no public prefix to pattern-match on."""
    mine = random_token(32, 9, "0123456789abcdef")
    text = f"voice:\n  token: {mine}\n"
    [finding] = leaks.scan_text(text, "c.yaml", known={"FISH_API_KEY": mine})
    assert finding.kind == "your FISH_API_KEY" and finding.line == 2


def test_a_key_with_no_known_prefix_is_found_where_it_is_assigned():
    """A Fish Audio key is random hex with nothing to pattern-match on; where
    people paste one is `SOMETHING_API_KEY = "..."`."""
    fish = random_token(32, 11, "0123456789abcdef")
    [finding] = leaks.scan_text(f'FISH_API_KEY = "{fish}"', "settings.py")
    assert finding.kind == "secret assigned to a variable"


def test_the_hook_never_reads_the_keychain(monkeypatch):
    """A Python other than Buddy's own asking macOS for a keychain item raises
    a permission dialog and waits - which hung a real `git commit`."""
    import importlib.util

    script = ROOT / "scripts" / "check-secrets.py"
    spec = importlib.util.spec_from_file_location("check_secrets", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    asked: list[bool] = []

    def record(**kwargs):
        asked.append(kwargs["keychain"])
        return {}

    monkeypatch.setattr(module.leaks, "known_secrets", record)
    monkeypatch.setattr(module, "staged", lambda known: [])
    assert module.main(["--staged"]) == 0
    assert module.main(["--staged", "--with-keychain"]) == 0
    assert asked == [False, True]


@pytest.mark.parametrize(
    "harmless",
    [
        '"sk-" + "ant-api03-" + "x" * 95',
        "sk-ant-from-the-keychain",
        "sk-test-not-a-real-key",
        "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "ghp_000000000000000000000000000000000000",
        'api_key_env = "ANTHROPIC_API_KEY"',
        "${keychain:OPENAI_API_KEY}",
        'token_env = "GITHUB_TOKEN_FOR_CI_RUNNERS_2"',
        "password: ${DATABASE_PASSWORD}",
        'api_key = "OPENAI_API_KEY_PRODUCTION_2"',
        'password = "DatabasePassword2024ForStaging"',
        'secret = "prodReadonlyUser_2024_v2"',
    ],
)
def test_placeholders_and_references_are_not_secrets(harmless):
    assert leaks.scan_text(harmless, "t.py") == []


def test_a_line_can_be_marked_as_intentionally_holding_a_lookalike():
    value = realistic()["google api key"]
    assert leaks.scan_text(f"{value}  # secret-scan: allow", "t.py") == []


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "app/.env.production",
        ".envrc",
        "deploy/server.pem",
        "id_rsa",
        "id_ed25519",
        "gcp/service-account.json",
        "credentials.json",
        ".netrc",
        "cert.p12",
    ],
)
def test_files_that_hold_secrets_by_their_nature(path):
    assert leaks.is_sensitive_path(path)


@pytest.mark.parametrize(
    "path", [".env.example", ".env.sample", "id_rsa.pub", "docs/env.md", "README.md", "keys.py"]
)
def test_their_templates_and_public_halves_are_fine(path):
    assert not leaks.is_sensitive_path(path)


# -- the command-line scanner and the hook ------------------------------------


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "core.hooksPath", str(ROOT / ".githooks"))
    return tmp_path


def test_the_pre_commit_hook_refuses_a_staged_key(repo):
    (repo / "config.py").write_text(f'KEY = "{realistic()["openai key"]}"\n')
    git(repo, "add", "config.py")
    result = git(repo, "commit", "-m", "oops")
    assert result.returncode != 0
    assert "openai key" in result.stderr + result.stdout
    assert git(repo, "log", "--oneline").stdout == ""


def test_the_pre_commit_hook_refuses_a_staged_env_file(repo):
    (repo / ".env").write_text("DEBUG=1\n")
    git(repo, "add", "-f", ".env")
    assert git(repo, "commit", "-m", "oops").returncode != 0


def test_the_pre_commit_hook_lets_ordinary_work_through(repo):
    (repo / "app.py").write_text('KEY_ENV = "OPENAI_API_KEY"\n')
    git(repo, "add", "app.py")
    assert git(repo, "commit", "-m", "fine").returncode == 0


def test_the_repository_holds_no_secrets():
    """Everything tracked, and everything that would be picked up by `git add`,
    scanned the way the pre-commit hook scans a commit - so CI and the local
    suite both fail before a key can be pushed."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check-secrets.py"), "--tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize(
    "alphabet",
    ["0123456789abcdef", "0123456789ABCDEF", string.ascii_letters + string.digits + "-_"],
)
def test_prefixless_keys_are_caught_reliably(seed, alphabet):
    """Not one lucky seed: a spread of random keys of each common alphabet."""
    value = random_token(32, 1000 + seed, alphabet)
    findings = leaks.scan_text(f'API_TOKEN = "{value}"', "config.py")
    assert [f.kind for f in findings] == ["secret assigned to a variable"]


def _oldest_python() -> str | None:
    """The oldest interpreter here: what a hook with no virtualenv may get."""
    import shutil

    for name in ("python3.8", "python3.9", "python3.10"):
        found = shutil.which(name)
        if found:
            return found
    system = shutil.which("python3", path="/usr/bin:/opt/anaconda3/bin:/usr/local/bin")
    if system:
        version = subprocess.run(
            [system, "-c", "import sys; print(sys.version_info[:2] < (3, 11))"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if version == "True":
            return system
    return None


def test_the_scanner_runs_on_the_python_a_hook_may_find():
    """Found in a fresh clone: the hook falls back to the system `python3`,
    which was 3.9, and `zip(strict=...)` - 3.10 - refused every commit with a
    traceback. The hook's code must run on older interpreters than Buddy's."""
    old = _oldest_python()
    if old is None:
        pytest.skip("no Python older than 3.11 on this machine")
    result = subprocess.run(
        [old, str(ROOT / "scripts" / "check-secrets.py"), "--tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_scanner_says_so_when_it_is_not_in_a_repository(tmp_path):
    """`docs/operating.md` tells people to run this on their own projects, so
    being pointed at an ordinary directory has to produce a sentence rather
    than a `CalledProcessError` traceback."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check-secrets.py"), "--tree"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "not one" in result.stderr and "checkout" in result.stderr
