"""Finding secrets before anything commits them.

Buddy hands agents your API keys - that is how a harness authenticates - and
then commits in their worktrees and merges their branches into yours. That is
two ways for a key to end up in git without anyone typing `git add`: an agent
that writes a key into a `.env` or a config file, and Buddy's own checkpoint,
which commits whatever is left in the worktree.

So this is consulted at every point something is committed:

* Buddy's checkpoint leaves secret-looking files and contents out of its
  commit (`Workspace.checkpoint`).
* `buddy merge` refuses a branch that adds one (`Workspace.merge`).
* The repository's pre-commit hook refuses a commit that stages one
  (`scripts/check-secrets.py`, `.githooks/pre-commit`).

Two ways to recognise a secret, because neither is enough alone:

* **Your actual keys**, matched exactly, whatever they look like. A Fish
  Audio key has no public prefix to pattern-match; your own keychain knows
  what it is.
* **The shapes providers use**, with an entropy check on the random part,
  so `"sk-ant-" + "x" * 95` in a test or `sk-proj-xxxx` in documentation is
  not mistaken for a real one.

Nothing here ever returns or prints a secret: a finding says where and what
kind, and that is all.

Standard library only, and Python 3.8 or later, so the git hook can import it
with whatever `python3` a machine has and none of Buddy's dependencies.
"""

from __future__ import annotations

import fnmatch
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

#: A line containing this is not scanned, for the rare lookalike that must
#: stay (a documented example, a test of this module).
ALLOW_MARKER = "secret-scan: allow"

#: The variables Buddy knows hold secrets. Their values, from the environment
#: or the keychain, are what "your own keys" means.
SECRET_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "FISH_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

#: A known value shorter than this is too likely to occur by chance.
MIN_KNOWN_LENGTH = 12


def _class(char: str) -> str:
    if char.islower():
        return "lower"
    if char.isupper():
        return "upper"
    return "digit" if char.isdigit() else "other"


def switch_ratio(text: str) -> float:
    """How often consecutive characters change class (lower, upper, digit, other).

    Random strings change class on around half their characters; names built
    from words rarely do.
    """
    if len(text) < 2:
        return 0.0
    # Indexing rather than `zip(..., strict=...)`, which needs Python 3.10:
    # the pre-commit hook runs on whatever `python3` a machine has.
    switches = sum(1 for i in range(1, len(text)) if _class(text[i - 1]) != _class(text[i]))
    return switches / (len(text) - 1)


def looks_random(value: str) -> bool:
    """A value that is a key rather than a name, for shapes with no prefix.

    Measured over 500 random keys against names like `GITHUB_TOKEN_FOR_CI_2`:
    neither entropy nor class switching separates them alone - random hex has
    low entropy and switches little - so hex is its own case, and everything
    else must be both high-entropy and switchy.
    """
    if len(value) < 24:
        return False
    if re.fullmatch(r"[0-9a-fA-F]+", value):
        return bool(re.search(r"\d", value)) and bool(re.search(r"[a-fA-F]", value))
    return entropy(value) >= 3.5 and switch_ratio(value) >= 0.40


@dataclass(frozen=True)
class Shape:
    kind: str
    pattern: re.Pattern[str]
    #: Bits per character the random part must reach. A real key is random;
    #: a placeholder repeats itself.
    min_entropy: float = 3.5
    #: Which regex group holds the random part, or None to skip the check.
    random_group: int | None = 1
    #: For shapes with no provider prefix: judged by `looks_random` instead.
    prefixless: bool = False


SHAPES = (
    Shape("anthropic key", re.compile(r"\bsk-ant-(?:api\d\d-|admin\d\d-)?([A-Za-z0-9_\-]{32,})")),
    Shape(
        "openai key",
        re.compile(r"\bsk-(?!ant-)(?:proj-|svcacct-|admin-)?([A-Za-z0-9_\-]{32,})"),
    ),
    Shape("google api key", re.compile(r"\bAIza([0-9A-Za-z_\-]{35})\b")),
    Shape("github token", re.compile(r"\bgh[pousr]_([A-Za-z0-9]{36,})\b")),
    Shape("github token", re.compile(r"\bgithub_pat_([A-Za-z0-9_]{50,})\b")),
    Shape("aws access key", re.compile(r"\b(?:AKIA|ASIA)([0-9A-Z]{16})\b"), min_entropy=2.8),
    Shape("slack token", re.compile(r"\bxox[baprs]-([A-Za-z0-9-]{20,})")),
    Shape(
        "private key",
        re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
        random_group=None,
    ),
    Shape(
        "service account key",
        re.compile(r'"private_key_id"\s*:\s*"([0-9a-f]{40})"'),
        min_entropy=3.0,
    ),
    # Keys with no public prefix - Fish Audio's, a self-hosted service's - are
    # caught where they are most often pasted: a random-looking value given
    # to a name that says what it is. Names like `api_key_env = "OPENAI_API_KEY"`
    # are not random enough to match.
    Shape(
        "secret assigned to a variable",
        re.compile(
            r"(?i)(?:api[_-]?key|secret|token|passw(?:or)?d|auth)[\"']?\s*[:=]\s*[\"']?"
            r"([A-Za-z0-9_\-+/=]{24,})"
        ),
        prefixless=True,
    ),
)

#: Files that hold secrets by their nature, whatever is in them today.
SENSITIVE_NAMES = (
    ".env",
    ".envrc",
    ".netrc",
    ".pypirc",
    "credentials.json",
    "application_default_credentials.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
)
SENSITIVE_GLOBS = (
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*service-account*.json",
    "*service_account*.json",
)
#: `.env.example` and friends are how a project documents its variables.
TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist", ".defaults")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str

    def describe(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.kind}"


def entropy(text: str) -> float:
    """Shannon entropy in bits per character."""
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def is_sensitive_path(path: str) -> bool:
    name = PurePosixPath(path.replace(os.sep, "/")).name
    if name.endswith(".pub") or name.endswith(TEMPLATE_SUFFIXES):
        return False
    return name in SENSITIVE_NAMES or any(fnmatch.fnmatch(name, g) for g in SENSITIVE_GLOBS)


def scan_text(
    text: str, path: str, known: dict[str, str] | None = None, *, first_line: int = 1
) -> list[Finding]:
    """Every secret in `text`, by line. `known` maps a name to your real value."""
    mine = {name: value for name, value in (known or {}).items() if len(value) >= MIN_KNOWN_LENGTH}
    findings: list[Finding] = []
    for offset, line in enumerate(text.splitlines()):
        if ALLOW_MARKER in line:
            continue
        number = first_line + offset
        for name, value in mine.items():
            if value in line:
                findings.append(Finding(path, number, f"your {name}"))
        claimed: list[tuple[int, int]] = []
        for shape in SHAPES:
            for match in shape.pattern.finditer(line):
                random_part = match.group(shape.random_group) if shape.random_group else None
                if shape.prefixless:
                    if shape.random_group is None or random_part is None:
                        continue
                    if not looks_random(random_part):
                        continue
                    span = match.span(shape.random_group)
                    if any(start < span[1] and span[0] < end for start, end in claimed):
                        continue  # a specific shape already named it
                elif random_part is not None and entropy(random_part) < shape.min_entropy:
                    continue
                token = match.group(0)
                if any(value in token or token in value for value in mine.values()):
                    continue  # already reported as yours, by name
                claimed.append(match.span())
                findings.append(Finding(path, number, shape.kind))
    return findings


def scan_added_lines(diff: str, known: dict[str, str] | None = None) -> list[Finding]:
    """Secrets on the lines a unified diff (`git diff -U0`) adds."""
    findings: list[Finding] = []
    path = ""
    number = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            path = target[2:] if target.startswith("b/") else target
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            number = int(match.group(1)) if match else 0
            continue
        if line.startswith("+") and path and path != "/dev/null":
            findings.extend(scan_text(line[1:], path, known, first_line=number))
            number += 1
    return findings


def known_secrets(
    names: tuple[str, ...] = SECRET_NAMES, *, keychain: bool = True
) -> dict[str, str]:
    """Your real values: the environment first, then Buddy's keychain entries.

    Never raises. A machine with no keychain backend - a CI runner - simply
    has no known values, and the shape checks still apply.

    `keychain=False` is for any process that is not Buddy's own: macOS guards
    each keychain item per application, and a different Python asking for one
    raises a permission dialog and blocks until someone answers it. Found as a
    `git commit` in a worktree that hung on exactly that.
    """
    found: dict[str, str] = {}
    keyring: Any = None
    if keychain:
        try:
            import keyring
        except Exception:  # noqa: BLE001 - optional here, by design
            keyring = None
    for name in names:
        value = os.environ.get(name)
        if not value and keyring is not None:
            try:
                value = keyring.get_password("buddy", name)
            except Exception:  # noqa: BLE001 - no backend, locked keychain, anything
                value = None
        if value and len(value) >= MIN_KNOWN_LENGTH:
            found[name] = value
    return found
