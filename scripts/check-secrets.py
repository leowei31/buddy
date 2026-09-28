#!/usr/bin/env python3
"""Refuse secrets before git records them.

    scripts/check-secrets.py --staged    what is about to be committed (the pre-commit hook)
    scripts/check-secrets.py --tree      every tracked file, and every file `git add` would pick up
    scripts/check-secrets.py --history   every file that was ever committed

Add --with-keychain to also match your real keys from Buddy's keychain
entries. It is off by default because macOS asks permission when a Python
other than Buddy's own reads them, and a hook must never wait on a dialog.

Exits 1 and lists `path:line: kind` for each finding - never the secret
itself. What counts as a secret is `buddy/leaks.py`: your real keys from the
environment or Buddy's keychain entries, provider key shapes, and files that
hold secrets by their nature (`.env`, private keys, service-account JSON).

A line that must keep a lookalike can carry `secret-scan: allow`.
If a real key was ever committed, removing it is not enough: rotate it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Buddy's own checkout, for importing the detector only.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buddy import leaks  # noqa: E402 - the path above has to come first

#: Larger than this, a file is not source a person would paste a key into.
MAX_BYTES = 2_000_000
#: Generated, and too large to be worth reading line by line.
SKIP = {"uv.lock"}


def git(*args: str) -> bytes:
    """In the repository being checked - wherever git runs this from - which
    is not necessarily Buddy's own."""
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def top() -> Path | None:
    """The top of the work tree, or None when this is not a git checkout.

    None rather than an exception because this script is documented for
    people to run on their own repositories: pointed at an ordinary
    directory it used to answer with a `CalledProcessError` traceback, which
    says nothing about what to do.
    """
    try:
        return Path(git("rev-parse", "--show-toplevel").decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def lines(output: bytes) -> list[str]:
    return [entry for entry in output.decode(errors="replace").split("\0") if entry]


def scan_blob(path: str, data: bytes, known: dict[str, str]) -> list[leaks.Finding]:
    if path in SKIP or len(data) > MAX_BYTES or b"\0" in data[:8000]:
        return []
    return leaks.scan_text(data.decode(errors="replace"), path, known)


def staged(known: dict[str, str]) -> list[leaks.Finding]:
    findings: list[leaks.Finding] = []
    for path in lines(git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")):
        if leaks.is_sensitive_path(path):
            findings.append(leaks.Finding(path, 0, "a file that holds secrets by its nature"))
            continue
        findings.extend(scan_blob(path, git("show", f":{path}"), known))
    return findings


def tree(known: dict[str, str]) -> list[leaks.Finding]:
    findings: list[leaks.Finding] = []
    for path in lines(git("ls-files", "-z", "--cached", "--others", "--exclude-standard")):
        if leaks.is_sensitive_path(path):
            findings.append(leaks.Finding(path, 0, "a file that holds secrets by its nature"))
            continue
        try:
            data = (top() / path).read_bytes()
        except OSError:
            continue  # deleted in the working tree but still in the index
        findings.extend(scan_blob(path, data, known))
    return findings


def history(known: dict[str, str]) -> list[leaks.Finding]:
    listing = git("rev-list", "--all", "--objects").decode(errors="replace").splitlines()
    seen: set[str] = set()
    findings: list[leaks.Finding] = []
    for entry in listing:
        sha, _, path = entry.partition(" ")
        if not path or sha in seen:
            continue
        seen.add(sha)
        if git("cat-file", "-t", sha).strip() != b"blob":
            continue
        if leaks.is_sensitive_path(path):
            findings.append(leaks.Finding(path, 0, f"a file that holds secrets (in {sha[:8]})"))
            continue
        for finding in scan_blob(path, git("cat-file", "blob", sha), known):
            kind = f"{finding.kind} (in {sha[:8]})"
            findings.append(leaks.Finding(finding.path, finding.line, kind))
    return findings


def main(argv: list[str]) -> int:
    import os

    root = top()
    if root is None:
        print(
            "check-secrets scans a git repository, and this is not one "
            f"({Path.cwd()}). Run it inside a checkout.",
            file=sys.stderr,
        )
        return 2
    os.chdir(root)  # git's paths are relative to the top of the work tree
    modes = {"--staged": staged, "--tree": tree, "--history": history}
    use_keychain = "--with-keychain" in argv
    chosen = [arg for arg in argv if arg != "--with-keychain"]
    if len(chosen) != 1 or chosen[0] not in modes:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    findings = modes[chosen[0]](leaks.known_secrets(keychain=use_keychain))
    if not findings:
        return 0
    print("Refusing: this would put secrets into git.", file=sys.stderr)
    for finding in findings:
        print(f"  {finding.describe()}", file=sys.stderr)
    print(
        "Move secrets into the environment or Buddy's keychain, and `.env` files out of git.\n"
        "A lookalike that must stay can carry `secret-scan: allow` on its line.\n"
        "If a real key was ever committed or pushed, rotate it: removing it is not enough.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
