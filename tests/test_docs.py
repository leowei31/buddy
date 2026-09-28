"""The documentation describes the tool that actually exists.

Built from the CLI itself rather than from a list someone has to remember to
update: every command, every option and every session command must appear in
the user documentation, and every command the documentation names must exist.
Written after a review found thirteen real options documented nowhere.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import typer.main

import buddy.config as config_module
from buddy.cli import COMMANDS, app

ROOT = Path(__file__).resolve().parent.parent
USER_DOCS = ["README.md", "docs/operating.md", "docs/configuration.md"]


def docs() -> str:
    return "\n".join((ROOT / name).read_text() for name in USER_DOCS)


def commands(command=None, prefix: str = ""):
    command = command or typer.main.get_command(app)
    for name, sub in sorted(getattr(command, "commands", {}).items()):
        full = f"{prefix} {name}".strip()
        yield full, sub
        yield from commands(sub, full)


def test_every_command_is_documented():
    text = docs()
    missing = [name for name, _ in commands() if not re.search(rf"buddy {re.escape(name)}\b", text)]
    assert missing == []


def test_every_option_is_documented():
    text = docs()
    top = typer.main.get_command(app)
    missing = [
        f"buddy {name} {opt}"
        for name, sub in [("", top), *commands()]
        for param in sub.params
        for opt in getattr(param, "opts", [])
        if opt.startswith("--") and opt != "--help" and opt not in text
    ]
    assert missing == []


def test_every_documented_command_exists():
    real = {name for name, _ in commands()}
    named = set(re.findall(r"`buddy ([a-z]+(?: [a-z]+)?)", docs()))
    assert {n for n in named if n not in real and n.split()[0] not in real} == set()


def test_every_session_command_is_documented():
    text = docs()
    assert [c for c in re.findall(r"^(/\w+)", COMMANDS, re.M) if c not in text] == []


def test_every_config_key_is_documented():
    reference = (ROOT / "docs" / "configuration.md").read_text()
    containers = {"provider_options", "clear_tool_uses", "fish_local", "fish_cloud"}
    missing = [
        f"{name}.{field.name}"
        for name in dir(config_module)
        if name.endswith("Section")
        and dataclasses.is_dataclass(section := getattr(config_module, name))
        for field in dataclasses.fields(section)
        if field.name not in containers and f"`{field.name}`" not in reference
    ]
    assert missing == []


def test_every_internal_link_resolves():
    broken = []
    every = [*USER_DOCS, "docs/architecture.md", "docs/development.md"]
    every += ["SECURITY.md", "CONTRIBUTING.md"]
    for doc in every:
        path = ROOT / doc
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, anchor = target.partition("#")
            linked = (path.parent / file_part).resolve() if file_part else path
            if not linked.exists():
                broken.append(f"{doc}: {target}")
                continue
            if anchor and linked.suffix == ".md" and anchor not in _anchors(linked):
                broken.append(f"{doc}: {target} (no such heading)")
    assert broken == []


def _anchors(path: Path) -> set[str]:
    """Heading anchors the way GitHub makes them, because that is where these
    documents are read.

    Underscores survive: `[harness.<name>.sandbox_command]` is
    `#harnessnamesandbox_command` there. This used to strip them, so a link
    that this test called good was a 404 on the rendered page - and a link
    nobody could follow is the one kind of documentation error that a reader
    finds before the author does.
    """
    found = set()
    for line in path.read_text().splitlines():
        heading = re.match(r"#{1,6}\s+(.*)", line)
        if heading:
            slug = re.sub(r"[`*\[\]()]", "", heading.group(1).strip().lower())
            slug = re.sub(r"[^\w\s-]", "", slug)
            found.add(re.sub(r"\s+", "-", slug).strip("-"))
    return found


def test_the_documented_sandbox_command_is_the_one_buddy_ships():
    """The command in `docs/configuration.md` is what a reader will paste into
    their config, so it has to be the real one. It drifted once already: `{env}`
    was added to forward `[harness.<name>.env]` into the container, and the
    documented line - still correct-looking - would have left a sandboxed
    harness with no credentials at all."""
    from buddy.sandbox import DEFAULT_SANDBOX_COMMAND

    text = (ROOT / "docs" / "configuration.md").read_text()
    [documented] = re.findall(r'^sandbox_command = "(.+)"$', text, re.MULTILINE)
    # The docs name a concrete image where the shipped template has a hole,
    # so everything but the image has to match exactly.
    expected = "".join(
        r"\S+" if part == "{image}" else re.escape(part)
        for part in re.split(r"(\{image\})", DEFAULT_SANDBOX_COMMAND)
    )
    assert re.fullmatch(expected, documented), documented


def test_every_sandbox_placeholder_is_documented():
    """`wrap_in_sandbox` substitutes these names; a reader can only use one
    they have been told about, and an undocumented one is a silent no-op in
    anybody else's command."""
    import inspect

    from buddy import sandbox

    source = inspect.getsource(sandbox.wrap_in_sandbox)
    substituted = set(re.findall(r"^\s*(\w+)=", source, re.MULTILINE)) & set(
        re.findall(r"\{(\w+)\}", sandbox.DEFAULT_SANDBOX_COMMAND)
    )
    text = (ROOT / "docs" / "configuration.md").read_text()
    assert substituted, "no placeholders found - has wrap_in_sandbox been rewritten?"
    assert [name for name in sorted(substituted) if f"`{{{name}}}`" not in text] == []
