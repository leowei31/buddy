"""Step 10: write `config.toml` from what the earlier steps verified.

Two halves, because two different things are asked of it:

- A machine with no config gets a fresh, fully commented file,
  filled in from what actually worked rather than from placeholders.
- A machine with a config gets *new keys merged in and nothing overwritten*.
  That is the promise that makes re-running setup safe on a config you have
  hand-edited, so the existing file is edited with `tomlkit` and keeps its
  comments, its ordering, and every value you chose.

No secret is ever written here. What goes in is the *name* of the variable
(`api_key_env`, `${keychain:NAME}`); the value stays in the keychain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit

from buddy.config import Config, ConfigError
from buddy.setup import BaseStep, CheckResult, SetupContext

TEMPLATE = """\
# Buddy's configuration. Written by `buddy setup`; yours to edit.
# Re-running setup merges in new keys and never overwrites what is here.
#
# Secrets are not in this file. They live in the OS keychain; what appears
# here is only the name to look up.

[buddy]
max_concurrent   = 7
default_priority = 3
stall_timeout    = "10m"          # no output for this long -> notify, never kill
max_runtime      = "2h"           # then checkpoint, requeue once, error the second time
web_port         = {web_port}
trust_mode       = false          # true = no spoken read-back when spawning

[brain]
provider               = "{provider}"
model                  = "{model}"
strategy               = "auto"   # server-side context edits where the provider has them
compact_trigger_tokens = 100000   # the API minimum is 50000
keep_recent_turns      = 6
tool_output_tail_lines = 60
read_file_max_bytes    = 20000

[brain.clear_tool_uses]
trigger = 40000
keep    = 5
"""


def _existing_dir(answer: str) -> Path | None:
    path = Path(answer).expanduser()
    return path if path.is_dir() else None


def _table(data: dict[str, Any]) -> Any:
    """A dict as a TOML table, recursing into nested dicts."""
    table = tomlkit.table()
    for key, value in data.items():
        table[key] = _table(value) if isinstance(value, dict) else value
    return table


def merge_missing(document: Any, plan: dict[str, Any]) -> list[str]:
    """Add what is absent, change nothing that is present.

    Returns the dotted paths that were added, so setup can say what it did
    rather than claiming to have written a file it may not have touched.
    """
    added: list[str] = []

    def walk(node: Any, values: dict[str, Any], prefix: str) -> None:
        for key, value in values.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                if key not in node:
                    node[key] = _table(value)
                    added.append(path)
                    continue
                walk(node[key], value, f"{path}.")
            elif key not in node:
                node[key] = value
                added.append(path)

    walk(document, plan, "")
    return added


class ConfigStep(BaseStep):
    name = "config"
    title = "Configuration"
    number = "10"

    async def check(self, ctx: SetupContext) -> CheckResult:
        path = ctx.paths.config_file
        if not path.exists():
            return CheckResult(False, f"no config at {path}")
        try:
            config = Config.load(home=ctx.home)
        except ConfigError as exc:
            return CheckResult(False, f"{path} does not parse: {exc}")
        document = tomlkit.parse(path.read_text())
        pending = merge_missing(tomlkit.parse(path.read_text()), ctx.plan)
        return CheckResult(
            satisfied=not pending,
            detail=(
                f"{path}, {len(config.projects)} project(s), {len(config.harnesses)} harness(es)"
                if not pending
                else f"{len(pending)} key(s) to add: {', '.join(pending[:4])}"
            ),
            pins={"config": str(path), "sections": ",".join(sorted(document.keys()))},
        )

    async def act(self, ctx: SetupContext) -> None:
        path = ctx.paths.config_file
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            brain = ctx.section("brain")
            path.write_text(
                TEMPLATE.format(
                    web_port=4321,
                    provider=brain.get("provider", "anthropic"),
                    model=brain.get("model", ""),
                )
            )
            ctx.ui.detail(f"Wrote a fresh {path}")

        await self._ask_for_a_project(ctx)

        document = tomlkit.parse(path.read_text())
        added = merge_missing(document, ctx.plan)
        path.write_text(tomlkit.dumps(document))
        path.chmod(0o600)
        if added:
            ctx.ui.detail(f"Added: {', '.join(added)}")
        else:
            ctx.ui.detail("Nothing to add; every key was already there.")

    async def _ask_for_a_project(self, ctx: SetupContext) -> None:
        """One project, or Buddy has nowhere to send work.

        Setup's goal is a *working* Buddy from one command, and with no
        `[projects.*]` every task would refuse to spawn. Asked once,
        skippable, and never asked again once one exists.
        """
        existing = Config.load(home=ctx.home).projects
        if existing or ctx.section("projects"):
            return
        ctx.ui.say("Buddy needs at least one project to send work to.")
        answer = ctx.ui.ask("Path to a git repo (blank to add it later by hand)").strip()
        if not answer:
            ctx.ui.detail("Skipped. Add a [projects.<name>] block to config.toml when ready.")
            return
        path = _existing_dir(answer)
        if path is None:
            ctx.ui.warn(f"{answer} is not a directory; skipping.")
            return
        name = ctx.ui.ask("Call it", default=path.name.replace("-", "_")).strip() or path.name
        block = ctx.section("projects", name)
        block["path"] = str(path)
        block["base_branch"] = "main"
        harnesses = sorted(ctx.section("harness"))
        if harnesses:
            block["default_harness"] = harnesses[0]

    async def verify(self, ctx: SetupContext) -> str:
        path = ctx.paths.config_file
        try:
            config = Config.load(home=ctx.home)
        except ConfigError as exc:
            raise self.fail(
                f"the config that was written does not validate: {exc}",
                hint=f"Fix {path} by hand, then: buddy setup --force config",
            ) from exc
        projects = ", ".join(sorted(config.projects)) or "none"
        harnesses = ", ".join(sorted(config.harnesses)) or "none"
        return (
            f"{path} parses - brain {config.brain.provider}, "
            f"projects {projects}, harnesses {harnesses}"
        )
