"""Step 8: probe the harness CLIs and write the ones that pass.

The rule that shapes this step: setup does not install
or log in to harness CLIs. They are the user's own tools, with their own
subscriptions and their own auth, and a orchestrator that installs them
behind your back is one that logs you in to something behind your back too.
`--install-harnesses` is the opt-in.

What setup *does* do is run each adapter's `preflight()` and only write the
ones that pass all four requirements, with the command template that was
actually verified rather than one remembered from an older release.
"""

from __future__ import annotations

import shutil

from buddy.harnesses import ADAPTERS
from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.setup.steps.keys import API_KEY_ENV

#: The binary each adapter drives, for the "is it even here?" probe.
BINARIES = {name: adapter.binary for name, adapter in ADAPTERS.items()}

#: `npm i -g` names, used only under --install-harnesses. A CLI without one
#: (Antigravity ships a native installer) is never installed for you: setup
#: prints its install line instead, because running someone else's install
#: script is a decision, not a default.
NPM_PACKAGES = {name: a.npm_package for name, a in ADAPTERS.items() if a.npm_package}


class HarnessesStep(BaseStep):
    name = "harnesses"
    title = "Harnesses"
    number = "8"

    #: The last preflight, so `contribute` does not run every CLI a third time.
    _cached: dict | None = None

    async def _preflight(self, ctx: SetupContext) -> dict[str, tuple[bool, str, list[str]]]:
        """Every known adapter, run against the installed CLI."""
        from buddy.config import Config, HarnessConfig

        results: dict[str, tuple[bool, str, list[str]]] = {}
        planned = ctx.section("harness")
        defaults = Config(home=ctx.home).harnesses
        for name, factory in ADAPTERS.items():
            if not shutil.which(BINARIES.get(name, name)):
                results[name] = (False, f"{BINARIES.get(name, name)} is not on PATH", [])
                continue
            block = planned.get(name) or {}
            if not block.get("command"):
                block = {**block, "command": factory.default_command}
            config = defaults.get(name) or HarnessConfig.from_dict(name, dict(block))
            report = await factory(config).preflight()
            notes = list(report.notes)
            if report.authenticated is False:
                notes.insert(0, f"not signed in: {report.auth_detail}")
            elif report.auth_detail:
                notes.insert(0, report.auth_detail)
            # Only a harness that can run a task now is written. A CLI that
            # is installed but signed out may be one you never meant Buddy to
            # use, and writing it would make doctor fail setup on your behalf
            # for it. It is named with its fix instead; logging in is yours
            # to do, and `--force harnesses` picks it up afterwards.
            results[name] = (report.usable, report.summary(), notes)
        self._cached = results
        return results

    async def contribute(self, ctx: SetupContext) -> None:
        results = self._cached or await self._preflight(ctx)
        for name, (ok, _, _) in results.items():
            if ok:
                self._plan(ctx, name)

    async def check(self, ctx: SetupContext) -> CheckResult:
        results = await self._preflight(ctx)
        passing = sorted(name for name, (ok, _, _) in results.items() if ok)
        detail = "; ".join(
            f"{name}: {summary}" for name, (_, summary, _) in sorted(results.items())
        )
        return CheckResult(
            # At least one working harness is the bar: with none, Buddy has
            # nothing to run and every task would queue forever.
            satisfied=bool(passing),
            detail=detail,
            pins={"harnesses": ",".join(passing)},
        )

    async def act(self, ctx: SetupContext) -> None:
        results = await self._preflight(ctx)
        missing = [
            name for name, (ok, summary, _) in results.items() if not ok and "PATH" in summary
        ]

        if missing and ctx.options.install_harnesses:
            await self._install(ctx, missing)
            results = await self._preflight(ctx)

        for name, (ok, summary, notes) in sorted(results.items()):
            ctx.ui.detail(f"{name}: {'ok' if ok else 'unavailable'} - {summary}")
            for note in notes:
                ctx.ui.warn(f"  {name}: {note}")

        if not any(ok for ok, _, _ in results.values()):
            hints = "; ".join(f"{name}: {a.install_hint}" for name, a in sorted(ADAPTERS.items()))
            raise self.fail(
                "no coding-agent CLI passed preflight, so Buddy would have nothing to run.",
                hint=(
                    f"Install one and log in to it - {hints}. Then: buddy setup --force "
                    "harnesses. Setup does not install them for you; "
                    "--install-harnesses opts in for the npm ones."
                ),
            )

    async def _install(self, ctx: SetupContext, names: list[str]) -> None:
        from buddy.setup.platform import run

        if not shutil.which("npm"):
            ctx.ui.warn("--install-harnesses needs npm, which is not on PATH. Skipping.")
            return
        for name in names:
            if name not in NPM_PACKAGES:
                hint = ADAPTERS[name].install_hint
                ctx.ui.detail(f"{name} is not on npm; install it yourself: {hint}")
        packages = [NPM_PACKAGES[name] for name in names if name in NPM_PACKAGES]
        if not packages:
            return
        printed = f"npm install -g {' '.join(packages)}"
        ctx.ui.say(f"Installing: {', '.join(packages)}")
        ctx.ui.detail(printed)
        if not ctx.ui.confirm("Run it?"):
            return
        code, out, err = await run("npm", "install", "-g", *packages, seconds=900)
        if code != 0:
            ctx.ui.warn(f"`{printed}` failed ({code}): {(err or out).strip()[-200:]}")
        else:
            ctx.ui.detail("Installed. You still have to log in to each one yourself.")

    def _plan(self, ctx: SetupContext, name: str) -> None:
        """Write the verified template, and wire the credentials from step 5.

        Each harness's `[env]` points at the provider
        credentials step 5 collected, through `${keychain:NAME}` so the
        secret itself never reaches config.toml.
        """
        from buddy.config import Config

        block = ctx.section("harness", name)
        adapter = ADAPTERS[name]
        installed = Config(home=ctx.home).harnesses.get(name)
        # The template that preflight actually verified, and never over the
        # top of one already in the config.
        verified = installed.command if installed and installed.command else ""
        block.setdefault("command", verified or adapter.default_command)
        chosen = (installed.default_model if installed else None) or adapter.default_model
        if chosen:
            block.setdefault("default_model", chosen)
        source = installed.waiting_patterns if installed else adapter.default_waiting_patterns
        patterns = list(source)
        if patterns:
            block.setdefault("waiting_patterns", [getattr(p, "pattern", p) for p in patterns])

        # Point the harness at the credentials step 5 collected -
        # but only under a name this CLI actually reads. Codex ignores
        # OPENAI_API_KEY in exec mode; exporting it there would look wired and
        # fail every run with "Missing bearer".
        provider = ctx.section("brain").get("provider", "")
        collected = API_KEY_ENV.get(provider)
        target = adapter.credential_env.get(collected) if collected else None
        if target:
            env = ctx.section("harness", name, "env")
            env.setdefault(target, f"${{keychain:{collected}}}")

    async def verify(self, ctx: SetupContext) -> str:
        result = await self.check(ctx)
        if not result.satisfied:
            raise self.fail(result.detail, hint="Install and log in to a harness CLI.")
        working = ", ".join(sorted(ctx.section("harness")))
        return f"{working} passed preflight"
