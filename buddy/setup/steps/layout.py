"""Step 4: the `~/.buddy` tree, the schema, and the FTS index.

The cheapest step to get right and the worst one to get wrong: everything
after it writes into this tree. Opening the `Store` is what runs the
migrations, so "create the layout" and "bring the schema up to date" are the
same act.
"""

from __future__ import annotations

from buddy.config import Config
from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.state import SCHEMA_VERSION, Store


class LayoutStep(BaseStep):
    name = "layout"
    title = "Layout and database"
    number = "4"

    def _paths(self, ctx: SetupContext) -> list:
        paths = ctx.paths
        return [paths.home, paths.tasks, paths.worktrees]

    async def check(self, ctx: SetupContext) -> CheckResult:
        missing = [str(path) for path in self._paths(ctx) if not path.is_dir()]
        paths = ctx.paths
        if missing:
            return CheckResult(False, f"missing: {', '.join(missing)}")
        if not paths.db.exists():
            return CheckResult(False, f"no database at {paths.db}")

        with Store(paths.db) as store:
            version = store.schema_version()
            tables = store.table_names()
        current = version == SCHEMA_VERSION
        # The FTS index backs `recall` (context layer 3); without it the brain
        # loses its memory of past conversations, silently.
        has_fts = "conversation_log_fts" in tables
        return CheckResult(
            satisfied=current and has_fts,
            detail=(
                f"schema v{version} of v{SCHEMA_VERSION}"
                + ("" if has_fts else ", FTS index missing")
            ),
            pins={"schema": str(SCHEMA_VERSION)},
        )

    async def act(self, ctx: SetupContext) -> None:
        paths = ctx.paths
        paths.ensure()
        # Secrets live in the keychain, but Vertex service-account JSON is a
        # file the user points at, so the directory that holds it is
        # created private rather than world-readable.
        paths.keys.mkdir(parents=True, exist_ok=True)
        paths.keys.chmod(0o700)
        with Store(paths.db) as store:
            store.ensure_slots()
        ctx.ui.detail(f"{paths.home} ready, schema v{SCHEMA_VERSION}")

    async def verify(self, ctx: SetupContext) -> str:
        result = await self.check(ctx)
        if not result.satisfied:
            raise self.fail(
                f"the layout is still not right: {result.detail}",
                hint=f"Check permissions on {ctx.paths.home}.",
            )
        with Store(ctx.paths.db) as store:
            slots = len(store.load_slots())
        config = Config(home=ctx.home)
        return f"{config.paths.home}, {result.detail}, {slots} slots"
