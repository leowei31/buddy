"""One module per setup step, each a `check -> act -> verify` triple.

A step never reaches for the terminal directly (it goes through `ctx.ui`),
never probes what step 1 already reported (it reads `ctx.platform()`), and
never edits `config.toml` (it fills `ctx.plan`, which step 10 renders once).
"""
