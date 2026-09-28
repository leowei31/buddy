"""Step 5: brain provider, credentials, and a probe that proves them.

Three rules:

- Secrets go to the OS keychain and never into `config.toml`.
  What lands in the config is the *name* of the variable, which is what
  `api_key_env` and `${keychain:NAME}` are for.
- An environment variable wins over the keychain, so a key exported for one
  session overrides a stored one without editing anything.
- `--yes` accepts every default, but a key is never a default: it is still
  prompted for, once.

Verification is a real call: "a one-token test call and a
context-feature probe per configured provider", which is exactly what
`provider.probe()` does, so a wrong key fails here rather than in the
middle of your first sentence.
"""

from __future__ import annotations

import os
from pathlib import Path

from buddy.config import (
    BrainSection,
    KeyringSecretResolver,
    keychain_available,
    keychain_get,
    keychain_set,
)
from buddy.providers import KNOWN as KNOWN_PROVIDERS
from buddy.providers import build as build_provider
from buddy.providers.base import ProviderError, ProviderNotInstalled
from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.setup.platform import run

#: What each provider needs before it can answer, and where config puts it.
API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

#: Providers that authenticate through Google Application Default Credentials
#: rather than a key Buddy can hold.
GOOGLE_PROVIDERS = ("anthropic_vertex", "vertex_gemini")

FISH_KEY_ENV = "FISH_API_KEY"

#: How each provider's keys begin. Used only to notice a paste that went
#: wrong, never to decide whether a key is real - that is the probe's job.
KEY_PREFIXES = {"ANTHROPIC_API_KEY": "sk-ant-", "OPENAI_API_KEY": "sk-"}


class BadKey(ValueError):
    """A paste that cannot be a key, with the reason in the message."""


def check_key(env: str, value: str) -> str:
    """Clean up a pasted secret, or say what is wrong with it.

    The prompt hides what you type, so a paste that did not register looks
    exactly like one that did - and pressing Cmd-V again concatenates. That
    is not a hypothetical: it is how a 319-character "key" got into a real
    keychain, three keys long, and every error after that pointed somewhere
    else.
    """
    cleaned = value.strip()
    if not cleaned:
        raise BadKey("nothing was entered")
    if any(character.isspace() for character in cleaned):
        raise BadKey("it contains a space or a newline, so something else came with it")

    prefix = KEY_PREFIXES.get(env)
    if prefix:
        if not cleaned.startswith(prefix):
            raise BadKey(f"{env} should start with {prefix!r}, and this starts {cleaned[:7]!r}")
        copies = cleaned.count(prefix)
        if copies > 1:
            raise BadKey(
                f"it looks like the key was pasted {copies} times "
                f"({len(cleaned)} characters, and {prefix!r} appears {copies} times). "
                "The prompt hides what you type, so a paste that did not seem to register "
                "often did."
            )
    return cleaned


def describe_key(value: str) -> str:
    """Enough to see that the right thing arrived, and no more."""
    return f"{len(value)} characters, ending {value[-4:]}"


def secret(name: str) -> str | None:
    """Env first, then the keychain - the precedence everything uses."""
    return os.environ.get(name) or keychain_get(name)


class KeysStep(BaseStep):
    name = "keys"
    title = "Keys and providers"
    number = "5"

    def _provider(self, ctx: SetupContext) -> str:
        planned = ctx.section("brain").get("provider")
        return ctx.options.brain or planned or "anthropic"

    async def contribute(self, ctx: SetupContext) -> None:
        provider = self._provider(ctx)
        ctx.section("brain")["provider"] = provider
        env = API_KEY_ENV.get(provider)
        if env:
            ctx.section("brain", provider)["api_key_env"] = env

    async def check(self, ctx: SetupContext) -> CheckResult:
        provider = self._provider(ctx)
        if provider not in KNOWN_PROVIDERS:
            return CheckResult(False, f"unknown provider {provider!r}")

        env = API_KEY_ENV.get(provider)
        pins = {"provider": provider}
        if env:
            if not secret(env):
                return CheckResult(False, f"{provider}: no {env} in the environment or keychain")
            pins["credential"] = env
        else:
            ok, detail = _google_credentials()
            if not ok:
                return CheckResult(False, f"{provider}: {detail}")
            pins["credential"] = detail
        return CheckResult(satisfied=True, detail=f"{provider} credentials present", pins=pins)

    async def act(self, ctx: SetupContext) -> None:
        provider = self._provider(ctx)
        if provider not in KNOWN_PROVIDERS:
            raise self.fail(
                f"unknown brain provider {provider!r}",
                hint=f"Pick one of: {', '.join(KNOWN_PROVIDERS)}",
            )
        ctx.section("brain")["provider"] = provider

        available, backend = keychain_available()
        if not available:
            ctx.ui.warn(
                f"No OS keychain here ({backend}), so keys cannot be stored. "
                "Export them in your shell instead; Buddy reads the environment first."
            )

        if provider in API_KEY_ENV:
            await self._collect_api_key(ctx, provider, available)
        else:
            await self._collect_google(ctx, provider)

        if not ctx.options.no_voice:
            await self._collect_fish_key(ctx, available)

    async def _collect_api_key(self, ctx: SetupContext, provider: str, keychain: bool) -> None:
        env = API_KEY_ENV[provider]
        stored = keychain_get(env)
        # `--force keys` has to mean "ask me again". Without this there is no
        # way to *replace* a stored key through setup, which is the one thing
        # you need when the stored one is wrong.
        forced = self.name in ctx.options.force
        stored_is_sane = True
        if stored and not forced:
            try:
                check_key(env, stored)
            except BadKey as bad:
                stored_is_sane = False
                ctx.ui.warn(f"The stored {env} cannot be right: {bad}")

        if os.environ.get(env):
            ctx.ui.detail(f"{env} is set in this environment; using it and storing nothing.")
        elif stored and not forced and stored_is_sane:
            ctx.ui.detail(f"{env} is already in the keychain ({describe_key(stored)}).")
        else:
            if stored:
                ctx.ui.say(f"Replacing the stored {env}.")
            # Prompted even under --yes: a key is the one thing
            # that is always asked for, and there is no sane default. Asked up
            # to three times, because a hidden prompt is easy to get wrong and
            # failing the whole run over a slipped paste helps nobody.
            value = ""
            for attempt in range(3):
                raw = ctx.ui.ask(f"{env} for {provider}", secret=True)
                try:
                    value = check_key(env, raw)
                    break
                except BadKey as bad:
                    ctx.ui.warn(f"That does not look right: {bad}")
                    if attempt == 2:
                        raise self.fail(
                            f"{provider} needs a usable {env}.",
                            hint=(
                                f"Export {env} in your shell, or re-run "
                                "`buddy setup --force keys` and paste it once."
                            ),
                        ) from bad
            if keychain:
                keychain_set(env, value)
                # Shown so a paste that did not register is visible immediately
                # rather than three commands later.
                ctx.ui.detail(f"Stored {env} in the OS keychain ({describe_key(value)}).")
            else:
                os.environ[env] = value
                ctx.ui.warn(f"Held {env} for this run only; export it to make it stick.")
        ctx.section("brain", provider)["api_key_env"] = env

    async def _collect_google(self, ctx: SetupContext, provider: str) -> None:
        ok, detail = _google_credentials()
        if not ok:
            ctx.ui.say(f"{provider} authenticates with Google Application Default Credentials.")
            ctx.ui.detail(detail)
            if ctx.ui.confirm("Run `gcloud auth application-default login` now?"):
                code, out, err = await run(
                    "gcloud", "auth", "application-default", "login", seconds=600
                )
                if code != 0:
                    raise self.fail(
                        f"gcloud login failed ({code}): {(err or out).strip()[-300:]}",
                        hint="Log in yourself, then: buddy setup --force keys",
                    )
            else:
                path = ctx.ui.ask("Path to a service-account JSON (blank to skip)").strip()
                if path:
                    resolved = _existing_file(path)
                    if resolved is None:
                        raise self.fail(f"no such file: {path}")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)
                    ctx.section("brain", provider)["credentials_file"] = str(resolved)

        block = ctx.section("brain", provider)
        project_key = "project_id" if provider == "anthropic_vertex" else "project"
        location_key = "region" if provider == "anthropic_vertex" else "location"
        default_project = block.get(project_key) or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        project = ctx.ui.ask(f"GCP project for {provider}", default=default_project).strip()
        if not project:
            raise self.fail(
                f"{provider} needs a GCP project.",
                hint="Find it with `gcloud config get-value project`.",
            )
        block[project_key] = project
        default_region = "global" if provider == "anthropic_vertex" else "us-central1"
        block[location_key] = ctx.ui.ask(
            "Region", default=block.get(location_key) or default_region
        ).strip()

    async def _collect_fish_key(self, ctx: SetupContext, keychain: bool) -> None:
        """Optional: cloud TTS is a fallback, not the primary."""
        if secret(FISH_KEY_ENV):
            ctx.ui.detail(f"{FISH_KEY_ENV} is available for cloud TTS.")
            return
        needs_cloud = ctx.options.cloud_tts or not ctx.platform().gpu.can_run_local_tts
        if not needs_cloud:
            return
        ctx.ui.say(
            "No GPU here that can run the local Fish server, so TTS would use the cloud."
            if not ctx.options.cloud_tts
            else "Cloud TTS was requested."
        )
        raw = ctx.ui.ask(f"{FISH_KEY_ENV} (blank to skip; voice will be text-only)", secret=True)
        if not raw.strip():
            ctx.ui.detail("Skipped. Buddy still runs; it just will not speak.")
            return
        try:
            value = check_key(FISH_KEY_ENV, raw)
        except BadKey as bad:
            ctx.ui.warn(f"Ignoring that {FISH_KEY_ENV}: {bad}")
            return
        if keychain:
            keychain_set(FISH_KEY_ENV, value)
        else:
            os.environ[FISH_KEY_ENV] = value
        ctx.section("voice", "fish_cloud")["api_key_env"] = FISH_KEY_ENV

    async def verify(self, ctx: SetupContext) -> str:
        """A one-token test call and a context-feature probe."""
        provider = self._provider(ctx)
        found = await self.check(ctx)
        if not found.satisfied:
            raise self.fail(found.detail, hint="Re-run: buddy setup --force keys")

        # The same resolver the session will use, so what setup verifies is
        # what `buddy` will actually do.
        brain = BrainSection(provider=provider, provider_options=_options(ctx, provider))
        try:
            built = build_provider(brain, resolver=KeyringSecretResolver())
        except (ProviderError, ProviderNotInstalled) as exc:
            raise self.fail(str(exc), hint="Install the extra it names, then re-run.") from exc

        result = await built.probe()
        if not result.reachable:
            raise self.fail(
                f"{provider} did not answer: {result.summary()}",
                hint="Check the key and the model name, then: buddy setup --force keys",
            )
        ctx.section("brain")["model"] = built.model
        # The probe is also what tells context management whether the server-side
        # edits exist here, so the answer is worth showing.
        return f"{provider} ({built.model}) answered - {result.summary()}"


def _options(ctx: SetupContext, provider: str) -> dict[str, dict]:
    """The provider's block, `api_key_env` included: `build` resolves it."""
    block = dict(ctx.section("brain", provider))
    return {provider: block} if block else {}


def _existing_file(path: str) -> Path | None:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_file() else None


def _google_credentials() -> tuple[bool, str]:
    """Whether Google ADC would work here, and what was found.

    Sync on purpose: it is three `stat` calls in an interactive wizard, and
    The non-blocking rule is about the orchestrator's loop while agents are
    running, not about a prompt that is already waiting on a human.
    """
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit and _existing_file(explicit):
        return True, f"service account at {explicit}"
    if _existing_file(str(Path.home() / ".config/gcloud/application_default_credentials.json")):
        return True, "gcloud application-default credentials"
    return False, "no application-default credentials found"
