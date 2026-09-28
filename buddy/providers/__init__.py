"""Model providers for the brain.

`build` is the only place that maps a config name to an implementation, so
adding a fifth provider touches this file and one new module - nothing else.
"""

from __future__ import annotations

from typing import Any

from buddy.config import BrainSection, EnvSecretResolver, SecretResolver
from buddy.providers.base import (
    Capabilities,
    ModelProvider,
    ProbeResult,
    ProviderError,
    ProviderNotInstalled,
    Turn,
)

KNOWN = ("anthropic", "anthropic_vertex", "openai", "vertex_gemini")


#: Providers that authenticate with a key Buddy can hold. The Vertex two use
#: Google Application Default Credentials, which are a file and a login, not
#: a string to pass in.
KEYED = ("anthropic", "openai")


def build(
    brain: BrainSection,
    *,
    provider: str | None = None,
    resolver: SecretResolver | None = None,
    **overrides: Any,
) -> ModelProvider:
    """The provider named by `[brain] provider`, with its own option block.

    `api_key_env` names *where* the key is, never the key itself, so it
    has to be resolved - through the environment first and then the OS
    keychain, which is the precedence setup uses.

    Skipping that step is what made `buddy setup` store a key, say so, and
    leave `buddy doctor` reporting the brain as unauthenticated: the SDK falls
    back to reading `ANTHROPIC_API_KEY` from the environment, and setup had
    deliberately not put it there.
    """
    name = provider or brain.provider
    options = dict(brain.options_for(name))
    # `[brain.<provider>]` goes straight into a constructor, so a renamed or
    # typo'd option - the ordinary "old config, new Buddy" case - raised a
    # bare TypeError past every caller, all of which catch only ProviderError.

    key_env = options.pop("api_key_env", None)
    options.update(overrides)
    model = options.pop("model", None) or brain.model

    if key_env and name in KEYED and "api_key" not in options:
        found = (resolver or EnvSecretResolver()).resolve(str(key_env))
        if found:
            options["api_key"] = found

    if name == "anthropic":
        from buddy.providers.anthropic import DEFAULT_MODEL, AnthropicProvider

        return _construct(AnthropicProvider, name, model or DEFAULT_MODEL, options)
    if name == "anthropic_vertex":
        from buddy.providers.anthropic import DEFAULT_MODEL
        from buddy.providers.anthropic_vertex import AnthropicVertexProvider

        return _construct(AnthropicVertexProvider, name, model or DEFAULT_MODEL, options)
    if name == "openai":
        from buddy.providers.openai import DEFAULT_MODEL, OpenAIProvider

        return _construct(OpenAIProvider, name, model or DEFAULT_MODEL, options)
    if name == "vertex_gemini":
        from buddy.providers.vertex_gemini import DEFAULT_MODEL, VertexGeminiProvider

        return _construct(VertexGeminiProvider, name, model or DEFAULT_MODEL, options)
    raise ProviderError(f"unknown provider {name!r} (known: {', '.join(KNOWN)})")


def _construct(factory: Any, name: str, model: str, options: dict[str, Any]) -> ModelProvider:
    """Build one, or say which config key it did not recognise."""
    try:
        return factory(model, **options)
    except TypeError as exc:
        unknown = ", ".join(sorted(options)) or "none"
        raise ProviderError(
            f"[brain.{name}] has an option this provider does not take ({exc}). "
            f"Keys given: {unknown}."
        ) from exc


__all__ = [
    "KEYED",
    "KNOWN",
    "Capabilities",
    "ModelProvider",
    "ProbeResult",
    "ProviderError",
    "ProviderNotInstalled",
    "Turn",
    "build",
]
