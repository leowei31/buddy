"""Claude on Google Cloud via AnthropicVertex.

The same Messages API, so everything about serialization, streaming, and the
context-management edits is inherited. Only the client and the auth differ:
GCP Application Default Credentials rather than an Anthropic key.

The context betas list Google Cloud as supported, but a given model or region
may still reject one, which is why `probe()` asks instead of assuming.
"""

from __future__ import annotations

from typing import Any

from buddy.providers.anthropic import DEFAULT_MODEL, AnthropicProvider
from buddy.providers.base import ProviderError, ProviderNotInstalled


class AnthropicVertexProvider(AnthropicProvider):
    name = "anthropic_vertex"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        project_id: str | None = None,
        region: str = "global",
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.project_id = project_id
        self.region = region
        super().__init__(model, client=client, **kwargs)

    def _make_client(self, api_key: str | None) -> Any:
        try:
            from anthropic import AsyncAnthropicVertex
        except ImportError as exc:
            raise ProviderNotInstalled(self.name, "vertex", "anthropic[vertex]") from exc
        # Auth is ADC (`gcloud auth application-default login`) or a service
        # account JSON via GOOGLE_APPLICATION_CREDENTIALS. No Anthropic key.
        # Only what is configured. The SDK treats `project_id=None` as given,
        # which overwrote its own ANTHROPIC_VERTEX_PROJECT_ID fallback and
        # quietly used the default-credentials project instead.
        options: dict[str, Any] = {"region": self.region}
        if self.project_id:
            options["project_id"] = self.project_id
        try:
            return AsyncAnthropicVertex(**options)
        except Exception as exc:
            raise ProviderError(f"vertex client could not be created: {exc}") from exc
