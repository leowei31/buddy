"""Gemini on Vertex via google-genai.

Auth is GCP Application Default Credentials, the same as `anthropic_vertex`.
No native context management, so Layers 1b and 2b handle it client-side.

Gemini's wire format differs more than OpenAI's: turns are `contents` with
`parts`, the system prompt is `system_instruction` rather than a message, and
a tool result is a `function_response` part keyed by the function *name*
rather than by a call id. All of that is contained here, which is the point
of the canonical format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from buddy.providers.base import (
    Capabilities,
    Finished,
    ProbeResult,
    ProviderError,
    ProviderNotInstalled,
    Role,
    StreamEvent,
    Text,
    TextDelta,
    ToolCall,
    ToolCallReady,
    ToolDef,
    ToolResult,
    Turn,
    Usage,
    estimate_tokens,
)

DEFAULT_MODEL = "gemini-3-pro"


class VertexGeminiProvider:
    name = "vertex_gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        project: str | None = None,
        location: str = "us-central1",
        client: Any = None,
        max_tokens: int = 16000,
    ) -> None:
        self.model = model
        self.project = project
        self.location = location
        self.max_tokens = max_tokens
        self._client = client or self._make_client()
        self.capabilities = Capabilities(
            server_compaction=False,
            tool_result_clearing=False,
            prompt_caching=True,
            mid_conversation_system=False,  # only `system_instruction`
            count_tokens=True,  # count_tokens exists on the models API
            max_context=1_000_000,
        )

    def _make_client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderNotInstalled(self.name, "gemini", "google-genai") from exc
        try:
            return genai.Client(vertexai=True, project=self.project, location=self.location)
        except Exception as exc:
            raise ProviderError(f"vertex gemini client could not be created: {exc}") from exc

    # -- serialization -----------------------------------------------------

    def to_wire(self, messages: Sequence[Turn]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        #: Gemini keys a function response by name, not by call id, so the
        #: id -> name mapping has to be carried forward.
        names: dict[str, str] = {}
        for turn in messages:
            parts: list[dict[str, Any]] = []
            for block in turn.blocks:
                if isinstance(block, Text) and block.text:
                    parts.append({"text": block.text})
                elif isinstance(block, ToolCall):
                    names[block.id] = block.name
                    parts.append({"function_call": {"name": block.name, "args": block.arguments}})
                elif isinstance(block, ToolResult):
                    parts.append(
                        {
                            "function_response": {
                                "name": names.get(block.call_id, block.call_id),
                                "response": {
                                    "error" if block.is_error else "output": block.content
                                },
                            }
                        }
                    )
            if not parts:
                continue
            # There is no system role in `contents`; an operator instruction
            # is delivered as a user part, which is the closest honest mapping.
            role = "model" if turn.role is Role.ASSISTANT else "user"
            contents.append({"role": role, "parts": parts})
        return contents

    def tools_to_wire(self, tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
        return [
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tools
                ]
            }
        ]

    def _config(self, system: str, tools: Sequence[ToolDef]) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
        }
        if tools:
            config["tools"] = self.tools_to_wire(tools)
        return config

    # -- streaming ---------------------------------------------------------

    async def stream(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef] = (),
        *,
        budgets: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=self.to_wire(messages),
                config=self._config(system, tools),
            )
            async for chunk in stream:
                metadata = getattr(chunk, "usage_metadata", None)
                if metadata is not None:
                    usage = Usage(
                        input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
                    )
                for candidate in getattr(chunk, "candidates", None) or []:
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        text = getattr(part, "text", None)
                        if text:
                            text_parts.append(text)
                            yield TextDelta(text)
                        function_call = getattr(part, "function_call", None)
                        if function_call is not None:
                            call = ToolCall(
                                id=getattr(function_call, "id", None) or f"call_{len(calls)}",
                                name=function_call.name,
                                arguments=dict(getattr(function_call, "args", None) or {}),
                            )
                            calls.append(call)
                            yield ToolCallReady(call)
        except Exception as exc:
            raise ProviderError(f"vertex gemini stream failed: {exc}") from exc

        blocks: list[Any] = []
        joined = "".join(text_parts)
        if joined:
            blocks.append(Text(joined))
        blocks.extend(calls)

        yield usage
        yield Finished(
            stop_reason="tool_use" if calls else "end_turn",
            turn=Turn(Role.ASSISTANT, blocks),
        )

    async def count_tokens(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef] = (),
    ) -> int:
        try:
            counted = await self._client.aio.models.count_tokens(
                model=self.model, contents=self.to_wire(messages)
            )
        except Exception:
            # Counting is a convenience here, never the thing that must work.
            return estimate_tokens(system, messages, tools)
        return int(getattr(counted, "total_tokens", 0))

    async def aclose(self) -> None:
        """google-genai's client owns no pool Buddy can close, so there is
        nothing to release here."""
        return None

    async def probe(self) -> ProbeResult:
        try:
            await self._client.aio.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": "hi"}]}],
                config={"max_output_tokens": 1},
            )
        except Exception as exc:
            return ProbeResult(
                provider=self.name,
                model=self.model,
                reachable=False,
                capabilities=Capabilities(),
                error=str(exc).splitlines()[0][:200],
            )
        return ProbeResult(
            provider=self.name,
            model=self.model,
            reachable=True,
            capabilities=self.capabilities,
            detail=f"project {self.project}, {self.location}; client-side context management",
        )
