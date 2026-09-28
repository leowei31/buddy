"""OpenAI, and anything speaking its protocol.

`base_url` makes this the adapter for any OpenAI-compatible endpoint - Azure
OpenAI, vLLM, Ollama - which is why it is worth more than one provider.

No native context management: Layers 1b and 2b do the work client-side, and
`Capabilities` says so, so `ContextManager` never asks for something the API
cannot do.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from buddy.providers.base import (
    Capabilities,
    Finished,
    ProbeResult,
    ProviderError,
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

DEFAULT_MODEL = "gpt-5"


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        max_tokens: int = 16000,
    ) -> None:
        self.model = model
        self.base_url = base_url or None
        self.max_tokens = max_tokens
        self._client = client or self._make_client(api_key)
        self.capabilities = Capabilities(
            server_compaction=False,
            tool_result_clearing=False,
            prompt_caching=True,  # automatic and transparent; nothing to declare
            mid_conversation_system=True,  # chat completions take system turns anywhere
            count_tokens=False,
            max_context=400_000,
        )

    def _make_client(self, api_key: str | None) -> Any:
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        try:
            return AsyncOpenAI(**kwargs)
        except Exception as exc:
            # Usually a missing OPENAI_API_KEY. The caller gets a sentence,
            # not a traceback.
            raise ProviderError(f"openai client could not be created: {exc}") from exc

    # -- serialization -----------------------------------------------------

    def to_wire(self, system: str, messages: Sequence[Turn]) -> list[dict[str, Any]]:
        """Canonical turns to chat-completions messages.

        Tool results become their own `role: "tool"` entries, and an assistant
        turn carries its calls in `tool_calls`, which is the shape this API
        needs and the reason serialization belongs in the provider.
        """
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in messages:
            if turn.role is Role.SYSTEM:
                wire.append({"role": "system", "content": turn.text})
                continue
            if turn.role is Role.USER:
                results = [b for b in turn.blocks if isinstance(b, ToolResult)]
                for result in results:
                    wire.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": (
                                f"ERROR: {result.content}" if result.is_error else result.content
                            ),
                        }
                    )
                text = turn.text
                if text:
                    wire.append({"role": "user", "content": text})
                continue

            calls = [b for b in turn.blocks if isinstance(b, ToolCall)]
            entry: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in calls
                ]
            if entry["content"] or calls:
                wire.append(entry)
        return wire

    def tools_to_wire(self, tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    # -- streaming ---------------------------------------------------------

    async def stream(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef] = (),
        *,
        budgets: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_wire(system, messages),
            "max_completion_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = self.tools_to_wire(tools)

        text_parts: list[str] = []
        partial: dict[int, dict[str, Any]] = {}
        usage = Usage()
        stop_reason = "end_turn"

        try:
            stream = await self._client.chat.completions.create(**request)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )
                for choice in getattr(chunk, "choices", []) or []:
                    if choice.finish_reason:
                        stop_reason = _stop_reason(choice.finish_reason)
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    if getattr(delta, "content", None):
                        text_parts.append(delta.content)
                        yield TextDelta(delta.content)
                    for call in getattr(delta, "tool_calls", None) or []:
                        slot = partial.setdefault(
                            call.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if call.id:
                            slot["id"] = call.id
                        function = getattr(call, "function", None)
                        if function is not None:
                            if getattr(function, "name", None):
                                slot["name"] = function.name
                            if getattr(function, "arguments", None):
                                slot["arguments"] += function.arguments
        except Exception as exc:
            raise ProviderError(f"openai stream failed: {exc}") from exc

        blocks: list[Any] = []
        joined = "".join(text_parts)
        if joined:
            blocks.append(Text(joined))
        for index in sorted(partial):
            slot = partial[index]
            arguments, problem = _parse_arguments(slot["arguments"])
            call = ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=arguments,
                arguments_error=problem,
            )
            blocks.append(call)
            yield ToolCallReady(call)

        yield usage
        yield Finished(stop_reason=stop_reason, turn=Turn(Role.ASSISTANT, blocks))

    async def count_tokens(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef] = (),
    ) -> int:
        """No counting endpoint here, so Buddy estimates."""
        return estimate_tokens(system, messages, tools)

    async def aclose(self) -> None:
        """Close the SDK client's connection pool."""
        closer = getattr(self._client, "close", None)
        if closer is not None:
            with contextlib.suppress(Exception):
                await closer()

    async def probe(self) -> ProbeResult:
        try:
            await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=1,
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
            detail="client-side context management"
            + (f"; base_url {self.base_url}" if self.base_url else ""),
        )


def _stop_reason(finish: str) -> str:
    return {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}.get(
        finish, finish
    )


def _parse_arguments(arguments: str) -> tuple[dict[str, Any], str]:
    """Tool arguments arrive as a JSON string, assembled from deltas.

    Never string-matched: escaping varies, so it is parsed, or the reason it
    could not be is returned alongside an empty dict.
    """
    if not arguments.strip():
        return {}, ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return {}, f"not valid JSON ({exc.msg} at character {exc.pos})"
    if not isinstance(parsed, dict):
        return {}, f"a JSON {type(parsed).__name__}, not an object"
    return parsed, ""


def _loads(arguments: str) -> dict[str, Any]:
    return _parse_arguments(arguments)[0]
