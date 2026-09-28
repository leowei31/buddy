"""Anthropic Messages API.

The only provider with native Layer 1 (tool-result clearing) and Layer 2
(server compaction), both of which live under `context_management.edits` but
carry different beta headers:

* `context-management-2025-06-27` + `clear_tool_uses_20250919` - clears old
  tool *results* while leaving the `tool_use` block, so the brain still knows
  it made the call and can re-call it (context layer 1).
* `compact-2026-01-12` + `compact_20260112` - summarizes the conversation
  into a `compaction` block (context layer 2).

Both are betas, so `probe()` establishes whether this model and account
actually accept them; `ContextManager` reads the answer rather than a table.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from buddy.providers.base import (
    Capabilities,
    Compaction,
    CompactionHappened,
    Finished,
    ProbeResult,
    ProviderError,
    Role,
    StreamEvent,
    Text,
    TextDelta,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolCallReady,
    ToolDef,
    ToolResult,
    Turn,
    Usage,
)

CONTEXT_EDIT_BETA = "context-management-2025-06-27"
COMPACTION_BETA = "compact-2026-01-12"
CLEAR_TOOL_USES = "clear_tool_uses_20250919"
COMPACT_EDIT = "compact_20260112"

DEFAULT_MODEL = "claude-opus-5"

#: Models that accept an operator instruction inside `messages` rather than
#: only in the top-level `system` field. That is where Layer 0's fresh agent
#: table belongs: putting volatile state in `system` would invalidate the
#: cached prefix on every single turn.
MID_CONVERSATION_SYSTEM_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-mythos-5-1",
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any = None,
        max_tokens: int = 16000,
        effort: str = "high",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = client or self._make_client(api_key)
        self.capabilities = self._assumed_capabilities()

    def _make_client(self, api_key: str | None) -> Any:
        import anthropic

        # A bare constructor is correct: the SDK resolves ANTHROPIC_API_KEY,
        # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile in that order,
        # so an unset env var does not mean there are no credentials.
        try:
            return (
                anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
            )
        except Exception as exc:
            raise ProviderError(f"anthropic client could not be created: {exc}") from exc

    def _assumed_capabilities(self) -> Capabilities:
        """What we expect before `probe()` confirms it."""
        return Capabilities(
            server_compaction=True,
            tool_result_clearing=True,
            prompt_caching=True,
            mid_conversation_system=any(
                self.model.startswith(name) for name in MID_CONVERSATION_SYSTEM_MODELS
            ),
            count_tokens=True,
            max_context=1_000_000,
        )

    # -- serialization -----------------------------------------------------

    def to_wire(self, messages: Sequence[Turn]) -> list[dict[str, Any]]:
        """Canonical turns to Anthropic message params."""
        wire: list[dict[str, Any]] = []
        for turn in messages:
            content = [block for block in (self._block_to_wire(b) for b in turn.blocks) if block]
            if not content:
                continue
            if turn.role is Role.SYSTEM:
                # A mid-conversation operator instruction. Providers without
                # it never see this role: `Brain` folds it in beforehand.
                wire.append({"role": "system", "content": turn.text})
                continue
            wire.append({"role": turn.role.value, "content": content})
        return wire

    def _block_to_wire(self, block: Any) -> dict[str, Any] | None:
        if isinstance(block, Text):
            return {"type": "text", "text": block.text} if block.text else None
        if isinstance(block, ToolCall):
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.arguments,
            }
        if isinstance(block, ToolResult):
            return {
                "type": "tool_result",
                "tool_use_id": block.call_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        if isinstance(block, Thinking):
            # Replayed verbatim, never reconstructed.
            return block.raw if block.raw else None
        if isinstance(block, Compaction):
            # Must go back untouched: the API replaces the compacted history
            # with it on the next request.
            return block.raw if block.raw else None
        return None

    def tools_to_wire(self, tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
            for tool in tools
        ]

    def _context_management(self, budgets: dict[str, Any] | None) -> tuple[list[str], dict]:
        """Both context-management edits, and the betas they need."""
        budgets = budgets or {}
        edits: list[dict[str, Any]] = []
        betas: list[str] = []
        if self.capabilities.tool_result_clearing:
            edit: dict[str, Any] = {"type": CLEAR_TOOL_USES}
            if "clear_trigger" in budgets:
                edit["trigger"] = {"type": "input_tokens", "value": budgets["clear_trigger"]}
            if "clear_keep" in budgets:
                edit["keep"] = {"type": "tool_uses", "value": budgets["clear_keep"]}
            edits.append(edit)
            betas.append(CONTEXT_EDIT_BETA)
        if self.capabilities.server_compaction:
            edit = {"type": COMPACT_EDIT}
            if "compact_trigger" in budgets:
                edit["trigger"] = {"type": "input_tokens", "value": budgets["compact_trigger"]}
            if budgets.get("instructions"):
                # Custom instructions *replace* the default prompt entirely,
                # which is why Buddy's are complete on their own.
                edit["instructions"] = budgets["instructions"]
            if budgets.get("pause_after_compaction"):
                edit["pause_after_compaction"] = True
            edits.append(edit)
            betas.append(COMPACTION_BETA)
        return betas, ({"edits": edits} if edits else {})

    # -- streaming ---------------------------------------------------------

    async def stream(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef] = (),
        *,
        budgets: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        betas, context_management = self._context_management(budgets)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.to_wire(messages),
            # A cache breakpoint on the system prompt, kept separate from the
            # conversation, so a compaction never invalidates it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "output_config": {"effort": self.effort},
        }
        if tools:
            request["tools"] = self.tools_to_wire(tools)
        if context_management:
            request["context_management"] = context_management
            request["betas"] = betas

        client = self._client.beta.messages if betas else self._client.messages
        blocks: list[Any] = []
        try:
            async with client.stream(**request) as stream:
                async for event in stream:
                    parsed = self._parse_event(event)
                    if parsed is not None:
                        yield parsed
                final = await stream.get_final_message()
        except Exception as exc:  # surfaced to the brain, never swallowed
            raise ProviderError(f"anthropic stream failed: {exc}") from exc

        blocks = self._final_blocks(final)
        stop_reason = getattr(final, "stop_reason", "end_turn") or "end_turn"

        # Typed events for the pieces the brain acts on, so nothing downstream
        # has to know the SDK's streaming shapes.
        for block in blocks:
            if isinstance(block, ToolCall):
                yield ToolCallReady(block)
            elif isinstance(block, Compaction):
                yield CompactionHappened(block, paused=stop_reason == "compaction")

        yield self._usage(final)
        yield Finished(stop_reason=stop_reason, turn=Turn(Role.ASSISTANT, blocks))

    def _parse_event(self, event: Any) -> StreamEvent | None:
        kind = getattr(event, "type", "")
        if kind == "text":
            return TextDelta(getattr(event, "text", ""))
        if kind == "thinking":
            return ThinkingDelta(getattr(event, "thinking", ""))
        return None

    def _final_blocks(self, final: Any) -> list[Any]:
        blocks: list[Any] = []
        for raw in getattr(final, "content", []) or []:
            kind = getattr(raw, "type", "")
            if kind == "text":
                blocks.append(Text(raw.text))
            elif kind == "thinking":
                blocks.append(Thinking(getattr(raw, "thinking", "") or "", raw=raw))
            elif kind == "tool_use":
                blocks.append(ToolCall(id=raw.id, name=raw.name, arguments=dict(raw.input or {})))
            elif kind == "compaction":
                blocks.append(Compaction(summary=_compaction_text(raw), raw=raw))
        return blocks

    def _usage(self, final: Any) -> Usage:
        usage = getattr(final, "usage", None)
        iterations = getattr(usage, "iterations", None) or []
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            iterations=max(1, len(iterations)),
        )

    def compaction_in(self, turn: Turn) -> Compaction | None:
        for block in turn.blocks:
            if isinstance(block, Compaction):
                return block
        return None

    # -- counting and probing ---------------------------------------------

    async def count_tokens(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef] = (),
    ) -> int:
        wire = [turn for turn in self.to_wire(messages) if turn.get("role") != "system"]
        if not wire:
            return 0
        request: dict[str, Any] = {"model": self.model, "messages": wire, "system": system}
        if tools:
            request["tools"] = self.tools_to_wire(tools)
        try:
            counted = await self._client.messages.count_tokens(**request)
        except Exception as exc:
            raise ProviderError(f"count_tokens failed: {exc}") from exc
        return int(getattr(counted, "input_tokens", 0))

    async def aclose(self) -> None:
        """Close the SDK client's connection pool."""
        closer = getattr(self._client, "close", None)
        if closer is not None:
            with contextlib.suppress(Exception):
                await closer()

    async def probe(self) -> ProbeResult:
        """A one-token call and a capability probe.

        The betas are asked for rather than assumed: a model or region that
        rejects one is a fact `ContextManager` needs, not a surprise later.
        """
        capabilities = self._assumed_capabilities()
        try:
            await self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception as exc:
            return ProviderResultError(self, capabilities, exc).result()

        detail: list[str] = []
        server_compaction = await self._accepts(
            betas=[COMPACTION_BETA],
            context_management={"edits": [{"type": COMPACT_EDIT}]},
        )
        clearing = await self._accepts(
            betas=[CONTEXT_EDIT_BETA],
            context_management={"edits": [{"type": CLEAR_TOOL_USES}]},
        )
        if not server_compaction:
            detail.append("server compaction rejected; falling back to client-side")
        if not clearing:
            detail.append("tool-result clearing rejected; falling back to client-side")

        return ProbeResult(
            provider=self.name,
            model=self.model,
            reachable=True,
            capabilities=Capabilities(
                server_compaction=server_compaction,
                tool_result_clearing=clearing,
                prompt_caching=capabilities.prompt_caching,
                mid_conversation_system=capabilities.mid_conversation_system,
                count_tokens=True,
                max_context=capabilities.max_context,
            ),
            detail="; ".join(detail),
        )

    async def _accepts(self, *, betas: list[str], context_management: dict) -> bool:
        try:
            await self._client.beta.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
                betas=betas,
                context_management=context_management,
            )
        except Exception:
            return False
        return True


class ProviderResultError:
    """Turns an exception from the one-token call into a ProbeResult."""

    def __init__(self, provider: AnthropicProvider, capabilities: Capabilities, exc: Exception):
        self.provider = provider
        self.capabilities = capabilities
        self.exc = exc

    def result(self) -> ProbeResult:
        return ProbeResult(
            provider=self.provider.name,
            model=self.provider.model,
            reachable=False,
            capabilities=Capabilities(),
            error=str(self.exc).splitlines()[0][:200],
        )


def _compaction_text(raw: Any) -> str:
    """The summary out of a compaction block.

    A documented failure mode is `content: null` when tools are defined and
    the instructions did not forbid tool calls; the caller detects the empty
    string and runs client-side compaction instead (context layer 2b).
    """
    content = getattr(raw, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)


def parse_summary(text: str) -> str:
    """Buddy's instructions ask for `<summary></summary>` tags."""
    start, end = text.find("<summary>"), text.rfind("</summary>")
    if start != -1 and end > start:
        return text[start + len("<summary>") : end].strip()
    return text.strip()


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
