"""ModelProvider protocol and Buddy's canonical message types.

Buddy keeps the conversation in its *own* representation and each provider
serializes to and from its wire format. That is what makes
`conversation_log`, `conversation_summaries`, `memory`, and the brief
template provider-independent, and it is why switching providers can
never leave one API looking at another API's tool-call blocks.

What differs between providers is contained here: tool-call serialization,
streaming event shapes, token counting, and which context-management layers
are native. What does not differ is everything Buddy actually reasons with.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# Canonical conversation
# --------------------------------------------------------------------------


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    #: An operator instruction placed mid-conversation rather than in the
    #: system prompt. Providers that cannot express it fold it into the
    #: neighbouring turn (see `Capabilities.mid_conversation_system`).
    SYSTEM = "system"


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Why `arguments` could not be read, when they could not. A provider
    #: that received unparseable arguments says so here rather than passing
    #: `{}` along, which reached the tool as a baffling "missing argument".
    arguments_error: str = ""


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Thinking:
    """Reasoning a provider returned.

    `raw` is the provider's own block, kept so it can be replayed verbatim on
    the same model. Thinking is bound to the model that produced it, so a
    provider switch drops these (and forces a compaction boundary for
    exactly this reason).
    """

    text: str = ""
    raw: Any = None


@dataclass(frozen=True)
class Compaction:
    """A summary the provider made of everything before it (context layer 2).

    `raw` must be handed back untouched: the API uses it to replace the
    compacted history on the next request, and appending only the text
    silently loses the compaction state.
    """

    summary: str
    raw: Any = None


Block = Text | ToolCall | ToolResult | Thinking | Compaction


@dataclass
class Turn:
    role: Role
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if isinstance(block, Text))

    def has(self, kind: type) -> bool:
        return any(isinstance(block, kind) for block in self.blocks)

    @classmethod
    def user(cls, text: str) -> Turn:
        return cls(Role.USER, [Text(text)])

    @classmethod
    def assistant(cls, text: str) -> Turn:
        return cls(Role.ASSISTANT, [Text(text)])

    @classmethod
    def system(cls, text: str) -> Turn:
        return cls(Role.SYSTEM, [Text(text)])


@dataclass(frozen=True)
class ToolDef:
    """A tool as Buddy defines it, in JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


# --------------------------------------------------------------------------
# Streaming events
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ToolCallReady:
    call: ToolCall


@dataclass(frozen=True)
class CompactionHappened:
    """The provider compacted mid-request (context layer 2).

    `paused` is True when the request stopped at the compaction rather than
    continuing, which is what lets Buddy persist the summary, re-insert the
    recent turns, refresh the agent table, and only then re-issue.
    """

    block: Compaction
    paused: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: Compaction is an extra sampling step, billed and rate-limited as one.
    #: Summed across iterations, because the top-level count under-reports
    #: when compaction fires.
    iterations: int = 1


@dataclass(frozen=True)
class Finished:
    stop_reason: str = "end_turn"
    #: Everything the assistant produced, in canonical form, ready to append.
    turn: Turn | None = None


StreamEvent = TextDelta | ThinkingDelta | ToolCallReady | CompactionHappened | Usage | Finished


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    """What a provider can do natively.

    `doctor` fills this from a real probe rather than from a table, so
    `ContextManager` picks server or client layers from evidence.
    """

    server_compaction: bool = False
    tool_result_clearing: bool = False
    prompt_caching: bool = False
    #: Whether an operator instruction can be appended to the message list
    #: without disturbing the cached prefix. This is where Layer 0's fresh
    #: agent table goes when it is available; otherwise the state is
    #: folded into the turn, which costs a cache miss every turn.
    mid_conversation_system: bool = False
    count_tokens: bool = False
    max_context: int = 200_000


@dataclass(frozen=True)
class ProbeResult:
    """What `buddy doctor` reports per configured provider."""

    provider: str
    model: str
    reachable: bool
    capabilities: Capabilities
    detail: str = ""
    error: str | None = None

    def summary(self) -> str:
        if not self.reachable:
            return f"{self.provider}: unreachable ({self.error})"
        native = [
            name
            for name, on in (
                ("compaction", self.capabilities.server_compaction),
                ("tool-result clearing", self.capabilities.tool_result_clearing),
                ("caching", self.capabilities.prompt_caching),
                ("mid-conversation system", self.capabilities.mid_conversation_system),
            )
            if on
        ]
        return f"{self.provider} ({self.model}): ok" + (
            f", native {', '.join(native)}" if native else ", client-side context management"
        )


class ProviderError(Exception):
    pass


class ProviderNotInstalled(ProviderError):
    """The provider's SDK is an optional extra that is not installed."""

    def __init__(self, provider: str, extra: str, package: str) -> None:
        super().__init__(
            f"the {provider} provider needs the {package!r} package. "
            f"Install it with: uv tool install 'buddy-orchestrator[{extra}]' "
            f"(or `uv add 'buddy-orchestrator[{extra}]'` from a checkout)."
        )
        self.provider = provider
        self.extra = extra


@runtime_checkable
class ModelProvider(Protocol):
    name: str
    model: str
    capabilities: Capabilities

    def stream(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef],
        *,
        budgets: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def count_tokens(
        self,
        system: str,
        messages: Sequence[Turn],
        tools: Sequence[ToolDef],
    ) -> int: ...

    async def probe(self) -> ProbeResult:
        """A one-token call plus a capability probe, for `doctor`."""
        ...

    async def aclose(self) -> None:
        """Release the SDK client's connection pool.

        Buddy's own process holds one provider for its lifetime, so this is
        not about the ordinary path - it is about `switch_provider`,
        which abandons a client every time it is called, and about tests,
        where a pool collected after its event loop has closed surfaces as an
        unraisable exception attributed to some unrelated test.
        """
        ...


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def estimate_tokens(system: str, messages: Sequence[Turn], tools: Sequence[ToolDef]) -> int:
    """A rough count for providers with no counting endpoint.

    Deliberately crude and deliberately an over-estimate: it only ever decides
    *when* to compact, and compacting slightly early is cheaper than
    discovering the context is full.
    """
    characters = len(system)
    for turn in messages:
        for block in turn.blocks:
            if isinstance(block, Text | Thinking):
                characters += len(block.text)
            elif isinstance(block, ToolCall):
                characters += len(block.name) + len(str(block.arguments))
            elif isinstance(block, ToolResult):
                characters += len(block.content)
            elif isinstance(block, Compaction):
                characters += len(block.summary)
    for tool in tools:
        characters += len(tool.name) + len(tool.description) + len(str(tool.parameters))
    return characters // 4 + 1


def split_recent(messages: Sequence[Turn], keep: int) -> tuple[list[Turn], list[Turn]]:
    """Everything before the last `keep` turns, and those turns.

    Used by client-side compaction to keep the immediate thread verbatim
    after the summary (context layer 2b).
    """
    if keep <= 0:
        return list(messages), []
    return list(messages[:-keep]), list(messages[-keep:])


def strip_thinking(turns: Sequence[Turn]) -> list[Turn]:
    """Drop thinking blocks from turns being replayed.

    They were produced against a history that no longer exists after a
    compaction, and they are bound to the model that made them.
    """
    stripped = []
    for turn in turns:
        blocks: list[Block] = [block for block in turn.blocks if not isinstance(block, Thinking)]
        if blocks:
            stripped.append(Turn(turn.role, blocks))
    return stripped
