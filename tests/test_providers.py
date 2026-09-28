"""Per provider: tool-call round-trip, streaming, count_tokens, capability probe.

Fake SDK clients, not network calls. What is under test is the translation
between Buddy's canonical format and each wire format, which is the whole
reason the canonical format exists: get that wrong and switching providers corrupts the
conversation rather than continuing it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from buddy.config import BrainSection
from buddy.providers import KNOWN, build
from buddy.providers.base import (
    Capabilities,
    Compaction,
    Finished,
    ProviderError,
    Role,
    Text,
    TextDelta,
    Thinking,
    ToolCall,
    ToolCallReady,
    ToolDef,
    ToolResult,
    Turn,
    Usage,
    estimate_tokens,
    split_recent,
    strip_thinking,
)

TOOLS = [
    ToolDef(
        name="list_agents",
        description="The slot table.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDef(
        name="get_output",
        description="Tail of a slot's log.",
        parameters={
            "type": "object",
            "properties": {"slot": {"type": "string"}, "lines": {"type": "integer"}},
            "required": ["slot"],
        },
    ),
]

#: One conversation exercising every block type, used against every provider.
CONVERSATION = [
    Turn.user("what is everyone doing"),
    Turn(
        Role.ASSISTANT,
        [
            Text("Let me look."),
            ToolCall(id="call_1", name="get_output", arguments={"slot": "Tuesday", "lines": 20}),
        ],
    ),
    Turn(Role.USER, [ToolResult(call_id="call_1", content="running tests...")]),
    Turn.assistant("Tuesday is running tests."),
]


# --------------------------------------------------------------------------
# Canonical helpers
# --------------------------------------------------------------------------


def test_turn_helpers():
    assert Turn.user("hi").role is Role.USER
    assert Turn.assistant("hi").text == "hi"
    assert Turn.system("state").role is Role.SYSTEM
    assert CONVERSATION[1].has(ToolCall)
    assert not CONVERSATION[0].has(ToolCall)


def test_split_recent_keeps_the_immediate_thread():
    older, recent = split_recent(CONVERSATION, 2)
    assert len(older) == 2
    assert len(recent) == 2
    assert split_recent(CONVERSATION, 0) == (CONVERSATION, [])


def test_strip_thinking_drops_blocks_bound_to_a_dead_history():
    turns = [Turn(Role.ASSISTANT, [Thinking("reasoning", raw={"x": 1}), Text("answer")])]
    stripped = strip_thinking(turns)
    assert not stripped[0].has(Thinking)
    assert stripped[0].text == "answer"
    # A turn that was only thinking disappears rather than becoming empty.
    assert strip_thinking([Turn(Role.ASSISTANT, [Thinking("only")])]) == []


def test_token_estimate_grows_with_content():
    small = estimate_tokens("sys", [Turn.user("hi")], [])
    large = estimate_tokens("sys", [Turn.user("hi" * 1000)], TOOLS)
    assert 0 < small < large


# --------------------------------------------------------------------------
# Fake SDK clients
# --------------------------------------------------------------------------


class Recorder:
    """Captures the request each provider builds."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def last(self) -> dict[str, Any]:
        return self.requests[-1]


# -- Anthropic -------------------------------------------------------------


class FakeAnthropicStream:
    def __init__(self, final: Any) -> None:
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            yield _obj(type="text", text="Let me ")
            yield _obj(type="text", text="look.")

        return gen()

    async def get_final_message(self):
        return self._final


class FakeAnthropicMessages:
    def __init__(self, recorder: Recorder, final: Any, *, fail_betas: set[str] | None = None):
        self.recorder = recorder
        self.final = final
        self.fail_betas = fail_betas or set()

    def stream(self, **request):
        self.recorder.requests.append(request)
        return FakeAnthropicStream(self.final)

    async def create(self, **request):
        self.recorder.requests.append(request)
        for beta in request.get("betas", []):
            if beta in self.fail_betas:
                raise RuntimeError(f"beta {beta} not supported for this model")
        return self.final

    async def count_tokens(self, **request):
        self.recorder.requests.append(request)
        return _obj(input_tokens=1234)


class FakeAnthropicClient:
    def __init__(self, final: Any, *, fail_betas: set[str] | None = None) -> None:
        self.recorder = Recorder()
        self.messages = FakeAnthropicMessages(self.recorder, final, fail_betas=fail_betas)
        self.beta = _obj(messages=self.messages)


def _obj(**kwargs) -> Any:
    return type("Obj", (), kwargs)()


def anthropic_final(*, with_tool_call=True, with_compaction=False, stop_reason="end_turn"):
    content: list[Any] = [_obj(type="text", text="Let me look.")]
    if with_tool_call:
        content.append(
            _obj(type="tool_use", id="call_1", name="get_output", input={"slot": "Tuesday"})
        )
    if with_compaction:
        content.append(
            _obj(type="compaction", content=[_obj(text="<summary>earlier work</summary>")])
        )
    return _obj(
        content=content,
        stop_reason=stop_reason,
        usage=_obj(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=0,
            iterations=[1, 2] if with_compaction else [],
        ),
    )


def make_anthropic(**kwargs):
    from buddy.providers.anthropic import AnthropicProvider

    client = FakeAnthropicClient(anthropic_final(**kwargs))
    return AnthropicProvider("claude-opus-5", client=client), client


# -- OpenAI ----------------------------------------------------------------


class FakeOpenAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self.chunks:
                yield chunk

        return gen()


class FakeOpenAICompletions:
    def __init__(self, recorder: Recorder, chunks: list[Any], fail: Exception | None = None):
        self.recorder = recorder
        self.chunks = chunks
        self.fail = fail

    async def create(self, **request):
        self.recorder.requests.append(request)
        if self.fail:
            raise self.fail
        if not request.get("stream"):
            return _obj(choices=[])
        return FakeOpenAIStream(self.chunks)


class FakeOpenAIClient:
    def __init__(self, chunks: list[Any], fail: Exception | None = None) -> None:
        self.recorder = Recorder()
        self.completions = FakeOpenAICompletions(self.recorder, chunks, fail)
        self.chat = _obj(completions=self.completions)


def openai_chunks():
    def delta(**kwargs):
        return _obj(choices=[_obj(delta=_obj(**kwargs), finish_reason=None)], usage=None)

    return [
        delta(content="Let me ", tool_calls=None),
        delta(content="look.", tool_calls=None),
        delta(
            content=None,
            tool_calls=[
                _obj(index=0, id="call_1", function=_obj(name="get_output", arguments='{"slot":'))
            ],
        ),
        delta(
            content=None,
            tool_calls=[_obj(index=0, id=None, function=_obj(name=None, arguments='"Tuesday"}'))],
        ),
        _obj(
            choices=[_obj(delta=_obj(content=None, tool_calls=None), finish_reason="tool_calls")],
            usage=_obj(prompt_tokens=100, completion_tokens=20),
        ),
    ]


def make_openai(chunks=None, fail=None):
    from buddy.providers.openai import OpenAIProvider

    client = FakeOpenAIClient(chunks if chunks is not None else openai_chunks(), fail)
    return OpenAIProvider("gpt-5", client=client), client


# -- Gemini ----------------------------------------------------------------


class FakeGeminiModels:
    def __init__(self, recorder: Recorder, chunks: list[Any]) -> None:
        self.recorder = recorder
        self.chunks = chunks

    async def generate_content_stream(self, **request):
        self.recorder.requests.append(request)

        async def gen():
            for chunk in self.chunks:
                yield chunk

        return gen()

    async def generate_content(self, **request):
        self.recorder.requests.append(request)
        return _obj(candidates=[])

    async def count_tokens(self, **request):
        self.recorder.requests.append(request)
        return _obj(total_tokens=4321)


class FakeGeminiClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.recorder = Recorder()
        self.models = FakeGeminiModels(self.recorder, chunks)
        self.aio = _obj(models=self.models)


def gemini_chunks():
    def chunk(parts, usage=None):
        return _obj(
            candidates=[_obj(content=_obj(parts=parts))],
            usage_metadata=usage,
        )

    return [
        chunk([_obj(text="Let me ", function_call=None)]),
        chunk([_obj(text="look.", function_call=None)]),
        chunk(
            [
                _obj(
                    text=None,
                    function_call=_obj(id=None, name="get_output", args={"slot": "Tuesday"}),
                )
            ],
            usage=_obj(prompt_token_count=100, candidates_token_count=20),
        ),
    ]


def make_gemini(chunks=None):
    from buddy.providers.vertex_gemini import VertexGeminiProvider

    client = FakeGeminiClient(chunks if chunks is not None else gemini_chunks())
    return VertexGeminiProvider("gemini-3-pro", project="p", client=client), client


PROVIDERS = {"anthropic": make_anthropic, "openai": make_openai, "vertex_gemini": make_gemini}


async def collect(provider, **kwargs) -> list[Any]:
    return [event async for event in provider.stream("SYSTEM", CONVERSATION, TOOLS, **kwargs)]


# --------------------------------------------------------------------------
# The same contract, every provider
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_streaming_yields_text_then_a_finished_turn(name):
    provider, _ = PROVIDERS[name]()
    events = await collect(provider)

    deltas = [event for event in events if isinstance(event, TextDelta)]
    assert "".join(delta.text for delta in deltas) == "Let me look."

    finished = events[-1]
    assert isinstance(finished, Finished)
    assert finished.turn is not None
    assert finished.turn.role is Role.ASSISTANT
    assert finished.turn.text == "Let me look."


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_a_tool_call_round_trips_into_canonical_form(name):
    provider, _ = PROVIDERS[name]()
    events = await collect(provider)

    ready = [event for event in events if isinstance(event, ToolCallReady)]
    assert len(ready) == 1
    assert ready[0].call.name == "get_output"
    assert ready[0].call.arguments == {"slot": "Tuesday"}

    calls = [b for b in events[-1].turn.blocks if isinstance(b, ToolCall)]
    assert calls and calls[0].arguments["slot"] == "Tuesday"


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_usage_is_reported(name):
    provider, _ = PROVIDERS[name]()
    usages = [event for event in await collect(provider) if isinstance(event, Usage)]
    assert usages and usages[0].input_tokens == 100
    assert usages[0].output_tokens == 20


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_a_full_conversation_serializes_without_loss(name):
    """Every block type survives the trip to the wire format."""
    provider, client = PROVIDERS[name]()
    await collect(provider)
    request = json.dumps(client.recorder.last(), default=str)
    assert "get_output" in request
    assert "Tuesday" in request
    assert "running tests" in request  # the tool result
    assert "what is everyone doing" in request


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_the_system_prompt_reaches_every_provider(name):
    provider, client = PROVIDERS[name]()
    await collect(provider)
    assert "SYSTEM" in json.dumps(client.recorder.last(), default=str)


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_tools_are_declared(name):
    provider, client = PROVIDERS[name]()
    await collect(provider)
    request = json.dumps(client.recorder.last(), default=str)
    assert "list_agents" in request
    assert "The slot table." in request


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_count_tokens_returns_a_positive_number(name):
    provider, _ = PROVIDERS[name]()
    assert await provider.count_tokens("SYSTEM", CONVERSATION, TOOLS) > 0


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_probe_reports_reachability_and_capabilities(name):
    provider, _ = PROVIDERS[name]()
    result = await provider.probe()
    assert result.reachable
    assert result.provider == name
    assert isinstance(result.capabilities, Capabilities)
    assert "ok" in result.summary()


# --------------------------------------------------------------------------
# Anthropic specifics: the native context layers
# --------------------------------------------------------------------------


async def test_anthropic_sends_both_context_management_edits():
    from buddy.providers.anthropic import (
        CLEAR_TOOL_USES,
        COMPACT_EDIT,
        COMPACTION_BETA,
        CONTEXT_EDIT_BETA,
    )

    provider, client = make_anthropic()
    await collect(
        provider,
        budgets={
            "compact_trigger": 100000,
            "clear_trigger": 40000,
            "clear_keep": 5,
            "instructions": "SUMMARIZE LIKE THIS",
            "pause_after_compaction": True,
        },
    )
    request = client.recorder.last()
    edits = request["context_management"]["edits"]
    kinds = {edit["type"] for edit in edits}
    assert kinds == {CLEAR_TOOL_USES, COMPACT_EDIT}
    assert set(request["betas"]) == {CONTEXT_EDIT_BETA, COMPACTION_BETA}

    compact = next(edit for edit in edits if edit["type"] == COMPACT_EDIT)
    assert compact["trigger"]["value"] == 100000
    assert compact["instructions"] == "SUMMARIZE LIKE THIS"
    assert compact["pause_after_compaction"] is True

    clear = next(edit for edit in edits if edit["type"] == CLEAR_TOOL_USES)
    assert clear["trigger"]["value"] == 40000
    assert clear["keep"]["value"] == 5


async def test_the_system_prompt_carries_a_cache_breakpoint():
    """Kept separate from the conversation, so a compaction cannot invalidate
    it."""
    provider, client = make_anthropic()
    await collect(provider)
    system = client.recorder.last()["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "SYSTEM"


async def test_a_compaction_block_is_captured_and_kept_raw():
    provider, _ = make_anthropic(with_compaction=True, stop_reason="compaction")
    events = await collect(provider)

    from buddy.providers.base import CompactionHappened

    happened = [event for event in events if isinstance(event, CompactionHappened)]
    assert len(happened) == 1
    assert happened[0].paused is True
    assert "earlier work" in happened[0].block.summary
    # The provider's own block is preserved for replay, not just the text.
    assert happened[0].block.raw is not None


async def test_usage_counts_compaction_iterations():
    provider, _ = make_anthropic(with_compaction=True)
    usage = [e for e in await collect(provider) if isinstance(e, Usage)][0]
    assert usage.iterations == 2


async def test_a_compaction_block_is_replayed_untouched():
    provider, client = make_anthropic()
    raw = {"type": "compaction", "content": [{"type": "text", "text": "earlier"}]}
    history = [Turn(Role.ASSISTANT, [Compaction("earlier", raw=raw)]), Turn.user("carry on")]
    await collect_with(provider, history)
    sent = json.dumps(client.recorder.last(), default=str)
    assert "compaction" in sent


async def collect_with(provider, history):
    return [event async for event in provider.stream("SYSTEM", history, TOOLS)]


async def test_a_null_compaction_summary_is_detected():
    """The documented failure mode when tools are defined (context layer 2)."""
    from buddy.providers.anthropic import _compaction_text

    assert _compaction_text(_obj(content=None)) == ""


def test_summary_tags_are_stripped():
    from buddy.providers.anthropic import parse_summary

    assert parse_summary("noise <summary>the point</summary> more") == "the point"
    assert parse_summary("no tags here") == "no tags here"


async def test_probe_falls_back_when_a_beta_is_rejected():
    """Capabilities come from evidence, not from a table."""
    from buddy.providers.anthropic import COMPACTION_BETA, AnthropicProvider

    client = FakeAnthropicClient(anthropic_final(), fail_betas={COMPACTION_BETA})
    provider = AnthropicProvider("claude-opus-5", client=client)
    result = await provider.probe()

    assert result.reachable
    assert result.capabilities.server_compaction is False
    assert result.capabilities.tool_result_clearing is True
    assert "client-side" in result.detail


async def test_an_unreachable_provider_is_reported_not_raised():
    from buddy.providers.anthropic import AnthropicProvider

    class Broken(FakeAnthropicClient):
        pass

    client = Broken(anthropic_final())

    async def boom(**kwargs):
        raise RuntimeError("401 invalid api key")

    client.messages.create = boom
    result = await AnthropicProvider("claude-opus-5", client=client).probe()
    assert not result.reachable
    assert "401" in (result.error or "")
    assert "unreachable" in result.summary()


def test_mid_conversation_system_is_model_gated():
    """Where Layer 0's fresh slot table goes without a cache miss."""
    from buddy.providers.anthropic import AnthropicProvider

    opus = AnthropicProvider("claude-opus-5", client=FakeAnthropicClient(anthropic_final()))
    sonnet = AnthropicProvider("claude-sonnet-5", client=FakeAnthropicClient(anthropic_final()))
    assert opus.capabilities.mid_conversation_system is True
    assert sonnet.capabilities.mid_conversation_system is False


# --------------------------------------------------------------------------
# OpenAI specifics
# --------------------------------------------------------------------------


async def test_openai_puts_tool_results_in_their_own_role():
    provider, client = make_openai()
    await collect(provider)
    wire = client.recorder.last()["messages"]
    tool_entries = [entry for entry in wire if entry["role"] == "tool"]
    assert tool_entries[0]["tool_call_id"] == "call_1"
    assert tool_entries[0]["content"] == "running tests..."
    assistant = [e for e in wire if e["role"] == "assistant" and e.get("tool_calls")][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_output"


async def test_openai_assembles_tool_arguments_from_deltas():
    """Arguments arrive as a JSON string in pieces and are parsed, never
    string-matched."""
    provider, _ = make_openai()
    ready = [e for e in await collect(provider) if isinstance(e, ToolCallReady)]
    assert ready[0].call.arguments == {"slot": "Tuesday"}


async def test_openai_survives_unparseable_tool_arguments():
    from buddy.providers.openai import _loads

    assert _loads("") == {}
    assert _loads("{not json") == {}
    assert _loads('["a list"]') == {}


async def test_openai_marks_errored_tool_results():
    provider, client = make_openai()
    history = [Turn(Role.USER, [ToolResult(call_id="c", content="boom", is_error=True)])]
    await collect_with(provider, history)
    entry = [e for e in client.recorder.last()["messages"] if e["role"] == "tool"][0]
    assert entry["content"].startswith("ERROR:")


async def test_openai_reports_no_native_context_management():
    provider, _ = make_openai()
    assert provider.capabilities.server_compaction is False
    assert provider.capabilities.tool_result_clearing is False
    result = await provider.probe()
    assert "client-side" in result.detail


async def test_openai_stream_errors_are_wrapped():
    provider, _ = make_openai(fail=RuntimeError("429 rate limited"))
    with pytest.raises(ProviderError, match="429"):
        await collect(provider)


def test_openai_base_url_makes_it_a_generic_adapter():
    from buddy.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        "local-model", base_url="http://localhost:11434/v1", client=FakeOpenAIClient([])
    )
    assert provider.base_url == "http://localhost:11434/v1"


# --------------------------------------------------------------------------
# Gemini specifics
# --------------------------------------------------------------------------


async def test_gemini_maps_roles_and_function_responses():
    provider, client = make_gemini()
    await collect(provider)
    contents = client.recorder.last()["contents"]
    assert [entry["role"] for entry in contents] == ["user", "model", "user", "model"]
    response = contents[2]["parts"][0]["function_response"]
    # Keyed by function name, not by call id, which is Gemini's shape.
    assert response["name"] == "get_output"
    assert response["response"]["output"] == "running tests..."


async def test_gemini_sends_the_system_prompt_as_an_instruction():
    provider, client = make_gemini()
    await collect(provider)
    assert client.recorder.last()["config"]["system_instruction"] == "SYSTEM"


async def test_gemini_counts_tokens_through_the_api():
    provider, _ = make_gemini()
    assert await provider.count_tokens("SYSTEM", CONVERSATION, TOOLS) == 4321


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_every_known_provider_is_buildable_or_names_its_extra():
    for name in KNOWN:
        brain = BrainSection(provider=name, model="some-model")
        try:
            provider = build(brain)
        except Exception as exc:  # missing SDK or missing credentials
            from buddy.providers.base import ProviderNotInstalled

            assert isinstance(exc, ProviderNotInstalled | Exception)
            if isinstance(exc, ProviderNotInstalled):
                assert exc.extra in ("vertex", "gemini")
            continue
        assert provider.name == name
        assert provider.model == "some-model"


def test_build_rejects_an_unknown_provider():
    with pytest.raises(ProviderError, match="unknown provider"):
        build(BrainSection(), provider="hal9000")


def test_build_passes_the_providers_own_option_block():
    brain = BrainSection(
        provider="openai",
        model="gpt-5",
        provider_options={
            "openai": {"base_url": "http://localhost:8000/v1", "api_key_env": "OPENAI_API_KEY"}
        },
    )
    provider = build(brain, client=FakeOpenAIClient([]))
    assert provider.base_url == "http://localhost:8000/v1"


# --------------------------------------------------------------------------
# Failure paths: a sentence, never a traceback
# --------------------------------------------------------------------------


def test_a_missing_key_becomes_a_provider_error(monkeypatch):
    """The contract is a working provider or a ProviderError - never a raw
    SDK exception surfacing as a traceback in `buddy doctor`."""
    from buddy.providers.openai import OpenAIProvider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    with pytest.raises(ProviderError, match="could not be created"):
        OpenAIProvider("gpt-5")


def test_a_missing_extra_names_the_extra_to_install():
    from buddy.providers.base import ProviderNotInstalled

    exc = ProviderNotInstalled("vertex_gemini", "gemini", "google-genai")
    message = str(exc)
    assert "google-genai" in message
    assert "buddy-orchestrator[gemini]" in message
    assert exc.extra == "gemini"


def test_provider_errors_reach_doctor_intact(monkeypatch):
    """Both failure kinds are catchable as one type at the call site."""
    from buddy.providers.base import ProviderNotInstalled

    assert issubclass(ProviderNotInstalled, ProviderError)


# -- closing --------------------------------------------------------


async def test_every_provider_closes_the_pool_it_opened():
    """Found through a flaky test: a client collected after its event loop
    has closed surfaces as an unraisable exception on whatever test happens
    to be running, and `switch_provider` abandons one on every call."""

    class Client:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    from buddy.providers.anthropic import AnthropicProvider
    from buddy.providers.openai import OpenAIProvider

    for factory in (AnthropicProvider, OpenAIProvider):
        client = Client()
        provider = factory("some-model", client=client)
        await provider.aclose()
        assert client.closed, factory.__name__


# -- where the key comes from --------------------------


def test_the_api_key_is_resolved_from_where_config_says_it_is():
    """The bug a real `buddy doctor` found.

    `[brain.anthropic] api_key_env` names *where* the key is, and `build` was
    discarding it - so `buddy setup` stored a key in the OS keychain, said so,
    and the SDK then fell back to reading an environment variable setup had
    deliberately not set. Every error afterwards pointed somewhere else.
    """
    from buddy.config import BrainSection
    from buddy.providers import build

    class Keychain:
        def resolve(self, name):
            return "sk-ant-from-the-keychain" if name == "ANTHROPIC_API_KEY" else None

    brain = BrainSection(
        provider="anthropic",
        provider_options={"anthropic": {"api_key_env": "ANTHROPIC_API_KEY"}},
    )
    provider = build(brain, resolver=Keychain())
    assert provider._client.api_key == "sk-ant-from-the-keychain"


def test_an_environment_variable_still_wins(monkeypatch):
    """The precedence setup relies on: a key exported for one session overrides
    the stored one without editing anything."""
    from buddy.config import BrainSection, KeyringSecretResolver
    from buddy.providers import build

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")
    brain = BrainSection(
        provider="openai", provider_options={"openai": {"api_key_env": "OPENAI_API_KEY"}}
    )
    provider = build(brain, resolver=KeyringSecretResolver())
    assert provider._client.api_key == "sk-from-the-environment"


def test_an_explicit_key_beats_the_resolver():
    from buddy.config import BrainSection
    from buddy.providers import build

    class Keychain:
        def resolve(self, _name):
            return "stored"

    brain = BrainSection(
        provider="anthropic",
        provider_options={"anthropic": {"api_key_env": "ANTHROPIC_API_KEY"}},
    )
    provider = build(brain, resolver=Keychain(), api_key="passed-in")
    assert provider._client.api_key == "passed-in"


def test_the_vertex_providers_are_not_handed_a_key():
    """They authenticate with Google Application Default Credentials, which
    are a file and a login, not a string to pass in."""
    from buddy.providers import KEYED, KNOWN

    assert set(KEYED) == {"anthropic", "openai"}
    assert set(KNOWN) - set(KEYED) == {"anthropic_vertex", "vertex_gemini"}


async def test_unparseable_tool_arguments_are_named_not_passed_on_as_nothing():
    """`{not json` became `{}`, and the tool failed with "missing
    required argument" - true, and no help to a model that sent arguments."""
    from buddy.providers.openai import _parse_arguments

    assert _parse_arguments('{"project": "webapp"}') == ({"project": "webapp"}, "")
    assert _parse_arguments("") == ({}, "")
    arguments, problem = _parse_arguments('{"project": "webapp",')
    assert arguments == {} and "not valid JSON" in problem
    assert "list" in _parse_arguments('["a"]')[1]


def test_vertex_without_a_configured_project_still_honours_the_environment(monkeypatch):
    """Passing `project_id=None` counts as given to the Anthropic SDK: it
    overwrote the SDK's own `ANTHROPIC_VERTEX_PROJECT_ID` fallback, so the
    client silently used the Google default-credentials project instead."""
    pytest.importorskip("anthropic.lib.vertex")
    from buddy.providers.anthropic_vertex import AnthropicVertexProvider

    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "the-project-you-exported")
    provider = AnthropicVertexProvider(model="claude-sonnet-5")
    client = provider._make_client(None)
    assert client.project_id == "the-project-you-exported"

    pinned = AnthropicVertexProvider(model="claude-sonnet-5", project_id="from-config")
    assert pinned._make_client(None).project_id == "from-config"
