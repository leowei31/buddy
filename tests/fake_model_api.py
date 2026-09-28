"""Model API servers that script a coding agent's turns.

The only way to run a *real* harness binary through a real tool call, offline
and for free, is to be the server it talks to. These are those servers, on
real sockets, streaming the server-sent events each CLI actually parses:

* `FakeResponses` speaks the OpenAI Responses API, which is all Codex speaks -
  `wire_api = "chat"` is refused outright since 0.15x.
* `FakeChatCompletions` speaks OpenAI-compatible chat completions, which
  OpenCode reaches through a custom provider in `OPENCODE_CONFIG`.

Scripted rather than clever: each tool-bearing request takes the next step. A
`run` step answers with a shell tool call, which the harness executes for
real in its working directory; a `say` step answers with the final message.
What the harness sends back - its tool output - is recorded for the test.

Antigravity has no equivalent: `agy` cannot be pointed at another server.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Step:
    run: str | None = None
    say: str | None = None


@dataclass
class Script:
    steps: list[Step]
    requests: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    _cursor: int = 0

    def next(self) -> Step:
        step = self.steps[min(self._cursor, len(self.steps) - 1)]
        self._cursor += 1
        return step


def _handler(script: Script) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            script.requests.append(body)
            for item in body.get("input", []):
                if item.get("type") == "function_call_output":
                    script.tool_outputs.append(str(item.get("output", "")))

            if not self.path.endswith("/responses"):
                self.send_response(404)
                self.end_headers()
                return
            step = script.next()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            number = len(script.requests)
            if step.run is not None:
                item = {
                    "type": "function_call",
                    "id": f"fc_{number}",
                    "call_id": f"call_{number}",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": step.run}),
                }
            else:
                item = {
                    "type": "message",
                    "role": "assistant",
                    "id": f"msg_{number}",
                    "content": [{"type": "output_text", "text": step.say or ""}],
                }
            for event in (
                {"type": "response.created", "response": {"id": f"resp_{number}"}},
                {"type": "response.output_item.done", "item": item},
                {
                    "type": "response.completed",
                    "response": {
                        "id": f"resp_{number}",
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    },
                },
            ):
                self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

    return Handler


class FakeResponses:
    """`with FakeResponses(script) as server:` - then point Codex at `server.url`."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(script))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def codex_flags(self, key_env: str = "FAKE_RESPONSES_KEY") -> str:
        """The `-c` overrides that send Codex here instead of to OpenAI."""
        provider = f'{{name="fake",base_url="{self.url}",env_key="{key_env}",wire_api="responses"}}'
        return f"-c model_provider=fake -c 'model_providers.fake={provider}' -m fake-model"

    def __enter__(self) -> FakeResponses:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _chat_handler(script: Script) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            script.requests.append(body)
            messages = body.get("messages", [])
            for message in messages:
                if message.get("role") == "tool":
                    script.tool_outputs.append(str(message.get("content", "")))

            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()

            def chunk(delta: dict[str, Any], finish: str | None = None) -> None:
                payload = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body.get("model", "fake-model"),
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            if not body.get("tools"):
                # OpenCode's first request names the session and offers no tools.
                chunk({"role": "assistant", "content": "Scripted task"})
                chunk({}, "stop")
            else:
                step = script.next()
                if step.run is not None:
                    call = {
                        "index": 0,
                        "id": f"call_{len(script.requests)}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": step.run, "description": "step"}),
                        },
                    }
                    chunk({"role": "assistant", "tool_calls": [call]})
                    chunk({}, "tool_calls")
                else:
                    chunk({"role": "assistant", "content": step.say or ""})
                    chunk({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


class FakeChatCompletions(FakeResponses):
    """`with FakeChatCompletions(script) as server:` - then `opencode_config`."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _chat_handler(script))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def opencode_config(self) -> str:
        """An `OPENCODE_CONFIG` document with this server as provider `fake`."""
        return json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "provider": {
                    "fake": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Fake",
                        "options": {"baseURL": self.url, "apiKey": "fake"},
                        "models": {"fake-model": {"name": "Fake", "tool_call": True}},
                    }
                },
            }
        )

    def __enter__(self) -> FakeChatCompletions:
        self._thread.start()
        return self
