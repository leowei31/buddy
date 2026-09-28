"""Buddy: a voice-driven, harness-agnostic multi-agent orchestrator.

The layer boundaries are load-bearing - one module knows each external thing:

    harnesses/*     the only modules that know a specific harness
    providers/*     the only modules that know a specific LLM API
    voice/tts.py    the only module that knows Fish Audio
    tmux_runner.py  the only module that knows tmux
    workspace.py    the only module that knows git
"""

__version__ = "0.1.0"
