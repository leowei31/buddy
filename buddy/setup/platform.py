"""OS, package manager, GPU, Docker and audio detection.

Step 1 of setup exists so that no later step has to guess. Everything here
is *observed* - a command that ran, a file that was read - never inferred
from another observation, because the whole point of the report is that step
7 can decide between a local GPU server and cloud TTS on evidence.

Detection never installs, never prompts, and never fails: an absent tool is a
fact about the machine, not an error.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from buddy.processes import kill_and_reap, release

#: Package managers, in the order they are looked for. The first one present
#: wins; a machine with both `apt` and `snap` is an apt machine.
PACKAGE_MANAGERS: tuple[tuple[str, str], ...] = (
    ("brew", "brew"),
    ("apt-get", "apt"),
    ("dnf", "dnf"),
    ("pacman", "pacman"),
    ("zypper", "zypper"),
)

#: Below this, setup step 7 will not stand up a local Fish server: it would
#: swap or OOM, and a three-second turn-around is worse than cloud.
MIN_LOCAL_TTS_VRAM_MB = 8000


async def run(*args: str, seconds: float = 20.0) -> tuple[int, str, str]:
    """A command's exit code and output, without ever raising.

    Every probe in this module is "does this work here?", so a missing binary
    and a binary that fails are the same answer, and neither is exceptional.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return 127, "", str(exc)
    try:
        async with asyncio.timeout(seconds):
            out, err = await proc.communicate()
    except TimeoutError:
        await kill_and_reap(proc)
        return 124, "", f"timed out after {seconds}s"
    finally:
        release(proc)
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


@dataclass(frozen=True)
class Gpu:
    kind: str = "none"  # "cuda" | "mps" | "none"
    name: str = ""
    vram_mb: int | None = None

    @property
    def can_run_local_tts(self) -> bool:
        """Setup step 7 wants CUDA with room. Apple's MPS is deliberately not
        enough: the Fish server image is CUDA-only."""
        return self.kind == "cuda" and (self.vram_mb or 0) >= MIN_LOCAL_TTS_VRAM_MB

    def describe(self) -> str:
        if self.kind == "none":
            return "none detected"
        vram = f", {self.vram_mb} MiB VRAM" if self.vram_mb else ""
        return f"{self.kind}: {self.name or 'unnamed'}{vram}"


@dataclass(frozen=True)
class Docker:
    installed: bool = False
    running: bool = False
    nvidia_toolkit: bool = False
    version: str = ""
    detail: str = ""

    def describe(self) -> str:
        if not self.installed:
            return "not installed"
        state = "running" if self.running else "installed but not running"
        toolkit = ", nvidia container toolkit" if self.nvidia_toolkit else ""
        return f"{self.version or 'unknown version'}, {state}{toolkit}"


@dataclass(frozen=True)
class PlatformReport:
    """What every later step reads instead of probing again."""

    system: str = "unknown"  # "macos" | "linux" | other, which is unsupported
    release: str = ""
    arch: str = ""
    is_wsl: bool = False
    package_manager: str = ""
    package_manager_path: str = ""
    gpu: Gpu = field(default_factory=Gpu)
    docker: Docker = field(default_factory=Docker)
    audio_backend: str = ""
    python: str = ""
    shell: str = ""

    @property
    def supported(self) -> bool:
        """Windows is out of scope because tmux is; WSL2 counts as Linux
        ."""
        return self.system in ("macos", "linux")

    @property
    def is_macos(self) -> bool:
        return self.system == "macos"

    def rows(self) -> list[tuple[str, str]]:
        """For the table `buddy setup` and `buddy doctor` print."""
        return [
            ("os", f"{self.system} {self.release}".strip() + (" (WSL)" if self.is_wsl else "")),
            ("arch", self.arch),
            ("package manager", self.package_manager or "none found"),
            ("gpu", self.gpu.describe()),
            ("docker", self.docker.describe()),
            ("audio", self.audio_backend or "none detected"),
            ("python", self.python),
        ]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: Path) -> PlatformReport | None:
        """The last report, or None if there is not one to read.

        A report from an older Buddy may not have every field; the missing
        ones take their defaults rather than making the file unreadable.
        """
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        gpu = Gpu(**_known(data.pop("gpu", None), Gpu))
        docker = Docker(**_known(data.pop("docker", None), Docker))
        return cls(**_known(data, cls), gpu=gpu, docker=docker)


def _known(data: object, kind: type) -> dict:
    """Only the fields this version of the dataclass has.

    A `platform.json` written by an older or newer Buddy stays readable; the
    fields it does not share simply take their defaults.
    """
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key in kind.__annotations__}


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _system() -> tuple[str, str]:
    name = platform.system()
    if name == "Darwin":
        return "macos", platform.mac_ver()[0]
    if name == "Linux":
        return "linux", platform.release()
    return name.lower() or "unknown", platform.release()


def _is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _package_manager() -> tuple[str, str]:
    for binary, name in PACKAGE_MANAGERS:
        found = shutil.which(binary)
        if found:
            return name, found
    return "", ""


async def detect_gpu(system: str, arch: str) -> Gpu:
    """CUDA if `nvidia-smi` answers, MPS on Apple Silicon, otherwise none.

    Apple Silicon is reported so setup step 6 can pick a whisper model for it,
    even though step 7's Fish image needs CUDA.
    """
    if shutil.which("nvidia-smi"):
        code, out, _ = await run(
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        )
        if code == 0 and out.strip():
            first = out.strip().splitlines()[0]
            name, _, memory = first.partition(",")
            try:
                vram = int(memory.strip())
            except ValueError:
                vram = None
            return Gpu(kind="cuda", name=name.strip(), vram_mb=vram)
    if system == "macos" and arch in ("arm64", "aarch64"):
        return Gpu(kind="mps", name=platform.processor() or "Apple Silicon")
    return Gpu()


async def detect_docker() -> Docker:
    if not shutil.which("docker"):
        return Docker(detail="docker is not on PATH")
    code, out, err = await run("docker", "--version")
    version = out.strip() if code == 0 else ""
    # `docker info` is the only honest test of "running": the client exists
    # whether or not a daemon is listening.
    code, out, err = await run("docker", "info", "--format", "{{json .}}", seconds=25)
    running = code == 0
    toolkit = False
    detail = ""
    if running:
        try:
            info = json.loads(out or "{}")
            toolkit = "nvidia" in {str(k).lower() for k in (info.get("Runtimes") or {})}
        except json.JSONDecodeError:
            toolkit = "nvidia" in out.lower()
    else:
        said = (err or out).strip()
        detail = said.splitlines()[-1] if said else "daemon unreachable"
    if not toolkit and shutil.which("nvidia-ctk"):
        toolkit = True
    return Docker(
        installed=True, running=running, nvidia_toolkit=toolkit, version=version, detail=detail
    )


async def detect_audio(system: str) -> str:
    if system == "macos":
        return "coreaudio"
    for binary, name in (("pw-cli", "pipewire"), ("pactl", "pulseaudio"), ("aplay", "alsa")):
        if shutil.which(binary):
            return name
    return ""


async def detect() -> PlatformReport:
    """The whole report. Runs the independent probes concurrently."""
    system, release = _system()
    arch = platform.machine()
    manager, manager_path = _package_manager()
    gpu, docker, audio = await asyncio.gather(
        detect_gpu(system, arch), detect_docker(), detect_audio(system)
    )
    return PlatformReport(
        system=system,
        release=release,
        arch=arch,
        is_wsl=_is_wsl(),
        package_manager=manager,
        package_manager_path=manager_path,
        gpu=gpu,
        docker=docker,
        audio_backend=audio,
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        shell=Path(os.environ.get("SHELL", "")).name,
    )
