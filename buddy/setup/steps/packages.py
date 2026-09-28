"""Step 2: tmux, git and portaudio through the detected manager.

Only what Buddy actually uses. ffmpeg was on this list once and nothing ever
called it - speech is decoded by PyAV, which carries its own FFmpeg - so
setup demanded, and offered to `sudo` install, a package no part of Buddy
needed, and failed outright on a machine without it.

Two rules shape this step. Nothing installs without the exact list
being shown first, and no `sudo` runs without saying what it is for - so the
command is printed, in full, before it is offered.

tmux's and git's versions are asked of the modules that own them, not
re-parsed here from a second `--version` call.
"""

from __future__ import annotations

from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.setup.platform import run
from buddy.tmux_runner import MIN_TMUX_VERSION, TmuxError, TmuxRunner
from buddy.workspace import MIN_GIT_VERSION, WorkspaceError, git_version

#: Package names differ per manager for the same library; the binaries do not.
PACKAGE_NAMES: dict[str, dict[str, str]] = {
    "brew": {"tmux": "tmux", "git": "git", "portaudio": "portaudio"},
    "apt": {"tmux": "tmux", "git": "git", "portaudio": "libportaudio2"},
    "dnf": {"tmux": "tmux", "git": "git", "portaudio": "portaudio"},
    "pacman": {"tmux": "tmux", "git": "git", "portaudio": "portaudio"},
    "zypper": {"tmux": "tmux", "git": "git", "portaudio": "portaudio"},
}

INSTALL_COMMAND: dict[str, list[str]] = {
    "brew": ["brew", "install"],
    "apt": ["sudo", "apt-get", "install", "-y"],
    "dnf": ["sudo", "dnf", "install", "-y"],
    "pacman": ["sudo", "pacman", "-S", "--noconfirm"],
    "zypper": ["sudo", "zypper", "install", "-y"],
}


def _version(parts: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in parts)


class PackagesStep(BaseStep):
    name = "packages"
    title = "System packages"
    number = "2"

    async def _missing(self, ctx: SetupContext) -> tuple[list[str], dict[str, str]]:
        """Which of the three are absent or too old, and what is present."""
        missing: list[str] = []
        pins: dict[str, str] = {}

        try:
            found = await TmuxRunner().check_version()
            pins["tmux"] = _version(found)
        except (TmuxError, FileNotFoundError):
            missing.append("tmux")

        try:
            found = await git_version()
            if found < MIN_GIT_VERSION:
                missing.append("git")
            pins["git"] = _version(found)
        except WorkspaceError:
            missing.append("git")

        # portaudio is a library, not a binary, so the package manager is the
        # only honest place to ask. It is needed for the microphone, so a
        # `--no-voice` machine does not need it at all.
        if not ctx.options.no_voice:
            installed = await self._portaudio_installed(ctx)
            if installed:
                pins["portaudio"] = "present"
            else:
                missing.append("portaudio")
        return missing, pins

    async def _portaudio_installed(self, ctx: SetupContext) -> bool:
        """Can audio actually be opened here?

        Asked of the thing that has to work, not of the package manager. The
        `sounddevice` wheel bundles its own PortAudio on macOS and Windows, so
        a machine where audio already works can have no system package at all
        - and installing one it does not need is exactly what setup must not do.
        The package manager is the fallback when Python cannot answer.
        """
        try:
            import sounddevice

            sounddevice.query_devices()
            return True
        except ImportError:
            pass  # the voice extra is not installed; ask the system instead
        except Exception:  # noqa: BLE001 - installed but no PortAudio behind it
            return False

        manager = ctx.platform().package_manager
        if manager == "brew":
            code, _, _ = await run("brew", "list", "--formula", "portaudio")
            return code == 0
        if manager == "apt":
            code, out, _ = await run("dpkg-query", "-W", "-f=${Status}", "libportaudio2")
            return code == 0 and "install ok installed" in out
        if manager in ("dnf", "zypper"):
            code, _, _ = await run("rpm", "-q", "portaudio")
            return code == 0
        if manager == "pacman":
            code, _, _ = await run("pacman", "-Q", "portaudio")
            return code == 0
        # No known manager: fall back to looking for the library itself rather
        # than claiming it is missing on a machine that has it.
        code, out, _ = await run("ldconfig", "-p")
        return "libportaudio" in out

    async def check(self, ctx: SetupContext) -> CheckResult:
        missing, pins = await self._missing(ctx)
        detail = (
            ", ".join(f"{name} {version}" for name, version in sorted(pins.items()))
            if not missing
            else f"missing: {', '.join(missing)}"
        )
        return CheckResult(satisfied=not missing, detail=detail, pins=pins)

    async def act(self, ctx: SetupContext) -> None:
        missing, _ = await self._missing(ctx)
        if not missing:
            return
        report = ctx.platform()
        manager = report.package_manager
        if manager not in INSTALL_COMMAND:
            raise self.fail(
                f"no supported package manager found, and these are missing: {', '.join(missing)}.",
                hint=(
                    "Install them by hand (brew, apt, dnf, pacman and zypper are the ones "
                    "Buddy drives), then re-run `buddy setup --force packages`."
                ),
            )

        names = [PACKAGE_NAMES[manager][item] for item in missing]
        command = [*INSTALL_COMMAND[manager], *names]
        printed = " ".join(command)
        ctx.ui.say(f"These are missing: {', '.join(missing)}")
        if command[0] == "sudo":
            # Never a silent sudo, and never one whose purpose is not
            # on screen next to it.
            ctx.ui.warn(f"This needs sudo to install system packages: {printed}")
        else:
            ctx.ui.detail(printed)
        if not ctx.ui.confirm("Install them now?"):
            raise self.fail(
                "declined the package install.",
                hint=f"Install them yourself with: {printed}",
            )

        code, out, err = await run(*command, seconds=900)
        if code != 0:
            tail = (err or out).strip().splitlines()[-3:]
            raise self.fail(
                f"`{printed}` failed ({code}): {' '.join(tail) or 'no output'}",
                hint=f"Run it yourself to see the whole error: {printed}",
            )

    async def verify(self, ctx: SetupContext) -> str:
        missing, pins = await self._missing(ctx)
        if missing:
            raise self.fail(
                f"still missing after the install: {', '.join(missing)}",
                hint="Check the package manager's output above.",
            )
        tmux_min = _version(MIN_TMUX_VERSION)
        git_min = _version(MIN_GIT_VERSION)
        found = ", ".join(f"{name} {version}" for name, version in sorted(pins.items()))
        return f"{found} (need tmux >= {tmux_min} for pane_dead_status, git >= {git_min})"
