"""Reading raw pane output: ANSI, tails, and the wrapper's sentinels."""

from pathlib import Path

from buddy.logs import (
    DONE_SENTINEL,
    START_SENTINEL,
    read_sentinels,
    read_tail,
    strip_ansi,
    tail,
)


def test_strip_ansi_removes_colour_titles_and_carriage_returns():
    raw = "\x1b[32mgreen\x1b[0m\r\n\x1b]0;window title\x07plain\r\n"
    assert strip_ansi(raw) == "green\nplain\n"


def test_strip_ansi_leaves_ordinary_text_alone():
    assert strip_ansi("just text\nmore text\n") == "just text\nmore text\n"


def test_tail_takes_the_last_lines():
    assert tail("a\nb\nc\nd", 2) == "c\nd"
    assert tail("a\nb", 10) == "a\nb"
    assert tail("a\nb", 0) == ""


def test_read_tail_strips_ansi_by_default(tmp_path: Path):
    log = tmp_path / "attempt-1.log"
    log.write_text("\x1b[31mone\x1b[0m\r\ntwo\r\nthree\r\n")
    assert read_tail(log, 2) == "two\nthree"
    assert "\x1b" in read_tail(log, 3, clean=False)


def test_read_tail_of_a_missing_log_is_empty(tmp_path: Path):
    assert read_tail(tmp_path / "nope.log") == ""


def test_read_tail_does_not_read_a_huge_log_whole(tmp_path: Path):
    log = tmp_path / "big.log"
    log.write_text("x" * 50_000 + "\nLAST LINE\n")
    assert "LAST LINE" in read_tail(log, 1, max_bytes=1000)


# -- sentinels -----------------------------------------


def write_log(tmp_path: Path, body: str) -> Path:
    log = tmp_path / "attempt-1.log"
    log.write_text(body)
    return log


def test_sentinels_report_a_finished_attempt(tmp_path: Path):
    log = write_log(
        tmp_path,
        f"{START_SENTINEL} t-0142 attempt=1 2026-09-08T10:00:00Z\n"
        "working...\n"
        f"{DONE_SENTINEL} t-0142 0 2026-09-08T10:05:00Z\n",
    )
    found = read_sentinels(log, "t-0142")
    assert found.started and found.finished
    assert found.attempt == 1
    assert found.exit_code == 0


def test_a_started_but_unfinished_attempt_is_the_interrupted_case(tmp_path: Path):
    """No DONE line means the run was interrupted, not completed."""
    log = write_log(tmp_path, f"{START_SENTINEL} t-0142 attempt=2 ts\nhalf-done\n")
    found = read_sentinels(log, "t-0142")
    assert found.started
    assert not found.finished
    assert found.exit_code is None
    assert found.attempt == 2


def test_a_stale_sentinel_from_another_task_is_ignored(tmp_path: Path):
    """A window is reused, so the id is what makes the record trustworthy
    ."""
    log = write_log(
        tmp_path,
        f"{DONE_SENTINEL} t-0001 0 ts\n{START_SENTINEL} t-0142 attempt=1 ts\nstill running\n",
    )
    found = read_sentinels(log, "t-0142")
    assert found.started
    assert not found.finished


def test_a_nonzero_exit_is_read_back(tmp_path: Path):
    log = write_log(tmp_path, f"{DONE_SENTINEL} t-0142 137 ts\n")
    assert read_sentinels(log, "t-0142").exit_code == 137


def test_sentinels_survive_ansi(tmp_path: Path):
    log = write_log(tmp_path, f"\x1b[32m{DONE_SENTINEL} t-0142 3\x1b[0m\r\n")
    assert read_sentinels(log, "t-0142").exit_code == 3


def test_no_log_means_nothing_known(tmp_path: Path):
    found = read_sentinels(tmp_path / "missing.log", "t-0142")
    assert not found.started and not found.finished


# -- rotation --------------------------------------------------------------


def test_sentinels_are_found_at_the_ends_of_a_huge_log_without_reading_its_middle(
    tmp_path, monkeypatch
):
    from buddy import logs

    log = tmp_path / "attempt-1.log"
    with log.open("wb") as handle:
        handle.write(b"__BUDDY_START__ t-0007 attempt=2 2026-09-14T00:00:00Z\r\n")
        handle.write(b"noise " * 2_000_000)  # 12 MB of middle
        handle.write(b"\r\n__BUDDY_DONE__ t-0007 0 2026-09-14T01:00:00Z\r\n")

    returned: list[int] = []
    real_open = Path.open

    def recording_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_read = handle.read

        def read(n=-1):
            data = real_read(n)
            returned.append(len(data))
            return data

        handle.read = read
        return handle

    monkeypatch.setattr(Path, "open", recording_open)
    found = logs.read_sentinels(log, "t-0007")

    assert found.started and found.attempt == 2
    assert found.finished and found.exit_code == 0
    assert sum(returned) <= 2 * logs.SENTINEL_WINDOW, f"read {sum(returned)} of 12 MB"


def test_the_start_sentinel_survives_rotation_into_the_previous_file(tmp_path):
    from buddy import logs

    log = tmp_path / "attempt-1.log"
    logs.rotated(log).write_text("__BUDDY_START__ t-0007 attempt=1 x\nearly output\n")
    log.write_text("late output\n__BUDDY_DONE__ t-0007 3 y\n")

    found = logs.read_sentinels(log, "t-0007")
    assert found.started and found.finished and found.exit_code == 3


def test_following_a_log_carries_on_through_a_rotation(tmp_path, capsys, monkeypatch):
    """`buddy logs -f` held its file handle, so the first rotation left it
    reading a file nothing writes to any more, silently, forever.

    Driven through the follower's own idle wait rather than a thread and a
    clock: the first version used sleeps, and failed on a loaded machine when
    the follower started after the first line was already written.
    """
    import os

    from buddy import cli

    log = tmp_path / "attempt-1.log"
    log.write_text("")

    def write(text: str) -> None:
        with log.open("a") as handle:
            handle.write(text)

    def rotate() -> None:
        os.replace(log, log.with_name(log.name + ".1"))
        write("after rotation\n")

    def stop() -> None:
        raise KeyboardInterrupt

    # Each time the follower has nothing to read and waits, the next thing happens.
    steps = iter([lambda: write("before rotation\n"), rotate, stop])
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: next(steps)())

    cli._follow(log)

    printed = capsys.readouterr().out.splitlines()
    assert printed == ["before rotation", "after rotation"]
