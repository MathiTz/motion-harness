"""A tiny supervisor run as the immediate child of every spawned command, so the command's whole
process tree dies if the harness itself is killed uncatchably (`kill -9`, an OOM kill, a crash, the
terminal closing hard) - not just on a clean, in-process cancellation (Esc/Ctrl+C), which
`core.workspace_tools._kill_process_tree` already handles by killing this supervisor's own process
group (the real command shares it - see below).

Verified with a real, external `kill -9` (not just a caught `CancelledError`): without this, a
`run_command` child survives the harness process indefinitely as an orphan, reparented to init,
still running whatever it was doing.

Not imported anywhere - always invoked as its own subprocess:

    python -m core.command_watchdog <harness_pid> <poll_interval_seconds> -- <argv...>

``<argv...>`` is the real command to run (already a complete argv - a plain command, or a sandbox's
wrapped form). It runs as this process's own child with no new session/process group of its own, so
it shares this supervisor's group; killing that group (either this module's own watchdog loop, or an
external `killpg` on the supervisor's pid) takes the real command down too. stdin/stdout/stderr are
inherited directly (no extra buffering), so the caller reading the supervisor's pipes sees the real
command's output exactly as if it had been spawned directly.

POSIX only (matches `core/workspace_tools.py`'s existing `start_new_session`/`killpg` use, which is
also POSIX-only); on Windows this module is never invoked and `run_command` behaves as it always has.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading

DEFAULT_POLL_INTERVAL = 2.0


def _parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else - not our case here, but not "gone" either
    return True


def main(argv: list) -> int:
    if len(argv) < 3 or argv[2] != "--":
        print("usage: python -m core.command_watchdog <harness_pid> <poll_interval> -- <argv...>", file=sys.stderr)
        return 2
    harness_pid = int(argv[0])
    poll_interval = float(argv[1])
    command = argv[3:]
    if not command:
        print("command_watchdog: no command given after '--'", file=sys.stderr)
        return 2

    try:
        proc = subprocess.Popen(command)  # same process group as this supervisor (no start_new_session here)
    except OSError as exc:
        print(f"command_watchdog: failed to start {command!r}: {exc}", file=sys.stderr)
        return 127

    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(poll_interval):
            if not _parent_alive(harness_pid):
                try:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                return  # unreachable once SIGKILL lands on our own group, but explicit is cheap

    threading.Thread(target=watch, daemon=True).start()
    code = proc.wait()
    stop.set()
    return code if code is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
