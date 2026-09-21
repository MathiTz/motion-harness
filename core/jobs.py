"""Background jobs: long-running processes (dev servers, watchers, builds)
the agent starts, then checks on and stops in later steps or turns.

Output (stdout+stderr merged) is kept in a bounded ring buffer. Each job runs
in its own process group so the whole tree can be stopped. Jobs live in the
``ToolSession`` so they survive across turns and are stopped when the session
ends.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

MAX_RUNNING_JOBS = 5
MAX_KEPT_JOBS = 20
MAX_BUFFER_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_WAIT_SECONDS = 30.0


class JobError(ValueError):
    pass


@dataclass
class Job:
    id: str
    name: str
    command: str
    started: float
    proc: "asyncio.subprocess.Process"
    lines: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_BUFFER_LINES))
    total_lines: int = 0          # lines ever produced (lines evicted from the buffer are counted)
    read_cursor: int = 0          # total_lines at the last job_output call
    exit_code: Optional[int] = None
    ended: Optional[float] = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: List["asyncio.Task"] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.exit_code is None

    @property
    def status(self) -> str:
        return "running" if self.running else f"exited({self.exit_code})"

    def summary(self) -> Dict[str, Any]:
        end = self.ended or time.monotonic()
        return {
            "job_id": self.id, "name": self.name, "command": self.command, "status": self.status,
            "pid": self.proc.pid, "uptime_s": round(end - self.started, 1), "lines": self.total_lines,
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self._counter = 0

    # ── lifecycle ────────────────────────────────────────────────────────
    def running(self) -> List[Job]:
        return [j for j in self.jobs.values() if j.running]

    async def start(
        self,
        argv: Sequence[str],
        *,
        command: str,
        cwd: str,
        env: Dict[str, str],
        name: Optional[str] = None,
    ) -> Job:
        if len(self.running()) >= MAX_RUNNING_JOBS:
            raise JobError(f"too many running jobs ({MAX_RUNNING_JOBS}); stop one with job_stop first")
        kwargs: Dict[str, Any] = dict(
            cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, limit=2 ** 20,
        )
        if os.name == "posix":
            kwargs["start_new_session"] = True
        try:
            proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
        except (FileNotFoundError, OSError) as exc:
            raise JobError(f"failed to start job: {exc}") from exc
        self._counter += 1
        job = Job(id=f"job{self._counter}", name=name or command.split()[0][:30], command=command,
                  started=time.monotonic(), proc=proc)
        self.jobs[job.id] = job
        job.tasks = [asyncio.create_task(self._read(job)), asyncio.create_task(self._wait(job))]
        self._prune()
        return job

    async def _read(self, job: Job) -> None:
        assert job.proc.stdout is not None
        try:
            while True:
                try:
                    raw = await job.proc.stdout.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    raw = b"[line too long, dropped]\n"
                if not raw:
                    return
                job.lines.append(raw.decode("utf-8", "replace").rstrip("\n")[:MAX_LINE_CHARS])
                job.total_lines += 1
                job.changed.set()
        except asyncio.CancelledError:
            pass

    async def _wait(self, job: Job) -> None:
        try:
            code = await job.proc.wait()
        except asyncio.CancelledError:
            return
        # Let the reader drain the last output before reporting the exit.
        await asyncio.sleep(0.05)
        job.exit_code = code if code is not None else -1
        job.ended = time.monotonic()
        job.changed.set()

    def _prune(self) -> None:
        finished = [j for j in self.jobs.values() if not j.running]
        for j in finished[: max(0, len(self.jobs) - MAX_KEPT_JOBS)]:
            self.jobs.pop(j.id, None)

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            known = ", ".join(self.jobs) or "none"
            raise JobError(f"unknown job '{job_id}' (known: {known})")
        return job

    # ── reading ──────────────────────────────────────────────────────────
    async def output(self, job_id: str, *, tail: int = 100, wait_seconds: float = 0.0, everything: bool = False) -> Dict[str, Any]:
        """New output since the previous call (or the last ``tail`` lines with
        ``everything``). ``wait_seconds`` waits for output / exit if none yet."""
        job = self.get(job_id)
        wait = max(0.0, min(float(wait_seconds or 0), MAX_WAIT_SECONDS))
        deadline = time.monotonic() + wait
        while not everything and job.total_lines <= job.read_cursor and job.running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            job.changed.clear()
            if job.total_lines > job.read_cursor or not job.running:
                break
            try:
                await asyncio.wait_for(job.changed.wait(), remaining)
            except asyncio.TimeoutError:
                break
        buffered = list(job.lines)
        first_kept = job.total_lines - len(buffered)  # index of buffered[0] in the overall stream
        if everything:
            chosen = buffered[-max(1, tail):]
            dropped = 0
        else:
            start = max(job.read_cursor, first_kept)
            dropped = max(0, first_kept - job.read_cursor)
            chosen = buffered[start - first_kept:]
            if len(chosen) > tail:
                dropped += len(chosen) - tail
                chosen = chosen[-tail:]
        job.read_cursor = job.total_lines
        return {
            **job.summary(),
            "output": "\n".join(chosen),
            "lines_returned": len(chosen),
            "lines_skipped": dropped,
        }

    def listing(self) -> List[Dict[str, Any]]:
        return [j.summary() for j in self.jobs.values()]

    # ── stopping ─────────────────────────────────────────────────────────
    async def stop(self, job_id: str, grace: float = 3.0) -> Dict[str, Any]:
        job = self.get(job_id)
        was_running = job.running
        if was_running:
            self._signal(job, signal.SIGTERM)
            try:
                await asyncio.wait_for(job.proc.wait(), grace)
            except asyncio.TimeoutError:
                self._signal(job, signal.SIGKILL)
                await job.proc.wait()
            await asyncio.sleep(0.06)  # let _wait() record the exit
            if job.running:
                job.exit_code = job.proc.returncode if job.proc.returncode is not None else -9
                job.ended = time.monotonic()
        return {**job.summary(), "was_running": was_running}

    @staticmethod
    def _signal(job: Job, sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(job.proc.pid, sig)
            else:
                job.proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    async def stop_all(self) -> int:
        stopped = 0
        for job in list(self.jobs.values()):
            if job.running:
                await self.stop(job.id, grace=1.0)
                stopped += 1
        # Let the readers see EOF and finish on their own (so the pipe transports
        # close cleanly), then cancel anything still pending.
        for job in self.jobs.values():
            pending = [t for t in job.tasks if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=1.0)
            for t in job.tasks:
                if not t.done():
                    t.cancel()
        return stopped
