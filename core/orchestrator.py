import asyncio
import json
import os
import uuid
import inspect
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from core.providers import ModelConfig, ProviderFactory
from core.session import state_dir
from core.toolstate import ToolSession
from main import MotionAgent

logger = logging.getLogger(__name__)

@dataclass
class TaskRequest:
    prompt: str
    model_id: Optional[str] = None
    priority: int = 0
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

@dataclass
class TaskStatus:
    task_id: str
    prompt: str
    status: str  # "PENDING", "RUNNING", "COMPLETED", "FAILED"
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # Full conversation trace
    conversation: List[Dict[str, str]] = field(default_factory=list)
    # Timestamped log lines for progress
    logs: List[str] = field(default_factory=list)
    # Path to saved artifact if any
    artifact_path: Optional[str] = None

    @property
    def duration(self) -> Optional[str]:
        """Human-readable duration string."""
        if self.start_time and self.end_time:
            delta = (self.end_time - self.start_time).total_seconds()
            if delta < 60:
                return f"{delta:.1f}s"
            minutes, secs = divmod(delta, 60)
            return f"{int(minutes)}m {int(secs)}s"
        return None


class TaskManager:
    """
    Orchestrates parallel execution of MotionAgent tasks.
    Handles hardware-aware concurrency and per-task model routing.

    Uses asyncio.Event per task for efficient notification instead of polling.
    Supports progress_callback for streaming status updates to the TUI.
    """
    def __init__(
        self,
        default_model_config: ModelConfig,
        workspace_path: str,
        mcp_manager: Any = None,
        permissions_config: Optional[dict] = None,
    ):
        self.default_config = default_model_config
        self.workspace_path = workspace_path
        self.mcp_manager = mcp_manager
        self.permissions_config = permissions_config or {}
        # Task transcripts live under <workspace>/.motion/ with the rest of the
        # harness's state, not as a loose tasks/ folder in the user's project.
        self.tasks_dir = str(state_dir(self.workspace_path, "tasks", create=False))
        
        # Hardware-aware concurrency: os.cpu_count() * 2
        self.max_workers = (os.cpu_count() or 1) * 2
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
        self.tasks: Dict[str, TaskStatus] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        self._running: Dict[str, "asyncio.Task"] = {}
        self.active_count = 0

    async def spawn_task(
        self,
        request: TaskRequest,
        model_override: Optional[ModelConfig] = None,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        Queue a new task for execution. Returns the task_id.
        Callers can await wait_for_task(task_id) instead of polling.

        progress_callback: optional async callable(status: TaskStatus) for streaming updates.
        """
        self.tasks[request.task_id] = TaskStatus(
            task_id=request.task_id,
            prompt=request.prompt,
            status="PENDING"
        )
        self._events[request.task_id] = asyncio.Event()
        if progress_callback:
            self._progress_callbacks.setdefault(request.task_id, []).append(progress_callback)
        
        # Schedule execution without blocking the main loop
        runner = asyncio.create_task(self._execute_task(request, model_override))
        self._running[request.task_id] = runner
        runner.add_done_callback(lambda _t, tid=request.task_id: self._running.pop(tid, None))

        return request.task_id

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or running task. Returns True if there was one."""
        runner = self._running.get(task_id)
        if runner is None or runner.done():
            return False
        runner.cancel()
        return True

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskStatus:
        """Wait for a task to complete. Returns the final TaskStatus."""
        event = self._events.get(task_id)
        if not event:
            raise ValueError(f"Unknown task: {task_id}")
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return self.tasks[task_id]

    async def _notify_progress(self, task_id: str) -> None:
        """Call the progress callback if one exists."""
        callbacks = list(self._progress_callbacks.get(task_id, []))
        for cb in callbacks:
            try:
                result = cb(self.tasks[task_id])
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

    def subscribe(self, task_id: str, callback: Callable) -> None:
        """Subscribe a callback for live updates on a specific task."""
        self._progress_callbacks.setdefault(task_id, [])
        if callback not in self._progress_callbacks[task_id]:
            self._progress_callbacks[task_id].append(callback)

    def unsubscribe(self, task_id: str, callback: Callable) -> None:
        """Unsubscribe a callback from a specific task."""
        callbacks = self._progress_callbacks.get(task_id, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def _log(self, task_id: str, message: str) -> None:
        """Append a timestamped log line to the task and notify."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.tasks[task_id].logs.append(f"[{ts}] {message}")

    def _save_artifact(self, task: TaskStatus) -> str:
        """Save task conversation + result to .motion/tasks/{task_id}.md."""
        state_dir(self.workspace_path, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        path = os.path.join(self.tasks_dir, f"{task.task_id}.md")
        lines = [
            f"# Task {task.task_id}",
            f"",
            f"**Status**: {task.status}",
            f"**Started**: {task.start_time}",
            f"**Completed**: {task.end_time}",
            f"**Duration**: {task.duration}",
            f"",
            f"## Prompt",
            f"",
            task.prompt,
            f"",
        ]
        if task.result:
            lines += ["## Result", "", task.result, ""]
        if task.error:
            lines += ["## Error", "", task.error, ""]
        if task.conversation:
            lines += ["## Conversation", ""]
            for turn in task.conversation:
                role = turn.get("role", "?")
                content = turn.get("content", "")
                lines.append(f"**{role}**: {content}")
            lines.append("")
        if task.logs:
            lines += ["## Log", ""] + task.logs + [""]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    async def _execute_task(self, request: TaskRequest, model_override: Optional[ModelConfig] = None):
        task = self.tasks[request.task_id]
        agent: Optional[MotionAgent] = None

        if self.semaphore.locked():
            self._log(request.task_id, f"Queued — {self.active_count}/{self.max_workers} workers busy")

        async with self.semaphore:
            self.active_count += 1
            task.status = "RUNNING"
            task.start_time = datetime.now()
            self._log(request.task_id, f"Started")
            await self._notify_progress(request.task_id)

            try:
                config = model_override or self.default_config
                # Each task gets a private in-memory store: parallel tasks must
                # not write to one shared SQLite file (or litter the repo with
                # per-task database files).
                agent = MotionAgent(config, memory_path=":memory:", mcp_manager=self.mcp_manager)
                agent.permissions_config = self.permissions_config

                # Record user turn
                task.conversation.append({"role": "user", "content": request.prompt})
                task.conversation.append({"role": "agent", "content": ""})
                self._log(request.task_id, f"Running agent ({config.name})")
                await self._notify_progress(request.task_id)

                step_text = ""

                async def on_stream_chunk(chunk: str) -> None:
                    """Live view of the answer. Marker chunks (see
                    core/agent_loop.py) are progress signals, not answer text."""
                    nonlocal step_text
                    if not chunk:
                        return
                    if chunk.startswith("_delta_ "):
                        step_text += chunk[8:]
                        task.result = step_text
                    elif chunk.startswith("_endstep_"):
                        step_text = ""
                        task.result = ""
                    elif chunk.startswith("_tool_ "):
                        self._log(request.task_id, chunk[len("_tool_ "):].strip())
                        await self._notify_progress(request.task_id)
                        return
                    elif chunk.startswith(("_step_ ", "_think_ ", "_out_ ")):
                        return
                    else:
                        task.result = (task.result or "") + chunk
                    if task.conversation and task.conversation[-1].get("role") == "agent":
                        task.conversation[-1]["content"] = task.result or ""
                    await self._notify_progress(request.task_id)

                # No interactive callbacks: background tasks cannot prompt the
                # user, so risky commands / private-network fetches / ask_user
                # are refused rather than silently approved.
                result = await agent.run(
                    request.prompt,
                    target="user",
                    on_stream_chunk=on_stream_chunk,
                    workspace=self.workspace_path,
                    agent_mode="build",
                    session=ToolSession(),
                )

                # The return value is the authoritative final answer.
                if result:
                    task.result = result
                    if task.conversation and task.conversation[-1].get("role") == "agent":
                        task.conversation[-1]["content"] = result
                task.status = "COMPLETED"
                self._log(request.task_id, f"Completed ({len(task.result or '')} chars)")
            except asyncio.CancelledError:
                task.status = "FAILED"
                task.error = "cancelled"
                self._log(request.task_id, "Cancelled")
                raise
            except Exception as e:
                task.error = str(e)
                task.status = "FAILED"
                if task.conversation and task.conversation[-1].get("role") == "agent" and not task.conversation[-1].get("content"):
                    task.conversation.pop()
                task.conversation.append({"role": "system", "content": f"Error: {e}"})
                self._log(request.task_id, f"Failed: {e}")
            finally:
                task.end_time = datetime.now()
                self.active_count -= 1
                if agent is not None:
                    # Release the HTTP client and the in-memory DB.
                    try:
                        await asyncio.wait_for(asyncio.shield(agent.provider.close()), timeout=2.0)
                    except Exception:
                        pass
                    try:
                        agent.memory.close()
                    except Exception:
                        pass
                # Save artifact
                try:
                    task.artifact_path = self._save_artifact(task)
                    self._log(request.task_id, f"Saved to {task.artifact_path}")
                except Exception as e:
                    self._log(request.task_id, f"Could not save artifact: {e}")
                self._events[request.task_id].set()
                await self._notify_progress(request.task_id)

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_workers": self.active_count,
            "max_workers": self.max_workers,
            "tasks": self.tasks
        }

    def get_task_result(self, task_id: str) -> Optional[str]:
        task = self.tasks.get(task_id)
        return task.result if task else None

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        return self.tasks.get(task_id)
