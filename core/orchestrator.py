import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from core.providers import ModelConfig, ProviderFactory
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


class TaskManager:
    """
    Orchestrates parallel execution of MotionAgent tasks.
    Handles hardware-aware concurrency and per-task model routing.

    Uses asyncio.Event per task for efficient notification instead of polling.
    """
    def __init__(self, default_model_config: ModelConfig, workspace_path: str):
        self.default_config = default_model_config
        self.workspace_path = workspace_path
        
        # Hardware-aware concurrency: os.cpu_count() * 2
        self.max_workers = (os.cpu_count() or 1) * 2
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
        self.tasks: Dict[str, TaskStatus] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self.active_count = 0

    async def spawn_task(self, request: TaskRequest, model_override: Optional[ModelConfig] = None) -> str:
        """
        Queue a new task for execution. Returns the task_id.
        Callers can await wait_for_task(task_id) instead of polling.
        """
        self.tasks[request.task_id] = TaskStatus(
            task_id=request.task_id,
            prompt=request.prompt,
            status="PENDING"
        )
        self._events[request.task_id] = asyncio.Event()
        
        # Schedule execution without blocking the main loop
        asyncio.create_task(self._execute_task(request, model_override))
        
        return request.task_id

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

    async def _execute_task(self, request: TaskRequest, model_override: Optional[ModelConfig] = None):
        if self.semaphore.locked():
            logger.info(f"Capacity reached: {self.active_count}/{self.max_workers} workers. Task {request.task_id} queued.")

        async with self.semaphore:
            self.active_count += 1
            self.tasks[request.task_id].status = "RUNNING"
            self.tasks[request.task_id].start_time = datetime.now()
            
            try:
                config = model_override or self.default_config
                agent = MotionAgent(config, memory_path=f"memory_{request.task_id}.db")
                result = await agent.run(request.prompt, target="agent")
                
                self.tasks[request.task_id].result = result
                self.tasks[request.task_id].status = "COMPLETED"
            except Exception as e:
                self.tasks[request.task_id].error = str(e)
                self.tasks[request.task_id].status = "FAILED"
            finally:
                self.tasks[request.task_id].end_time = datetime.now()
                self.active_count -= 1
                self._events[request.task_id].set()

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_workers": self.active_count,
            "max_workers": self.max_workers,
            "tasks": self.tasks
        }

    def get_task_result(self, task_id: str) -> Optional[str]:
        task = self.tasks.get(task_id)
        return task.result if task else None
