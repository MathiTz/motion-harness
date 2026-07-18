import asyncio
import os
import tempfile

from core.orchestrator import TaskManager, TaskRequest
from core.providers import ModelConfig


class _FakeAgent:
    async def run(self, prompt, target="user", on_stream_chunk=None):
        for chunk in ("hello ", "world"):
            if on_stream_chunk:
                maybe = on_stream_chunk(chunk)
                if asyncio.iscoroutine(maybe):
                    await maybe
        return "hello world"


async def test_task_manager_streaming_callbacks_and_artifact_path():
    updates = []
    with tempfile.TemporaryDirectory() as workspace:
        manager = TaskManager(
            default_model_config=ModelConfig(
                name="test-model",
                endpoint="http://localhost:11434",
                provider_type="local",
            ),
            workspace_path=workspace,
        )

        async def on_progress(status):
            updates.append((status.status, bool(status.result), status.artifact_path))


        original_execute = manager._execute_task

        async def patched_execute_task(request, model_override=None):
            task = manager.tasks[request.task_id]
            task.status = "RUNNING"
            task.start_time = __import__("datetime").datetime.now()
            task.conversation.append({"role": "user", "content": request.prompt})
            task.conversation.append({"role": "agent", "content": ""})
            await manager._notify_progress(request.task_id)

            async def on_stream_chunk(chunk):
                task.result = (task.result or "") + chunk
                task.conversation[-1]["content"] = task.result
                await manager._notify_progress(request.task_id)

            result = await _FakeAgent().run(request.prompt, on_stream_chunk=on_stream_chunk)
            if result and not task.result:
                task.result = result
            task.status = "COMPLETED"
            task.end_time = __import__("datetime").datetime.now()
            task.artifact_path = manager._save_artifact(task)
            manager._events[request.task_id].set()
            await manager._notify_progress(request.task_id)

        manager._execute_task = patched_execute_task
        request = TaskRequest(prompt="test prompt")
        task_id = await manager.spawn_task(
            request=request,
            model_override=manager.default_config,
            progress_callback=on_progress,
        )
        final = await manager.wait_for_task(task_id, timeout=5)
        manager._execute_task = original_execute

        assert final.status == "COMPLETED"
        assert final.result == "hello world"
        assert final.artifact_path is not None
        assert final.artifact_path.startswith(os.path.join(workspace, "tasks") + os.sep)
        assert os.path.exists(final.artifact_path)
        assert any(status == "RUNNING" for status, _, _ in updates)
        assert any(status == "COMPLETED" and has_result for status, has_result, _ in updates)


if __name__ == "__main__":
    asyncio.run(test_task_manager_streaming_callbacks_and_artifact_path())
