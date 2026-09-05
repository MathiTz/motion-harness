from core.providers import ModelConfig, ProviderFactory, LocalProvider
from core.config import ConfigManager
from core.caveman import CavemanProtocol
from core.learning import SkillSynthesizer, Trajectory
from core import auth
from memory.db import MemoryDB, EMBEDDING_DIM
from memory.retriever import HybridRetriever
from core.workspace_tools import (
    WorkspaceTools,
    format_tool_result,
    parse_tool_call,
)
import asyncio
import inspect
import hashlib
import logging
import os
import re
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# The installed location of the harness itself. Persistent harness state
# (memory DB, config.yml, .env, auto-synthesized skills) lives here so it
# stays put no matter which directory `motion` is invoked/pointed at - only
# the *workspace* (files the agent reads/writes) should follow the caller's
# current directory. See core/config.py and core/learning.py for the same
# pattern applied to config and skill-synthesis paths.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Per-turn cap on the tool-call agent loop (list/read/write/replace calls).
# This is a last-resort safety valve against a truly stuck/looping model, not
# a task-size limit - large multi-file builds are expected to run for many
# steps. Users can watch progress live (each tool op is streamed) and cancel
# at any time, or queue a follow-up message, so this is set high rather than
# tight. If it's ever hit, whatever progress was made is still reported (see
# the tool loop's `else` branch below) instead of being silently discarded.
MAX_TOOL_STEPS = 150

class MotionAgent:
    def __init__(self, model_config: ModelConfig, memory_path: Optional[str] = None, auto_skill_synthesis: bool = False):
        self.provider = ProviderFactory.get_provider(model_config)
        self.memory = MemoryDB(memory_path or os.path.join(REPO_DIR, "motion_memory.db"))
        self.retriever = HybridRetriever(self.memory, self)
        self.caveman = CavemanProtocol(enabled=True)
        self.synthesizer = SkillSynthesizer(model_config, self.memory, embedding_provider=self)
        self.auto_skill_synthesis = auto_skill_synthesis

    async def get_embedding(self, text: str):
        """Generate embeddings using the provider's embedding endpoint.
        Falls back to a deterministic hash-based vector if the provider
        does not support embeddings (e.g. cloud-only setups without a
        local model).
        """
        if isinstance(self.provider, LocalProvider):
            try:
                return await self.provider.embed(text)
            except Exception as e:
                logger.warning(f"Embedding call failed, using fallback: {e}")

        # Fallback: deterministic hash-based vector for testing / cloud-only setups
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) / 255.0 for b in h]  # 32 floats from 32 bytes
        vec = (raw * ((EMBEDDING_DIM // len(raw)) + 1))[:EMBEDDING_DIM]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    async def run(
        self,
        prompt: str,
        target: str = "user",
        on_stream_chunk=None,
        on_trace_event=None,
        history: Optional[list] = None,
        context_query: Optional[str] = None,
        workspace: Optional[str] = None,
        agent_mode: str = "build",
    ):
        async def emit_trace(stage: str, message: str, **extra):
            if not on_trace_event:
                return
            payload = {"stage": stage, "message": message, **extra}
            try:
                try:
                    maybe = on_trace_event(stage, payload)
                except TypeError:
                    maybe = on_trace_event(payload)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception:
                pass
        # 1. Memory Recall
        await emit_trace("memory_recall_start", "Running retriever.retrieve")
        context_chunks = await self.retriever.retrieve(prompt)
        # Augment recall with the session context query so we don't repeat ourselves.
        if context_query:
            context_chunks += await self.retriever.retrieve(context_query)
        # De-duplicate by content, keep order. Applied unconditionally (not just
        # when context_query is set) since a single retrieve() call can itself
        # surface duplicate content if the DB has duplicate rows.
        seen = set()
        deduped = []
        for c in context_chunks:
            key = c["content"]
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        context_chunks = deduped[:5]
        await emit_trace("memory_recall_done", "Memory recall complete", chunks=len(context_chunks))
        context_text = "\n".join([c["content"] for c in context_chunks])

        # 2. Construct System Prompt
        tools = WorkspaceTools(
            workspace or os.getcwd(),
            read_only=agent_mode == "plan",
        )
        system_prompt = (
            f"You are Motion Agent.\n\n{tools.instructions}\n\n"
            f"Memory Context:\n{context_text}"
        )

        # Tool calls require a short agent loop. The XML envelope works across
        # Ollama, OpenAI-compatible, Anthropic, and proxy providers without
        # requiring each provider to implement a different native tool API.
        tool_history = list(history or [])
        tool_response = None
        used_tool = False
        empty_retries = 0
        inspection_only_loops = 0
        tool_operations: list[str] = []
        for tool_step in range(MAX_TOOL_STEPS):
            step_prompt = prompt if tool_step == 0 else "Continue the task using the tool result above."
            candidate = await self.provider.complete(
                step_prompt,
                system_prompt=system_prompt,
                history=tool_history or None,
            )

            # Stream visible progress for the current step so the UI doesn't stay
            # blank while tools are running. Strip any tool markup from previews.
            visible = re.sub(r"<[^>]+>", "", candidate or "").strip()
            if on_stream_chunk and visible:
                maybe = on_stream_chunk(f"_step_ {visible[:500]}")
                if inspect.isawaitable(maybe):
                    await maybe

            try:
                tool_call = parse_tool_call(candidate)
            except Exception as exc:
                # Treat a malformed tool call as context, not a fatal stop.
                # The model sees the error and can self-correct on the next turn,
                # while the user stays in control via Esc.
                await emit_trace("tool_error", "Invalid tool call", error=str(exc))
                tool_history.extend([
                    {"role": "assistant", "content": candidate},
                    {"role": "user", "content": format_tool_result("invalid", error=str(exc))},
                ])
                continue

            if tool_call is None and not (candidate or "").strip() and used_tool:
                empty_retries += 1
                if empty_retries <= 2:
                    tool_history.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty. Continue the task: use another "
                            "tool if work remains, otherwise provide a concise completion summary."
                        ),
                    })
                    continue
                break

            if tool_call is None:
                tool_response = candidate
                break

            name, arguments = tool_call
            used_tool = True

            # Track loops that only inspect. If the model keeps listing/reading
            # without writing on a build-mode task, nudge it to create files.
            if name in {"list_files", "read_file"}:
                inspection_only_loops += 1
            else:
                inspection_only_loops = 0

            await emit_trace(
                "tool_start",
                f"Running {name}",
                tool=name,
                path=str(arguments.get("path", "")),
            )
            # Plan mode rejects writes by design. Treat this as a soft policy
            # nudge rather than a tool_error, so the model can recover with a
            # real plan instead of the loop stopping on the first write attempt.
            if agent_mode == "plan" and name in {"write_file", "replace_in_file"}:
                result_message = format_tool_result(name, error="write tools are disabled in plan mode")
                tool_history.extend([
                    {"role": "assistant", "content": candidate},
                    {"role": "user", "content": result_message},
                ])
                tool_history.append({
                    "role": "user",
                    "content": (
                        "You are in read-only Plan mode - file writes are disabled here. "
                        "Do not retry write_file/replace_in_file. Respond now with a concrete "
                        "written plan: proposed files/directories, the approach for each major "
                        "piece, and any libraries you'd use. The user will review this and switch "
                        "you to Build mode to implement it."
                    ),
                })
                await emit_trace("tool_done", f"{name} blocked in plan mode", tool=name)
                continue
            path = str(arguments.get("path", "") or "").strip()
            try:
                result = tools.execute(name, arguments)
                result_message = format_tool_result(name, result=result)
                path = str(result.get("path") or path or "").strip()
                if name == "write_file":
                    operation = f"wrote `{path}`"
                    stream_text = f"wrote `{path}` ({result.get('bytes_written', 0)} bytes)"
                elif name == "replace_in_file":
                    operation = f"updated `{path}`"
                    stream_text = operation
                elif name == "read_file":
                    operation = f"read `{path}`"
                    stream_text = operation
                elif name == "list_files":
                    operation = f"listed `{path or '.'}`"
                    stream_text = operation
                else:
                    operation = f"ran `{name}`"
                    stream_text = operation
                tool_operations.append(operation)
                # Stream every tool op (not just writes) so the UI can show
                # live step-by-step progress for the whole loop, however long
                # it runs.
                if on_stream_chunk:
                    maybe = on_stream_chunk(f"_tool_ {stream_text}")
                    if inspect.isawaitable(maybe):
                        await maybe
                await emit_trace("tool_done", f"Completed {name}", tool=name, path=path)
            except Exception as exc:
                operation = f"`{name}` failed: {exc}"
                result_message = format_tool_result(name, error=str(exc))
                if on_stream_chunk:
                    maybe = on_stream_chunk(f"_tool_ {operation}")
                    if inspect.isawaitable(maybe):
                        await maybe
                await emit_trace("tool_error", f"{name} failed", tool=name, path=path, error=str(exc))
                tool_history.extend([
                    {"role": "assistant", "content": candidate},
                    {"role": "user", "content": result_message},
                ])
                # Treat tool execution errors as context rather than a hard stop.
                # The model can see the failure and decide how to proceed; only
                # the user (via Esc) interrupts the interaction.
                continue
            tool_history.extend([
                {"role": "assistant", "content": candidate},
                {"role": "user", "content": result_message},
            ])

            if (
                agent_mode == "build"
                and inspection_only_loops >= 2
                and not any(op.startswith(("wrote ", "updated ")) for op in tool_operations)
            ):
                tool_history.append({
                    "role": "user",
                    "content": (
                        "You have inspected the workspace enough. The user asked you to create "
                        "something. Now use write_file to create the requested files with concrete, "
                        "complete content. Do not ask for clarification and do not return a script."
                    ),
                })
        else:
            # Never discard real progress: if tools actually ran before the cap
            # was hit, tell the user what was done and how to resume, instead
            # of a bare "narrow the task" message that hides completed writes.
            # Hitting this at all is unusual given how high the ceiling is -
            # it almost always means the model is stuck looping rather than
            # that the task was too big.
            if tool_operations:
                completed = "\n".join(f"- {operation}" for operation in tool_operations)
                tool_response = (
                    f"Hit the internal safety limit ({MAX_TOOL_STEPS} tool calls) before "
                    f"finishing - this usually means something got stuck. Progress so far:\n"
                    f"{completed}\n\nSay \"continue\" and I'll pick up from here."
                )
            else:
                tool_response = (
                    f"Hit the internal safety limit ({MAX_TOOL_STEPS} tool calls) without "
                    "making any progress. Please narrow the task and try again."
                )

        # The final non-tool response is already complete. Tool markup is never
        # streamed into the chat UI.
        raw_response = (tool_response or "").strip()
        if not raw_response and used_tool:
            write_operations = [
                operation
                for operation in tool_operations
                if operation.startswith(("wrote ", "updated "))
            ]
            if write_operations:
                raw_response = "Completed filesystem changes:\n" + "\n".join(
                    f"- {operation}" for operation in write_operations
                )
            elif agent_mode == "plan":
                raw_response = (
                    "I inspected the workspace but couldn't finish a plan in the space "
                    "available. Ask me again, or narrow the scope, and I'll lay out the "
                    "file/directory approach here in Plan mode before you switch to Build."
                )
            else:
                raw_response = (
                    "I inspected the workspace but did not make any filesystem changes. "
                    "If you want me to create files, say exactly what to build and I will "
                    "use write_file to create it."
                )

        # Ensure the user always sees the final text, whether streaming or not.
        if on_stream_chunk:
            if raw_response:
                maybe = on_stream_chunk(raw_response)
                if inspect.isawaitable(maybe):
                    await maybe
        await emit_trace(
            "model_done",
            "Agent tool loop finished" if used_tool else "Completion finished",
            chars=len(raw_response),
        )

        # 3. Model Completion (streaming if callback is provided)
        # If the model already returned a natural-language answer in the tool
        # loop, use that. Otherwise fall back to a streaming/oneshot completion.
        stream_chunk_count = 0
        provider_type = getattr(getattr(self.provider, "config", None), "provider_type", "unknown")
        await emit_trace(
            "model_start",
            "Calling provider for completion",
            mode="stream" if on_stream_chunk else "oneshot",
            provider=provider_type,
        )
        if not raw_response:
            if on_stream_chunk:
                raw_chunks = []
                async for chunk in self.provider.stream_complete(prompt, system_prompt=system_prompt, history=history):
                    stream_chunk_count += 1
                    raw_chunks.append(chunk)
                    await emit_trace("stream_chunk", "Received stream chunk", chunk_index=stream_chunk_count, chars=len(chunk or ""))
                    try:
                        maybe = on_stream_chunk(chunk)
                        if inspect.isawaitable(maybe):
                            await maybe
                    except Exception:
                        pass
                raw_response = "".join(raw_chunks)
                await emit_trace("model_done", "Streaming completion finished", stream_chunks=stream_chunk_count, chars=len(raw_response))
            else:
                raw_response = await self.provider.complete(prompt, system_prompt=system_prompt, history=history)
                await emit_trace("model_done", "One-shot completion finished", chars=len(raw_response or ""))

        # 4. Caveman Compression
        final_response = self.caveman.process_outgoing(raw_response, target=target)
        await emit_trace("finalize", "Post-processing completed", chars=len(final_response or ""))

        # 5. Skill Crystallization (manual-first: disabled by default)
        if self.auto_skill_synthesis:
            try:
                await emit_trace("skill_synthesis_start", "Running skill synthesizer")
                trajectory = Trajectory(
                    task_id="single",
                    prompt=prompt,
                    steps=[{"tool": "model", "input": prompt, "output": raw_response}],
                    final_result=raw_response,
                    success=True,
                )
                skill_path = await self.synthesizer.synthesize(trajectory)
                if skill_path:
                    logger.info(f"Skill crystallized: {skill_path}")
                    await emit_trace("skill_synthesis_done", "Skill synthesized", path=skill_path)
                else:
                    await emit_trace("skill_synthesis_done", "Skill synthesis skipped")
            except Exception as e:
                logger.debug(f"Skill synthesis skipped: {e}")
                await emit_trace("skill_synthesis_error", f"Skill synthesis error: {e}")
        return final_response

def load_agent_from_config(config_path: str = "", provider_id: str | None = None) -> MotionAgent:
    """Load a MotionAgent using settings from config.yml (or config.example.yml) and .env."""
    # Load .env file if present. Anchored to REPO_DIR (not CWD) so `motion`
    # finds its own .env regardless of which directory it's invoked from.
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else REPO_DIR
    env_path = os.path.join(config_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key and value:
                        os.environ.setdefault(key, value)

    cm = ConfigManager(config_path)
    provider_id = provider_id or cm.get_default_provider()
    provider_cfg = cm.get_provider_config(provider_id)

    model_config = ModelConfig(
        name=provider_cfg.get("name", provider_id),
        endpoint=provider_cfg["endpoint"],
        api_key=provider_cfg.get("api_key"),
        provider_type=provider_cfg.get("provider_type", "cloud"),
        options=provider_cfg.get("options", {}),
    )
    return MotionAgent(model_config)


def list_providers(config_path: str = ""):
    """Print available providers and models from config."""
    cm = ConfigManager(config_path)
    default = cm.get_default_provider()
    print("Available providers:")
    for pid, name, models, is_default, has_key in cm.list_providers():
        marker = " ← default" if is_default and '/' not in default else ""
        key_icon = "🔑" if has_key else "🔒"
        if len(models) > 1:
            print(f"  {key_icon} {pid:20s} {name}")
            for m in models:
                sel = "*" if (is_default and f"{pid}/{m}" == default) or (is_default and m == models[0] and '/' not in default) else " "
                print(f"    {sel} {m}")
        else:
            m = models[0] if models else "?"
            sel = "*" if is_default else " "
            print(f"  {key_icon} {pid:20s} {name:30s} model={m}{marker}")
    print(f"\nUsage: python main.py --provider ollama-cloud/gemma4:31b")
    print(f"       python main.py --provider ollama-cloud          # uses default model")


async def test_compression():
    """Test Caveman compression without needing a live model."""
    config = ModelConfig(name="Test", endpoint="https://ollama.com/v1", provider_type="cloud")
    agent = MotionAgent(config)

    fluffy_response = "Certainly! I have analyzed the files and found that the bug is in line 42. I'm sorry for the inconvenience. Please let me know if you need further assistance."

    # Case 1: Target is User (Should NOT be compressed)
    user_output = agent.caveman.process_outgoing(fluffy_response, target="user")

    # Case 2: Target is another Agent (Should be compressed)
    agent_output = agent.caveman.process_outgoing(fluffy_response, target="agent")

    print(f"Original: {fluffy_response}")
    print(f"To User:   {user_output}")
    print(f"To Agent:  {agent_output}")

    # Verify bidirectional decompression
    decompressed = agent.caveman.process_incoming(agent_output)
    print(f"Decompressed: {decompressed}")

    assert user_output == fluffy_response
    assert "Certainly!" not in agent_output
    assert len(agent_output) < len(fluffy_response)
    print("\n✅ Caveman integration verified: Tokens reduced for internal communication!")


async def interactive_chat(provider_id: str | None = None):
    """Interactive chat using the configured provider (fallback non-TUI mode)."""
    agent = load_agent_from_config(provider_id=provider_id)
    provider_name = agent.provider.config.name
    print(f"🤖 Motion Agent — using {provider_name}")
    print("Type a message (or 'quit' to exit):\n")

    try:
        while True:
            try:
                prompt = input("You> ").strip()
            except EOFError:
                break
            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                break
            try:
                response = await agent.run(prompt)
                print(f"\nAgent> {response}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n👋 Bye!")
    finally:
        try:
            await asyncio.wait_for(agent.provider.close(), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        agent.memory.close()


def cmd_auth(args) -> None:
    """Handle `motion auth login|logout|list`."""
    action = args.auth_action
    if action == "list":
        keys = auth.list_keys()
        if not keys:
            print("No API keys stored.")
            return
        print("Stored API keys:")
        for provider, key in sorted(keys.items()):
            masked = f"{key[:4]}…{key[-4:]}" if len(key) > 8 else "…"
            print(f"  {provider:20s} {masked}")
        return

    if action == "login":
        provider = args.auth_provider
        if not provider:
            print("Usage: motion auth login <provider>")
            return
        cm = ConfigManager()
        try:
            cm.get_provider_config(provider)
        except ValueError as e:
            print(f"Unknown provider: {provider}")
            print("Available providers:")
            for pid, name, models, is_default, has_key in cm.list_providers():
                print(f"  {pid:20s} {name}")
            return
        import getpass
        key = getpass.getpass(f"API key for {provider}: ").strip()
        if not key:
            print("No key entered; aborting.")
            return
        auth.set_key(provider, key)
        print(f"Saved API key for {provider} → {auth.AUTH_FILE}")
        return

    if action == "logout":
        provider = args.auth_provider
        if not provider:
            print("Usage: motion auth logout <provider>")
            return
        if auth.remove_key(provider):
            print(f"Removed API key for {provider}.")
        else:
            print(f"No stored key for {provider}.")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Motion Agent")
    parser.add_argument("--test", action="store_true", help="Run Caveman compression test (no model needed)")
    parser.add_argument("--list", action="store_true", help="List available providers")
    parser.add_argument("--provider", type=str, default=None, help="Provider to use (e.g. ollama-cloud, ollama-cloud/gemma4:31b, claude, openai)")
    parser.add_argument("--chat", action="store_true", help="Launch in chat REPL mode instead of TUI")
    sub = parser.add_subparsers(dest="command")
    auth_parser = sub.add_parser("auth", help="Manage provider API keys")
    auth_sub = auth_parser.add_subparsers(dest="auth_action", required=True)
    auth_sub.add_parser("list", help="List stored API keys")
    login_p = auth_sub.add_parser("login", help="Store an API key for a provider")
    login_p.add_argument("auth_provider", nargs="?", help="Provider id (e.g. ollama-cloud)")
    logout_p = auth_sub.add_parser("logout", help="Remove a stored API key")
    logout_p.add_argument("auth_provider", nargs="?", help="Provider id (e.g. ollama-cloud)")
    args = parser.parse_args()

    if args.command == "auth":
        cmd_auth(args)
    elif args.list:
        list_providers()
    elif args.test:
        asyncio.run(test_compression())
    elif args.chat:
        asyncio.run(interactive_chat(provider_id=args.provider))
    else:
        # Launch the TUI by default
        from ui.tui import launch_tui
        config = ConfigManager()
        provider_id = args.provider or config.get_default_provider()
        provider_cfg = config.get_provider_config(provider_id)
        # Normalize to full provider/model id so the TUI dropdown matches.
        if "/" not in provider_id:
            model = provider_cfg.get("options", {}).get("model")
            if model:
                provider_id = f"{provider_id}/{model}"
        model_config = ModelConfig(
            name=provider_cfg.get("name", provider_id),
            endpoint=provider_cfg["endpoint"],
            api_key=provider_cfg.get("api_key"),
            provider_type=provider_cfg.get("provider_type", "cloud"),
            options=provider_cfg.get("options", {}),
        )
        launch_tui(model_config, provider_id=provider_id)
