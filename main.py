from core.providers import ModelConfig, ProviderFactory, LocalProvider
from core.config import ConfigManager
from core.caveman import CavemanProtocol
from core.learning import SkillSynthesizer, Trajectory
from memory.db import MemoryDB, EMBEDDING_DIM
from memory.retriever import HybridRetriever
import asyncio
import inspect
import hashlib
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class MotionAgent:
    def __init__(self, model_config: ModelConfig, memory_path: str = "motion_memory.db", auto_skill_synthesis: bool = False):
        self.provider = ProviderFactory.get_provider(model_config)
        self.memory = MemoryDB(memory_path)
        self.retriever = HybridRetriever(self.memory, self)
        self.caveman = CavemanProtocol(enabled=True)
        self.synthesizer = SkillSynthesizer(model_config, self.memory)
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

    async def run(self, prompt: str, target: str = "user", on_stream_chunk=None, on_trace_event=None):
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
        await emit_trace("memory_recall_done", "Memory recall complete", chunks=len(context_chunks))
        context_text = "\n".join([c["content"] for c in context_chunks])

        # 2. Construct System Prompt
        system_prompt = f"You are Motion Agent. Memory Context:\n{context_text}"

        # 3. Model Completion (streaming if callback is provided)
        stream_chunk_count = 0
        provider_type = getattr(getattr(self.provider, "config", None), "provider_type", "unknown")
        await emit_trace(
            "model_start",
            "Calling provider for completion",
            mode="stream" if on_stream_chunk else "oneshot",
            provider=provider_type,
        )
        if on_stream_chunk:
            raw_chunks = []
            async for chunk in self.provider.stream_complete(prompt, system_prompt=system_prompt):
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
            raw_response = await self.provider.complete(prompt, system_prompt=system_prompt)
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
        else:
            await emit_trace("skill_synthesis_done", "Skill synthesis disabled (manual mode)")

        return final_response

def load_agent_from_config(config_path: str = "", provider_id: str | None = None) -> MotionAgent:
    """Load a MotionAgent using settings from config.yml (or config.example.yml) and .env."""
    # Load .env file if present
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else "."
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


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Motion Agent")
    parser.add_argument("--test", action="store_true", help="Run Caveman compression test (no model needed)")
    parser.add_argument("--list", action="store_true", help="List available providers")
    parser.add_argument("--provider", type=str, default=None, help="Provider to use (e.g. ollama-cloud, ollama-cloud/gemma4:31b, claude-3-5)")
    parser.add_argument("--chat", action="store_true", help="Launch in chat REPL mode instead of TUI")
    args = parser.parse_args()

    if args.list:
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
        model_config = ModelConfig(
            name=provider_cfg.get("name", provider_id),
            endpoint=provider_cfg["endpoint"],
            api_key=provider_cfg.get("api_key"),
            provider_type=provider_cfg.get("provider_type", "cloud"),
            options=provider_cfg.get("options", {}),
        )
        launch_tui(model_config, provider_id=provider_id)
