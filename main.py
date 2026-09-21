from core.providers import ModelConfig, ProviderFactory, LocalProvider
from core.config import ConfigManager
from core.caveman import CavemanProtocol
from core.learning import SkillSynthesizer
from core import auth
from core.agent_loop import MAX_TOOL_STEPS, TurnRunner  # noqa: F401  (MAX_TOOL_STEPS re-exported)
from memory.db import MemoryDB, MemoryChunk, NoteStore, EMBEDDING_DIM
from memory.retriever import HybridRetriever
import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# The installed location of the harness itself. Persistent harness state
# (memory DB, config.yml, .env, auto-synthesized skills) lives here so it
# stays put no matter which directory `motion` is invoked/pointed at - only
# the *workspace* (files the agent reads/writes) should follow the caller's
# current directory. See core/config.py and core/learning.py for the same
# pattern applied to config and skill-synthesis paths.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

class MotionAgent:
    def __init__(self, model_config: ModelConfig, memory_path: Optional[str] = None, auto_skill_synthesis: bool = False, mcp_manager: Optional[Any] = None):
        self.provider = ProviderFactory.get_provider(model_config)
        self.memory = MemoryDB(memory_path or os.path.join(REPO_DIR, "motion_memory.db"))
        self.retriever = HybridRetriever(self.memory, self)
        self.caveman = CavemanProtocol(enabled=True)
        self.synthesizer = SkillSynthesizer(model_config, self.memory, embedding_provider=self)
        self.auto_skill_synthesis = auto_skill_synthesis
        # Optional MCP manager exposing external MCP servers' tools to the agent.
        self.mcp_manager = mcp_manager
        # Persistent backend for the memory_save / memory_get tools.
        self.notes = NoteStore(self.memory)
        # `permissions:` block from config.yml (command allow/ask/deny rules).
        self.permissions_config: dict = {}
        # "auto" = confine shell/Python writes with an OS sandbox when the
        # platform has one; "off" disables it (config.yml: sandbox: off).
        self.sandbox_mode = "auto"
        # Memory recall must never stall a turn: give up after this many seconds.
        self.recall_timeout = 2.0
        # Store substantive turns in memory (off by default; the TUI enables it
        # from config). Runs in the background so it never delays the answer.
        self.auto_remember = False
        # True when the last get_embedding() came from a real embedding model
        # (retrieval skips semantic search otherwise - hash vectors carry no meaning).
        self.semantic_available = False
        self._embed_cache: "OrderedDict[str, list]" = OrderedDict()
        self._warned_dim = False
        self._bg_tasks: set = set()

    async def get_embedding(self, text: str):
        """Embed ``text`` with the provider's embedding endpoint (Ollama
        /api/embeddings, or an OpenAI-compatible /embeddings when
        ``embed_model`` is configured). Falls back to a deterministic hash
        vector when none is available; ``semantic_available`` then reads
        False so retrieval relies on keyword search instead.
        """
        cached = self._embed_cache.get(text)
        if cached is not None:
            self._embed_cache.move_to_end(text)
            self.semantic_available = True
            return cached
        vec = None
        provider = self.provider
        if isinstance(provider, LocalProvider) or getattr(provider, "can_embed", False):
            try:
                vec = await provider.embed(text)
            except Exception as e:
                logger.warning(f"Embedding call failed, using fallback: {e}")
        if vec:
            vec = self._fit_dim(list(vec))
            self.semantic_available = True
            self._embed_cache[text] = vec
            if len(self._embed_cache) > 256:
                self._embed_cache.popitem(last=False)
            return vec

        # Fallback: deterministic hash-based vector for cloud-only setups.
        self.semantic_available = False
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) / 255.0 for b in h]  # 32 floats from 32 bytes
        vec = (raw * ((EMBEDDING_DIM // len(raw)) + 1))[:EMBEDDING_DIM]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def _fit_dim(self, vec: list) -> list:
        """Match the memory DB's fixed dimension. Real embedders return e.g.
        768 floats, which the 128-wide vector table rejects; truncating and
        re-normalizing works for Matryoshka-style models (nomic-embed-text)."""
        if len(vec) == EMBEDDING_DIM:
            return vec
        if not self._warned_dim:
            logger.warning("embedding has %d dims; adapting to the memory DB's %d", len(vec), EMBEDDING_DIM)
            self._warned_dim = True
        vec = (vec + [0.0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def schedule_remember(self, prompt: str, response: str, operations: list) -> None:
        """Store this turn in memory without delaying the reply."""
        try:
            task = asyncio.get_running_loop().create_task(self.remember_turn(prompt, response, operations))
        except RuntimeError:
            return
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def remember_turn(self, prompt: str, response: str, operations: list) -> None:
        try:
            tools = f"\nActions: {'; '.join(operations[:12])}" if operations else ""
            content = f"Q: {prompt[:600]}\nA: {response[:900]}{tools}"
            embedding = await self.get_embedding(content)
            self.memory.add_memory(MemoryChunk(
                content=content, embedding=embedding, metadata={"type": "EPISODE"}, mem_type="EPISODE",
            ))
        except Exception as e:  # never let bookkeeping surface as a turn failure
            logger.debug(f"remember_turn skipped: {e}")

    async def summarize(self, turns: list) -> str:
        """Condense conversation turns ``[(prompt, response), ...]`` into a
        short factual summary (used by /compact)."""
        transcript = "\n\n".join(f"User: {p[:1500]}\nAssistant: {r[:1500]}" for p, r in turns)
        text = await self.provider.complete(
            "Summarize this conversation so it can replace the original as context. Keep decisions, "
            "file paths, commands, errors and open tasks; drop pleasantries. Use terse bullet points.\n\n"
            + transcript,
            system_prompt="You write compact, factual conversation summaries.",
        )
        return (text or "").strip()

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
        on_permission_request=None,
        allowed_paths: Optional[set] = None,
        *,
        session=None,
        on_ask_user=None,
        on_approval=None,
        on_todo=None,
        images: Optional[list] = None,
    ):
        """Run one turn (see ``core/agent_loop.py`` for the loop itself).

        ``on_permission_request(path) -> "once" | "session" | "deny"`` is asked
        when a tool call would escape the workspace root; ``"session"``
        persists the approval into ``allowed_paths``. ``on_approval(kind,
        subject, reason)`` is asked for risky shell commands and private-network
        fetches. ``on_ask_user(question, options)`` answers the model's
        ``ask_user`` tool. ``session`` (a ``ToolSession``) carries approvals,
        undo checkpoints and read-tracking across turns.
        """
        runner = TurnRunner(
            self,
            prompt,
            target=target,
            on_stream_chunk=on_stream_chunk,
            on_trace_event=on_trace_event,
            history=history,
            context_query=context_query,
            workspace=workspace,
            agent_mode=agent_mode,
            on_permission_request=on_permission_request,
            allowed_paths=allowed_paths,
            session=session,
            on_ask_user=on_ask_user,
            on_approval=on_approval,
            on_todo=on_todo,
            images=images,
        )
        return await runner.run()

def _load_dotenv(config_dir: str) -> None:
    """Load a .env file (if present) into os.environ without overwriting
    variables the shell/OS already set.

    Anchored to the given directory (normally REPO_DIR, not CWD) so `motion`
    finds its own .env regardless of which directory it's invoked from. Used
    by both the TUI launch path and the --chat/REPL path so API keys placed
    in .env (e.g. OLLAMA_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY) are
    always picked up, not just when going through load_agent_from_config.
    """
    env_path = os.path.join(config_dir, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and value:
                    os.environ.setdefault(key, value)


def load_agent_from_config(config_path: str = "", provider_id: str | None = None) -> MotionAgent:
    """Load a MotionAgent using settings from config.yml (or config.example.yml) and .env."""
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else REPO_DIR
    _load_dotenv(config_dir)

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


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Motion Agent")
    parser.add_argument("--test", action="store_true", help="Run Caveman compression test (no model needed)")
    parser.add_argument("--list", action="store_true", help="List available providers")
    parser.add_argument("--provider", type=str, default=None, help="Provider to use (e.g. ollama-cloud, ollama-cloud/gemma4:31b, claude, openai)")
    parser.add_argument("--chat", action="store_true", help="Launch in chat REPL mode instead of TUI")
    headless = parser.add_argument_group("headless (non-interactive) mode")
    headless.add_argument("-p", "--prompt", nargs="?", const="-", default=None, metavar="PROMPT",
                          help="Run one prompt non-interactively and print the result ('-' or no value reads the prompt from stdin)")
    headless.add_argument("--output-format", choices=("text", "json", "stream-json"), default="text",
                          help="Headless output: final text (default), one JSON object, or NDJSON events")
    headless.add_argument("--plan", action="store_true", help="Headless: read-only plan mode (no writes or commands)")
    headless.add_argument("--workspace", default=None, help="Headless: directory the agent works in (default: current directory)")
    headless.add_argument("--stdin", action="store_true", help="Headless: append piped stdin to the prompt as context")
    headless.add_argument("--verbose", action="store_true", help="Headless: print tool activity to stderr")
    sub = parser.add_subparsers(dest="command")
    auth_parser = sub.add_parser("auth", help="Manage provider API keys")
    auth_sub = auth_parser.add_subparsers(dest="auth_action", required=True)
    auth_sub.add_parser("list", help="List stored API keys")
    login_p = auth_sub.add_parser("login", help="Store an API key for a provider")
    login_p.add_argument("auth_provider", nargs="?", help="Provider id (e.g. ollama-cloud)")
    logout_p = auth_sub.add_parser("logout", help="Remove a stored API key")
    logout_p.add_argument("auth_provider", nargs="?", help="Provider id (e.g. ollama-cloud)")
    return parser


def main(argv=None) -> int:
    import sys

    # Load .env before touching ConfigManager anywhere below - the TUI launch
    # path (the default, no-flags invocation) previously skipped this
    # entirely, so API keys placed in .env never became visible to
    # has_api_key()/ModelDialog and no models appeared to choose from.
    _load_dotenv(REPO_DIR)

    args = build_parser().parse_args(argv)

    if args.prompt is not None:
        from core.headless import main_headless

        return main_headless(args)
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
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
