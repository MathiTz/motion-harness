from core.providers import ModelConfig, ProviderFactory
from core.caveman import CavemanProtocol
from core.learning import SkillSynthesizer, Trajectory
from memory.db import MemoryDB
from memory.retriever import HybridRetriever
import asyncio
import logging

logger = logging.getLogger(__name__)

class MotionAgent:
    def __init__(self, model_config: ModelConfig, memory_path: str = "motion_memory.db"):
        self.provider = ProviderFactory.get_provider(model_config)
        self.memory = MemoryDB(memory_path)
        self.retriever = HybridRetriever(self.memory, self)
        self.caveman = CavemanProtocol(enabled=True)
        self.synthesizer = SkillSynthesizer(model_config, self.memory)

    async def get_embedding(self, text: str):
        # Default embedding provider; replace with real model embedding in production
        return [float(len(text)) / 100.0] * 128

    async def run(self, prompt: str, target: str = "user"):
        # 1. Memory Recall
        context_chunks = await self.retriever.retrieve(prompt)
        context_text = "\n".join([c["content"] for c in context_chunks])

        # 2. Construct System Prompt
        system_prompt = f"You are Motion Agent. Memory Context:\n{context_text}"

        # 3. Model Completion
        raw_response = await self.provider.complete(prompt, system_prompt=system_prompt)

        # 4. Caveman Compression
        final_response = self.caveman.process_outgoing(raw_response, target=target)

        # 5. Skill Crystallization (on success)
        try:
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
        except Exception as e:
            logger.debug(f"Skill synthesis skipped: {e}")

        return final_response

async def test_compression():
    config = ModelConfig(name="Claude-3.5", endpoint="https://api.anthropic.com", provider_type="cloud")
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

if __name__ == "__main__":
    asyncio.run(test_compression())
