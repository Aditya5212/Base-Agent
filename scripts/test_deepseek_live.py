"""Interactive / verification script testing live DeepSeek models and PostgreSQL persistence.

Usage:
  uv run python -m scripts.test_deepseek_live
"""

import asyncio
import uuid
from app.services.agent import execute_agent
from app.services.memory import get_thread_messages, delete_thread_memory, close_checkpointer_pool


async def main():
    thread_id = f"deepseek-live-{uuid.uuid4().hex[:6]}"
    print(f"=== Starting Live DeepSeek + Neon Test [Thread: {thread_id}] ===")

    # Turn 1
    print("\n--- Turn 1: Sending user introduction to deepseek-flash ---")
    state1, model1 = await execute_agent(
        message="Hello! My name is Aditya and my favorite framework is LangGraph. Remember this.",
        thread_id=thread_id,
        user_id="aditya",
        model="deepseek-flash",
    )
    print(f"Model: {model1}")
    print(f"Assistant: {state1['messages'][-1].content}")

    # Turn 2: Memory test
    print("\n--- Turn 2: Testing Memory Recall ---")
    state2, model2 = await execute_agent(
        message="What is my name and what is my favorite framework?",
        thread_id=thread_id,
        user_id="aditya",
        model="deepseek-flash",
    )
    print(f"Model: {model2}")
    print(f"Assistant: {state2['messages'][-1].content}")

    # Turn 3: Switch to deepseek-v4-pro
    print("\n--- Turn 3: Testing deepseek-v4-pro on same thread ---")
    state3, model3 = await execute_agent(
        message="In one short sentence, why is PostgreSQL good for agent checkpoints?",
        thread_id=thread_id,
        user_id="aditya",
        model="deepseek-v4-pro",
    )
    print(f"Model: {model3}")
    print(f"Assistant: {state3['messages'][-1].content}")

    # Verify storage in Neon PostgreSQL
    saved_messages = await get_thread_messages(thread_id)
    print(f"\nPersisted messages in Neon PostgreSQL: {len(saved_messages)}")
    for i, m in enumerate(saved_messages, 1):
        role = getattr(m, "type", "unknown")
        print(f"  [{i}] ({role}): {m.content[:70]}...")

    # Cleanup
    await delete_thread_memory(thread_id)
    print("\nCleaned up test thread.")
    await close_checkpointer_pool()


if __name__ == "__main__":
    asyncio.run(main())
