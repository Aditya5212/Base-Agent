"""Interactive verification script testing live DeepSeek streaming and PostgreSQL persistence.

Usage:
  uv run python -m scripts.test_deepseek_stream_live
"""

import asyncio
import uuid
from app.services.agent import stream_agent_events
from app.services.memory import get_thread_messages, delete_thread_memory, close_checkpointer_pool


async def run_turn(prompt: str, thread_id: str, model: str):
    print(f"\n--- User: {prompt} (Model: {model}) ---")
    print("Assistant: ", end="", flush=True)

    full_response = ""
    history_len = 0
    async for event in stream_agent_events(
        message=prompt,
        thread_id=thread_id,
        user_id="aditya",
        model=model,
    ):
        if event.event == "token":
            delta = event.data.get("delta", "")
            print(delta, end="", flush=True)
            full_response += delta
        elif event.event == "reasoning":
            r_delta = event.data.get("delta", "")
            print(f"[thinking: {r_delta}]", end="", flush=True)
        elif event.event == "end":
            history_len = event.data.get("history_length", 0)
        elif event.event == "error":
            print(f"\n[Error: {event.data.get('error')}]")

    print(f"\n[Turn finished | Thread history count: {history_len}]")


async def main():
    thread_id = f"stream-live-{uuid.uuid4().hex[:6]}"
    print(f"=== Starting Live DeepSeek SSE Streaming Test [Thread: {thread_id}] ===")

    # Turn 1: Stream intro
    await run_turn(
        prompt="Hi! My name is Aditya and my favorite framework is LangGraph. Keep it short.",
        thread_id=thread_id,
        model="deepseek-flash",
    )

    # Turn 2: Memory recall via streaming
    await run_turn(
        prompt="What is my name and what is my favorite framework?",
        thread_id=thread_id,
        model="deepseek-flash",
    )

    # Turn 3: Switch to deepseek-v4-pro streaming
    await run_turn(
        prompt="In one short sentence, summarize how streaming improves user experience.",
        thread_id=thread_id,
        model="deepseek-v4-pro",
    )

    # Verify storage in Neon PostgreSQL
    saved_messages = await get_thread_messages(thread_id)
    print(f"\nPersisted messages in Neon PostgreSQL: {len(saved_messages)}")
    for i, m in enumerate(saved_messages, 1):
        role = getattr(m, "type", "unknown")
        text = getattr(m, "text", None) or str(m.content)
        print(f"  [{i}] ({role}): {text[:70]}...")

    # Cleanup
    await delete_thread_memory(thread_id)
    print("\nCleaned up test thread.")
    await close_checkpointer_pool()


if __name__ == "__main__":
    asyncio.run(main())
