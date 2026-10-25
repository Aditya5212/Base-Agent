import asyncio
import uuid
from app.core.taskiq import broker, init_taskiq_broker, shutdown_taskiq_broker
from app.tasks.document_tasks import process_document_task
from sqlalchemy import text
from app.core.database import async_session_factory

async def test_live():
    print("1. Initializing broker...")
    await init_taskiq_broker()
    print("   Broker initialized:", broker.is_worker_process)

    print("2. Sending a test task message via kiq()...")
    # We call kiq with dummy parameters
    task = await process_document_task.kiq(
        document_id=f"test-live-{uuid.uuid4().hex[:6]}",
        r2_key="dummy_key",
        user_id="root",
        file_name="dummy.txt",
        preset="balanced",
        is_reindex=False,
    )
    print(f"   Task sent! Taskiq task_id: {task.task_id}")

    print("3. Querying taskiq_messages table in PostgreSQL to verify the message was stored...")
    async with async_session_factory() as session:
        # Check column names
        cols_res = await session.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'taskiq_messages' ORDER BY ordinal_position"
        ))
        cols = cols_res.fetchall()
        print("   taskiq_messages schema:", [(c[0], c[1]) for c in cols])

        result = await session.execute(
            text("SELECT * FROM taskiq_messages WHERE task_id = :task_id"),
            {"task_id": task.task_id}
        )
        row = result.fetchone()
        if row:
            print(f"   SUCCESS! Found queued message in database for task_id={task.task_id}!")
            # Clean up the dummy test message
            await session.execute(
                text("DELETE FROM taskiq_messages WHERE task_id = :task_id"),
                {"task_id": task.task_id}
            )
            await session.commit()
            print("   Cleaned up test row.")
        else:
            print("   WARNING: Message not found in taskiq_messages table for task_id:", task.task_id)

    print("4. Shutting down broker...")
    await shutdown_taskiq_broker()
    print("   Broker shutdown complete.")

if __name__ == "__main__":
    asyncio.run(test_live())
