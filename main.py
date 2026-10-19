import asyncio
import sys
import uvicorn

# On Windows, psycopg 3 async requires SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.main import app


def main() -> None:
    """Run development server directly via python main.py."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
