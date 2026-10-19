"""Base Agent application package."""

import asyncio
import sys

# On Windows, psycopg 3 async requires SelectorEventLoop instead of the default ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
