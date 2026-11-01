"""Killable child-process execution for trusted Python transformations.

This is resource/lifecycle separation, not an OS sandbox for hostile code.
"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


async def run_code_process(payload: dict[str, Any], timeout: float) -> Any:
    with tempfile.TemporaryDirectory(prefix="base-agent-code-") as workdir:
        env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
        env.update({"TEMP": workdir, "TMP": workdir})
        request = json.dumps(payload).encode()
        if len(request) > 2_000_000:
            raise ValueError("Code input exceeds 2 MB")
        # Popen + a managed I/O thread works with Windows' Selector loop, which
        # psycopg requires. The actual submitted code always runs in the child.
        proc = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("code_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=workdir, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        communication = asyncio.create_task(asyncio.to_thread(proc.communicate, request))
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout)
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            await communication
            raise
        if proc.returncode:
            raise ValueError("Code worker failed")
        if len(stdout) > 2_000_000:
            raise ValueError("Code output exceeds 2 MB")
        result = json.loads(stdout)
        if "error" in result:
            raise ValueError(result["error"])
        return result["result"]
