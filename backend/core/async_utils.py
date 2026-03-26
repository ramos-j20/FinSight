"""Utilities for handling async operations from synchronous code."""
import asyncio
import threading
from typing import Coroutine, Any


def run_sync(coro: Coroutine, wait: bool = False) -> Any:
    """
    Safely run an async coroutine from synchronous code.
    
    If a loop is already running (e.g. FastAPI, Benchmark loop):
    - If wait=True: Runs in a separate thread and blocks until done.
    - If wait=False: Creates a background task in the existing loop (fire-and-forget).
    
    If no loop is running:
    - Uses asyncio.run() to execute the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        if not wait:
            # Fire-and-forget in the existing loop
            return loop.create_task(coro)
        
        # We need to wait for the result from within a running loop.
        # This is the tricky part that can cause pool issues.
        # For now, let's use the thread-based wait as a fallback.
        result = [None]
        exception = [None]

        def target():
            try:
                # Use a fresh loop in the new thread
                result[0] = asyncio.run(coro)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()

        if exception[0]:
            raise exception[0]
        return result[0]
    else:
        # No loop running, asyncio.run is safe
        return asyncio.run(coro)
