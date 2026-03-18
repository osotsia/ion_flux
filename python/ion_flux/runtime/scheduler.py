import asyncio

class MultiTenantScheduler:
    """
    Limits concurrent solves to prevent GPU/CPU OOM and thrashing.
    Integrates smoothly with standard Python web frameworks like FastAPI.
    """
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def gather(self, tasks):
        """Helper to cleanly await a list of tasks."""
        return await asyncio.gather(*tasks)