import asyncio
import logging

class MultiTenantScheduler:
    """
    Limits concurrent solves to prevent GPU/CPU Out-Of-Memory conditions.
    Maintains a semaphore and monitors queue depth for observability.
    """
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks = 0

    async def gather(self, tasks: list):
        """Helper to cleanly await a list of tasks."""
        return await asyncio.gather(*tasks)

    # Added context manager properties to accurately monitor active load
    async def __aenter__(self):
        await self.semaphore.acquire()
        self._active_tasks += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._active_tasks -= 1
        self.semaphore.release()
        
    @property
    def utilization(self) -> float:
        """Returns the current saturation of the scheduler (0.0 to 1.0)."""
        return min(self._active_tasks / self.max_concurrent, 1.0)