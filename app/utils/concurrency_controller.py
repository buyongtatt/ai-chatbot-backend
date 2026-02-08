import asyncio
import time
from typing import Optional, Callable
from collections import deque
from app.config.settings import settings

class ConcurrencyController:
    """
    Controls concurrent requests to prevent overwhelming Ollama.
    Uses a semaphore to limit simultaneous processing.
    """
    
    def __init__(self, max_concurrent: int = None):
        self.max_concurrent = max_concurrent or settings.MAX_CONCURRENT_REQUESTS
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.active_requests = 0
        self.queued_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    async def acquire(self, request_id: str = None) -> None:
        """
        Acquire a slot for request processing.
        Blocks if max concurrent requests reached.
        """
        self.queued_requests += 1
        self.total_requests += 1
        
        request_id = request_id or f"req_{self.total_requests}"
        print(f"[ConcurrencyControl] Request {request_id} queued. "
              f"Active: {self.active_requests}, Queued: {self.queued_requests}, "
              f"Limit: {self.max_concurrent}")
        
        await self.semaphore.acquire()
        
        self.active_requests += 1
        self.queued_requests -= 1
        
        print(f"[ConcurrencyControl] Request {request_id} acquired slot. "
              f"Active: {self.active_requests}, Queued: {self.queued_requests}")
    
    def release(self, request_id: str = None) -> None:
        """
        Release a slot after request completes.
        """
        self.semaphore.release()
        self.active_requests = max(0, self.active_requests - 1)
        
        request_id = request_id or "unknown"
        print(f"[ConcurrencyControl] Request {request_id} released. "
              f"Active: {self.active_requests}, Queued: {self.queued_requests}")
    
    async def context_manager_acquire(self, request_id: str = None):
        """Context manager version for use with 'async with'"""
        await self.acquire(request_id)
        return self
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def get_stats(self) -> dict:
        """Get current stats"""
        uptime = time.time() - self.start_time
        return {
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_requests_processed": self.total_requests,
            "max_concurrent": self.max_concurrent,
            "uptime_seconds": uptime,
            "avg_requests_per_minute": (self.total_requests / uptime) * 60 if uptime > 0 else 0
        }


# Global instance
concurrency_controller = ConcurrencyController()
