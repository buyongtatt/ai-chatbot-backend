#!/usr/bin/env python3
"""
Startup script for AI Chatbot Backend with optimized concurrency settings
"""

import uvicorn
import argparse
from app.config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="AI Chatbot Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    
    args = parser.parse_args()
    
    # Use configured defaults if not specified
    workers = args.workers or settings.MAX_WORKER_PROCESSES
    
    # Run the server with optimized settings for concurrency
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=workers,
        reload=args.reload,
        # Optimize for concurrent connections
        loop="asyncio",  # Use asyncio event loop
        http="h11",      # HTTP/1.1 protocol
        # Connection handling
        limit_concurrency=settings.CONNECTION_LIMIT,     # Max concurrent connections
        limit_max_requests=10000,   # Max requests per worker before recycling
        # Timeout settings
        timeout_keep_alive=5,       # Keep-alive timeout
        # Backlog for incoming connections
        backlog=2048,
    )

if __name__ == "__main__":
    main()