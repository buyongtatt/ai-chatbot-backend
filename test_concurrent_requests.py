#!/usr/bin/env python3
"""
Test script to verify concurrent request handling
"""

import asyncio
import aiohttp
import time

async def make_request(session, url, question, request_id):
    """Make a single request to the ask endpoint"""
    try:
        start_time = time.time()
        async with session.post(url, data={'question': question}) as response:
            # Read the streaming response
            response_text = await response.text()
            end_time = time.time()
            print(f"Request {request_id} completed in {end_time - start_time:.2f} seconds")
            return response_text
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None

async def test_concurrent_requests():
    """Test multiple concurrent requests"""
    url = "http://localhost:8000/ask_stream"
    
    # Sample questions for testing
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "How do I make a chocolate cake?",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis"
    ]
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for concurrent requests
        tasks = []
        for i, question in enumerate(questions):
            task = make_request(session, url, question, i+1)
            tasks.append(task)
        
        print(f"Starting {len(tasks)} concurrent requests...")
        start_time = time.time()
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        print(f"All requests completed in {end_time - start_time:.2f} seconds")
        
        # Print summary
        successful_requests = sum(1 for result in results if result is not None)
        print(f"Successful requests: {successful_requests}/{len(tasks)}")
        
        return results

if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())