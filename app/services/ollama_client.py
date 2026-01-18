# app/services/ollama_client.py
import time
from ollama import Client
from app.config.settings import settings

# Create a singleton client (e.g., during app startup)
client = Client(host=settings.OLLAMA_HOST)  # e.g., "http://localhost:11434"

def chat_stream(messages, model: str = None):
    model = model or settings.OLLAMA_MODEL
    t0 = time.time()
    print(f"[chat_stream] START model={model}; building request...")

    # Kick off Ollama request in streaming mode
    stream = client.chat(model=model, messages=messages, stream=True)
    t1 = time.time()
    print(f"[chat_stream] REQUEST SENT in {t1 - t0:.2f}s; waiting for first token...")

    got_first = False
    token_count = 0
    last_token_time = t1
    
    for chunk in stream:
        current_time = time.time()
        
        if not got_first:
            print(f"[chat_stream] FIRST TOKEN after {current_time - t1:.2f}s (total {current_time - t0:.2f}s)")
            got_first = True

        piece = chunk.get("message", {}).get("content", "")
        
        if piece:  # Only count and log non-empty pieces
            token_count += 1
            time_since_last_token = current_time - last_token_time
            
            # Log every 5 tokens or for the first few tokens
            if token_count <= 10 or token_count % 5 == 0:
                print(f"[chat_stream] Token #{token_count}: {len(piece)} chars, "
                      f"time_since_last: {time_since_last_token:.3f}s, "
                      f"total_time: {current_time - t0:.2f}s")
            elif piece.endswith(('.', '!', '?', '\n')):  # Log at sentence endings
                print(f"[chat_stream] Token #{token_count}: {len(piece)} chars, "
                      f"time_since_last: {time_since_last_token:.3f}s (sentence end)")
            
            last_token_time = current_time

        yield piece

    # Final summary
    total_time = time.time() - t0
    if token_count > 0:
        avg_time_per_token = (time.time() - t1) / token_count
        print(f"[chat_stream] STREAM END - Total tokens: {token_count}, "
              f"Total time: {total_time:.2f}s, "
              f"Avg time/token: {avg_time_per_token:.3f}s")
    else:
        print(f"[chat_stream] STREAM END - No tokens received, Total time: {total_time:.2f}s")