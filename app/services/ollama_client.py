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
    for chunk in stream:
        if not got_first:
            t2 = time.time()
            print(f"[chat_stream] FIRST TOKEN after {t2 - t1:.2f}s (total {t2 - t0:.2f}s)")
            got_first = True

        piece = chunk.get("message", {}).get("content", "")
        # Optional: print token sizes or token/sec estimates
        # print(f"[chat_stream] chunk: {len(piece)} chars")
        yield piece

    print("[chat_stream] STREAM END")
