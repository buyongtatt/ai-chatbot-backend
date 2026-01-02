import ollama
from app.config.settings import settings

def chat(messages, model: str = None) -> str:
    """Return full answer (non-streaming)."""
    model = model or settings.OLLAMA_MODEL
    resp = ollama.chat(model=model, messages=messages)
    return resp["message"]["content"]

def chat_stream(messages, model: str = None):
    """Yield chunks of answer progressively (streaming)."""
    model = model or settings.OLLAMA_MODEL
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        # Each chunk contains partial message content
        yield chunk["message"]["content"]
