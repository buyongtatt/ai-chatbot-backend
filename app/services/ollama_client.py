
from ollama import Client
from app.config.settings import settings

# Create a singleton client (e.g., during app startup)
client = Client(host=settings.OLLAMA_HOST)  # e.g., "http://localhost:11434"

def chat(messages, model: str = None) -> str:
    model = model or settings.OLLAMA_MODEL
    resp = client.chat(model=model, messages=messages)
    return resp["message"]["content"]

def chat_stream(messages, model: str = None):
    model = model or settings.OLLAMA_MODEL
    stream = client.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]
