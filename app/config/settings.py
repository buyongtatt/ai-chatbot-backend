from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_HOST: str = "http://localhost:11434"
    ROOT_URL: str = "https://docs.example.com"
    OLLAMA_MODEL: str = "llama3"
    MAX_PAGES: int = 50
    MAX_DEPTH: int = 3
    REQUEST_TIMEOUT: int = 15
    KB_CONFIG_PATH: str = "storage/knowledge_bases.json"
    
    # Concurrency settings
    MAX_CONCURRENT_REQUESTS: int = 100
    MAX_WORKER_PROCESSES: int = 4
    CONNECTION_LIMIT: int = 1000

    class Config:
        env_file = ".env"

settings = Settings()
