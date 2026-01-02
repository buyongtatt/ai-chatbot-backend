from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ROOT_URL: str = "https://docs.example.com"
    OLLAMA_MODEL: str = "llama3"
    MAX_PAGES: int = 50
    MAX_DEPTH: int = 3
    REQUEST_TIMEOUT: int = 15

    class Config:
        env_file = ".env"

settings = Settings()
