from fastapi import FastAPI
from app.config.settings import settings
from app.services.crawler import crawl
from app.services.retriever import global_index
from app.routers.ask import router as ask_router
from app.utils.cache import encode_content_for_cache, decode_content_from_cache
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="AI Assistant")

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ✅ any origin can call your API
    allow_credentials=True,
    allow_methods=["*"],       # allow all HTTP methods
    allow_headers=["*"],       # allow all headers
)


CACHE_PATH = os.path.join(os.path.dirname(__file__), "storage", "cache_index.json")
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global_index.docs.clear()  # always start fresh

    try:
        result = await crawl(settings.ROOT_URL)
        for doc_id, content in result.pages.items():
            global_index.add_document(doc_id, content)
        cache_pages = {doc_id: encode_content_for_cache(content) for doc_id, content in result.pages.items()}
        with open(CACHE_PATH, "w") as f:
            json.dump({"root_url": settings.ROOT_URL, "pages": cache_pages}, f)
        print(f"Crawled and cached {len(result.pages)} pages from {settings.ROOT_URL}")
    except Exception as e:
        print(f"Startup crawl failed: {e}")


app.include_router(ask_router, tags=["ask"])
