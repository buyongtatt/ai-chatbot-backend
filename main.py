from fastapi import FastAPI
from app.config.settings import settings
from app.services.crawler import crawl
from app.services.retriever import global_index
from app.routers.ask import router as ask_router
from app.utils.cache import encode_content_for_cache, decode_content_from_cache
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import uvicorn

app = FastAPI(
    title="AI Assistant",
    # Configure for better concurrency handling
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "storage", "cache_index.json")
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup - crawling will happen on-demand when user asks questions"""
    print("=== SERVER STARTUP ===")
    print(f"Ready to crawl on-demand from: {settings.ROOT_URL}")
    print(f"Cache path: {CACHE_PATH}")

@app.get("/config")
def get_config():
    return settings.dict()

# Debug endpoint to see all documents
@app.get("/debug/documents")
async def debug_documents():
    """List all indexed documents with details"""
    docs_info = []
    for doc_id, doc_content in global_index.documents.items():
        docs_info.append({
            "doc_id": doc_id,
            "content_type": doc_content.get("content_type", "unknown"),
            "text_length": len(doc_content.get("text", "")),
            "images_count": len(doc_content.get("images", [])),
            "files_count": len(doc_content.get("files", [])),
            "chunks_count": len(global_index.doc_to_chunks.get(doc_id, [])),
            "image_details": [
                {
                    "source": img.get("source", "unknown"),
                    "filename": img.get("filename", "unnamed"),
                    "mime": img.get("mime", "unknown"),
                    "size_bytes": len(img.get("content", b"")) if img.get("content") else 0
                }
                for img in doc_content.get("images", [])[:5]  # Show first 5 images
            ]
        })
    return {"documents": docs_info, "total": len(docs_info)}

# Debug endpoint to see chunks
@app.get("/debug/chunks")
async def debug_chunks():
    """List all chunks"""
    chunks_info = []
    for i, chunk in enumerate(global_index.chunks):
        # Only show first 50 chunks to avoid huge responses
        if i >= 50:
            chunks_info.append({"message": f"... and {len(global_index.chunks) - 50} more chunks"})
            break
            
        chunks_info.append({
            "index": i,
            "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
            "doc_id": chunk.get("doc_id", "unknown"),
            "text_length": len(chunk.get("text", "")),
            "text_preview": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
        })
    return {"chunks": chunks_info, "total": len(global_index.chunks)}

# Debug endpoint for specific document
@app.get("/debug/document/{doc_id:path}")
async def debug_document(doc_id: str):
    """Get detailed info about a specific document"""
    # Try exact match first
    document = global_index.documents.get(doc_id)
    
    # If not found, try partial matching
    if not document:
        matching_docs = [id for id in global_index.documents.keys() if doc_id in id]
        if matching_docs:
            return {
                "error": f"Exact document ID not found",
                "similar_ids": matching_docs,
                "message": "Try one of the similar IDs above"
            }
        else:
            return {"error": f"Document ID '{doc_id}' not found"}
    
    # Get chunks for this document
    chunk_indices = global_index.doc_to_chunks.get(doc_id, [])
    chunks = [global_index.chunks[i] for i in chunk_indices if i < len(global_index.chunks)]
    
    return {
        "document_id": doc_id,
        "content_type": document.get("content_type"),
        "text_length": len(document.get("text", "")),
        "metadata": document.get("meta", {}),
        "images": [
            {
                "source": img.get("source", "unknown"),
                "filename": img.get("filename", "unnamed"),
                "mime": img.get("mime", "unknown"),
                "size_bytes": len(img.get("content", b"")) if img.get("content") else 0
            }
            for img in document.get("images", [])
        ],
        "files": len(document.get("files", [])),
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "text_preview": chunk.get("text", "")[:300] + "..." if len(chunk.get("text", "")) > 300 else chunk.get("text", ""),
                "length": len(chunk.get("text", ""))
            }
            for i, chunk in enumerate(chunks)
        ]
    }

app.include_router(ask_router, tags=["ask"])