import re
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import uuid

# app/services/retriever.py

class AIRetrieverIndex:
    def __init__(
        self,
        max_chunk_chars: int = 2500,
        overlap_chars: int = 400,
        max_context_tokens: int = 40000,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.max_context_tokens = max_context_tokens

        # Store original documents with metadata
        self.documents: Dict[str, Dict[str, Any]] = {}
        
        # Store chunks for retrieval
        self.chunks: List[Dict[str, Any]] = []
        self.doc_to_chunks: Dict[str, List[int]] = defaultdict(list)

    def add_document(self, doc_id: str, content: Dict[str, Any]) -> None:
        """Add a document and create chunks for retrieval"""
        # Store the complete document including images
        self.documents[doc_id] = content

        # Create chunks from text content
        text = content.get("text", "") or ""
        chunks = self._split_text_into_chunks(text, doc_id)

        # Add chunks to index
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.doc_to_chunks[doc_id].append(start_idx + i)

    def _split_text_into_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Split text into chunks only if document is large enough"""
        if not text or not text.strip():
            return []

        # If document is small enough, return it as a single chunk
        if len(text) <= self.max_chunk_chars:
            return [{
                "chunk_id": f"{doc_id}#chunk-0",
                "doc_id": doc_id,
                "chunk_index": 0,
                "text": text.strip(),
                "char_length": len(text)
            }]

        # Only chunk if document is large
        paragraphs = []
        current_para = ""
        
        for line in text.split('\n'):
            if line.strip().startswith('[PAGE:') or line.strip().startswith('[/PAGE:'):
                # Treat PAGE markers as separate paragraphs
                if current_para.strip():
                    paragraphs.append(current_para.strip())
                paragraphs.append(line.strip())
                current_para = ""
            elif line.strip() == "":
                if current_para.strip():
                    paragraphs.append(current_para.strip())
                    current_para = ""
            else:
                current_para += line + "\n"
        
        if current_para.strip():
            paragraphs.append(current_para.strip())

        chunks = []
        current_chunk_paragraphs = []
        current_length = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if (current_length + paragraph_length > self.max_chunk_chars and 
                current_length >= 500 and current_chunk_paragraphs):
                
                # Create chunk
                chunk_text = "\n".join(current_chunk_paragraphs)
                if chunk_text.strip():
                    chunks.append({
                        "chunk_id": f"{doc_id}#chunk-{chunk_index}",
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "text": chunk_text.strip(),
                        "char_length": len(chunk_text)
                    })
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.overlap_chars > 0 and current_chunk_paragraphs:
                    # Take last part of previous chunk as overlap
                    combined_text = "\n".join(current_chunk_paragraphs)
                    if len(combined_text) > self.overlap_chars:
                        overlap_text = combined_text[-self.overlap_chars:]
                    else:
                        overlap_text = combined_text
                    current_chunk_paragraphs = [overlap_text, paragraph]
                    current_length = len(overlap_text) + paragraph_length
                else:
                    current_chunk_paragraphs = [paragraph]
                    current_length = paragraph_length
            else:
                current_chunk_paragraphs.append(paragraph)
                current_length += paragraph_length + 1  # +1 for \n

        # Handle final chunk
        if current_chunk_paragraphs:
            chunk_text = "\n".join(current_chunk_paragraphs)
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"{doc_id}#chunk-{chunk_index}",
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text.strip(),
                    "char_length": len(chunk_text)
                })

        return chunks

    def retrieve_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve all relevant chunks without AI scoring for faster response"""
        if not self.chunks:
            return []

        print(f"[Retriever] Retrieved {len(self.chunks)} chunks for query (no scoring needed)")
        
        # Return all chunks with a default relevance score
        result_chunks = []
        for chunk in self.chunks:
            chunk_with_score = chunk.copy()
            chunk_with_score["relevance_score"] = 0.9  # Default high score since all chunks are relevant
            result_chunks.append(chunk_with_score)
        
        return result_chunks

    
    def get_document_assets(self, marker: str) -> Dict[str, Any]:
        """Get assets (images/files) for a document or image source"""
        
        print(f"[DEBUG] Looking for assets for marker: {marker}")
        
        # Case 1: Marker is a full document ID
        if marker.startswith("docs://"):
            document = self.documents.get(marker)
            
            if document:
                # Check if this document has its own images/files
                images = document.get("images", [])
                files = document.get("files", [])
                
                # If this document has images/files, return them directly
                if images or files:
                    print(f"[DEBUG] Found {len(images)} images and {len(files)} files in document directly")
                    return {"images": images, "files": files}
                
                # If this is a chunk, look up images from parent document
                meta = document.get("meta", {})
                if meta.get("is_chunk") and meta.get("parent_document"):
                    parent_doc_id = meta["parent_document"]
                    print(f"[DEBUG] This is a chunk, looking up parent: {parent_doc_id}")
                    parent_document = self.documents.get(parent_doc_id)
                    if parent_document:
                        parent_images = parent_document.get("images", [])
                        parent_files = parent_document.get("files", [])
                        print(f"[DEBUG] Found {len(parent_images)} images and {len(parent_files)} files from parent")
                        return {"images": parent_images, "files": parent_files}
            
            # If no document found, try manual parent document lookup
            if '#' in marker:
                base_doc_id = marker.split('#')[0]
                print(f"[DEBUG] Trying manual parent lookup: {base_doc_id}")
                if base_doc_id != marker:
                    parent_document = self.documents.get(base_doc_id)
                    if parent_document:
                        images = parent_document.get("images", [])
                        files = parent_document.get("files", [])
                        print(f"[DEBUG] Found {len(images)} images and {len(files)} files from manual parent lookup")
                        return {"images": images, "files": files}
        
        # Case 2: Marker is just an image source ID (like "fftester_page7_area12_18")
        # Search all documents for this image source
        else:
            print(f"[DEBUG] Searching for image source: {marker}")
            for doc_id, document in self.documents.items():
                images = document.get("images", [])
                for img in images:
                    if img.get("source") == marker:
                        print(f"[DEBUG] Found image source {marker} in document {doc_id}")
                        return {"images": [img], "files": []}  # Return just this image
                
                # Also check parent documents if this is a chunk
                meta = document.get("meta", {})
                if meta.get("is_chunk") and meta.get("parent_document"):
                    parent_doc_id = meta["parent_document"]
                    parent_document = self.documents.get(parent_doc_id)
                    if parent_document:
                        parent_images = parent_document.get("images", [])
                        for img in parent_images:
                            if img.get("source") == marker:
                                print(f"[DEBUG] Found image source {marker} in parent document {parent_doc_id}")
                                return {"images": [img], "files": []}  # Return just this image
        
        print(f"[DEBUG] No assets found for marker: {marker}")
        return {"images": [], "files": []}

# Global instance
global_index = AIRetrieverIndex(
    max_chunk_chars=2500,
    overlap_chars=400,
    max_context_tokens=40000,
)