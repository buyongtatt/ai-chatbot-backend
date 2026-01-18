import base64
import re
import io
import json
import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.services.retriever import global_index
from app.services.ingest import extract_any
from app.services.ollama_client import chat_stream

router = APIRouter()

def _sniff_image_bytes(data: bytes) -> Optional[bytes]:
    """Return normalized image bytes (JPEG) if data looks like an image; else None."""
    try:
        img = Image.open(io.BytesIO(data))
        buf = io.BytesIO()
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None

def analyze_question_intent(question: str) -> Dict[str, bool]:
    """Analyze what type of content the question is asking for"""
    question_lower = question.lower()
    
    visual_keywords = ['image', 'chart', 'graph', 'diagram', 'figure', 'photo', 'picture', 
                       'visual', 'screenshot', 'show me', 'display', 'look like', 'appear']
    
    data_keywords = ['data', 'file', 'table', 'number', 'report', 'document', 'content',
                     'extract', 'download', 'spreadsheet', 'information', 'configuration']
    
    procedural_keywords = ['how to', 'steps', 'procedure', 'process', 'configure', 'setup']
    
    return {
        'wants_visual': any(keyword in question_lower for keyword in visual_keywords),
        'wants_data': any(keyword in question_lower for keyword in data_keywords),
        'wants_procedure': any(keyword in question_lower for keyword in procedural_keywords)
    }

def build_context_for_ai(relevant_chunks: List[Dict], question: str) -> str:
    """Build optimized context for AI with chunk-level relevance"""
    
    context_parts = []
    context_parts.append(f"USER QUESTION: {question}")
    
    # Add question intent analysis
    intent = analyze_question_intent(question)
    intent_desc = []
    if intent['wants_visual']:
        intent_desc.append("Looking for visual content")
    if intent['wants_data']:
        intent_desc.append("Looking for data content") 
    if intent['wants_procedure']:
        intent_desc.append("Looking for procedural information")
    
    if intent_desc:
        context_parts.append("ANALYSIS: " + ", ".join(intent_desc))
    
    context_parts.append("\nRELEVANT CONTEXT CHUNKS (ordered by AI relevance score):\n")
    context_parts.append("=" * 60)
    
    total_chars = 0
    MAX_CONTEXT_CHARS = 25000
    
    for i, chunk in enumerate(relevant_chunks):
        chunk_text = f"\nCHUNK {i+1} (Relevance Score: {chunk.get('relevance_score', 0):.3f}):\n"
        chunk_text += f"Source Document: {chunk.get('doc_id', 'unknown')}\n"
        chunk_text += f"Content: {chunk.get('text', '')}\n"
        
        if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS and i > 0:
            context_parts.append("\n[Additional context omitted due to length limits]")
            break
            
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)
    
    return "\n".join(context_parts)

@router.post("/ask_stream")
async def ask_stream(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    uploaded_doc_id: Optional[str] = None
    uploaded_meta: Optional[Dict[str, Any]] = None
    uploaded_img_bytes: Optional[bytes] = None
    uploaded_is_image = False

    # Handle uploaded file
    if file:
        data = await file.read()
        fname = (file.filename or "").strip()
        ctype = (file.content_type or "").lower()

        print(f"[ask_stream] Processing upload: {fname} ({len(data)} bytes)")

        # Detect if it's an image
        if ctype.startswith("image/"):
            uploaded_is_image = True
            uploaded_img_bytes = data
        else:
            sniffed = _sniff_image_bytes(data)
            if sniffed:
                uploaded_is_image = True
                uploaded_img_bytes = sniffed
                ctype = "image/jpeg"

        # Generate doc_id
        if not fname:
            inferred_ext = ".jpg" if uploaded_is_image else ""
            fname = f"blob{inferred_ext}"
        uploaded_doc_id = f"uploaded://{uuid.uuid4().hex}-{fname}"

        # Extract content
        text, meta = extract_any(data, fname)
        meta = dict(meta or {})
        meta.setdefault("text", text or "")
        uploaded_meta = meta

        # Add to index
        global_index.add_document(uploaded_doc_id, {
            "doc_id": uploaded_doc_id,
            "text": meta.get("text"),
            "images": meta.get("images", []) or ([{
                "source": fname,
                "mime": ctype or "image/jpeg",
                "content": uploaded_img_bytes,
            }] if uploaded_is_image and not meta.get("images") else []),
            "files": meta.get("files", []),
            "source_url": None,
            "content_type": ctype,
            "meta": meta
        })

    # Retrieve relevant context using AI-based scoring
    print(f"[ask_stream] Retrieving context for question: {question}")
    relevant_chunks = global_index.retrieve_relevant_context(question)
    print(f"[ask_stream] Found {len(relevant_chunks)} relevant chunks")

    # Build context for AI
    context_text = build_context_for_ai(relevant_chunks, question)
    
    # Track which documents have assets for later use
    relevant_doc_ids = list(set(chunk.get('doc_id') for chunk in relevant_chunks))

    prompt = f"""{context_text}

QUESTION: {question}

Please answer the question based ONLY on the context provided above. The context chunks are ordered by AI-calculated relevance (highest first). Focus on the most relevant chunks and ignore chunks with low relevance scores."""

    def build_user_message(prompt: str) -> Dict[str, Any]:
        return {"role": "user", "content": prompt}

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant who answers questions based ONLY on provided context.\n\n"
                "CONTEXT FORMAT:\n"
                "- Context contains numbered chunks ordered by AI relevance score (1.0 = highest)\n"
                "- Each chunk shows its relevance score and source document\n"
                "- Focus primarily on chunks with scores > 0.5\n\n"
                "YOUR RESPONSE SHOULD:\n"
                "- Answer ONLY what the question asks\n"
                "- Use information ONLY from relevant context chunks\n"
                "- Ignore chunks with relevance < 0.3\n"
                "- If information isn't in context, say so clearly\n"
                "- Be concise and focused\n"
                "- Do NOT include technical implementation details\n"
                "- Do NOT explain how the scoring worked\n\n"
                "ASSET HANDLING:\n"
                "- Assets (images/files) are available from source documents\n"
                "- Reference assets ONLY when they directly help answer the question\n"
                "- Use format: [[IMAGE:document_id]] or [[FILE:document_id]]\n"
                "- Reference BEFORE describing the asset content\n\n"
                "EXAMPLE FORMAT:\n"
                "'Based on the context, sales increased by 15%. As shown in [[IMAGE:docs://example.pdf]], the chart illustrates this trend.'\n"
                "'The configuration steps are provided in [[FILE:docs://config.pdf]].'"
            ),
        },
        build_user_message(prompt),
    ]

    def _as_bytes(x) -> Optional[bytes]:
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            return bytes(x)
        if isinstance(x, str):
            try:
                return base64.b64decode(x, validate=False)
            except Exception:
                return None
        return None

    def _is_httpish(u: str) -> bool:
        return isinstance(u, str) and u.startswith(("http://", "https://"))

    def json_stream():
        full_answer = ""
        try:
            for chunk in chat_stream(messages):
                full_answer += chunk
                yield json.dumps({"type": "text", "content": chunk}) + "\n"
        finally:
            print("[ask_stream] Response streaming completed")

        # Extract asset references from the answer
        if full_answer.strip():
            file_markers = re.findall(r"\[\[FILE:(.*?)\]\]", full_answer)
            image_markers = re.findall(r"\[\[IMAGE:(.*?)\]\]", full_answer)

            print(f"[ask_stream] Full answer preview: {full_answer[:300]}...")
            print(f"[ask_stream] Found asset references - Files: {len(file_markers)}, Images: {len(image_markers)}")
            print(f"[ask_stream] Image markers found: {image_markers}")

        # Track what we've already sent to avoid duplicates
        processed_assets = set()  # Tracks (marker, asset_type, source, size) tuples
        processed_markers = set()  # Tracks processed markers to avoid processing same marker twice
        
        # Process image references
        for marker in image_markers:
            print(f"[ask_stream] Processing image marker: {marker}")
            
            # Avoid processing the same marker multiple times
            if marker in processed_markers:
                print(f"[ask_stream] Skipping already processed marker: {marker}")
                continue
            processed_markers.add(marker)
            
            # Get assets for this document
            assets = global_index.get_document_assets(marker)
            print(f"[ask_stream] Assets retrieved: {len(assets.get('images', []))} images")
            
            # If no assets found, try the base document ID
            if not assets or (not assets.get("images") and not assets.get("files")):
                if '#' in marker:
                    base_marker = marker.split('#')[0]
                    print(f"[ask_stream] No assets found, trying base marker: {base_marker}")
                    assets = global_index.get_document_assets(base_marker)
                    print(f"[ask_stream] Base marker assets: {len(assets.get('images', []))} images")
            
            if not assets or not assets.get("images"):
                print(f"[ask_stream] No images found for marker: {marker}")
                continue
                
            # Track images attached for this marker to avoid duplicates within same marker
            marker_images_attached = set()
                
            for img_idx, img in enumerate(assets["images"]):
                content = _as_bytes(img.get("content"))
                if not content or len(content) == 0:
                    print(f"[ask_stream] Skipping empty image #{img_idx+1}: {img.get('source', 'unknown')}")
                    continue
                    
                mime = img.get("mime", "image/png")
                src = img.get("source", "embedded")
                filename = img.get("filename", f"image_{img_idx+1}")
                
                # Create unique identifier for this specific image to prevent duplicates
                image_identifier = (src, len(content), mime)
                if image_identifier in marker_images_attached:
                    print(f"[ask_stream] Skipping duplicate image (same content): {src}")
                    continue
                marker_images_attached.add(image_identifier)
                
                # Create global unique identifier to prevent cross-marker duplicates
                asset_key = (marker, "image", src, len(content))
                if asset_key in processed_assets:
                    print(f"[ask_stream] Skipping globally duplicate image: {src}")
                    continue
                processed_assets.add(asset_key)
                
                payload = {
                    "type": "image",
                    "doc_id": marker,
                    "mime": mime,
                    "size": len(content),
                    "filename": filename
                }
                
                if _is_httpish(src):
                    payload["url"] = src
                else:
                    payload["content_b64"] = base64.b64encode(content).decode("ascii")
                
                print(f"[ask_stream] Attaching image #{img_idx+1}: {filename} ({mime}, {len(content)} bytes)")
                yield json.dumps(payload) + "\n"

        # Process file references
        for marker in file_markers:
            print(f"[ask_stream] Processing file marker: {marker}")
            
            # Avoid processing the same marker multiple times
            if marker in processed_markers:
                print(f"[ask_stream] Skipping already processed file marker: {marker}")
                continue
            processed_markers.add(marker)
            
            # Get assets for this document
            assets = global_index.get_document_assets(marker)
            if not assets or not assets.get("files"):
                print(f"[ask_stream] No files found for marker: {marker}")
                continue
                
            # Track files attached for this marker to avoid duplicates within same marker
            marker_files_attached = set()
            
            for f in assets["files"]:
                content = _as_bytes(f.get("content"))
                if not content or len(content) == 0:
                    continue
                    
                # Skip very large files
                if len(content) > 5000000:  # 5MB limit
                    print(f"[ask_stream] Skipping large file: {len(content)} bytes")
                    continue
                    
                mime = f.get("mime", "application/octet-stream")
                src = f.get("source", "embedded")
                filename = f.get("filename", f"file_from_{marker.split('//')[-1].split('/')[0] if '//' in marker else 'document'}")
                
                # Create unique identifier for this specific file to prevent duplicates
                file_identifier = (src, len(content), mime)
                if file_identifier in marker_files_attached:
                    print(f"[ask_stream] Skipping duplicate file (same content): {src}")
                    continue
                marker_files_attached.add(file_identifier)
                
                # Create global unique identifier to prevent cross-marker duplicates
                asset_key = (marker, "file", src, len(content))
                if asset_key in processed_assets:
                    print(f"[ask_stream] Skipping globally duplicate file: {src}")
                    continue
                processed_assets.add(asset_key)
                
                payload = {
                    "type": "file",
                    "doc_id": marker,
                    "mime": mime,
                    "size": len(content),
                    "filename": filename
                }
                
                if _is_httpish(src):
                    payload["url"] = src
                else:
                    payload["content_b64"] = base64.b64encode(content).decode("ascii")
                
                print(f"[ask_stream] Attaching file: {filename} ({mime}, {len(content)} bytes)")
                yield json.dumps(payload) + "\n"

    return StreamingResponse(json_stream(), media_type="application/x-ndjson")