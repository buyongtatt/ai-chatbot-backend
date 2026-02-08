import base64
import re
import io
import json
import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import asyncio
from app.utils.knowledge_base_manager import kb_manager

from app.services.retriever import global_index
from app.services.ingest import extract_any
from app.services.ollama_client import chat_stream, async_chat_stream
from app.services.crawler import crawl

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
    """Build context for AI without exposing internal mechanics"""
    
    context_parts = []
    
    # Just provide the actual content without mentioning chunks or scores
    if relevant_chunks:
        context_parts.append("BACKGROUND INFORMATION:")
        context_parts.append("-" * 50)
        
        for chunk in relevant_chunks:
            content = chunk.get('text', '').strip()
            if content:
                context_parts.append(content)
        
        context_parts.append("-" * 50)
    return "\n".join(context_parts)

@router.post("/ask_stream")
async def ask_stream(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None),
    area_name: Optional[str] = Form(None)
):
    uploaded_doc_id: Optional[str] = None
    uploaded_meta: Optional[Dict[str, Any]] = None
    uploaded_img_bytes: Optional[bytes] = None
    uploaded_is_image = False

    # Handle knowledge base URL crawling
    if area_name:
        area_name = area_name.strip()
        if area_name:
            print(f"[ask_stream] Looking up knowledge base area: {area_name}")
            
            # Validate and get URL for the area
            is_valid, kb_url = kb_manager.validate_area(area_name)
            
            if not is_valid or not kb_url:
                available_areas = kb_manager.list_area_names()
                error_msg = f"Knowledge base area '{area_name}' not found. Available areas: {', '.join(available_areas)}"
                print(f"[ask_stream] {error_msg}")
                # Continue anyway - user might rely on uploaded files only
            else:
                print(f"[ask_stream] Crawling knowledge base from: {kb_url}")
                try:
                    crawl_result = await crawl(kb_url)

                    # Index all crawled documents
                    for doc_id, content in crawl_result.pages.items():
                        global_index.add_document(doc_id, content)
                        print(f"[ask_stream] Indexed document: {doc_id}")

                    print(f"[ask_stream] Successfully crawled and indexed {len(crawl_result.pages)} documents")
                except Exception as e:
                    print(f"[ask_stream] Error crawling knowledge base: {e}")
                    import traceback
                    traceback.print_exc()

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

    # If user uploaded content, include extracted text in context
    if uploaded_doc_id:
        try:
            # Include the extracted text from uploaded file/image directly in context
            if uploaded_meta:
                extracted_text = uploaded_meta.get("text", "").strip()
                fname = uploaded_meta.get("filename", "uploaded file")
                mime = uploaded_meta.get("mime", "unknown")
                
                if extracted_text:
                    context_text += f"\n\n{'#'*70}\n"
                    context_text += f"# ⭐ UPLOADED CONTENT (USER-PROVIDED) ⭐\n"
                    context_text += f"# File: {fname} | Type: {mime}\n"
                    context_text += f"{'#'*70}\n"
                    context_text += f"{extracted_text}\n"
                    context_text += f"{'#'*70}\n"
                    print(f"[ask_stream] Added extracted content from {fname} ({len(extracted_text)} chars)")
                else:
                    # Even if no text extracted, note that the file exists
                    context_text += f"\n\n{'#'*70}\n"
                    context_text += f"# ⭐ UPLOADED CONTENT (USER-PROVIDED) ⭐\n"
                    context_text += f"# File: {fname} | Type: {mime}\n"
                    context_text += f"# Content is available for analysis\n"
                    context_text += f"{'#'*70}\n"
            
            # Also make images/files available as references
            assets = global_index.get_document_assets(uploaded_doc_id)
            if assets:
                if uploaded_is_image and assets.get('images'):
                    fname = uploaded_meta.get("filename", "image") if uploaded_meta else "uploaded image"
                    context_text += f"\n[Image available as [[IMAGE:{uploaded_doc_id}]] - {fname}]\n"
                elif assets.get('files'):
                    fname = uploaded_meta.get("filename", "file") if uploaded_meta else "uploaded file"
                    context_text += f"\n[File available as [[FILE:{uploaded_doc_id}]] - {fname}]\n"
            
            relevant_doc_ids.append(uploaded_doc_id)
        except Exception as e:
            print(f"[ask_stream] Error processing uploaded content: {e}")

    prompt = f"""{context_text}

{question}"""

    def build_user_message(prompt: str) -> Dict[str, Any]:
        return {"role": "user", "content": prompt}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant answering questions based on provided information.\n\n"
                "KEY INSTRUCTIONS:\n"
                "- If user uploaded content, analyze it thoroughly and use it in your answer\n"
                "- Provide natural, conversational responses without mentioning how you process information\n"
                "- Answer only based on the information available\n"
                "- If you can reference images or files that help explain your answer, use: [[IMAGE:id]] or [[FILE:id]]\n"
                "- Be concise, helpful, and friendly\n"
                "- Do NOT mention chunks, relevance scores, or technical processing details"
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

    async def async_json_stream():
        full_answer = ""
        try:
            # Use async_chat_stream for non-blocking processing
            async for chunk in async_chat_stream(messages):
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

    return StreamingResponse(async_json_stream(), media_type="application/x-ndjson")


    @router.get("/debug/knowledge_bases")
    async def debug_knowledge_bases():
        """List all available knowledge base areas"""
        all_areas = kb_manager.get_all_areas()
    
        areas_info = []
        for kb in all_areas:
            areas_info.append({
                "area_name": kb.get("area_name"),
                "display_name": kb.get("display_name"),
                "url": kb.get("url"),
                "description": kb.get("description", "")
            })
    
        return {
            "total": len(areas_info),
            "knowledge_bases": areas_info
        }