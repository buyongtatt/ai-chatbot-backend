
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
from app.services.ollama_client import chat_stream  # wrapper to Ollama

router = APIRouter()

def _sniff_image_bytes(data: bytes) -> Optional[bytes]:
    """Return normalized image bytes (JPEG) if data looks like an image; else None."""
    try:
        img = Image.open(io.BytesIO(data))
        # If you want to keep original format, you can `img.format`; but normalize helps latency.
        buf = io.BytesIO()
        img = img.convert("RGB")  # normalize mode
        img.save(buf, format="JPEG", quality=85)  # downscale could be added here
        return buf.getvalue()
    except Exception:
        return None

@router.post("/ask_stream")
async def ask_stream(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    uploaded_doc_id: Optional[str] = None
    uploaded_meta: Optional[Dict[str, Any]] = None

    # --- 1) Read uploaded file (if any) and decide if it's an image ---
    uploaded_img_bytes: Optional[bytes] = None
    uploaded_is_image = False

    if file:
        data = await file.read()
        fname = (file.filename or "").strip()
        ctype = (file.content_type or "").lower()

        print(f"[ask_stream] Received upload: filename={fname!r} content_type={ctype!r} bytes={len(data)}")

        # Robust image detection:
        # a) trust content_type when it starts with image/
        # b) else sniff bytes with Pillow
        if ctype.startswith("image/"):
            uploaded_is_image = True
            uploaded_img_bytes = data
        else:
            sniffed = _sniff_image_bytes(data)
            if sniffed:
                uploaded_is_image = True
                uploaded_img_bytes = sniffed  # normalized JPEG bytes
                # If client provided a non-image content_type, fix it:
                ctype = "image/jpeg"

        # Generate a safe doc_id even if filename is missing
        if not fname:
            # Assign a generated name with an image extension if we detected image
            inferred_ext = ".jpg" if uploaded_is_image else ""
            fname = f"blob{inferred_ext}"
        uploaded_doc_id = f"uploaded://img-{uuid.uuid4().hex}-{fname}"

        # Ingest for RAG (extract text, and images/files if recognized)
        text, meta = extract_any(data, fname)
        uploaded_meta = meta

        # If ingest didn't detect image (due to wrong extension), but we sniffed image, add it
        if uploaded_is_image and not meta.get("images"):
            meta["images"] = [{
                "source": fname,
                "mime": ctype or "image/jpeg",
                "content": uploaded_img_bytes,
            }]

        # Index the uploaded doc (text + images/files)
        global_index.add_document(uploaded_doc_id, {
            "doc_id": uploaded_doc_id,
            "text": meta.get("text", ""),
            "images": meta.get("images", []),
            "files": meta.get("files", []),
        })

    # --- 2) Retrieve contexts and ensure uploaded doc is included ---
    contexts: List[Dict[str, Any]] = global_index.top_k(question, 5) or []
    if uploaded_doc_id and uploaded_meta:
        if not any(c.get("doc_id") == uploaded_doc_id for c in contexts):
            contexts.append({
                "doc_id": uploaded_doc_id,
                "text": uploaded_meta.get("text", ""),
                "images": uploaded_meta.get("images", []),
                "files": uploaded_meta.get("files", []),
            })

    # --- 3) Build context text: explicitly list the uploaded image doc_id ---
    context_text = ""
    for c in contexts:
        context_text += f"[{c['doc_id']}]\n{c.get('text', '')}\n"
        if c.get("images"):
            context_text += f"Images available under: {c['doc_id']}\n"
        if c.get("files"):
            context_text += f"Files available under: {c['doc_id']}\n"

    if uploaded_doc_id and uploaded_img_bytes:
        context_text += f"\nAttached image doc_id: {uploaded_doc_id}\n"

    prompt = (
        f"Context:\n{context_text}\n\n"
        "If an image is attached, use it for visual reasoning (only one image is provided).\n"
        "When referencing images or files, use markers like [[IMAGE:doc_id]] or [[FILE:doc_id]].\n"
        "Only reference doc_ids listed in the context.\n"
        f"Question:\n{question}\n\nAnswer:"
    )

    # --- 4) Build user message with **ONE** image (raw bytes) ---
    def build_user_message(prompt: str, ctxs: List[Dict[str, Any]]) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "user", "content": prompt}
        attached = False

        # Prefer the uploaded image bytes
        if uploaded_img_bytes:
            msg["images"] = [uploaded_img_bytes]  # raw bytes for Ollama Python SDK
            attached = True

        # Fall back to the first image in contexts (requires 'content' raw bytes)
        if not attached:
            for c in ctxs:
                for img in c.get("images", []):
                    if img.get("content"):
                        msg["images"] = [img["content"]]  # raw bytes
                        attached = True
                        break
                if attached:
                    break

        print(f"[ask_stream] Image attached? {bool(msg.get('images'))}; "
              f"count={len(msg.get('images', []))}; "
              f"type={type(msg['images'][0]) if msg.get('images') else None}; "
              f"size={len(msg['images'][0]) if msg.get('images') else 0}")

        return msg

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Stream the answer progressively. "
                "If you want to reference a file or image, use markers like [[FILE:doc_id]] or [[IMAGE:doc_id]]. "
                "Only reference doc_ids listed in the context. Do not invent filenames or raw URLs. "
                "You may receive a single image; use it for visual reasoning if present."
            ),
        },
        build_user_message(prompt, contexts),
    ]

    # --- 5) Resolve markers back to context items (unchanged) ---
    def resolve_marker(marker: str, ctxs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for c in ctxs:
            if c.get("doc_id") == marker:
                return c
        for c in ctxs:
            if marker in str(c.get("doc_id", "")):
                return c
        cleaned = re.sub(r"^(docs://|uploaded://)", "", marker)
        for c in ctxs:
            clean_id = re.sub(r"^(docs://|uploaded://)", "", str(c.get("doc_id", "")))
            if cleaned in clean_id:
                return c
        return None

    # --- 6) Stream text then emit asset events ---
    def json_stream():
        full_answer = ""
        try:
            # Force the vision model here
            for chunk in chat_stream(messages, model="llama3.2-vision"):
                full_answer += chunk
                yield json.dumps({"type": "text", "content": chunk}) + "\n"
        finally:
            print("[ask_stream] Stream ended")

        # markers (unchanged)
        if full_answer.strip():
            file_markers = re.findall(r"\[\[FILE:(.*?)\]\]", full_answer)
            image_markers = re.findall(r"\[\[IMAGE:(.*?)\]\]", full_answer)

            print("[ask_stream] Available doc_ids:", [c.get("doc_id") for c in contexts])
            print("[ask_stream] File markers:", file_markers)
            print("[ask_stream] Image markers:", image_markers)

            for marker in file_markers:
                c = resolve_marker(marker, contexts)
                if c:
                    for f in c.get("files", []):
                        src = f.get("source"); mime = f.get("mime")
                        size = len(f.get("content", b"") or b"")
                        print(f"[ask_stream] Attaching file: {src} ({mime}, {size} bytes)")
                        yield json.dumps({
                            "type": "file",
                            "doc_id": c.get("doc_id"),
                            "url": src,
                            "mime": mime,
                        }) + "\n"

            for marker in image_markers:
                c = resolve_marker(marker, contexts)
                if c:
                    for img in c.get("images", []):
                        src = img.get("source"); mime = img.get("mime")
                        size = len(img.get("content", b"") or b"")
                        print(f"[ask_stream] Attaching image: {src} ({mime}, {size} bytes)")
                        yield json.dumps({
                            "type": "image",
                            "doc_id": c.get("doc_id"),
                            "url": src,
                            "mime": mime,
                        }) + "\n"

    return StreamingResponse(json_stream(), media_type="application/x-ndjson")
