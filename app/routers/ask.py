
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
        meta = dict(meta or {})
        meta.setdefault("text", text or "")
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
            "text": meta.get("text"),
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
        "You may receive **multiple images** attached through the message's `images` field.\n"
        "Use all provided images for visual reasoning.\n"
        "If the question involves visual content, analyze the image(s) directly.\n\n"
        "When referencing images or files in your answer, use markers like [[IMAGE:doc_id]] or [[FILE:doc_id]].\n"
        "Only reference doc_ids that are present in the context.\n\n"
        "If no images are actually attached, say so explicitly.\n"
        "If images are attached, do NOT deny their existence.\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    
    def _as_bytes(x) -> Optional[bytes]:
        """Return bytes from raw bytes or base64 str; otherwise None."""
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            return bytes(x)
        if isinstance(x, str):
            try:
                # Allow non-strict (some clients add whitespace)
                return base64.b64decode(x, validate=False)
            except Exception:
                return None
        return None

    def _gather_all_context_images(ctxs: List[Dict[str, Any]]) -> List[bytes]:
        out: List[bytes] = []
        for c in ctxs or []:
            for img in c.get("images", []):
                b = _as_bytes(img.get("content"))
                if b:
                    out.append(b)
        return out

    def build_user_message(prompt: str,
                        ctxs: List[Dict[str, Any]],
                        include_uploaded_first: bool = True,
                        max_images: int = 8,
                        min_bytes: int = 2_000) -> Dict[str, Any]:
        """
        - include_uploaded_first=True: put uploaded image first if present
        - max_images: hard cap to prevent OOM
        - min_bytes: skip tiny icons/thumbnails
        """
        msg: Dict[str, Any] = {"role": "user", "content": prompt}
        imgs: List[bytes] = []

        # 1) Include uploaded image (if any)
        if include_uploaded_first and uploaded_img_bytes:
            if isinstance(uploaded_img_bytes, (bytes, bytearray)) and len(uploaded_img_bytes) >= min_bytes:
                imgs.append(bytes(uploaded_img_bytes))

        # 2) Add ALL images from contexts
        ctx_imgs = _gather_all_context_images(ctxs)
        for b in ctx_imgs:
            # if len(imgs) >= max_images:
            #     break
            if b and len(b) >= min_bytes:
                imgs.append(b)

        # 3) Attach if any found
        if imgs:
            msg["images"] = imgs

        # Logging
        total_bytes = sum(len(b) for b in imgs) if imgs else 0
        print("[ask_stream]"
            f"count={len(imgs)}; total_bytes={total_bytes}; ")

        return msg

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, grounded assistant. Stream your answer progressively.\n\n"
                "Context & assets:\n"
                "- You will receive a text context that lists one or more doc_ids.\n"
                "- Some doc_ids include FILES and/or IMAGES that the user may have uploaded or were retrieved from the knowledge base.\n"
                "- Only refer to assets using markers [[FILE:doc_id]] and [[IMAGE:doc_id]]. Do not invent doc_ids, filenames, or URLs.\n\n"
                "Instructions:\n"
                "1) First, read and synthesize the provided context text. If images are attached, assume they relate to the context unless stated otherwise.\n"
                "2) If one or more images are attached, incorporate visual reasoning:\n"
                "   - Describe only what is visible; avoid assumptions beyond the image content.\n"
                "   - If there are multiple images, compare/contrast them when relevant.\n"
                "3) When citing or pointing users to assets, use [[FILE:doc_id]] or [[IMAGE:doc_id]] from the context only.\n"
                "4) Ground your answer in the context text (and images if present). If the answer is not in the provided context and cannot be reliably inferred from the image(s), say so and propose next steps.\n"
                "5) Keep the answer concise, structured, and directly responsive to the user's question. If a quick step-by-step helps, use it.\n"
                "6) Ask a brief clarifying question only if the user’s request is ambiguous and clarification is necessary to proceed.\n"
                "7) Do not output raw base64 or any links other than markers. Do not reveal internal reasoning.\n"
            ),
        },
        build_user_message(prompt, contexts),  # your function that now attaches ALL images
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

    
    def _is_httpish(u: str) -> bool:
        return isinstance(u, str) and u.startswith(("http://", "https://"))

    def json_stream():
        full_answer = ""
        try:
            # Force the vision model here
            for chunk in chat_stream(messages):
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
            if not c:
                continue

            for i, f in enumerate(c.get("files", []) or []):
                src  = f.get("source")
                mime = f.get("mime") or "application/octet-stream"
                buf  = f.get("content") or b""
                size = len(buf)
                filename = f.get("filename") or "file.bin"

                payload = {
                    "type": "file",
                    "doc_id": c.get("doc_id"),
                    "mime": mime,
                    "filename": filename,
                    "size": size,
                }

                if _is_httpish(src):
                    # Good: client can download directly
                    payload["url"] = src
                else:
                    payload["content_b64"] = base64.b64encode(buf).decode("ascii")

                print(f"[ask_stream] Attaching file: {src} ({mime}, {size} bytes)")
                yield json.dumps(payload) + "\n"


            
            for marker in image_markers:
                c = resolve_marker(marker, contexts)
                if c:
                    for idx, img in enumerate(c.get("images", [])):
                        src = img.get("source"); mime = img.get("mime")
                        content = img.get("content") or b""
                        size = len(content)
                        payload = {
                            "type": "image",
                            "doc_id": c.get("doc_id"),
                            "mime": mime,
                            "size": size,
                        }
                        if _is_httpish(src):
                            payload["url"] = src
                        else:
                            # Attach base64 so the client can render or save
                            payload["content_b64"] = base64.b64encode(content).decode("ascii")
                            # optional filename fallback
                            if img.get("filename"):
                                payload["filename"] = str(img["filename"])
                        print(f"[ask_stream] Attaching image: {src} ({mime}, {size} bytes)")
                        yield json.dumps(payload) + "\n"


    return StreamingResponse(json_stream(), media_type="application/x-ndjson")
