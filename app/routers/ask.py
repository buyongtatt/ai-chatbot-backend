import re
import json
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from app.services.retriever import global_index

from app.services.ingest import extract_any
from app.services.ollama_client import chat_stream

router = APIRouter()

@router.post("/ask_stream")
async def ask_stream(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    file_text = ""
    uploaded_doc_id = None
    if file:
        data = await file.read()
        file_text, meta = extract_any(data, file.filename)
        uploaded_doc_id = f"uploaded://{file.filename}"
        if file_text.strip():
            global_index.add_document(uploaded_doc_id, {
                "doc_id": uploaded_doc_id,
                "text": file_text,
                "images": [],
                "files": []
            })

    contexts: List[Dict[str, Any]] = global_index.top_k(question, 5)

    context_text = ""
    for c in contexts:
        context_text += f"[{c['doc_id']}]\n{c.get('text','')}\n"
        if c.get("images"):
            context_text += f"Images available: {c['doc_id']}\n"
        if c.get("files"):
            context_text += f"Files available: {c['doc_id']}\n"

    prompt = f"Context:\n{context_text}\n\n"
    if file_text:
        prompt += f"Uploaded file content:\n{file_text}\n\n"
    prompt += f"Question:\n{question}\n\nAnswer:"

    # Build user message with content + vision inputs
    def build_user_message(prompt: str, contexts: List[Dict[str, Any]]): 
        msg = {"role": "user", "content": prompt} 
        images = [] 
        for c in contexts: 
            for img in c.get("images", []): 
                if "content" in img: # raw bytes stored during crawl 
                    images.append(img["content"]) 
                    break
            if "images" in msg: 
                break
            
        file_texts = [] 
        for c in contexts: 
            for f in c.get("files", []): 
                if "text" in f: 
                    file_texts.append(f["text"]) 
        if file_texts: 
            msg["content"] += "\n\nAttached file contents:\n" + "\n".join(file_texts) 
            
        return msg 
    
    messages = [ 
        {"role": "system", "content": ( 
            "You are a helpful assistant. Stream the answer progressively. " 
            "If you want to reference a file or image, use markers like [[FILE:doc_id]] or [[IMAGE:doc_id]]. " 
            "Only reference doc_ids listed in the context. Do not invent filenames or raw URLs." 
        )}, 
        build_user_message(prompt, contexts) 
    ]

    def resolve_marker(marker: str, contexts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]: 
        # 1. Exact match 
        for c in contexts: 
            if c["doc_id"] == marker: 
                return c 
        # 2. Substring match 
        for c in contexts: 
            if marker in c["doc_id"]: 
                return c 
        # 3. Strip docs:// scheme and compare 
        for c in contexts: 
            clean_id = re.sub(r"^docs://", "", c["doc_id"]) 
            if marker in clean_id: 
                return c 
        return None

    def json_stream():
        full_answer = ""
        try:
            # 1. Stream text chunks progressively
            for chunk in chat_stream(messages):
                full_answer += chunk
                yield json.dumps({"type": "text", "content": chunk}) + "\n"
        finally: 
            print("[ask_stream] Stream ended")

        # 2. Parse markers in the final answer
        if full_answer.strip():
            file_markers = re.findall(r"\[\[FILE:(.*?)\]\]", full_answer)
            image_markers = re.findall(r"\[\[IMAGE:(.*?)\]\]", full_answer)

            print("[ask_stream] Available doc_ids:", [c["doc_id"] for c in contexts])
            print("[ask_stream] File markers:", file_markers)
            print("[ask_stream] Image markers:", image_markers)

            # 3. Attach referenced files 
            for marker in file_markers: 
                c = resolve_marker(marker, contexts) 
                if c: 
                    for f in c.get("files", []): 
                        print(f"[ask_stream] Attaching file: {f['source']} ({f['mime']}, {len(f['content'])} bytes)") 
                        yield json.dumps({ 
                            "type": "file", 
                            "url": f["source"], 
                            "mime": f["mime"] 
                        }) + "\n" 
                else: 
                    print(f"[ask_stream] No file match for marker: {marker}") 
                
            # 4. Attach referenced images 
            for marker in image_markers: 
                c = resolve_marker(marker, contexts) 
                if c: 
                    for img in c.get("images", []): 
                        print(f"[ask_stream] Attaching image: {img['source']} ({img['mime']}, {len(img['content'])} bytes)") 
                        yield json.dumps({ 
                            "type": "image", 
                            "url": img["source"], 
                            "mime": img["mime"] 
                        }) + "\n" 
                else: 
                    print(f"[ask_stream] No image match for marker: {marker}")


    return StreamingResponse(json_stream(), media_type="application/json")
