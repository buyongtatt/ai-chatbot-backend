
import os
import re
import mimetypes
from urllib.parse import urlparse

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _safe_filename(name: str) -> str:
    # Keep it filesystem-safe and short
    name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)
    return name[:200] if len(name) > 200 else name

def _guess_ext_from_mime(mime: str) -> str:
    if not mime:
        return ".bin"
    # Common overrides
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "application/pdf": ".pdf",
    }
    if mime.lower() in mapping:
        return mapping[mime.lower()]
    ext = mimetypes.guess_extension(mime)
    return ext or ".bin"

def _unique_path(base_dir: str, filename: str) -> str:
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(filename)
    i = 1
    while True:
        cand = os.path.join(base_dir, f"{root}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1


def save_extracted_text_and_images(
    text: str,
    images: list,
    source_url: str,
    out_root_dir: str = "output_docs"
) -> dict:
    """
    Saves:
      - text → out_root_dir/<host>/<basename>_text.txt
      - each image in images → out_root_dir/<host>/images/<filename or generated>

    `images` is a list of dicts like:
      { "content": <bytes>, "mime": "image/png", "source": "...", "filename": "foo.png" }

    Returns paths used (for logging/debug).
    """
    parsed = urlparse(source_url)
    host = _safe_filename(parsed.netloc or "unknown_host")
    # Make a stable base name from URL path (fallback to 'document')
    base_name = _safe_filename(os.path.basename(parsed.path) or "document")

    # Create dirs
    doc_dir = os.path.join(out_root_dir, host)
    img_dir = os.path.join(doc_dir, "images")
    _ensure_dir(doc_dir)
    _ensure_dir(img_dir)

    saved = {"text_path": None, "image_paths": []}

    # ---- Save text
    if text and text.strip():
        text_filename = f"{base_name}_text.txt"
        text_path = _unique_path(doc_dir, text_filename)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        saved["text_path"] = text_path

    # ---- Save images
    for idx, img in enumerate(images or []):
        content = img.get("content")
        if not content:
            continue

        # filename preference: provided filename → from source → mime-based
        filename = img.get("filename")
        if filename:
            filename = _safe_filename(os.path.basename(str(filename)))
        else:
            # Try infer from source
            src = img.get("source") or ""
            tail = _safe_filename(os.path.basename(urlparse(src).path or "")) if src else ""
            if tail:
                filename = tail
            else:
                # Fallback to synthesized name
                ext = _guess_ext_from_mime(img.get("mime"))
                filename = f"{base_name}_img_{idx}{ext}"

        # Ensure extension exists
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = _guess_ext_from_mime(img.get("mime"))
            filename = f"{root}{ext}"

        out_path = _unique_path(img_dir, filename)
        with open(out_path, "wb") as wf:
            wf.write(content)
        saved["image_paths"].append(out_path)

    return saved

def _remove_fragment(u: str) -> str:
    p = urlparse(u)
    return p._replace(fragment="").geturl()
