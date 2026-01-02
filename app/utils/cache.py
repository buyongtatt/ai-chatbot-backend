import base64
from typing import Dict, Any

def encode_content_for_cache(content: Dict[str, Any]) -> Dict[str, Any]:
    # Convert binary assets to base64 so JSON can store them
    enc = {
        "doc_id": content.get("doc_id"),
        "text": content.get("text", ""),
        "images": [],
        "files": [],
    }
    for img in content.get("images", []):
        enc["images"].append({
            "content_b64": base64.b64encode(img["content"]).decode("utf-8"),
            "mime": img["mime"],
            "source": img.get("source"),
        })
    for f in content.get("files", []):
        enc["files"].append({
            "content_b64": base64.b64encode(f["content"]).decode("utf-8"),
            "mime": f["mime"],
            "source": f.get("source"),
        })
    return enc

def decode_content_from_cache(content: Dict[str, Any]) -> Dict[str, Any]:
    dec = {
        "doc_id": content.get("doc_id"),
        "text": content.get("text", ""),
        "images": [],
        "files": [],
    }
    for img in content.get("images", []):
        dec["images"].append({
            "content": base64.b64decode(img["content_b64"]),
            "mime": img["mime"],
            "source": img.get("source"),
        })
    for f in content.get("files", []):
        dec["files"].append({
            "content": base64.b64decode(f["content_b64"]),
            "mime": f["mime"],
            "source": f.get("source"),
        })
    return dec
