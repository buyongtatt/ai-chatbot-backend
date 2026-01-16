
import base64
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import Dict, Any, List, Set, Tuple, Optional
import mimetypes
import io
import re
from app.config.settings import settings
from app.services.helper import _remove_fragment

# Optional imports for better extraction
HAVE_FITZ = False
try:
    import fitz  # PyMuPDF for robust PDF text + images
    HAVE_FITZ = True
except Exception:
    HAVE_FITZ = False

HAVE_PYPDF2 = False
try:
    import PyPDF2
    HAVE_PYPDF2 = True
except Exception:
    HAVE_PYPDF2 = False

HAVE_DOCX = False
try:
    import docx  # python-docx
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

HAVE_PPTX = False
try:
    from pptx import Presentation  # python-pptx
    HAVE_PPTX = True
except Exception:
    HAVE_PPTX = False

HAVE_OPENPYXL = False
try:
    import openpyxl  # for .xlsx text extraction
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False


class CrawlResult:
    def __init__(self):
        self.pages: Dict[str, Dict[str, Any]] = {}

def _registrable_domain(netloc: str) -> str:
    parts = netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc

def _same_domain_or_subdomain(target: str, base_reg_domain: str) -> bool:
    return target == base_reg_domain or target.endswith("." + base_reg_domain)

def _is_http_url(u: str) -> bool:
    p = urlparse(u)
    return p.scheme in ("http", "https")

def _normalize_url(base: str, href: str) -> str:
    return urljoin(base, href)

def _filename_from_url(url: str) -> Optional[str]:
    path = urlparse(url).path
    if not path:
        return None
    name = path.split("/")[-1]
    return name or None

def _fetch(url: str, headers: Dict[str, str]) -> Tuple[int, str, bytes, str]:
    r = requests.get(url, headers=headers, timeout=settings.REQUEST_TIMEOUT)
    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    return r.status_code, ctype, r.content, r.text  # raw bytes + best-effort text

def _fetch_bytes(url: str, headers: Dict[str, str]) -> Tuple[bytes, str]:
    try:
        r = requests.get(url, headers=headers, timeout=settings.REQUEST_TIMEOUT)
        if r.status_code == 200:
            content_type = (r.headers.get("Content-Type") or "").split(";")[0].strip() \
                           or mimetypes.guess_type(url)[0] \
                           or "application/octet-stream"
            return r.content, content_type
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return b"", "application/octet-stream"

# ---------------- HTML Extraction ----------------

def _extract_visible_text_from_html(html: str) -> Tuple[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style
    for bad in soup(["script", "style", "noscript"]):
        bad.extract()

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Capture a wider set of elements that typically contain content
    text_parts = []
    for tag in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","blockquote","figcaption","pre","code","td","th"]):
        t = tag.get_text(" ", strip=True)
        if t:
            text_parts.append(t)

    # Fallback: if nothing captured, try body text
    if not text_parts and soup.body:
        all_text = soup.body.get_text(" ", strip=True)
        if all_text:
            text_parts.append(all_text)

    text = "\n".join(text_parts)
    # De-duplicate repeated spaces/newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, title

def _download_html_images(base_url: str, soup: BeautifulSoup, headers: Dict[str, str], downloaded: Set[str]) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for img in soup.find_all("img", src=True):
        src = _normalize_url(base_url, img["src"])

        # if src.startswith("data:image"):
        #     # Handle base64 inline image
        #     header, b64data = src.split(",", 1)
        #     mime = header.split(":")[1].split(";")[0]
        #     img_bytes = base64.b64decode(b64data)

        #     images.append({
        #         "content": img_bytes,
        #         "mime": mime,
        #         "source": "inline",
        #         "filename": f"inline_image.{mime.split('/')[-1]}"
        #     })
        #     continue
                
        if not _is_http_url(src):
            continue
        if src in downloaded:
            continue
        content, mime = _fetch_bytes(src, headers)
        if content:
            images.append({
                "content": content,
                "mime": mime,
                "source": src,
                "filename": _filename_from_url(src)
            })
            downloaded.add(src)
    return images

def _discover_and_download_linked_files(base_url: str, soup: BeautifulSoup, headers: Dict[str, str], downloaded: Set[str]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    exts = [".pdf", ".docx", ".pptx", ".xlsx", ".zip"]
    for a in soup.find_all("a", href=True):
        href = _normalize_url(base_url, a["href"])
        lower = href.lower()
        if any(lower.endswith(ext) for ext in exts):
            if not _is_http_url(href):
                continue
            if href in downloaded:
                continue
            content, mime = _fetch_bytes(href, headers)
            if content:
                files.append({
                    "content": content,
                    "mime": mime,
                    "source": href,
                    "filename": _filename_from_url(href)
                })
                downloaded.add(href)
    return files

# ---------------- PDF Extraction ----------------

def _extract_pdf_with_pymupdf(raw: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    images: List[Dict[str, Any]] = []

    with fitz.open(stream=raw, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # Text
            text_parts.append(page.get_text("text"))  # layout-aware; could use "blocks" if needed
            
            # Images
            for img in page.get_images(full=True):
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                img_ext = base.get("ext", "png").lower()
                mime = f"image/{'jpeg' if img_ext in ('jpg','jpeg') else img_ext}"
                images.append({
                    "content": img_bytes,
                    "mime": mime,
                    "source": f"embedded:page{page_index+1}:xref{xref}",
                    "filename": f"page{page_index+1}_img{xref}.{img_ext}"
                })

    # Clean text a bit
    text = "\n".join(p.strip() for p in text_parts if p and p.strip())
    return text, images

def _extract_pdf_with_pypdf2(raw: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    # Fallback: text only; PyPDF2 does not reliably extract images
    text_parts: List[str] = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t.strip())
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
    text = "\n".join(text_parts)
    return text, []  # no images in fallback

def _extract_pdf(raw: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    if HAVE_FITZ:
        return _extract_pdf_with_pymupdf(raw)
    elif HAVE_PYPDF2:
        return _extract_pdf_with_pypdf2(raw)
    else:
        # No extractor available
        return "", []

# ---------------- DOCX / PPTX / XLSX (Optional) ----------------

def _extract_docx(raw: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    if not HAVE_DOCX:
        return "", []
    text_parts: List[str] = []
    images: List[Dict[str, Any]] = []
    try:
        doc = docx.Document(io.BytesIO(raw))
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                text_parts.append(p.text.strip())
        # Extract images via package parts
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_part = rel.target_part
                img_bytes = image_part.blob
                # Guess mime from content_type
                mime = getattr(image_part, "content_type", "image/png")
                images.append({
                    "content": img_bytes,
                    "mime": mime,
                    "source": "embedded:docx",
                    "filename": getattr(image_part, "partname", None)
                })
    except Exception as e:
        print(f"DOCX extraction failed: {e}")
    return "\n".join(text_parts), images

def _extract_pptx(raw: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    if not HAVE_PPTX:
        return "", []
    text_parts: List[str] = []
    images: List[Dict[str, Any]] = []
    try:
        prs = Presentation(io.BytesIO(raw))
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text and shape.text.strip():
                    text_parts.append(shape.text.strip())
                # Pictures
                if shape.shape_type == 13 and hasattr(shape, "image"):  # MSO_SHAPE_TYPE.PICTURE
                    image = shape.image
                    img_bytes = image.blob
                    mime = image.content_type or "image/png"
                    images.append({
                        "content": img_bytes,
                        "mime": mime,
                        "source": "embedded:pptx",
                        "filename": getattr(image, "filename", None)
                    })
    except Exception as e:
        print(f"PPTX extraction failed: {e}")
    return "\n".join(text_parts), images

def _extract_xlsx(raw: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    if not HAVE_OPENPYXL:
        return "", []
    text_parts: List[str] = []
    images: List[Dict[str, Any]] = []
    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True)
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                # Join cell values per row
                cells = [str(c) for c in row if c is not None]
                if cells:
                    text_parts.append(" | ".join(cells))
        # Images are not straightforward in openpyxl; skip or implement if needed.
    except Exception as e:
        print(f"XLSX extraction failed: {e}")
    return "\n".join(text_parts), images

# ---------------- Router ----------------

def _extract_from_binary(raw: bytes, mime: str, url: str) -> Tuple[str, List[Dict[str, Any]]]:
    mime = (mime or "").lower()
    url_l = url.lower()

    if "pdf" in mime or url_l.endswith(".pdf"):
        return _extract_pdf(raw)
    if "officedocument.wordprocessingml.document" in mime or url_l.endswith(".docx"):
        return _extract_docx(raw)
    if "officedocument.presentationml.presentation" in mime or url_l.endswith(".pptx"):
        return _extract_pptx(raw)
    if "officedocument.spreadsheetml.sheet" in mime or url_l.endswith(".xlsx"):
        return _extract_xlsx(raw)

    # Unsupported binary: no text/images extraction
    return "", []

# ---------------- Main crawl ----------------

async def crawl(root_url: str) -> CrawlResult:
    result = CrawlResult()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI Assistant/1.0)"}

    root_parsed = urlparse(root_url)
    base_reg = _registrable_domain(root_parsed.netloc)

    queue = deque([(root_url, 0)])
    visited: Set[str] = set()
    downloaded_binary_sources: Set[str] = set()  # de-dup images/files

    while queue and len(result.pages) < settings.MAX_PAGES:
        url, depth = queue.popleft()
        if url in visited or depth > settings.MAX_DEPTH:
            continue
        visited.add(url)

        try:
            status, content_type, raw, text_guess = _fetch(url, headers)
            if status != 200:
                print(f"Skip {url}: status {status}")
                continue

            doc_id = f"docs://{url}"

            # Non-HTML → treat as a document (e.g., PDF) and extract text + images
            if content_type and content_type != "text/html":
                text, extracted_images = _extract_from_binary(raw, content_type, url)
                result.pages[doc_id] = {
                    "doc_id": doc_id,
                    "source_url": url,
                    "content_type": content_type,
                    "text": text,
                    "images": extracted_images,
                    "files": [{
                        "content": raw,
                        "mime": content_type or (mimetypes.guess_type(url)[0] or "application/octet-stream"),
                        "source": url,
                        "filename": _filename_from_url(url),
                    }],
                    "meta": {"title": None, "status": status}
                }
                print(f"Fetched binary {url}: type={content_type}, text_len={len(text)}, images={len(extracted_images)}")
                # Typically do not enqueue from binaries
                continue

            # HTML flow
            soup = BeautifulSoup(text_guess, "html.parser")
            text, title = _extract_visible_text_from_html(text_guess)

            # Download HTML images
            images = _download_html_images(url, soup, headers, downloaded_binary_sources)
            
            # Download linked documents (and keep original binaries in `files`)
            files = _discover_and_download_linked_files(url, soup, headers, downloaded_binary_sources)

            # For any linked documents we just downloaded, attempt extraction too (optional inline expansion)
            # If you prefer to store them as separate doc_ids, move this out and push to the queue.
            for f in files:
                f_text, f_imgs = _extract_from_binary(f["content"], f["mime"], f["source"])
                # Append extracted elements to this page’s content
                if f_text:
                    text += ("\n\n" + f_text)
                images.extend(f_imgs)

            result.pages[doc_id] = {
                "doc_id": doc_id,
                "source_url": url,
                "content_type": content_type or "text/html",
                "text": text,
                "images": images,
                "files": files,  # original binaries captured
                "meta": {"title": title, "status": status}
            }
            print(f"Crawled {url}: text_len={len(text)}, images={len(images)}, files={len(files)}")

            # Enqueue internal links (same domain or subdomain)
            for a in soup.find_all("a", href=True):
                link = _normalize_url(url, a["href"])
                link = _remove_fragment(link)
                if not _is_http_url(link):
                    continue
                parsed = urlparse(link)
                if _same_domain_or_subdomain(parsed.netloc, base_reg):
                    if link not in visited and len(result.pages) + len(queue) < settings.MAX_PAGES:
                        queue.append((link, depth + 1))

        except Exception as e:
            print(f"Error crawling {url}: {e}")
    
    return result
