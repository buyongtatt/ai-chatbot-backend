import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import Dict, Any, List, Set, Tuple
import mimetypes
from app.config.settings import settings

class CrawlResult:
    def __init__(self):
        # doc_id → content dict
        self.pages: Dict[str, Dict[str, Any]] = {}

def _registrable_domain(netloc: str) -> str:
    # naive: last two labels (example.com). For complex TLDs, consider a PSL library.
    parts = netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc

def _same_domain_or_subdomain(target: str, base_reg_domain: str) -> bool:
    # allow exact registrable domain or any subdomain ending with base_reg_domain
    return target == base_reg_domain or target.endswith("." + base_reg_domain)

def _is_http_url(u: str) -> bool:
    p = urlparse(u)
    return p.scheme in ("http", "https")

def _normalize_url(base: str, href: str) -> str:
    return urljoin(base, href)

def _fetch_page(url: str, headers: Dict[str, str]) -> Tuple[int, str]:
    r = requests.get(url, headers=headers, timeout=settings.REQUEST_TIMEOUT)
    return r.status_code, r.text

def _fetch_bytes(url: str, headers: Dict[str, str]) -> Tuple[bytes, str]:
    try:
        r = requests.get(url, headers=headers, timeout=settings.REQUEST_TIMEOUT)
        if r.status_code == 200:
            content_type = r.headers.get("Content-Type") or mimetypes.guess_type(url)[0] or "application/octet-stream"
            return r.content, content_type
    except Exception:
        print(f"Error fetching {url}: {e}") 
        return b"", "application/octet-stream"
    return b"", "application/octet-stream"

async def crawl(root_url: str) -> CrawlResult:
    result = CrawlResult()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI Assistant/1.0)"}

    root_parsed = urlparse(root_url)
    base_reg = _registrable_domain(root_parsed.netloc)

    queue = deque([(root_url, 0)])
    visited: Set[str] = set()

    while queue and len(result.pages) < settings.MAX_PAGES:
        url, depth = queue.popleft()
        if url in visited or depth > settings.MAX_DEPTH:
            continue
        visited.add(url)

        try:
            status, html = _fetch_page(url, headers)
            if status != 200:
                print(f"Skip {url}: status {status}")
                continue

            soup = BeautifulSoup(html, "html.parser")

            # Extract text (paragraphs, headings, code/pre)
            parts = []
            for tag in soup.find_all(["h1", "h2", "h3", "p", "pre", "code"]):
                text = tag.get_text(" ", strip=True)
                if text:
                    parts.append(text)
            text = "\n".join(parts)

            # Images as bytes
            images: List[Dict[str, Any]] = []
            for img in soup.find_all("img", src=True):
                src = _normalize_url(url, img["src"])
                if _is_http_url(src):
                    content, mime = _fetch_bytes(src, headers)
                    if content:
                        images.append({"content": content, "mime": mime, "source": src})

            # Files as bytes (common doc formats)
            files: List[Dict[str, Any]] = []
            for a in soup.find_all("a", href=True):
                href = _normalize_url(url, a["href"])
                lower = href.lower()
                if any(lower.endswith(ext) for ext in [".pdf", ".docx", ".pptx", ".xlsx", ".zip"]):
                    if _is_http_url(href):
                        content, mime = _fetch_bytes(href, headers)
                        if content:
                            files.append({"content": content, "mime": mime, "source": href})

            # Store page
            doc_id = f"docs://{url}"
            result.pages[doc_id] = {
                "doc_id": doc_id,
                "text": text,
                "images": images,
                "files": files,
            }
            print(f"Crawled {url}: text_parts={len(parts)}, images={len(images)}, files={len(files)}")

            # Enqueue links (internal + subdomains)
            for a in soup.find_all("a", href=True):
                link = _normalize_url(url, a["href"])
                if not _is_http_url(link):
                    continue
                parsed = urlparse(link)
                if _same_domain_or_subdomain(parsed.netloc, base_reg):
                    if link not in visited and len(result.pages) + len(queue) < settings.MAX_PAGES:
                        queue.append((link, depth + 1))

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    return result
