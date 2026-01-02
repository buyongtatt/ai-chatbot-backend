import io, zipfile, tarfile, traceback
import fitz  # PyMuPDF
import docx, pptx, openpyxl
from PIL import Image
import pytesseract

def guess_mime(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"): return "application/pdf"
    if name.endswith(".docx"): return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if name.endswith(".pptx"): return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if name.endswith(".xlsx"): return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if name.endswith(".zip"): return "application/zip"
    if name.endswith(".tar") or name.endswith(".gz"): return "application/x-tar"
    if name.endswith(".txt"): return "text/plain"
    if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"): return "image/png"
    return "application/octet-stream"

def extract_any(file_bytes: bytes, filename: str):
    meta = {"filename": filename}
    mime = guess_mime(filename)

    try:
        if mime.startswith("text/"):
            return file_bytes.decode("utf-8", errors="ignore"), meta

        if mime == "application/pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = " ".join(page.get_text() for page in doc)
            return text, meta

        if mime.endswith("wordprocessingml.document"):
            docx_file = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in docx_file.paragraphs), meta

        if mime.endswith("presentationml.presentation"):
            pres = pptx.Presentation(io.BytesIO(file_bytes))
            texts = [shape.text for slide in pres.slides for shape in slide.shapes if hasattr(shape, "text")]
            return "\n".join(texts), meta

        if mime.endswith("spreadsheetml.sheet"):
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
            rows = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    rows.append(" ".join("" if c is None else str(c) for c in row))
            return "\n".join(rows), meta

        if mime.startswith("image/"):
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img)
            return text, meta

        if mime == "application/zip":
            texts = []
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for name in z.namelist():
                    with z.open(name) as f:
                        data = f.read()
                        t, _ = extract_any(data, name)
                        texts.append(f"\n# {name}\n{t}")
            return "\n".join(texts), meta

        if mime in ["application/x-tar", "application/gzip", "application/x-gzip"]:
            texts = []
            with tarfile.open(fileobj=io.BytesIO(file_bytes), mode="r:*") as tar:
                for m in tar.getmembers():
                    if m.isfile():
                        f = tar.extractfile(m)
                        if f:
                            data = f.read()
                            t, _ = extract_any(data, m.name)
                            texts.append(f"\n# {m.name}\n{t}")
            return "\n".join(texts), meta

    except Exception as e:
        print(f"Extraction failed for {filename}: {e}")
        traceback.print_exc()
        return "", meta

    return "", meta
