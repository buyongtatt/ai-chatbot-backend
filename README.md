# FastAPI Backend with Ollama Vision Model

## 📖 Project Description

This backend is a FastAPI service that crawls a given root URL, indexes text, images, and files, and serves a streaming Q&A endpoint powered by Ollama. The system uses a requirements‑first workflow: crawl and store assets with explicit metadata (source, MIME type, byte size), then at ask‑time feed relevant text and one image (due to model limits) so the model reasons about content before deciding to attach assets via `doc_id` markers.

---

## ⚙️ Environment Variables

- **ROOT_URL**: Documentation root URL where the crawler starts and follows internal/subdomain links.  
  Example: `https://www.allrecipes.com/recipe/46822/indian-chicken-curry-ii/`

- **OLLAMA_MODEL**: Ollama model used for inference.  
  Example: `llama3.2-vision`

- **MAX_PAGES**: Maximum number of pages to crawl.  
  Example: `10`

- **MAX_DEPTH**: Maximum link depth from the root.  
  Example: `3`

- **REQUEST_TIMEOUT**: Timeout (in seconds) for each HTTP request.  
  Example: `15`

---

## 🛠️ Setup Instructions

1. **Prerequisites**
   - Python 3.10+
   - [Ollama](https://ollama.ai) installed and the model pulled:
     ```bash
     ollama pull llama3.2-vision
     ```
   - Recommended: virtual environment (`venv` or `conda`)

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Typical packages:
   - `fastapi` — web framework
   - `uvicorn` — ASGI server
   - `httpx` or `requests` — HTTP client
   - `beautifulsoup4` and `lxml` — HTML parsing
   - `python-dotenv` — environment variable loading
   - `pydantic` — data validation
   - `aiofiles` — async file handling

3. **Environment file**
   Create a `.env` file at the project root:
   ```env
   ROOT_URL=https://www.allrecipes.com/recipe/46822/indian-chicken-curry-ii/
   OLLAMA_MODEL=llama3.2-vision
   MAX_PAGES=10
   MAX_DEPTH=3
   REQUEST_TIMEOUT=15
   ```

---

## 🚀 Starting the Project

1. **Run the crawler**  
   Depending on your structure, either:

   ```bash
   python -m app.crawl
   ```

   or let the backend crawl on first request. Logs will show assets collected.

2. **Start the FastAPI server**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   You should see startup logs confirming environment variables.

---

## 💡 Example Usage

### Ask (streaming)

- **Endpoint:** `POST /ask_stream`
- **Form fields:**
  - `question`: string
  - `file`: optional upload (PDF/DOCX/TXT)

Example:

```bash
curl -N -X POST http://localhost:8000/ask_stream \
  -F 'question=What are the key steps to make Indian chicken curry?'
```

With an uploaded file:

```bash
curl -N -X POST http://localhost:8000/ask_stream \
  -F 'question=Summarize the instructions' \
  -F 'file=@instructions.pdf'
```

Response stream:

- `{"type":"text","content":"..."}` chunks as the answer is generated.
- If the model references assets by `doc_id`:
  - `{"type":"image","url":"https://...","mime":"image/jpeg"}`
  - `{"type":"file","url":"https://...","mime":"application/pdf"}`

---

## � Running with Concurrency Optimizations

To handle multiple concurrent users efficiently, use the dedicated startup script:

```bash
python start_server.py --workers 4 --port 8000
```

Options:

- `--workers N`: Number of worker processes (default: 4)
- `--port N`: Port to listen on (default: 8000)
- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `--reload`: Enable auto-reload on code changes

Example for high-concurrency deployment:

```bash
python start_server.py --workers 8 --port 8000
```

## 🧪 Testing Concurrent Requests

To test concurrent request handling, run the test script:

```bash
python test_concurrent_requests.py
```

This will send multiple simultaneous requests to verify that they are processed concurrently rather than sequentially.

## 🔍 Operational Notes

- **Doc_id discipline:** Prompts list available doc_ids; model instructed to reference them exactly.
- **Hybrid resolver:** Backend matches imperfect markers via substring/scheme‑stripped fallback.
- **Single image limit:** Only one image is fed to the model per request; others can still be attached if referenced.
- **Timeouts:** Stream generator closes cleanly to prevent "loading forever" after interruption.
- **Debug logs:** Byte sizes, MIME types, and doc_ids printed for validation.

---
