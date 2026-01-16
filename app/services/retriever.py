
# app/services/retriever.py
import re
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

class ChunkingRetrieverIndex:
    """
    A drop-in replacement for your current RetrieverIndex that:
      - Chunks documents on add_document
      - Ranks chunks lexically (TF-IDF-like)
      - Enforces a token budget for returned contexts
      - Includes parent images/files ONLY in the first returned chunk for that parent
    """
    def __init__(
        self,
        max_chunk_chars: int = 1200,
        overlap_chars: int = 200,
        approx_chars_per_token: int = 4,   # rule of thumb for many LLMs
        max_context_tokens: int = 3500,    # budget for context_text only
        min_chunk_chars: int = 200,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.approx_chars_per_token = approx_chars_per_token
        self.max_context_tokens = max_context_tokens
        self.min_chunk_chars = min_chunk_chars

        # Store original parents (for images/files)
        self.parents: Dict[str, Dict[str, Any]] = {}

        # Flat chunk list and indices
        self.chunks: List[Dict[str, Any]] = []  # each has: doc_id, parent_doc_id, chunk_index, text
        self.parent_to_chunk_idxs: Dict[str, List[int]] = defaultdict(list)

        # For scoring
        self.df: Counter = Counter()  # document frequency per term across chunks
        self.total_chunks: int = 0

    # -------------------
    # Utility: tokenization & token estimates
    # -------------------
    _word_re = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

    def _words(self, text: str) -> List[str]:
        return [w.lower() for w in self._word_re.findall(text or "")]

    def _estimate_tokens(self, text: str) -> int:
        # Very rough heuristic; safe for local Gemma/Ollama usage.
        # You can tune to your model if you later add a real tokenizer.
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.approx_chars_per_token))

    # -------------------
    # Chunking
    # -------------------
    def _split_paragraphs(self, text: str) -> List[str]:
        # Split on blank lines first (paragraphs)
        paras = re.split(r"\n\s*\n", text.strip())
        # Normalize whitespace in paragraphs
        paras = [re.sub(r"[ \t]+", " ", p.strip()) for p in paras if p.strip()]
        return paras

    def _build_chunks(self, text: str) -> List[str]:
        """
        Greedy paragraph packer with overlap (by characters).
        Produces chunks close to `max_chunk_chars`, with overlap across boundaries.
        """
        if not text:
            return []

        paras = self._split_paragraphs(text)
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        for p in paras:
            if not p:
                continue
            if cur_len + len(p) + 1 <= self.max_chunk_chars or cur_len < self.min_chunk_chars:
                cur.append(p)
                cur_len += len(p) + 1
            else:
                # finalize current chunk
                chunk_text = "\n\n".join(cur).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # seed next chunk with overlap tail from previous chunk
                if self.overlap_chars > 0 and chunk_text:
                    tail = chunk_text[-self.overlap_chars:]
                    cur = [tail, p]
                    cur_len = len(tail) + 1 + len(p)
                else:
                    cur = [p]
                    cur_len = len(p)

        # last one
        if cur:
            chunk_text = "\n\n".join(cur).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    # -------------------
    # Index maintenance
    # -------------------
    def _update_df(self, chunk_texts: List[str]) -> None:
        # update document frequency with set of words in each chunk
        for t in chunk_texts:
            terms = set(self._words(t))
            for w in terms:
                self.df[w] += 1
        self.total_chunks += len(chunk_texts)

    def add_document(self, doc_id: str, content: Dict[str, Any]) -> None:
        """
        Ingest a parent document. We:
          - store the parent
          - make text chunks and append to global chunk list
          - update DF statistics
        Images/files are kept ONLY on the parent to avoid duplication.
        """
        # store/overwrite parent
        self.parents[doc_id] = content

        text = (content or {}).get("text") or ""
        # if there's no text (e.g., image-only doc), still add one empty chunk to carry assets
        chunk_texts = self._build_chunks(text) or [""]

        start_idx = len(self.chunks)
        for i, t in enumerate(chunk_texts):
            chunk = {
                "doc_id": f"{doc_id}#chunk-{i}",
                "parent_doc_id": doc_id,
                "chunk_index": i,
                "text": t,
                # DO NOT inline large images/files here for every chunk
                "images": [],
                "files": [],
            }
            self.chunks.append(chunk)
            self.parent_to_chunk_idxs[doc_id].append(start_idx + i)

        # update DF
        self._update_df(chunk_texts)

    # -------------------
    # Scoring & retrieval
    # -------------------
    def _score(self, query: str, chunk_text: str) -> float:
        """
        Simple TF-IDF-like score (no external deps).
        """
        if not query or not chunk_text:
            return 0.0

        q_terms = self._words(query)
        if not q_terms:
            return 0.0

        q_tf = Counter(q_terms)

        c_terms = self._words(chunk_text)
        c_tf = Counter(c_terms)

        score = 0.0
        N = max(1, self.total_chunks)
        for w, qf in q_tf.items():
            df = self.df.get(w, 0)
            # idf with +1 smoothing
            idf = math.log((N + 1) / (df + 1)) + 1.0
            # chunk tf
            cf = c_tf.get(w, 0)
            score += (qf * cf) * idf
        return score

    def _select_under_token_budget(
        self,
        ranked_chunk_idxs: List[int],
        k: int
    ) -> List[int]:
        """
        Keep best chunks until:
          - we have k chunks OR
          - we reach max_context_tokens budget
        We estimate tokens from chunk text length.
        """
        selected: List[int] = []
        tokens_used = 0

        for idx in ranked_chunk_idxs:
            if len(selected) >= k:
                break
            t = self.chunks[idx].get("text", "")
            est = self._estimate_tokens(t)
            if tokens_used + est > self.max_context_tokens:
                # skip overly large chunk if it alone exceeds budget
                if est > self.max_context_tokens and not selected:
                    selected.append(idx)
                break
            selected.append(idx)
            tokens_used += est

        return selected

    def _merge_parent_assets_into_first_hit(self, selected_idxs: List[int]) -> List[Dict[str, Any]]:
        """
        For each parent_doc_id present in selected chunks:
          - include images/files ONLY in the FIRST selected chunk for that parent
          - others remain text-only
        This keeps your /ask_stream behavior that attaches images from contexts.
        """
        out: List[Dict[str, Any]] = []
        seen_parent: set = set()

        for idx in selected_idxs:
            ch = dict(self.chunks[idx])  # copy
            parent_id = ch.get("parent_doc_id")
            if parent_id and parent_id not in seen_parent:
                parent = self.parents.get(parent_id, {})
                # Only attach heavy assets once
                ch["images"] = parent.get("images", []) or []
                ch["files"]  = parent.get("files", []) or []
                seen_parent.add(parent_id)
            else:
                ch["images"] = []
                ch["files"]  = []
            out.append(ch)
        return out

    def top_k(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search over chunks, return up to k chunks with token budget enforced.
        Returned entries have:
          - doc_id (chunk id like docs://...#chunk-i)
          - text (chunk text)
          - images/files only on the first returned chunk per parent
        """
        if not self.chunks:
            return []

        # rank
        scores: List[Tuple[float, int]] = []
        for i, ch in enumerate(self.chunks):
            s = self._score(query, ch.get("text", ""))
            if s > 0:
                scores.append((s, i))

        # if no scores matched, fallback to recency/insertion order
        if not scores:
            ranked_idxs = list(range(min(k, len(self.chunks))))
        else:
            ranked_idxs = [i for _, i in sorted(scores, key=lambda x: x[0], reverse=True)]

        # reduce under token budget
        selected_idxs = self._select_under_token_budget(ranked_idxs, k)

        # attach parent images/files on first hit for that parent
        return self._merge_parent_assets_into_first_hit(selected_idxs)


# Keep your global symbol the same so the rest of your app doesn't change
global_index = ChunkingRetrieverIndex(
    max_chunk_chars=1200,
    overlap_chars=200,
    approx_chars_per_token=4,
    max_context_tokens=65000,  # tune this based on your model/context window
)
