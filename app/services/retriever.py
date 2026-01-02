from typing import Dict, Any, List

class RetrieverIndex:
    def __init__(self):
        # doc_id → rich content object
        self.docs: Dict[str, Dict[str, Any]] = {}

    def add_document(self, doc_id: str, content: Dict[str, Any]) -> None:
        self.docs[doc_id] = content

    def top_k(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Minimal baseline: return first k. Replace with semantic search later.
        return list(self.docs.values())[:k]

global_index = RetrieverIndex()
