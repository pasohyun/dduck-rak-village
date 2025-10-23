from typing import List, Dict

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

from config import settings

class ReRanker:
    def __init__(self):
        if settings.use_rerank and CrossEncoder is not None:
            self.model = CrossEncoder(settings.reranker_model)
        else:
            self.model = None

    def rerank(self, query: str, hits: List[Dict], top_k: int = 5):
        if not self.model:
            return hits[:top_k]
        pairs = [(query, h["doc"]) for h in hits]
        scores = self.model.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        hits.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return hits[:top_k]
