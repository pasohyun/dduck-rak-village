from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from config import settings
from ingest import embed_texts

class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.col = self.client.get_collection(settings.collection_name)
        self.emb_model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)

    def search(self, query: str, filters: Dict[str, Any] | None = None, top_k: int | None = None):
        top_k = top_k or settings.top_k
        q_emb = embed_texts(self.emb_model, [query], is_query=True)[0]
        res = self.col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
            where=filters or {}
        )
        # Normalize output
        hits = []
        for i in range(len(res.get("ids", [[]])[0])):
            hits.append({
                "id": res["ids"][0][i],
                "doc": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "score": res["distances"][0][i] if "distances" in res else None,
            })
        return hits
