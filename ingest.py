import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from config import settings
import os, math

# ---- Helpers ----
def chunk_text(text: str, size: int, overlap: int):
    if not text:
        return []
    tokens = list(text)
    chunks = []
    step = max(size - overlap, 1)
    for i in range(0, len(tokens), step):
        chunk = "".join(tokens[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks


def load_embedding():
    # E5/multilingual: add prefixes at call time
    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    return model


def embed_texts(model, texts, is_query=False):
    # E5 style prefixes for better retrieval
    prefix = "query: " if is_query else "passage: "
    inputs = [prefix + (t or "") for t in texts]
    return model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def build_collection(client):
    if settings.collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(settings.collection_name)
    return client.create_collection(settings.collection_name, metadata={"hnsw:space": "cosine"})


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    df = pd.read_csv("data/policies_sample.csv")
    df = df.fillna("")

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    col = build_collection(client)

    emb = load_embedding()

    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        base_id = row["doc_id"]
        # 정책 본문 구성 (검색 용이하도록 핵심 필드 결합)
        full = "\n\n".join([
            f"제목: {row['title']}",
            f"정책유형: {row['policy_type']}",
            f"지역: {row['region']} / 주관: {row['issuer']}",
            f"기간: {row['start_date']} ~ {row['end_date']}",
            f"자격요건: {row['eligibility_text']}",
            f"지원내용: {row['benefit_text']}",
            f"구비서류: {row['required_docs_text']}",
            f"신청: {row['apply_url']}",
            f"원문: {row['full_text']}"
        ])
        parts = chunk_text(full, settings.chunk_size, settings.chunk_overlap)
        for j, ch in enumerate(parts):
            ids.append(f"{base_id}::{j}")
            docs.append(ch)
            metas.append({
                "doc_id": base_id,
                "title": row["title"],
                "policy_type": row["policy_type"],
                "region": row["region"],
                "issuer": row["issuer"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "apply_url": row["apply_url"],
            })

    embs = embed_texts(emb, docs, is_query=False)

    # batched add (to avoid giant payloads)
    B = 512
    for i in range(0, len(ids), B):
        col.add(
            ids=ids[i:i+B],
            embeddings=embs[i:i+B].tolist(),
            documents=docs[i:i+B],
            metadatas=metas[i:i+B]
        )

    print(f"✅ Ingested chunks: {len(ids)} → collection='{settings.collection_name}' at {settings.chroma_persist_dir}")
