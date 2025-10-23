from pydantic import BaseModel
from typing import Optional
import os

class Settings(BaseModel):
    # === Embeddings ===
    # E5 계열은 instruction prefix 필요: query는 "query: ", 문서는 "passage: " 추천
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cuda" 권장

    # === Vector DB ===
    chroma_persist_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "sme_policies")

    # === LLM (llama.cpp) ===
    llama_model_path: str = os.getenv("LLAMA_MODEL", "./models/llama-3.1-8b-instruct.Q4_K_M.gguf")
    llama_ctx: int = int(os.getenv("LLAMA_CTX", 8192))
    llama_n_gpu_layers: int = int(os.getenv("LLAMA_N_GPU_LAYERS", 35))  # 0=CPU only
    llama_temperature: float = float(os.getenv("LLAMA_TEMP", 0.2))
    llama_top_p: float = float(os.getenv("LLAMA_TOP_P", 0.9))
    llama_threads: int = int(os.getenv("LLAMA_THREADS", max(os.cpu_count() or 4, 4)))

    # === Chunking ===
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 900))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 150))

    # === RAG ===
    top_k: int = int(os.getenv("TOP_K", 8))
    use_rerank: bool = os.getenv("USE_RERANK", "false").lower() == "true"
    reranker_model: Optional[str] = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # optional

settings = Settings()
