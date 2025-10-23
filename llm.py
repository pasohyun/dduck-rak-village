# llm.py
import os, requests

BACKEND = os.getenv("LLM_BACKEND", "llama.cpp")  # "ollama" or "llama.cpp"

if BACKEND == "ollama":
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")   # ex) llama3, mistral, qwen2.5:7b-instruct
    OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")

    def generate(messages, temperature=0.2, top_p=0.9, max_tokens=768):
        # messages: [{"role":"system"/"user"/"assistant","content":"..."}]
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
else:
    # 기존 llama.cpp 경로 (그대로 유지)
    from llama_cpp import Llama
    from config import settings
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def load_llm():
        return Llama(
            model_path=settings.llama_model_path,
            n_ctx=settings.llama_ctx,
            n_gpu_layers=settings.llama_n_gpu_layers,
            n_threads=settings.llama_threads,
            verbose=False,
        )

    def generate(messages, temperature=None, top_p=None, max_tokens=768):
        llm = load_llm()
        out = llm.create_chat_completion(
            messages=messages,
            temperature=temperature if temperature is not None else 0.2,
            top_p=top_p if top_p is not None else 0.9,
            max_tokens=max_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()
