import gc
import os
import re
import time

import psutil

# -------------------------------------------------------
# ⚙️ CONFIGURATION — Optimized for Render (2 GB)
# -------------------------------------------------------
MODEL_NAME = os.getenv("SUMMARIZATION_MODEL_NAME", "sshleifer/distilbart-cnn-6-6")
MAX_TOKENS_PER_CHUNK = 250
OVERLAP = 15
PER_CHUNK_MAX_NEW_TOKENS = 50
REDUCE_MAX_NEW_TOKENS = 80
MAX_WORKERS = 1

LOG_PREFIX = "[SUMMARY]"
_tokenizer_cache, _summarizer_cache = {}, {}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------------------------------------
# 🧠 Memory utility
# -------------------------------------------------------
def _mem(label=""):
    """Print current memory usage (MB)."""
    try:
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(f"[MEMORY] {label}: {rss:.1f} MB")
    except Exception:
        pass


# -------------------------------------------------------
# 🧹 Text Cleaning
# -------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"(TextColor|Filename|escription|▬|■|□|▲|▶|►|▪|●|–|—|‐|-)", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:!?%'\-\s]", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


# -------------------------------------------------------
# 🔤 Model + Tokenizer (loaded once)
# -------------------------------------------------------
def _get_tokenizer():
    from transformers import AutoTokenizer

    if MODEL_NAME not in _tokenizer_cache:
        _tokenizer_cache[MODEL_NAME] = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    return _tokenizer_cache[MODEL_NAME]


def _get_summarizer():
    import torch
    from transformers import pipeline

    if MODEL_NAME not in _summarizer_cache:
        torch.set_num_threads(1)
        summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=_get_tokenizer(),
            device=-1,  # CPU only
            framework="pt",
        )
        _summarizer_cache[MODEL_NAME] = summarizer
        print(f"{LOG_PREFIX} 🚀 Loaded summarizer: {MODEL_NAME} (device=-1)")
        _mem("after model load")
    return _summarizer_cache[MODEL_NAME]


# -------------------------------------------------------
# 🧩 Token-safe chunking
# -------------------------------------------------------
def _chunk_text_token_safe(text, max_tokens=MAX_TOKENS_PER_CHUNK):
    tok = _get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    stride = max_tokens - OVERLAP
    return [
        tok.decode(ids[i:i + max_tokens], skip_special_tokens=True)
        for i in range(0, len(ids), stride)
    ]


# -------------------------------------------------------
# ⚡ Main Summarization
# -------------------------------------------------------
async def summarize_text(text: str, max_tokens: int = 512, summary_type: str = "medium") -> dict:
    if not text.strip():
        raise ValueError("Empty text cannot be summarized.")

    start = time.time()
    cleaned = clean_text(text)

    # Safety: truncate huge documents
    if len(cleaned) > 180_000:
        cleaned = cleaned[:180_000]

    if len(cleaned.split()) < 120:
        return {
            "summary": cleaned[:800] + "...",
            "chunks": 1,
            "successful_chunks": 1,
            "duration_ms": 0,
            "model_name": MODEL_NAME,
            "cached": True,
        }

    # Adjust summarization intensity
    if summary_type == "detailed":
        max_new_tokens = REDUCE_MAX_NEW_TOKENS
    elif summary_type == "short":
        max_new_tokens = int(PER_CHUNK_MAX_NEW_TOKENS * 0.7)
    else:
        max_new_tokens = PER_CHUNK_MAX_NEW_TOKENS

    summarizer = _get_summarizer()
    tokenizer = _get_tokenizer()
    chunks = _chunk_text_token_safe(cleaned, MAX_TOKENS_PER_CHUNK)

    if len(chunks) > 8:
        print(f"{LOG_PREFIX} ⚠️ Too many chunks ({len(chunks)}). Truncating to 8.")
        chunks = chunks[:8]

    print(f"{LOG_PREFIX} 🕒 Start summarization ({summary_type}) with {len(chunks)} chunks")
    _mem("before summarize")

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        try:
            token_len = len(tokenizer.encode(chunk, add_special_tokens=False))
            print(f"{LOG_PREFIX} Summarizing chunk {i}/{len(chunks)} ({token_len} tokens)")
            out = summarizer(
                chunk,
                max_new_tokens=max_new_tokens,
                min_length=int(max_new_tokens * 0.4),
                do_sample=False,
            )[0]["summary_text"].strip()
            summaries.append(out)

            if i % 2 == 0:
                gc.collect()
        except Exception as e:
            print(f"{LOG_PREFIX} ❌ Chunk {i} failed: {e}")

    combined = " ".join(summaries).strip()

    # Optional refinement for long outputs
    if len(summaries) > 1 and len(combined.split()) > 400:
        try:
            print(f"{LOG_PREFIX} 🔁 Refining final summary...")
            combined = summarizer(
                combined,
                max_new_tokens=max_new_tokens,
                min_length=int(max_new_tokens * 0.5),
                do_sample=False,
            )[0]["summary_text"].strip()
        except Exception as e:
            print(f"{LOG_PREFIX} ⚠️ Refinement failed: {e}")

    duration_ms = int((time.time() - start) * 1000)
    _mem("after summarize")
    gc.collect()

    return {
        "summary": combined,
        "chunks": len(chunks),
        "successful_chunks": len(summaries),
        "duration_ms": duration_ms,
        "model_name": MODEL_NAME,
        "max_new_tokens": max_new_tokens,
        "cached": True,
    }
