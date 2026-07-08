"""Shared RAG engine for Baseline and Engineered systems.

One code path, driven by config.RAG_PRESETS, so the only thing that differs
between the two systems is the preset -- never the logic. Phase 2 exercises
the "baseline" preset; Phase 3 turns on the "engineered" levers.

  build_index(preset)          -> embed chunks (cached) into a Chroma collection
  retrieve(question, preset)   -> top-K chunks (optionally Year/Month filtered)
  generate(question, ctx, ...) -> Claude answer over the retrieved context
  answer(question, preset)     -> {answer, retrieved}

CLI:
  python rag.py build baseline
  python rag.py ask   baseline "What was the federal deficit in fiscal 2014?"
"""
import json
import re
import sys
from functools import lru_cache

import numpy as np

import config

# Pin threads so embeddings are reproducible across processes/runs. MiniLM's
# multi-threaded float32 reductions vary slightly run-to-run, which can flip
# borderline top-5 retrievals (this caused a 52.5% vs 27.5% discrepancy).
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# --------------------------------------------------------------------------
# Embedding model (local, no API key)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Encode texts to L2-normalized vectors (cosine == dot product), robustly.

    MiniLM on this CPU maps some degenerate table-markup chunks (e.g. rows of
    '| nan | --- |') to ~zero raw vectors; normalizing those yields NaN, which
    Chroma rejects. So we normalize MANUALLY: NaN-scrub the raw output, and give
    any zero-norm row a fixed placeholder direction (it simply won't match
    queries). Combined with pinned threads this is finite AND reproducible."""
    m = embedder()
    v = m.encode(texts, batch_size=batch_size, normalize_embeddings=False,
                 show_progress_bar=len(texts) > 2000, convert_to_numpy=True)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norms = np.linalg.norm(v, axis=1)
    dead = norms < 1e-8
    if dead.any():                      # placeholder unit vector for dead rows
        v[dead] = 0.0
        v[dead, 0] = 1.0
        norms[dead] = 1.0
    return v / norms[:, None]


def load_chunks(preset_name: str) -> list[dict]:
    path = config.OUT_DIR / config.RAG_PRESETS[preset_name]["chunks"]
    # drop empty/whitespace-only chunks: they embed to a zero vector, which
    # normalization turns into NaN (and Chroma rejects).
    return [c for c in (json.loads(line) for line in open(path)) if c["text"].strip()]


# --------------------------------------------------------------------------
# Index build (Chroma, cosine). Embeddings cached to .npy so a rebuild is free.
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _client():
    import chromadb
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _cached_embeddings(preset_name: str, chunks: list[dict]) -> np.ndarray:
    config.EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.EMB_CACHE_DIR / f"{preset_name}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(chunks) and np.isfinite(arr).all():
            return arr
    arr = embed([c["text"] for c in chunks])
    np.save(cache, arr)
    return arr


def build_index(preset_name: str, rebuild: bool = False):
    chunks = load_chunks(preset_name)
    client = _client()
    if rebuild:
        try:
            client.delete_collection(preset_name)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        preset_name, metadata={"hnsw:space": "cosine"}
    )
    if coll.count() == len(chunks):
        print(f"[{preset_name}] index already built ({coll.count()} chunks)")
        return coll

    print(f"[{preset_name}] embedding {len(chunks)} chunks ...")
    vecs = _cached_embeddings(preset_name, chunks)
    print(f"[{preset_name}] adding to Chroma ...")
    B = 4000
    for i in range(0, len(chunks), B):
        part = chunks[i : i + B]
        coll.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=vecs[i : i + B].tolist(),
            documents=[c["text"] for c in part],
            metadatas=[
                {"source_file": c["source_file"], "year": c["year"], "month": c["month"]}
                for c in part
            ],
        )
    print(f"[{preset_name}] done -> {coll.count()} chunks")
    return coll


# --------------------------------------------------------------------------
# Year/Month detection (used by the engineered metadata filter in Phase 3)
# --------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def detect_years(question: str) -> list[int]:
    yrs = {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)
           if config.START_YEAR <= int(y) <= config.END_YEAR}
    return sorted(yrs)


def build_where(question: str, preset_name: str):
    """Soft filter: only constrain by year when years are detected AND lie in
    range; otherwise return None (fall back to plain semantic search). Never
    returns a filter that could empty the result set."""
    if not config.RAG_PRESETS[preset_name]["use_metadata_filter"]:
        return None
    yrs = detect_years(question)
    if not yrs:
        return None
    return {"year": {"$in": yrs}}


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(question: str, preset_name: str, k: int = config.TOP_K) -> list[dict]:
    coll = _client().get_collection(preset_name)
    where = build_where(question, preset_name)
    qvec = embed([question])[0].tolist()
    res = coll.query(query_embeddings=[qvec], n_results=k, where=where)
    # soft fallback: if the metadata filter starved the result, retry unfiltered
    if where is not None and len(res["ids"][0]) < k:
        res = coll.query(query_embeddings=[qvec], n_results=k)
    out = []
    for rank, (cid, doc, meta, dist) in enumerate(zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
        out.append({"rank": rank, "chunk_id": cid, "text": doc,
                    "source_file": meta["source_file"], "year": meta["year"],
                    "month": meta["month"], "distance": dist})
    return out


# --------------------------------------------------------------------------
# Generation (needs ANTHROPIC_API_KEY in the environment)
# --------------------------------------------------------------------------
PROMPTS = {
    "simple": (
        "Use the Treasury Bulletin excerpts below to answer the question. "
        "Give a specific answer -- a number with its units if the answer is "
        "numeric. End with a line 'Final answer: <value>'.\n\n"
        "{context}\n\nQuestion: {question}\nAnswer:"
    ),
    "strict": (
        "You are a precise financial analyst. Using ONLY the Treasury Bulletin "
        "excerpts below, answer the question. The answer usually requires "
        "locating values in the tables and may require simple computation "
        "(a difference, percentage, growth rate, or sum) -- work it out step by "
        "step. Report the result as a number in the same units the source uses. "
        "Only if the excerpts genuinely do not contain the needed values, reply "
        "'Not found'. Cite the source_file(s) you used, and end with a line "
        "'Final answer: <value>'.\n\n"
        "{context}\n\nQuestion: {question}\nAnswer:"
    ),
}


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['source_file']}]\n{c['text']}" for c in chunks
    )


@lru_cache(maxsize=1)
def _anthropic():
    import anthropic
    # short timeout so a stalled connection fails fast and our visible retry
    # layer (below) handles the backoff, instead of the SDK hanging ~10 min.
    return anthropic.Anthropic(max_retries=2, timeout=90.0)


def messages_create(**kwargs):
    """messages.create with backoff on transient overload/rate-limit/5xx, so a
    long evaluation run survives a busy API instead of crashing mid-way."""
    import time
    import random
    import anthropic
    transient = (anthropic.OverloadedError, anthropic.RateLimitError,
                 anthropic.InternalServerError, anthropic.APIConnectionError,
                 anthropic.APITimeoutError)
    tries = 12
    for attempt in range(tries):
        try:
            return _anthropic().messages.create(**kwargs)
        except transient as e:
            if attempt == tries - 1:
                raise
            wait = min(60, 3 * 2 ** attempt) + random.random() * 2
            print(f"    (API busy: {type(e).__name__}; retry {attempt+1}/{tries} in {wait:.0f}s)",
                  flush=True)
            time.sleep(wait)


def resp_text(resp) -> str:
    """Join the text blocks of a response, skipping any thinking blocks
    (newer models can return a ThinkingBlock as content[0])."""
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def generate(question: str, chunks: list[dict], preset_name: str) -> str:
    style = config.RAG_PRESETS[preset_name]["prompt_style"]
    prompt = PROMPTS[style].format(context=format_context(chunks), question=question)
    resp = messages_create(
        model=config.GEN_MODEL, max_tokens=config.GEN_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp_text(resp).strip()


def answer(question: str, preset_name: str, k: int = config.TOP_K) -> dict:
    chunks = retrieve(question, preset_name, k)
    return {"answer": generate(question, chunks, preset_name), "retrieved": chunks}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    preset = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    if cmd == "build":
        build_index(preset, rebuild="--rebuild" in sys.argv)
    elif cmd == "retrieve":
        for c in retrieve(sys.argv[3], preset):
            print(f"#{c['rank']} {c['source_file']} (d={c['distance']:.3f})  "
                  f"{c['text'][:90]!r}")
    elif cmd == "ask":
        out = answer(sys.argv[3], preset)
        print("SOURCES:", [c["source_file"] for c in out["retrieved"]])
        print("ANSWER :", out["answer"])
    else:
        print(__doc__)
