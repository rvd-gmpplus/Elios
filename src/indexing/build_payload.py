# --- add / replace imports at the top ---
from __future__ import annotations
from typing import Dict, Any, List
import re, collections, os, hashlib
from openai import OpenAI

_WORD = re.compile(r"[A-Za-z0-9_\-]+")

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]

def bm25_sparser(doc: str, avgdl: float = 100.0, k1: float = 1.2, b: float = 0.75) -> Dict[str, float]:
    """
    Lightweight TF-length BM25-ish weights (no corpus IDF).
    Returns a dict: term -> weight (float).
    """
    tf = collections.Counter(_tokenize(doc))
    if not tf:
        return {}
    dl = sum(tf.values())
    out: Dict[str, float] = {}
    for term, f in tf.items():
        denom = f + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
        out[term] = float((f * (k1 + 1.0)) / max(1.0, denom))
    return out

def _term_to_id(term: str) -> int:
    """
    Stable 32-bit integer for a term using blake2b(4 bytes).
    Pinecone expects int indices (int32 is fine).
    """
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")

def dense_embed(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=model, input=texts)
    return [e.embedding for e in resp.data]

def build_vectors(records: List[Dict[str, Any]], embed_model: str = "text-embedding-3-large") -> List[Dict[str, Any]]:
    texts = [r["text"] for r in records]
    embeddings = dense_embed(texts, model=embed_model)
    out: List[Dict[str, Any]] = []
    for r, vec in zip(records, embeddings):
        # Keep your existing sparse builder (None for dense-only indexes)
        try:
            sparse_map = bm25_sparser(r["text"])
            if sparse_map:
                indices = [_term_to_id(t) for t in sparse_map.keys()]
                values  = list(sparse_map.values())
                sparse_values = {"indices": indices, "values": values}
            else:
                sparse_values = None
        except Exception:
            sparse_values = None

        out.append({
            "id": r["id"],
            "values": vec,
            # ✅ store text inside metadata so QA can build context without refetching
            "metadata": {**r.get("metadata", {}), "text": r["text"]},
            "sparse_values": sparse_values
        })
    return out
