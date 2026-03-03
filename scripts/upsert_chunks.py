from __future__ import annotations
import os, json, pathlib, math, itertools
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
import yaml
from pinecone import Pinecone
from src.indexing.build_payload import build_vectors

def batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch

def main():
    load_dotenv(Path.cwd()/".env")

    cfg = yaml.safe_load(open("configs/indexing.yaml", "r", encoding="utf-8"))
    index_name = os.environ["PINECONE_INDEX"]
    namespace = os.environ.get("PINECONE_NAMESPACE", "default")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)   # use index name, not host

    in_path = Path("data/processed/chunks.jsonl")
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}. Run the ingest step first.")
    rows = [json.loads(l) for l in in_path.open("r", encoding="utf-8")]

    batch_size = int(cfg.get("upsert", {}).get("batch_size", 64))
    model = cfg.get("hybrid", {}).get("dense_embedding_model", "text-embedding-3-large")

    total = 0
    for batch in batched(rows, batch_size):
        vecs = build_vectors(batch, embed_model=model)
        index.upsert(vectors=vecs, namespace=namespace)
        total += len(vecs)
        print(f"Upserted {total}/{len(rows)}")

    print(f"✔ Done. Upserted {total} vectors into namespace '{namespace}'.")

if __name__ == "__main__":
    main()
