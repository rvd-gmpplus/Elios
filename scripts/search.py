from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

# import the same helpers we used for upsert
from src.indexing.build_payload import dense_embed, bm25_sparser, _term_to_id

def hybrid_query(index, query: str, top_k: int = 10, namespace: str = "default"):
    # dense embedding
    qvec = dense_embed([query])[0]

    # sparse vector (BM25-ish) -> convert TERMS -> INT indices
    q_sparse = bm25_sparser(query)
    if q_sparse:
        indices = [_term_to_id(t) for t in q_sparse.keys()]   # ints
        values  = list(q_sparse.values())                     # floats
        sparse_values = {"indices": indices, "values": values}
    else:
        sparse_values = None

    res = index.query(
        vector=qvec,
        sparse_vector=sparse_values,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    return res

def main():
    load_dotenv(Path.cwd()/".env")
    namespace = os.environ.get("PINECONE_NAMESPACE", "default")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX"])  # use index NAME

    query = input("Query: ").strip()
    res = hybrid_query(index, query, top_k=10, namespace=namespace)
    for i, m in enumerate(res["matches"], 1):
        md = m.get("metadata", {})
        print(f"{i:>2}. score={m['score']:.4f} id={m['id']}  section={md.get('section_path','')}  title={md.get('doc_title','')}")

if __name__ == "__main__":
    main()
