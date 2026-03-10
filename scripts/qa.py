from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

from src.indexing.build_payload import dense_embed, bm25_sparser, _term_to_id
from src.retrieval.expand import expand_with_siblings, build_context

TOP_K_RETRIEVE = 24
TOP_K_RERANK = 12
SIBLINGS_PER_PARENT = 2
MAX_PARENTS = 8
MAX_CONTEXT_CHARS = 12000
USE_HYBRID_DEFAULT = True
USE_LLM_RERANK_DEFAULT = True


# ----------------------- Retrieval -----------------------
def retrieve(index, query: str, top_k: int, namespace: str, try_hybrid: bool = True) -> List[Dict[str, Any]]:
    qvec = dense_embed([query])[0]
    kwargs = dict(vector=qvec, top_k=top_k, namespace=namespace, include_metadata=True, include_values=False)
    if try_hybrid:
        try:
            q_sparse = bm25_sparser(query)
            if q_sparse:
                indices = [_term_to_id(t) for t in q_sparse.keys()]
                values = list(q_sparse.values())
                kwargs["sparse_vector"] = {"indices": indices, "values": values}
        except Exception:
            pass

    res = index.query(**kwargs)
    for m in res["matches"]:
        if "text" not in m:
            m["text"] = m.get("metadata", {}).get("text", "")
    return res["matches"]


# ----------------------- Re-ranker -----------------------
def rerank_with_llm(query: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not matches:
        return matches
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    items = []
    for i, m in enumerate(matches, 1):
        md = m.get("metadata", {})
        items.append({
            "id": m.get("id", f"m{i}"),
            "title": md.get("doc_title", ""),
            "path": md.get("section_path", ""),
            "text": (m.get("text") or "")[:600],
        })

    system_msg = "You are a re-ranking engine. Score each chunk 0–5 for how well it helps answer the user's query."
    user_payload = {
        "query": query,
        "chunks": items,
        "instruction": "Return JSON: [{\"id\": str, \"score\": float between 0 and 5}...] in the SAME ORDER as input."
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    content = resp.choices[0].message.content.strip()
    try:
        scores = json.loads(content)
        score_map = {s.get("id"): float(s.get("score", 0.0)) for s in scores if "id" in s}
        for m in matches:
            m["rerank_score"] = float(score_map.get(m.get("id"), 0.0))
        matches.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception:
        pass
    return matches


# ----------------------- Pretty print scores -----------------------
def print_scores(matches: List[Dict[str, Any]], limit: int | None = None):
    print("\n=== Scores ===")
    print(f"{'#':>2}  {'pinecone':>8}  {'rerank':>6}  {'title'}  —  section path")
    print("-" * 80)
    for i, m in enumerate(matches[:limit] if limit else matches, 1):
        md = m.get("metadata", {})
        title = md.get("doc_title", "") or "-"
        path = md.get("section_path", "") or "-"
        pine = m.get("score", 0.0)
        rrs  = m.get("rerank_score", None)
        rrs_s = f"{rrs:.2f}" if isinstance(rrs, (int, float)) else "-"
        print(f"{i:>2}. {pine:8.4f}  {rrs_s:>6}  {title} — {path}")


# ----------------------- Answering -----------------------
def answer_with_openai(question: str, context: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "You are a GMP+ Document Assistant. Answer based on the provided context only. "
                "Quote exact lines when helpful. "
                "If the context contains relevant guidance or methodology (even without a specific numeric limit), "
                "summarise that guidance and explain how it applies to the question. "
                "Only say you do not know if the context contains nothing relevant to the question at all."
            )},
            {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
        ],
    )
    return resp.choices[0].message.content


# ----------------------- Main -----------------------
def main():
    load_dotenv(Path.cwd() / ".env")
    namespace = os.environ.get("PINECONE_NAMESPACE", "default")
    index_name = os.environ["PINECONE_INDEX"]

    # --- parse CLI flags ---
    use_rerank = USE_LLM_RERANK_DEFAULT
    use_hybrid = USE_HYBRID_DEFAULT
    show_scores = False

    if "--no-rerank" in sys.argv:
        use_rerank = False
        sys.argv.remove("--no-rerank")
    elif "--rerank" in sys.argv:
        use_rerank = True
        sys.argv.remove("--rerank")

    if "--no-hybrid" in sys.argv:
        use_hybrid = False
        sys.argv.remove("--no-hybrid")
    elif "--hybrid" in sys.argv:
        use_hybrid = True
        sys.argv.remove("--hybrid")

    if "--show-scores" in sys.argv:
        show_scores = True
        sys.argv.remove("--show-scores")

    # --- compose question text ---
    question = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--")).strip()
    if not question:
        question = input("Question: ").strip()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    # --- retrieval ---
    matches = retrieve(index, question, top_k=TOP_K_RETRIEVE, namespace=namespace, try_hybrid=use_hybrid)

    # --- optional LLM re-rank ---
    if use_rerank and matches:
        head = matches[:TOP_K_RERANK]
        tail = matches[TOP_K_RERANK:]
        head = rerank_with_llm(question, head)
        matches = head + tail

    # --- optional score printout (before expansion) ---
    if show_scores:
        print_scores(matches, limit=TOP_K_RERANK)

    # --- expansion & answer ---
    expanded = expand_with_siblings(matches, siblings=SIBLINGS_PER_PARENT, max_parents=MAX_PARENTS)
    context, cites = build_context(expanded, max_chars=MAX_CONTEXT_CHARS)
    answer = answer_with_openai(question, context)

    print("\n=== Answer ===\n")
    print(answer.strip())
    print("\n=== Sources ===")
    for mid, label in cites[:10]:
        print(f"- {label} ({mid})")


if __name__ == "__main__":
    main()
