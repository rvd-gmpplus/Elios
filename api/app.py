# api/app.py
from __future__ import annotations
import os, sys, json, time, uuid, re, logging, contextvars, csv
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from starlette.middleware.base import BaseHTTPMiddleware
from src.indexing.build_payload import bm25_sparser, _term_to_id

load_dotenv()  # handy for local dev

# =========================
# Environment & Clients
# =========================
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "gmp-v2prod")

# Auth: prefer BEARER_TOKEN; accept legacy RAILWAY_BEARER
BEARER_TOKEN = os.environ.get("BEARER_TOKEN") or os.environ.get("RAILWAY_BEARER")

# Logging knobs (Railway variables)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TOP_HITS = int(os.getenv("LOG_TOP_HITS", "8"))
LOG_PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "180"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "60"))  # tiny snippets are filtered from logs/context

# Configure root logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s | request_id=%(request_id)s",
    stream=sys.stdout,
)
# Quiet noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)

log = logging.getLogger("cbv2")

# Per-request request_id via ContextVar so every log line can include it
REQ_ID: contextvars.ContextVar[str] = contextvars.ContextVar("req_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQ_ID.get("-")
        return True

# Ensure our logger has the filter
for h in logging.getLogger().handlers:
    h.addFilter(RequestIdFilter())

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
ocli = OpenAI(api_key=OPENAI_API_KEY)

APP_NAME = "CBV2 RAG API"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load source URL mapping from sources.csv (title → original GMP+ website URL)
# Builds two lookups: exact title match and document code prefix match (e.g. "CR3.0", "S9.92")
_DOC_CODE_RE = re.compile(r"^([A-Z]+\s?\d+\.\d+)")

def _load_source_urls():
    by_title = {}
    by_code = {}
    csv_path = Path(__file__).resolve().parent.parent / "sources.csv"
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                title = (row.get("title") or "").strip()
                url = (row.get("url") or "").strip()
                if title and url:
                    by_title[title] = url
                    m = _DOC_CODE_RE.match(title)
                    if m:
                        by_code[m.group(1).replace(" ", "")] = url
    except FileNotFoundError:
        pass
    return by_title, by_code

SOURCE_URLS, SOURCE_URLS_BY_CODE = _load_source_urls()

def _resolve_source_url(title: str, fallback: str) -> str:
    """Look up the original GMP+ URL by exact title, then by doc code prefix, then fallback."""
    url = SOURCE_URLS.get(title)
    if url:
        return url
    m = _DOC_CODE_RE.match(title)
    if m:
        url = SOURCE_URLS_BY_CODE.get(m.group(1).replace(" ", ""))
        if url:
            return url
    return fallback

# =========================
# Helpers
# =========================
_ws_re = re.compile(r"\s+")

def _singleline(s: str, limit: int = LOG_PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate for compact, single-line previews."""
    if not s:
        return ""
    s = _ws_re.sub(" ", str(s)).strip()
    return s[:limit].rstrip() + ("…" if len(s) > limit else "")

def _auth_check(auth_header: Optional[str]):
    """Enforce Bearer auth if BEARER_TOKEN is configured."""
    if BEARER_TOKEN:
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = auth_header.split(" ", 1)[1].strip()
        if token != BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

def embed(texts: List[str]) -> List[List[float]]:
    resp = ocli.embeddings.create(model="text-embedding-3-large", input=texts)
    return [d.embedding for d in resp.data]

def _sparse_vector(query: str):
    """Build Pinecone sparse_vector dict from BM25 term weights, or None."""
    try:
        sw = bm25_sparser(query)
        if sw:
            return {"indices": [_term_to_id(t) for t in sw], "values": list(sw.values())}
    except Exception:
        pass
    return None

def _md(m: Dict[str, Any]) -> Dict[str, Any]:
    return (m.get("metadata") or {})

def _text_from_md(md: Dict[str, Any]) -> str:
    # Normalize and single-line chunk text
    s = (md.get("text") or md.get("chunk_text") or md.get("text_content") or "")
    return _singleline(s, limit=10_000)  # normalize but do not truncate aggressively here

def _has_body(m: Dict[str, Any]) -> bool:
    # Drop ultra-short or header like snippets from context and logs
    txt = _text_from_md(_md(m))
    return len(txt) >= MIN_CHUNK_CHARS

def answer_with_openai(question: str, context: str) -> str:
    resp = ocli.chat.completions.create(
        model="gpt-5-mini",
        reasoning_effort="minimal",
        verbosity="low",
        messages=[
            {"role": "system", "content": (
                "You are a GMP+ Document Assistant. Answer based on the provided context only. "
                "If the context contains relevant guidance or methodology (even without a specific numeric limit), "
                "summarise that guidance and explain how it applies to the question. "
                "Only say you do not know if the context contains nothing relevant to the question at all."
            )},
            {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"}
        ]
    )
    return resp.choices[0].message.content

def rerank_with_llm(query: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign a rerank_score (0..5) and sort descending by it; preserve Pinecone order on parse errors."""
    if not matches:
        return matches
    items = []
    for m in matches:
        md = _md(m)
        items.append({
            "id": m.get("id"),
            "title": md.get("doc_title", ""),
            "path": md.get("section_path", ""),
            "text": _text_from_md(md)[:650],
        })
    prompt = {
        "query": query,
        "chunks": items,
        "instruction": 'Return JSON: [{"id": str, "score": float 0..5}] in same order.'
    }
    resp = ocli.chat.completions.create(
        model="gpt-5-nano",
        reasoning_effort="minimal",
        verbosity="low",
        messages=[
            {"role": "system", "content": "You are a re-ranking engine. Score each chunk 0 to 5 by how well it answers the question."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ]
    )
    try:
        scores = json.loads(resp.choices[0].message.content.strip())
        score_map = {s["id"]: float(s.get("score", 0.0)) for s in scores if "id" in s}
        for m in matches:
            m["rerank_score"] = float(score_map.get(m.get("id"), 0.0))
        matches.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    except Exception:
        pass
    return matches

def as_context(matches: List[Dict[str, Any]], max_chars: int = 12000) -> Tuple[str, List[Dict[str, str]]]:
    parts, cites = [], []
    total = 0
    for m in matches:
        md = _md(m)
        title = md.get("doc_title", "")
        path = md.get("section_path", "")
        url   = _resolve_source_url(title, md.get("url", ""))
        text = _text_from_md(md)

        header = f"[{title} — {path}]"
        if url:
            header += f" ({url})"

        para = f"{header}\n{text}\n"
        if total + len(para) > max_chars:
            break
        parts.append(para)
        cites.append({"id": m.get("id"), "title": title, "section": path, "url": url})
        total += len(para)
    return "\n".join(parts).strip(), cites

# =========================
# FastAPI + Middleware
# =========================
app = FastAPI(
    title=APP_NAME,
    version="1.4.0",
    description="Health, search, retrieve, answer, and debug endpoints for GMP+ RAG on Railway."
)

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        REQ_ID.set(request_id)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # One-line access summary (our logger adds request_id automatically)
        log.info(
            "HTTP %s %s | status=%s | elapsed=%.1fms",
            request.method,
            request.url.path,
            getattr(response, "status_code", None),
            elapsed_ms,
        )
        response.headers["x-request-id"] = request_id
        return response

app.add_middleware(RequestContextMiddleware)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------- Web UI ----------
@app.get("/", response_class=FileResponse, include_in_schema=False)
def web_ui():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

class WebAskPayload(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class WebAskSource(BaseModel):
    title: str = ""
    section: str = ""
    url: str = ""

class WebAskResponse(BaseModel):
    answer: str
    sources: list[WebAskSource]

@app.post("/web/ask", response_model=WebAskResponse, include_in_schema=False)
def web_ask(payload: WebAskPayload):
    question = payload.question.strip()

    log.info("WebAsk | q='%s'", _singleline(question))

    # 1) Embed (dense only, no hybrid for public)
    qvec = embed([question])[0]

    # 2) Pinecone query with safe defaults
    res = index.query(
        vector=qvec,
        top_k=24,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
        include_values=False,
    )
    matches = res.get("matches", []) or []
    matches = [m for m in matches if (m.get("score") or 0.0) >= 0.28]
    matches = [m for m in matches if _has_body(m)]

    # 3) Rerank top 12
    if matches:
        head = matches[:12]
        tail = matches[12:]
        head = rerank_with_llm(question, head)
        matches = head + tail

    # 4) Build context and answer
    context, cites = as_context(matches)
    ans = answer_with_openai(question, context)

    sources = [
        WebAskSource(title=c.get("title", ""), section=c.get("section", ""), url=c.get("url", ""))
        for c in cites
    ]

    return WebAskResponse(answer=ans, sources=sources)

# ---------- Models ----------
class QueryPayload(BaseModel):
    question: Optional[str] = None
    top_k: int = Field(24, ge=1, le=50)
    rerank_top: int = Field(12, ge=1, le=50)
    use_rerank: bool = True
    use_hybrid: bool = False
    namespace: Optional[str] = None
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class DebugRequest(BaseModel):
    question: Optional[str] = None
    top_k: int = Field(24, ge=1, le=50)
    rerank_top: int = Field(12, ge=1, le=50)
    use_rerank: bool = True
    use_hybrid: bool = False
    namespace: Optional[str] = None
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_text: bool = True
    snippet_chars: int = Field(700, ge=50, le=5000)

class RetrieveHit(BaseModel):
    id: str
    pinecone: Optional[float] = None
    rerank: Optional[float] = None
    title: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None

class RetrieveResponse(BaseModel):
    question: str
    namespace: str
    params: Dict[str, Any]
    hits: List[RetrieveHit]

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "index": PINECONE_INDEX, "namespace_default": PINECONE_NAMESPACE}

# ---------- SEARCH (GET) ----------
@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50),
    use_hybrid: bool = Query(False),
    namespace: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    authorization: Optional[str] = Header(None)
):
    _auth_check(authorization)
    ns = namespace or PINECONE_NAMESPACE

    log.info("Search | q='%s' | top_k=%s | use_hybrid=%s | min_score=%s | ns=%s", _singleline(q), top_k, use_hybrid, min_score, ns)

    t0 = time.perf_counter()
    qvec = embed([q])[0]
    sparse = _sparse_vector(q) if use_hybrid else None
    t1 = time.perf_counter()
    res = index.query(
        vector=qvec,
        sparse_vector=sparse,
        top_k=top_k,
        namespace=ns,
        include_metadata=True,
        include_values=False
    )
    t2 = time.perf_counter()

    matches = res.get("matches", []) or []
    if min_score is not None:
        matches = [m for m in matches if (m.get("score") or 0.0) >= float(min_score)]
    # Drop very short junk snippets from output and logs
    before = len(matches)
    matches = [m for m in matches if _has_body(m)]
    if before != len(matches):
        log.info(
            "Search | filtered_short_snippets | before=%s after=%s min_chars=%s",
            before,
            len(matches),
            MIN_CHUNK_CHARS,
        )

    # Log top hits
    for i, m in enumerate(matches[:LOG_TOP_HITS], start=1):
        md = _md(m)
        log.info(
            "SearchTop | #%d | score=%.4f | title=%s | section=%s | preview=%s",
            i,
            (m.get("score") or 0.0),
            _singleline(md.get("doc_title")),
            _singleline(md.get("section_path")),
            _singleline(_text_from_md(md)),
        )

    log.info("Timing | embed=%.1fms | pinecone=%.1fms", (t1 - t0)*1000, (t2 - t1)*1000)

    hits = []
    for m in matches:
        md = _md(m)
        hits.append({
            "id": m.get("id"),
            "score": m.get("score"),
            "metadata": {
                "doc_title": md.get("doc_title"),
                "section_path": md.get("section_path"),
                "text": _text_from_md(md)[:500]
            }
        })
    return {"q": q, "namespace": ns, "top_k": top_k, "min_score": min_score, "hits": hits}

# ---------- RETRIEVE (POST) ----------
@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: QueryPayload, authorization: Optional[str] = Header(None)):
    _auth_check(authorization)

    # Guard: if question is missing or empty, return clean response (avoid 422 from ChatGPT Actions)
    if not payload.question or not payload.question.strip():
        ns = payload.namespace or PINECONE_NAMESPACE
        return RetrieveResponse(
            question="",
            namespace=ns,
            params={
                "top_k": payload.top_k,
                "rerank_top": payload.rerank_top,
                "use_rerank": payload.use_rerank,
                "use_hybrid": payload.use_hybrid,
                "min_score": payload.min_score,
            },
            hits=[],
        )

    # clamp
    payload.top_k = max(1, min(50, int(payload.top_k)))
    payload.rerank_top = max(1, min(payload.top_k, int(payload.rerank_top)))
    ns = payload.namespace or PINECONE_NAMESPACE

    log.info(
        "RetrieveReq | q='%s' | top_k=%s | rerank_top=%s | use_rerank=%s | use_hybrid=%s | min_score=%s | ns=%s",
        _singleline(payload.question),
        payload.top_k,
        payload.rerank_top,
        payload.use_rerank,
        payload.use_hybrid,
        payload.min_score,
        ns,
    )

    # 1) Embed
    t0 = time.perf_counter()
    qvec = embed([payload.question])[0]
    sparse = _sparse_vector(payload.question) if payload.use_hybrid else None
    t1 = time.perf_counter()

    # 2) Pinecone
    res = index.query(
        vector=qvec,
        sparse_vector=sparse,
        top_k=payload.top_k,
        namespace=ns,
        include_metadata=True,
        include_values=False,
    )
    t2 = time.perf_counter()
    matches = res.get("matches", []) or []

    # 3) Filter by score and remove tiny snippets
    if payload.min_score is not None:
        ms = float(payload.min_score)
        matches = [m for m in matches if (m.get("score") or 0.0) >= ms]
    before = len(matches)
    matches = [m for m in matches if _has_body(m)]
    if before != len(matches):
        log.info(
            "Retrieve | filtered_short_snippets | before=%s after=%s min_chars=%s",
            before,
            len(matches),
            MIN_CHUNK_CHARS,
        )

    log.info(
        "Timing | retrieve_embed=%.1fms | retrieve_pinecone=%.1fms",
        (t1 - t0) * 1000,
        (t2 - t1) * 1000,
    )

    # 4) Optional rerank
    t_r0 = time.perf_counter()
    post_matches = matches[:]
    if payload.use_rerank and post_matches:
        head = post_matches[: payload.rerank_top]
        tail = post_matches[payload.rerank_top:]
        head = rerank_with_llm(payload.question, head)  # adds rerank_score and sorts
        post_matches = head + tail
    t_r1 = time.perf_counter()
    if payload.use_rerank:
        log.info("Timing | retrieve_rerank=%.1fms", (t_r1 - t_r0) * 1000)

    # 5) Build hits list from post rerank matches
    hits: List[RetrieveHit] = []
    for i, m in enumerate(post_matches, start=1):
        md = _md(m)
        txt = _text_from_md(md)
        hit = RetrieveHit(
            id=m.get("id"),
            pinecone=m.get("score"),
            rerank=m.get("rerank_score", None),
            title=md.get("doc_title"),
            section=md.get("section_path"),
            url=md.get("url"),
            text=txt,
        )
        hits.append(hit)
        if i <= LOG_TOP_HITS:
            log.info(
                "RetrievePost | #%d | pinecone=%.4f | rerank=%s | title=%s | section=%s | preview=%s",
                i,
                (m.get("score") or 0.0),
                f"{m.get('rerank_score'):.2f}" if m.get("rerank_score") is not None else "NA",
                _singleline(md.get("doc_title")),
                _singleline(md.get("section_path")),
                _singleline(txt),
            )

    return RetrieveResponse(
        question=payload.question,
        namespace=ns,
        params={
            "top_k": payload.top_k,
            "rerank_top": payload.rerank_top,
            "use_rerank": payload.use_rerank,
            "use_hybrid": payload.use_hybrid,
            "min_score": payload.min_score,
        },
        hits=hits,
    )

# ---------- ANSWER (POST) ----------
@app.post("/answer")
def answer(payload: QueryPayload, authorization: Optional[str] = Header(None)):
    _auth_check(authorization)

    # Guard: if question is missing or empty, return clean response (avoid 422 from ChatGPT Actions)
    if not payload.question or not payload.question.strip():
        return {"answer": "", "sources": [], "scores": []}

    # clamp
    payload.top_k = max(1, min(50, int(payload.top_k)))
    payload.rerank_top = max(1, min(payload.top_k, int(payload.rerank_top)))
    ns = payload.namespace or PINECONE_NAMESPACE

    log.info(
        "AnswerReq | q='%s' | top_k=%s | rerank_top=%s | use_rerank=%s | use_hybrid=%s | min_score=%s | ns=%s",
        _singleline(payload.question),
        payload.top_k,
        payload.rerank_top,
        payload.use_rerank,
        payload.use_hybrid,
        payload.min_score,
        ns,
    )

    # 1) Embed
    t0 = time.perf_counter()
    qvec = embed([payload.question])[0]
    sparse = _sparse_vector(payload.question) if payload.use_hybrid else None
    t1 = time.perf_counter()

    # 2) Pinecone
    res = index.query(
        vector=qvec,
        sparse_vector=sparse,
        top_k=payload.top_k,
        namespace=ns,
        include_metadata=True,
        include_values=False
    )
    t2 = time.perf_counter()
    matches = res.get("matches", []) or []

    # Filter by score, then remove tiny snippets
    if payload.min_score is not None:
        ms = float(payload.min_score)
        matches = [m for m in matches if (m.get("score") or 0.0) >= ms]
    before = len(matches)
    matches = [m for m in matches if _has_body(m)]
    if before != len(matches):
        log.info(
            "Answer | filtered_short_snippets | before=%s after=%s min_chars=%s",
            before,
            len(matches),
            MIN_CHUNK_CHARS,
        )

    log.info("Timing | embed=%.1fms | pinecone=%.1fms", (t1 - t0)*1000, (t2 - t1)*1000)

    # 3) Rerank on head
    t_r0 = time.perf_counter()
    if payload.use_rerank and matches:
        head = matches[:payload.rerank_top]
        tail = matches[payload.rerank_top:]
        head = rerank_with_llm(payload.question, head)
        matches = head + tail
    t_r1 = time.perf_counter()
    if payload.use_rerank:
        log.info("Timing | rerank=%.1fms", (t_r1 - t_r0)*1000)

    # Log post-rerank head
    for i, m in enumerate(matches[:payload.rerank_top], start=1):
        md = _md(m)
        log.info(
            "Post | #%d | pinecone=%.4f | rerank=%s | title=%s | section=%s | preview=%s",
            i,
            (m.get("score") or 0.0),
            f"{m.get('rerank_score'):.2f}" if m.get("rerank_score") is not None else "NA",
            _singleline(md.get("doc_title")),
            _singleline(md.get("section_path")),
            _singleline(_text_from_md(md)),
        )

    # 4) Build context and answer
    t_a0 = time.perf_counter()
    context, cites = as_context(matches)
    ans = answer_with_openai(payload.question, context)
    t_a1 = time.perf_counter()

    log.info("Answer | produced | preview=%s", _singleline(ans))
    log.info("Timing | answer=%.1fms | total=%.1fms", (t_a1 - t_a0)*1000, (t_a1 - t0)*1000)

    scored = [{
        "id": m.get("id"),
        "pinecone": m.get("score"),
        "rerank": m.get("rerank_score", None),
        "title": _md(m).get("doc_title"),
        "section": _md(m).get("section_path")
    } for m in matches[:payload.rerank_top]]

    return {"answer": ans, "sources": cites, "scores": scored}

# ---------- DEBUG (POST) ----------
class DebugRow(BaseModel):
    id: str
    pinecone: Optional[float] = None
    rerank: Optional[float] = None
    title: Optional[str] = None
    section: Optional[str] = None
    text: Optional[str] = None

class DebugRequestOut(BaseModel):
    question: str
    namespace: str
    params: Dict[str, Any]
    pre_rerank: List[DebugRow]
    post_rerank: List[DebugRow]
    context_preview: str
    sources: List[Dict[str, str]]

@app.post("/debug", response_model=DebugRequestOut)
def debug(payload: DebugRequest, authorization: Optional[str] = Header(None)):
    _auth_check(authorization)

    # Guard: if question is missing or empty, return clean response (avoid 422 from ChatGPT Actions)
    if not payload.question or not payload.question.strip():
        ns = payload.namespace or PINECONE_NAMESPACE
        return {
            "question": "",
            "namespace": ns,
            "params": {
                "top_k": payload.top_k,
                "rerank_top": payload.rerank_top,
                "use_rerank": payload.use_rerank,
                "use_hybrid": payload.use_hybrid,
                "min_score": payload.min_score,
                "include_text": payload.include_text,
                "snippet_chars": payload.snippet_chars,
            },
            "pre_rerank": [],
            "post_rerank": [],
            "context_preview": "",
            "sources": [],
        }

    payload.top_k = max(1, min(50, int(payload.top_k)))
    payload.rerank_top = max(1, min(payload.top_k, int(payload.rerank_top)))
    ns = payload.namespace or PINECONE_NAMESPACE

    log.info(
        "DebugReq | q='%s' | top_k=%s | rerank_top=%s | use_rerank=%s | use_hybrid=%s | min_score=%s | ns=%s",
        _singleline(payload.question),
        payload.top_k,
        payload.rerank_top,
        payload.use_rerank,
        payload.use_hybrid,
        payload.min_score,
        ns,
    )

    # 1) Embed
    t0 = time.perf_counter()
    qvec = embed([payload.question])[0]
    sparse = _sparse_vector(payload.question) if payload.use_hybrid else None
    t1 = time.perf_counter()

    # 2) Pinecone
    res = index.query(
        vector=qvec,
        sparse_vector=sparse,
        top_k=payload.top_k,
        namespace=ns,
        include_metadata=True,
        include_values=False
    )
    t2 = time.perf_counter()
    matches = res.get("matches", []) or []

    # Filters
    if payload.min_score is not None:
        ms = float(payload.min_score)
        matches = [m for m in matches if (m.get("score") or 0.0) >= ms]
    before = len(matches)
    matches = [m for m in matches if _has_body(m)]
    if before != len(matches):
        log.info(
            "Debug | filtered_short_snippets | before=%s after=%s min_chars=%s",
            before,
            len(matches),
            MIN_CHUNK_CHARS,
        )

    log.info("Timing | embed=%.1fms | pinecone=%.1fms", (t1 - t0)*1000, (t2 - t1)*1000)

    # 3) pre-rerank table (log top)
    pre: List[Dict[str, Any]] = []
    for i, m in enumerate(matches, start=1):
        md = _md(m)
        row = {
            "id": m.get("id"),
            "pinecone": m.get("score"),
            "title": md.get("doc_title"),
            "section": md.get("section_path"),
        }
        if payload.include_text:
            row["text"] = _text_from_md(md)[: int(payload.snippet_chars)]
        pre.append(row)
        if i <= LOG_TOP_HITS:
            log.info(
                "Pre | #%d | pinecone=%.4f | title=%s | section=%s | preview=%s",
                i,
                (m.get("score") or 0.0),
                _singleline(md.get("doc_title")),
                _singleline(md.get("section_path")),
                _singleline(_text_from_md(md)),
            )

    # 4) optional rerank
    t_r0 = time.perf_counter()
    post_matches = matches[:]
    if payload.use_rerank and matches:
        head = post_matches[: payload.rerank_top]
        tail = post_matches[payload.rerank_top:]
        head = rerank_with_llm(payload.question, head)  # adds rerank_score and sorts
        post_matches = head + tail
    t_r1 = time.perf_counter()
    if payload.use_rerank:
        log.info("Timing | rerank=%.1fms", (t_r1 - t_r0)*1000)

    # 5) post-rerank table (top rerank_top)
    post: List[Dict[str, Any]] = []
    for i, m in enumerate(post_matches[: payload.rerank_top], start=1):
        md = _md(m)
        row = {
            "id": m.get("id"),
            "pinecone": m.get("score"),
            "rerank": m.get("rerank_score", None),
            "title": md.get("doc_title"),
            "section": md.get("section_path"),
        }
        if payload.include_text:
            row["text"] = _text_from_md(md)[: int(payload.snippet_chars)]
        post.append(row)
        if i <= LOG_TOP_HITS:
            log.info(
                "Post | #%d | pinecone=%.4f | rerank=%s | title=%s | section=%s | preview=%s",
                i,
                (m.get("score") or 0.0),
                f"{m.get('rerank_score'):.2f}" if m.get("rerank_score") is not None else "NA",
                _singleline(md.get("doc_title")),
                _singleline(md.get("section_path")),
                _singleline(_text_from_md(md)),
            )

    # 6) context preview and sources
    context, cites = as_context(post_matches)
    context_preview = context[:2000]
    log.info("Ctx | preview=%s", _singleline(context_preview, limit=300))

    return {
        "question": payload.question,
        "namespace": ns,
        "params": {
            "top_k": payload.top_k,
            "rerank_top": payload.rerank_top,
            "use_rerank": payload.use_rerank,
            "use_hybrid": payload.use_hybrid,
            "min_score": payload.min_score,
            "include_text": payload.include_text,
            "snippet_chars": payload.snippet_chars,
        },
        "pre_rerank": pre,
        "post_rerank": post,
        "context_preview": context_preview,
        "sources": cites,
    }

# Run locally:
# uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
