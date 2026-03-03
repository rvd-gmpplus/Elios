
from __future__ import annotations
import os, json, pathlib, re
from typing import Dict, Any, List, Tuple
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path
import argparse
import yaml

from src.chunker.hier_chunker import HierarchicalChunker

def read_text_from_path(p: pathlib.Path) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"url": str(p), "doc_title": p.stem, "language": "en"}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(txt, "lxml")
        article = None
        for sel in ["main article", "article", "main", "div.rich-text", "section"]:
            article = soup.select_one(sel)
            if article: break
        core = article.get_text("\n") if article else soup.get_text("\n")
        title = None
        for ts in ["header h1","main h1","h1","h2"]:
            t = soup.select_one(ts)
            if t and t.get_text(strip=True):
                title = t.get_text(strip=True); break
        if title: meta["doc_title"] = title
        txt = core
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt, meta

def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw", help="Where input documents live")
    ap.add_argument("--out", default="data/processed/chunks.jsonl", help="Output JSONL with chunks")
    ap.add_argument("--chunk_cfg", default="configs/chunking.yaml", help="Chunking config")
    args = ap.parse_args()

    load_dotenv(dotenv_path=Path.cwd()/".env")

    with open(args.chunk_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    chunker = HierarchicalChunker(cfg)

    raw_dir = pathlib.Path(args.raw_dir)
    docs = []
    for p in raw_dir.glob("**/*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in {".txt",".md",".html",".htm"}:
            continue
        text, meta = read_text_from_path(p)
        doc_id = p.stem
        meta["doc_id"] = doc_id
        chunks = chunker.split(text, meta)
        for ch in chunks:
            docs.append({
                "id": ch.id,
                "parent_id": ch.parent_id,
                "text": ch.text,
                "metadata": ch.metadata,
                "n_tokens": len(ch.text.split())
            })

    out_path = pathlib.Path(args.out)
    write_jsonl(out_path, docs)
    print(f"\u2714 Wrote {len(docs)} chunks -> {out_path}")

if __name__ == "__main__":
    main()
