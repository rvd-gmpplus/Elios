from __future__ import annotations
from typing import List, Dict, Any, Tuple

def _parent_id(m: Dict[str, Any]) -> str:
    md = m.get("metadata", {})
    return md.get("parent_id") or m.get("parent_id") or md.get("section_path") or "root"

def group_by_parent(matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    byp: Dict[str, List[Dict[str, Any]]] = {}
    for m in matches:
        pid = _parent_id(m)
        byp.setdefault(pid, []).append(m)
    for pid in byp:
        byp[pid].sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return byp

def expand_with_siblings(matches: List[Dict[str, Any]], siblings: int = 2, max_parents: int = 8) -> List[Dict[str, Any]]:
    byp = group_by_parent(matches)
    parent_best = [(pid, kids[0]) for pid, kids in byp.items()]
    parent_best.sort(key=lambda kv: kv[1].get("score", 0.0), reverse=True)
    parent_best = parent_best[:max_parents]

    expanded: List[Dict[str, Any]] = []
    for pid, best in parent_best:
        expanded.append(best)
        expanded.extend(byp[pid][1:1+siblings])  # top siblings

    seen, out = set(), []
    for m in expanded:
        mid = m.get("id")
        if mid in seen: continue
        seen.add(mid); out.append(m)
    return out

def build_context(chunks: List[Dict[str, Any]], max_chars: int = 12000) -> Tuple[str, List[Tuple[str, str]]]:
    parts, cites, total = [], [], 0
    for m in chunks:
        md = m.get("metadata", {})
        title = md.get("doc_title", "")
        path  = md.get("section_path", "")
        txt   = m.get("text") or md.get("text") or ""
        header = f"[{title} — {path}]".strip()
        block  = (header + "\n" + txt.strip()).strip()
        if not block: continue
        if total + len(block) > max_chars: break
        parts.append(block); total += len(block)
        cites.append((m.get("id", ""), f"{title} — {path}"))
    return "\n\n---\n\n".join(parts), cites
