from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import unicodedata
import hashlib

# -------- token counting (tiktoken if available, else fallback) --------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _tok_count(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:
    def _tok_count(text: str) -> int:
        # coarse fallback: word count approximates token count
        return max(1, len((text or "").split()))

# -------- optional sentence splitter (spaCy) with fallback --------
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "textcat"])
        _HAS_SPACY = True
    except Exception:
        _NLP = None
        _HAS_SPACY = False
except Exception:
    _NLP = None
    _HAS_SPACY = False

# regex sentence heuristic when spaCy unavailable
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

@dataclass
class Chunk:
    id: str
    parent_id: Optional[str]
    text: str
    metadata: Dict[str, Any]

class HierarchicalChunker:
    """
    Hierarchical + token-aware chunker for GMP+-style docs.

    Features:
      - Heading detection for numeric (1, 1.1, 1.2.3), code headings (TS/CR/BA/MI/S 1.8), Annex/Appendix,
        roman/letter sub-sections, and numbered/bullet lists.
      - Sentence splitting (spaCy if available, regex fallback).
      - Window packing with adaptive overlap and hard token cap.
      - Merge very small/listy windows forward to avoid title-only slivers.
      - Do NOT duplicate headings inside body; heading is injected once per window.
      - Short-section forward merge (<30 words) to the next section.
      - ASCII-safe bounded IDs for Pinecone (slugify + MD5 tail when needed).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        pat = cfg.get("patterns", {})

        # --- heading patterns ---
        self.re_chapter = re.compile(pat.get("chapter", r"^(?P<chapter>\d+(?:\.\d+)*)\b"))
        self.re_code = re.compile(pat.get("code_heading", r"^(?P<code>(TS|CR|BA|MI|S)\s*\d+(?:\.\d+)*)\b"), re.IGNORECASE)
        self.re_letter = re.compile(pat.get("section_letter", r"^(?P<section>[a-z])\.(?:\s|$)"), re.IGNORECASE)
        self.re_annex  = re.compile(pat.get("annex", r"^(annex|appendix)\s+[ivxlcdm0-9]+(?:\s*$|\s*[:\-–—]\s+.+$)"), re.IGNORECASE)
        self.re_roman  = re.compile(pat.get("roman_list", r"^(?P<roman>[ivxlcdm]+)\.(?:\s|$)"), re.IGNORECASE)
        self.re_num_inline = re.compile(pat.get("numbered_list_inline", r"(?<!\d)(?P<num>\d{1,2})\.\s+"))

        # --- window sizing (defaults mirror tuned config) ---
        self.max_tokens        = int(cfg.get("max_tokens", 900))
        self.min_tokens        = int(cfg.get("min_tokens", 180))
        self.hard_max_tokens   = int(cfg.get("hard_max_tokens", 1000))
        self.base_overlap      = int(cfg.get("base_overlap_tokens", 90))
        self.max_overlap       = int(cfg.get("max_overlap_tokens", 180))
        self.include_heading   = bool(cfg.get("include_heading_in_child", True))

        # --- tuning knobs ---
        self.min_words_merge         = int(cfg.get("min_words_merge", 80))   # merge tiny/listy windows
        self.strip_heading_duplicate = bool(cfg.get("strip_heading_duplicate", True))

        # --- HTML heading helpers ---
        # Detect a full <hN>...</hN> on a single physical line (common in preprocessed HTML)
        self._re_html_h = re.compile(r"<\s*h([1-6])\b[^>]*>(.*?)</\s*h\1\s*>", re.IGNORECASE | re.DOTALL)

    # ---------------- helpers ----------------
    def _sentences(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if _HAS_SPACY and _NLP is not None:
            doc = _NLP(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        # fallback: heuristic split
        parts = _SENT_RE.split(text.strip())
        out, acc = [], ""
        for p in parts:
            acc = (acc + " " + p).strip() if acc else p.strip()
            if acc and acc[-1:] in ".!?":
                out.append(acc); acc = ""
        if acc:
            out.append(acc)
        return out

    def _annex_tail_is_heading(self, tail: str) -> bool:
        """Annex must be EOL or followed by colon/dash + title to count as a heading."""
        tail = (tail or "").strip()
        return (tail == "") or bool(re.match(r'^[:\-–—]\s+\S+', tail))

    def _looks_heading_textonly(self, line: str) -> bool:
        """Plaintext heading detection with annex tail safeguard."""
        s = (line or "").strip()
        m_ann = self.re_annex.match(s)
        if m_ann:
            tail = s[m_ann.end():]
            if not self._annex_tail_is_heading(tail):
                m_ann = None  # inline mention like "see Annex 1 ..." → not a heading
        return any((
            self.re_chapter.match(s),
            self.re_code.match(s),
            bool(m_ann),
            self.re_letter.match(s),
            self.re_roman.match(s),
        ))

    def _looks_heading(self, line: str) -> bool:
        # kept for backward-compat usage in code paths that expect this name
        return self._looks_heading_textonly(line)

    def _top_level_kind(self, title: str) -> str:
        s = (title or "").strip()
        if self.re_annex.match(s):
            return "annex"
        if self.re_chapter.match(s) or self.re_code.match(s):
            return "chapter"
        if self.re_letter.match(s) or self.re_roman.match(s):
            return "sub"
        return "other"

    def _canonicalise_annex_title(self, title: str) -> str:
        m = self.re_annex.match(title or "")
        if not m:
            return title
        kw = m.group(1).lower()
        m_num = re.search(r"\b(\d+|[ivxlcdm]+)\b", title or "", flags=re.I)
        num = m_num.group(1) if m_num else ""
        return f"{kw} {num}".strip()

    def _slug_ascii(self, s: str, maxlen: int = 120) -> str:
        """ASCII-safe, lowercased, dashed slug; trims long strings with md5 tail when needed."""
        s = unicodedata.normalize("NFKD", s or "")
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"[^a-z0-9:]+", "-", s).strip("-")
        s = re.sub(r"-{2,}", "-", s)
        if not s:
            s = "root"
        if len(s) > maxlen:
            h = hashlib.md5(s.encode()).hexdigest()[:8]
            s = s[:maxlen] + "-" + h
        return s

    # ---------------- sectionization ----------------
    def _sectionize(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Detect headings and build sections; DO NOT include the heading line
        inside the body text (heading is injected later). Then merge sections
        with ultra-short bodies (<30 words) into the next section.
        """
        lines = [ln.rstrip() for ln in (text or "").splitlines()]
        sections: List[Tuple[str, Dict[str, Any]]] = []
        cur_lines: List[str] = []
        cur_meta: Dict[str, Any] = {"section_title": "", "section_path": ""}

        def flush():
            nonlocal cur_lines, cur_meta, sections
            if cur_lines:
                blob = "\n".join(cur_lines).strip()
                if blob:
                    sections.append((blob, cur_meta.copy()))
            cur_lines = []

        # HTML mode detection: if we see any H-tag, we only accept headings inside H1–H6
        html_mode = bool(re.search(r"</\s*h[1-6]\s*>", text or "", flags=re.IGNORECASE))

        current_path: List[str] = []
        for idx, ln in enumerate(lines):
            stripped = ln.strip()

            if html_mode:
                # Try to locate a complete H1–H6 line; if present, extract its inner text
                m_h = self._re_html_h.search(stripped)
                if m_h:
                    inner = re.sub(r"\s+", " ", m_h.group(2)).strip()
                    # Heading detection only allowed on the inner heading text
                    if self._looks_heading_textonly(inner):
                        flush()
                        if self.re_annex.match(inner):
                            inner = self._canonicalise_annex_title(inner)
                        current_path = [inner]
                        cur_meta = {
                            "section_title": inner,
                            "section_path": " > ".join(current_path)
                        }
                        # do NOT add the heading itself to body lines
                        continue
                    else:
                        # H-tag but not matching our heading patterns → treat as paragraph text
                        cur_lines.append(ln)
                        continue
                else:
                    # Not an H-tag line → never run heading regexes in HTML mode
                    cur_lines.append(ln)
                    continue
            else:
                # -------- PLAINTEXT MODE --------
                s = stripped

                # First, test for annex with tail safety
                m_ann = self.re_annex.match(s)
                is_heading = False
                if m_ann:
                    tail = s[m_ann.end():]
                    if self._annex_tail_is_heading(tail):
                        if tail.strip() == "":
                            # Bare "Annex 1" at start of a physical line.
                            # Accept as heading only if this starts a *new block*:
                            # - start of document, or
                            # - previous original line is blank, or
                            # - we've just flushed (i.e., cur_lines empty)
                            prev_blank = (idx == 0) or (not lines[idx - 1].strip()) or (len(cur_lines) == 0)
                            if prev_blank:
                                is_heading = True
                            else:
                                # Likely a wrapped inline mention → reject as heading
                                m_ann = None
                        else:
                            # Has colon/dash + title → treat as heading
                            is_heading = True
                    else:
                        m_ann = None  # inline form like "... see Annex 1 to ..."

                # If not annex, test other heading shapes — but only at a *new block*
                if not is_heading:
                    prev_blank = (idx == 0) or (not lines[idx - 1].strip()) or (len(cur_lines) == 0)

                    # Allow chapter/code only at new block; else it's likely a wrapped inline ref
                    if prev_blank and (self.re_chapter.match(s) or self.re_code.match(s)):
                        is_heading = True
                    elif prev_blank and (self.re_letter.match(s) or self.re_roman.match(s)):
                        # optional: also require new-block for letter/roman to avoid line-wrap false positives
                        is_heading = True
                    else:
                        is_heading = False

                if is_heading:
                    flush()
                    if m_ann:
                        s = self._canonicalise_annex_title(s)
                    current_path = [s]
                    cur_meta = {
                        "section_title": s,
                        "section_path": " > ".join(current_path)
                    }
                    # do NOT add the heading itself to body lines
                    continue

                # avoid accidental duplicate of heading as first body line
                if cur_meta.get("section_title") and s == cur_meta["section_title"]:
                    continue
                cur_lines.append(ln)

        flush()

        if not sections:
            return [((text or "").strip(), {"section_title": "", "section_path": ""})]

        # second pass: merge forward ultra-short bodies
        merged: List[Tuple[str, Dict[str, Any]]] = []
        i = 0
        while i < len(sections):
            body, meta = sections[i]
            words = len(body.split())
            if words < 30 and (i + 1) < len(sections):
                next_body, next_meta = sections[i + 1]
                # A+B: only merge forward within same top-level kind (chapter/annex)
                kind_now  = self._top_level_kind(meta.get("section_title", ""))
                kind_next = self._top_level_kind(next_meta.get("section_title", ""))
                same_top  = (kind_now == kind_next) and (kind_now in {"chapter", "annex"})
                if same_top:
                    combined = (body + "\n\n" + next_body).strip()
                    # adopt next section's meta so path/title remain intuitive
                    merged.append((combined, next_meta))
                    i += 2
                    continue
                else:
                    merged.append((body, meta))
                    i += 1
                    continue
            else:
                merged.append((body, meta))
                i += 1

        return merged

    # ---------------- window packing ----------------
    def _pack(self, sentences: List[str], heading: str = "") -> List[List[str]]:
        """
        Pack sentences into token windows with overlap; inject heading once per window.
        Merge tiny windows and windows starting with list/bullet/number into previous.
        """
        windows: List[List[str]] = []
        cur: List[str] = []; cur_tok = 0

        def push():
            nonlocal cur, cur_tok, windows
            if cur:
                windows.append(cur)
            cur = []; cur_tok = 0

        for sent in sentences:
            t = _tok_count(sent)
            if (cur_tok + t) > self.max_tokens and cur:
                push()
            cur.append(sent); cur_tok += t
            if cur_tok >= self.hard_max_tokens:
                push()
        if cur:
            push()

        # overlap + heading injection
        overlapped: List[List[str]] = []
        prev: Optional[List[str]] = None
        for w in windows:
            if prev is None:
                base = w
            else:
                starts_listy = bool(self.re_num_inline.match(w[0])) or w[0].lstrip().startswith(("-", "•"))
                carry = prev[-2:] if starts_listy and len(prev) >= 2 else prev[-1:]
                base = (carry or []) + w

            # enforce hard cap
            while _tok_count(" ".join(base)) > self.hard_max_tokens and len(base) > 1:
                base = base[:-1]

            if self.include_heading and heading:
                candidate = [heading] + base
                while _tok_count(" ".join(candidate)) > self.hard_max_tokens and len(candidate) > 1:
                    candidate = candidate[:-1]
                overlapped.append(candidate)
            else:
                overlapped.append(base)
            prev = w

        # merge tiny/listy windows into previous
        def is_listy_first(sentence: str) -> bool:
            s = sentence.lstrip()
            return s.startswith(("-", "•")) or bool(self.re_num_inline.match(s))

        merged: List[List[str]] = []
        for w in overlapped:
            words = len(" ".join(w).split())
            starts_list = len(w) > 0 and is_listy_first(w[0])
            if merged and (words < self.min_words_merge or starts_list):
                merged[-1].extend(w)
                # still respect hard cap
                while _tok_count(" ".join(merged[-1])) > self.hard_max_tokens and len(merged[-1]) > 1:
                    merged[-1].pop()
            else:
                merged.append(w)

        # final de-dup of heading if body first sentence equals heading
        out: List[List[str]] = []
        for w in merged:
            if not w:
                continue
            if self.strip_heading_duplicate and heading:
                body = w[:]
                if body and body[0].strip() == heading.strip():
                    body = body[1:]
                out.append(body if body else w)
            else:
                out.append(w)
        return out

    # ---------------- public API ----------------
    def split(self, doc_text: str, doc_meta: Dict[str, Any]) -> List[Chunk]:
        """Return chunk list for a single document."""
        sections = self._sectionize(doc_text or "")
        chunks: List[Chunk] = []

        # prefer caller-supplied doc_id; fallback to title or 'doc'
        raw_doc_id = doc_meta.get("doc_id") or doc_meta.get("doc_title") or "doc"

        cidx = 0
        for sec_text, meta in sections:
            heading = meta.get("section_title", "")
            sents = self._sentences(sec_text)
            if not sents:
                continue
            windows = self._pack(sents, heading=heading if heading else "")

            # ASCII-safe bounded IDs (Pinecone requirement)
            base_doc = self._slug_ascii(str(raw_doc_id), 64)
            base_sec = self._slug_ascii(str(meta.get("section_path", "root") or "root"), 200)
            parent_id = f"{base_doc}:{base_sec}"

            parent_meta = {**doc_meta, **meta, "parent_id": parent_id}
            for w in windows:
                text = " ".join(w).strip()
                if not text:
                    continue
                chunk_id = f"{parent_id}:c{cidx}"
                chunks.append(Chunk(
                    id=chunk_id,
                    parent_id=parent_id,
                    text=text,
                    metadata=parent_meta.copy()
                ))
                cidx += 1

        if not chunks and (doc_text or "").strip():
            # single fallback chunk
            base_doc = self._slug_ascii(str(raw_doc_id), 64)
            chunk_id = f"{base_doc}:c0"
            chunks.append(Chunk(
                id=chunk_id,
                parent_id=None,
                text=(doc_text or "").strip(),
                metadata=doc_meta
            ))
        return chunks
