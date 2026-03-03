
from src.chunker.hier_chunker import HierarchicalChunker

cfg = {
  "max_tokens": 120,
  "hard_max_tokens": 180,
  "base_overlap_tokens": 30,
  "max_overlap_tokens": 80,
  "include_heading_in_child": True,
  "patterns": {
    "chapter": r"^(?P<chapter>\d+(?:\.\d+)*)",
    "section_letter": r"^(?P<section>[a-z])\.",
    "annex": r"(?i)^(annex|appendix)\s+[ivxlcdm0-9]+",
    "roman_list": r"^(?P<roman>[ivxlcdm]+)\.",
    "numbered_list_inline": r"(?<!\d)(?P<num>\d{1,2})\.\s+",
  }
}
chunker = HierarchicalChunker(cfg)

doc = '''1 Introduction
This is the intro. It has two sentences.

1.1 Scope
This section contains bullets:
1. First item. It explains something.
2. Second item. More text here.
- A dash bullet continues the list.
Annex I
More annex text here.
'''
chunks = chunker.split(doc, {"doc_id": "demo", "doc_title":"Demo"})
print(f"chunks: {len(chunks)}")
for c in chunks[:5]:
    print('---')
    print(c.id, c.metadata.get("section_path"))
    print(c.text[:140].replace("\n"," "))
