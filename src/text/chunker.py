# WHAT THIS MODULE DOES:
# - Splits long text into overlapping chunks.

import re
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!\n])\s+')

def split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text_smart(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    sentences = split_sentences(text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_size:
            current = (current + " " + s).strip()
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    if overlap <= 0:
        return chunks
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            tail = overlapped[-1][-overlap:]
            overlapped.append((tail + " " + chunk).strip())
    return overlapped
