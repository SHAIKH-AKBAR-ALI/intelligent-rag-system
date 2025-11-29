# WHAT THIS MODULE DOES:
# - Extractors for uploaded files and URLs.

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pdfplumber
import docx
from src.ingestion.fetch_and_clean import fetch_and_clean_url

def _make_meta(source_name: str, source_type: str, extra: Dict[str, Any]=None) -> Dict[str, Any]:
    meta = {"source_name": source_name, "source_type": source_type, "crawl_ts": datetime.utcnow().isoformat() + "Z"}
    if extra:
        meta.update(extra)
    return meta

def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                texts.append(f"[Page {i+1}]\n{page_text}")
    text = "\n\n".join(texts).strip()
    source_name = os.path.basename(file_path)
    metadata = _make_meta(source_name, "pdf", {"pages": len(texts)})
    return {"source_name": source_name, "source_type": "pdf", "text": text, "metadata": metadata}

def extract_text_from_docx(file_path: str) -> Dict[str, Any]:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"DOCX not found: {file_path}")
    doc = docx.Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs).strip()
    source_name = os.path.basename(file_path)
    metadata = _make_meta(source_name, "docx", {"paragraphs": len(paragraphs)})
    return {"source_name": source_name, "source_type": "docx", "text": text, "metadata": metadata}

def extract_text_from_txt(file_path: str, encoding: str="utf-8") -> Dict[str, Any]:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"TXT not found: {file_path}")
    with open(file_path, "r", encoding=encoding) as f:
        text = f.read()
    source_name = os.path.basename(file_path)
    metadata = _make_meta(source_name, "txt", {"length": len(text)})
    return {"source_name": source_name, "source_type": "txt", "text": text, "metadata": metadata}

def extract_text_from_upload(uploaded_file) -> Dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(tmp_path)
        elif suffix in [".docx", ".doc"]:
            return extract_text_from_docx(tmp_path)
        elif suffix in [".txt", ".md"]:
            return extract_text_from_txt(tmp_path)
        else:
            try:
                return extract_text_from_txt(tmp_path)
            except Exception:
                raise RuntimeError(f"Unsupported uploaded file type: {suffix}")
    finally:
        pass

def extract_text_from_url(url: str) -> Dict[str, Any]:
    res = fetch_and_clean_url(url)
    source_name = res.get("title") or res.get("source_url") or url
    metadata = _make_meta(source_name, "url", {"source_url": res.get("source_url"), "text_hash": res.get("text_hash")})
    return {"source_name": source_name, "source_type": "url", "text": res.get("text", ""), "metadata": metadata}
