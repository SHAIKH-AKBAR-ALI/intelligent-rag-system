# WHAT THIS MODULE DOES:
# - Fetches a URL and returns cleaned text and metadata.

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import hashlib
import time
import re
from urllib.parse import urlparse

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rag-agent/1.0)"}

def _safe_get(url: str, timeout: int = 10, max_retries: int = 3) -> str:
    backoff = 1
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return resp.text
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2

def _clean_html_to_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav"]):
        tag.decompose()
    for sel in ["div[class*='ads']", "div[id*='ad']", "aside"]:
        for el in soup.select(sel):
            el.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    blocks = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if len(text) > 20 or el.name.startswith("h"):
            blocks.append(text)
    if not blocks:
        blocks = [soup.get_text(" ", strip=True)]
    text = "\n\n".join(blocks)
    text = re.sub(r"\s+", " ", text).strip()
    return {"title": title, "text": text}

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fetch_and_clean_url(url: str, timeout: int = 10) -> dict:
    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError("URL must include scheme (https://...)")
    html = _safe_get(url, timeout)
    cleaned = _clean_html_to_text(html)
    cleaned["source_url"] = url
    cleaned["crawl_ts"] = datetime.utcnow().isoformat() + "Z"
    cleaned["text_hash"] = _text_hash(cleaned["text"])
    return cleaned
