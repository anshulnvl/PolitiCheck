import re
import requests
import ftfy
import langdetect
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from newspaper import Article


@dataclass
class PreprocessedDocument:
    title: str = ""
    body: str = ""
    author: str = ""
    publish_date: Optional[str] = None
    source_domain: str = ""
    language: str = ""
    word_count: int = 0
    error: Optional[str] = None


def process(input_text: str) -> PreprocessedDocument:
    """
    Entry point. Accepts either a URL or raw text.
    Returns a cleaned PreprocessedDocument ready for the signal pipeline.
    """
    if _is_url(input_text.strip()):
        return _process_url(input_text.strip())
    else:
        return _process_raw_text(input_text)


# ─── URL path ─────────────────────────────────────────────────────────────────

def _process_url(url: str) -> PreprocessedDocument:
    domain = urlparse(url).netloc.replace("www.", "")

    # 1. Try newspaper3k first (handles most news sites)
    try:
        article = Article(url)
        article.download()
        article.parse()

        title = article.title or ""
        body  = article.text or ""
        author = ", ".join(article.authors) if article.authors else ""
        pub_date = str(article.publish_date.date()) if article.publish_date else None

        # If newspaper got almost nothing, fall back to BS4
        if len(body.strip()) < 200:
            raise ValueError("newspaper3k returned insufficient content")

        return _build_doc(title, body, author, pub_date, domain)

    except Exception:
        pass   # fall through to BeautifulSoup scrape

    # 2. Fallback: raw requests + BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")

        title = soup.title.string if soup.title else ""
        # Remove nav, footer, ads, scripts
        for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
            tag.decompose()

        body = " ".join(soup.stripped_strings)
        return _build_doc(title, body, "", None, domain)

    except Exception as e:
        return PreprocessedDocument(source_domain=domain, error=f"Scrape failed: {e}")


# ─── Raw text path ────────────────────────────────────────────────────────────

def _process_raw_text(text: str) -> PreprocessedDocument:
    return _build_doc(title="", body=text, author="", pub_date=None, domain="")


# ─── Shared cleaning + validation ─────────────────────────────────────────────

def _build_doc(title: str, body: str, author: str,
               pub_date: Optional[str], domain: str) -> PreprocessedDocument:
    # Fix encoding issues (smart quotes, mojibake, etc.)
    title = ftfy.fix_text(title or "").strip()
    body  = ftfy.fix_text(body  or "").strip()

    # Strip residual HTML tags
    body = _strip_html(body)

    # Collapse whitespace
    body = re.sub(r'\s+', ' ', body).strip()

    # Language detection — reject non-English for v1
    lang = _detect_language(body)
    if lang != "en":
        return PreprocessedDocument(
            source_domain=domain,
            language=lang,
            error=f"Non-English content detected ({lang}). v1 supports English only."
        )

    return PreprocessedDocument(
        title=title,
        body=body,
        author=author,
        publish_date=pub_date,
        source_domain=domain,
        language=lang,
        word_count=len(body.split()),
    )


def _strip_html(text: str) -> str:
    """Remove any remaining HTML tags from text."""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def _detect_language(text: str) -> str:
    try:
        sample = text[:1000]   # langdetect only needs a sample
        return langdetect.detect(sample)
    except Exception:
        return "unknown"


def _is_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")