from html.parser import HTMLParser
import urllib.request
import urllib.error
import urllib.parse
import json
import zlib
import re
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("S2_API_KEY", "")
OUTPUT_FILE = "datasets/phimp_corpus.txt"
MAX_DOC_CHARS = 200_000
MIN_DOC_CHARS = 2_000
REQUEST_DELAY = 1.1  # S2 rate limit: 1 req/sec with key
REQUEST_TIMEOUT = 30

SEARCH_QUERIES = [
    "philosophy of art",
    "aesthetics art theory",
    "what is art ontology",
    "aesthetic experience perception",
    "art representation expression",
    "philosophy of music",
    "philosophy of literature",
    "art interpretation criticism",
    "beauty sublime aesthetic",
    "art emotion imagination",
]

S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "paperId,title,abstract,openAccessPdf,publicationTypes,year"
# ─────────────────────────────────────────────────────────────────────────────


def s2_headers() -> dict:
    h = {"User-Agent": "Mozilla/5.0 (research corpus builder; contact: daemoncat.io)"}
    if API_KEY:
        h["x-api-key"] = API_KEY
    return h


def fetch_json(url: str, params: dict | None = None) -> dict | None:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=s2_headers())
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url}")
        return None
    except Exception as e:
        print(f"  SKIP {url}: {e}")
        return None


def fetch_bytes(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers=s2_headers())
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return resp.read()
    except Exception as e:
        print(f"  SKIP {url}: {e}")
        return None


# ── Minimal PDF text extractor ────────────────────────────────────────────────


def _inflate(data: bytes) -> bytes:
    try:
        return zlib.decompress(data)
    except zlib.error:
        try:
            return zlib.decompress(data, -15)
        except zlib.error:
            return b""


def _decode_pdf_string(s: bytes) -> str:
    """Decode a PDF text string — handles hex <...> and literal (...)."""
    out: list[str] = []
    i = 0
    while i < len(s):
        b = s[i]
        if b == 0x5C:  # backslash escape
            i += 1
            if i < len(s):
                esc = s[i]
                out.append(
                    {
                        0x6E: "\n",
                        0x72: "\r",
                        0x74: "\t",
                        0x28: "(",
                        0x29: ")",
                        0x5C: "\\",
                    }.get(esc, chr(esc))
                )
        else:
            try:
                out.append(chr(b))
            except ValueError:
                pass
        i += 1
    return "".join(out)


def extract_pdf_text(data: bytes) -> str:
    """
    Minimal PDF text extractor.
    Decompress all FlateDecode streams, then pull text from BT...ET blocks.
    Good enough for modern single-column academic PDFs.
    """
    # decompress all flate streams
    streams: list[bytes] = []
    for m in re.finditer(
        rb"stream\r?\n(.*?)\r?\nendstream",
        data,
        re.DOTALL,
    ):
        raw = m.group(1)
        inflated = _inflate(raw)
        if inflated:
            streams.append(inflated)
        else:
            streams.append(raw)

    paragraphs: list[str] = []

    for stream in streams:
        try:
            text = stream.decode("latin-1", errors="ignore")
        except Exception:
            continue

        # extract text from BT...ET blocks
        for block in re.finditer(r"BT\s*(.*?)\s*ET", text, re.DOTALL):
            buf: list[str] = []
            content = block.group(1)

            # Tj — single string
            for m in re.finditer(r"\(([^)]*)\)\s*Tj", content):
                buf.append(_decode_pdf_string(m.group(1).encode("latin-1")))

            # TJ — array of strings and kerning offsets
            for m in re.finditer(r"\[([^\]]*)\]\s*TJ", content):
                inner = m.group(1)
                for sm in re.finditer(r"\(([^)]*)\)", inner):
                    buf.append(_decode_pdf_string(sm.group(1).encode("latin-1")))

            joined = " ".join(buf).strip()
            # basic noise filter: skip blocks that are mostly non-alpha
            alpha = sum(c.isalpha() for c in joined)
            if joined and alpha / max(len(joined), 1) > 0.5:
                paragraphs.append(joined)

    return "\n".join(paragraphs)


# ── S2 search + pagination ────────────────────────────────────────────────────


def search_papers(query: str) -> list[dict]:
    """Paginate through bulk search results for one query."""
    params: dict = {
        "query": query,
        "fields": FIELDS,
        "publicationTypes": "JournalArticle",
        "openAccessPdf": "",  # presence of param = filter for OA PDFs
        "fieldsOfStudy": "Philosophy",
    }

    papers: list[dict] = []
    url = S2_BULK_URL

    while True:
        data = fetch_json(url, params)
        if not data:
            break

        batch = data.get("data", [])
        papers.extend(batch)

        token = data.get("token")
        if not token:
            break

        # next page: token replaces all other params
        params = {"token": token}
        time.sleep(REQUEST_DELAY)

    return papers


def collect_papers() -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []

    for query in SEARCH_QUERIES:
        print(f"  query: {query!r}")
        batch = search_papers(query)
        before = len(unique)
        for p in batch:
            pid = p.get("paperId", "")
            if pid and pid not in seen:
                seen.add(pid)
                unique.append(p)
        print(f"    {len(batch)} results, {len(unique) - before} new")
        time.sleep(REQUEST_DELAY)

    return unique


# ── Per-paper text extraction ─────────────────────────────────────────────────


def get_text(paper: dict) -> str:
    """Try OA PDF first, fall back to abstract."""
    pdf_info = paper.get("openAccessPdf") or {}
    pdf_url = pdf_info.get("url", "")

    if pdf_url:
        pdf_bytes = fetch_bytes(pdf_url)
        if pdf_bytes:
            text = extract_pdf_text(pdf_bytes)
            if len(text) >= MIN_DOC_CHARS:
                return text

    # fallback: abstract
    return paper.get("abstract") or ""


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not API_KEY:
        print("Warning: no S2_API_KEY set — unauthenticated rate limits apply\n")

    print("Collecting papers from Semantic Scholar...")
    papers = collect_papers()
    print(f"\n{len(papers)} unique papers collected\n")

    if not papers:
        return

    collected = 0
    written = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, paper in enumerate(papers):
            pid = paper.get("paperId", "?")
            title = paper.get("title", "untitled")
            year = paper.get("year", "?")
            print(f"[{i+1}/{len(papers)}] {title[:60]}", end=" ... ", flush=True)

            content = get_text(paper)

            if not (MIN_DOC_CHARS <= len(content) <= MAX_DOC_CHARS):
                print(f"skip ({len(content):,} chars)")
                skipped += 1
            else:
                out.write(content + "\n\n")
                if written % 20 == 0:
                    out.flush()
                collected += len(content.encode("utf-8"))
                written += 1
                print(f"ok ({len(content):,} chars)")

            time.sleep(REQUEST_DELAY)

    print(
        f"\nDone: {collected / (1024**2):.1f} MB"
        f" | {written} entries"
        f" | {skipped} skipped"
    )


if __name__ == "__main__":
    main()
