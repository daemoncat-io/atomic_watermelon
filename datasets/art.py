from html.parser import HTMLParser
import urllib.request
import urllib.error
import time

# ── Config ────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research corpus builder; respectful crawl; delay=2s)"
}
TOC_URL = "https://smarthistory.org/smarthistory-table-of-contents/"
OUTPUT_FILE = "datasets/smarthistory_corpus.txt"
BASE_URL = "https://smarthistory.org"
MAX_DOC_CHARS = 200_000
MIN_DOC_CHARS = 2_000
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 2.0
# ─────────────────────────────────────────────────────────────────────────────


def fetch(url: str) -> str | None:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url}")
        return None
    except Exception as e:
        print(f"  SKIP {url}: {e}")
        return None


class TOCParser(HTMLParser):
    """
    Extracts article URLs from Smarthistory's table of contents page.
    TOC links are <a href="https://smarthistory.org/some-article/"> inside
    the main post content — we collect all internal article hrefs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self._in_content = False
        self._content_depth = 0
        self._all_hrefs: list[str] = []  # debug

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        el_class = attr_dict.get("class", "") or ""

        # Smarthistory uses WordPress — article body is div.entry-content
        if tag == "div" and "entry-content" in el_class:
            self._in_content = True
            self._content_depth = 1
            return

        if tag == "div" and self._in_content:
            self._content_depth += 1

        if tag == "a":
            href = attr_dict.get("href", "") or ""
            self._all_hrefs.append(href)
            if (
                self._in_content
                and href.startswith(BASE_URL)
                and href != TOC_URL
                and "?" not in href
                and "#" not in href
                # exclude category/tag/author/page paths
                and "/category/" not in href
                and "/tag/" not in href
                and "/author/" not in href
                and "/page/" not in href
                and href.rstrip("/") != BASE_URL
            ):
                self.urls.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._in_content:
            self._content_depth -= 1
            if self._content_depth == 0:
                self._in_content = False


class ArticleParser(HTMLParser):
    """
    Extracts clean prose from a Smarthistory article page.
    Body lives in <div class="entry-content">.
    Skips: figures, captions, nav, related posts, comments, scripts, styles.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_content = False
        self._content_depth = 0
        self._in_skip = False
        self._skip_depth = 0
        self._in_collect = False
        self._buf: list[str] = []
        self.paragraphs: list[str] = []

    SKIP_TAGS = {
        "figure",
        "figcaption",
        "nav",
        "footer",
        "header",
        "script",
        "style",
        "aside",
    }
    SKIP_CLASSES = {
        "sharedaddy",
        "jp-relatedposts",
        "wpcnt",
        "entry-footer",
        "post-navigation",
        "comments-area",
        "site-footer",
        "related-posts",
    }
    PARA_TAGS = {"p", "h2", "h3", "h4", "blockquote"}

    def _is_skip(self, tag: str, attr_dict: dict) -> bool:
        if tag in self.SKIP_TAGS:
            return True
        el_class = attr_dict.get("class", "") or ""
        return any(sc in el_class for sc in self.SKIP_CLASSES)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        el_class = attr_dict.get("class", "") or ""

        if tag == "div" and "entry-content" in el_class:
            self._in_content = True
            self._content_depth = 1
            return

        if not self._in_content:
            return

        if tag == "div":
            self._content_depth += 1

        if self._in_skip:
            if tag in ("div", "figure", "aside", "nav", "footer"):
                self._skip_depth += 1
            return

        if self._is_skip(tag, attr_dict):
            self._in_skip = True
            self._skip_depth = 1
            return

        if tag in self.PARA_TAGS:
            self._in_collect = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if not self._in_content:
            return

        if self._in_skip:
            if tag in ("div", "figure", "aside", "nav", "footer"):
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._in_skip = False
            return

        if tag == "div":
            self._content_depth -= 1
            if self._content_depth == 0:
                self._in_content = False
            return

        if tag in self.PARA_TAGS and self._in_collect:
            text = " ".join(self._buf).strip()
            if text:
                self.paragraphs.append(text)
            self._in_collect = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_content and self._in_collect and not self._in_skip:
            cleaned = data.strip()
            if cleaned:
                self._buf.append(cleaned)


def parse_toc(html: str) -> list[str]:
    parser = TOCParser()
    parser.feed(html)

    if not parser.urls:
        print("  DEBUG — no article URLs matched. Sample hrefs:")
        for href in parser._all_hrefs[:30]:
            print(f"    {href}")

    seen: set[str] = set()
    unique: list[str] = []
    for url in parser.urls:
        norm = url.rstrip("/")
        if norm not in seen:
            seen.add(norm)
            unique.append(url)
    return unique


def parse_article(html: str) -> str:
    parser = ArticleParser()
    parser.feed(html)
    return "\n".join(parser.paragraphs)


def main() -> None:
    print("Fetching Smarthistory table of contents...")
    toc_html = fetch(TOC_URL)
    if not toc_html:
        print("Failed to fetch TOC page.")
        return

    urls = parse_toc(toc_html)
    print(f"  {len(urls)} article URLs found\n")

    if not urls:
        return

    collected = 0
    written = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, url in enumerate(urls):
            slug = url.rstrip("/").split("/")[-1]
            print(f"[{i+1}/{len(urls)}] {slug}", end=" ... ", flush=True)

            html = fetch(url)
            if not html:
                skipped += 1
                continue

            content = parse_article(html)

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
