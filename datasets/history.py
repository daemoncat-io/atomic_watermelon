from html.parser import HTMLParser
import urllib.request
import urllib.error
import time

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "datasets/worldhistory_corpus.txt"
BASE_URL = "https://www.worldhistory.org"
INDEX_URL = "https://www.worldhistory.org/article/"
REQUEST_DELAY = 2.0
MIN_DOC_CHARS = 2_000
MAX_DOC_CHARS = 200_000
REQUEST_TIMEOUT = 30
# WHE paginates the article index — scrape up to this many pages
MAX_INDEX_PAGES = 100
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research corpus builder; respectful crawl; delay=2s)"
}
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


class ArticleIndexParser(HTMLParser):
    """
    Extracts article hrefs from a WHE article-listing page.
    Links follow the pattern /article/<id>/<slug>/
    """

    def __init__(self) -> None:
        super().__init__()
        self.article_urls: list[str] = []
        self._all_hrefs: list[str] = []  # debug

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_dict = dict(attrs)
        href = attr_dict.get("href", "") or ""
        self._all_hrefs.append(href)

        # article paths look like /article/1234/slug-here/
        # skip the bare index page itself
        if href.startswith("/article/") and href.count("/") >= 3:
            full = BASE_URL + href if not href.startswith("http") else href
            self.article_urls.append(full)


class ArticleParser(HTMLParser):
    """
    Extracts clean prose from a WHE article page.
    Body lives in <div class="content-text"> (primary) or
    <article> / <div class="article-details"> (fallback).
    Skips: aside, figure, nav, footer, header, related, scripts, styles.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_body = False
        self._body_depth = 0
        self._in_skip = False
        self._skip_depth = 0
        self._in_collect = False
        self._buf: list[str] = []
        self.paragraphs: list[str] = []

    BODY_CLASSES = {
        "content-text",
        "article-body",
        "article-content",
        "entry-body",
        "definition-body",
    }
    SKIP_TAGS = {
        "aside",
        "figure",
        "figcaption",
        "nav",
        "footer",
        "header",
        "script",
        "style",
    }
    SKIP_CLASSES = {
        "related-content",
        "article-sidebar",
        "breadcrumb",
        "social-share",
        "newsletter",
        "advertisement",
        "author-bio",
        "cite-this",
        "license",
    }
    PARA_TAGS = {"p", "h2", "h3", "h4", "blockquote"}

    def _is_body(self, tag: str, attr_dict: dict) -> bool:
        if tag not in ("div", "section", "article"):
            return False
        el_class = attr_dict.get("class", "") or ""
        return any(bc in el_class for bc in self.BODY_CLASSES)

    def _is_skip(self, tag: str, attr_dict: dict) -> bool:
        if tag in self.SKIP_TAGS:
            return True
        el_class = attr_dict.get("class", "") or ""
        el_id = attr_dict.get("id", "") or ""
        return any(sc in el_class or sc in el_id for sc in self.SKIP_CLASSES)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)

        if not self._in_body and self._is_body(tag, attr_dict):
            self._in_body = True
            self._body_depth = 1
            return

        if not self._in_body:
            return

        if tag in ("div", "section", "article"):
            self._body_depth += 1

        if self._in_skip:
            if tag in ("div", "section", "article", "aside", "figure"):
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
        if not self._in_body:
            return

        if self._in_skip:
            if tag in ("div", "section", "article", "aside", "figure"):
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._in_skip = False
            return

        if tag in ("div", "section", "article"):
            self._body_depth -= 1
            if self._body_depth == 0:
                self._in_body = False
            return

        if tag in self.PARA_TAGS and self._in_collect:
            text = " ".join(self._buf).strip()
            if text:
                self.paragraphs.append(text)
            self._in_collect = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_body and self._in_collect and not self._in_skip:
            cleaned = data.strip()
            if cleaned:
                self._buf.append(cleaned)


def collect_article_urls() -> list[str]:
    """
    Paginate through WHE's article index to collect all article URLs.
    WHE paginates as /article/?page=2, /article/?page=3, etc.
    """
    seen: set[str] = set()
    unique: list[str] = []

    for page in range(1, MAX_INDEX_PAGES + 1):
        if page == 1:
            url = INDEX_URL
        else:
            url = f"{INDEX_URL}?page={page}"

        print(f"  Fetching index page {page}...", flush=True)
        html = fetch(url)
        if not html:
            break

        parser = ArticleIndexParser()
        parser.feed(html)

        if not parser.article_urls:
            if page == 1:
                print("  DEBUG — no article URLs matched on first page. Sample hrefs:")
                for href in parser._all_hrefs[:30]:
                    print(f"    {href}")
            else:
                print(f"  No more articles at page {page}, stopping index crawl.")
            break

        new_this_page = 0
        for u in parser.article_urls:
            norm = u.rstrip("/")
            if norm not in seen:
                seen.add(norm)
                unique.append(u)
                new_this_page += 1

        print(f"    +{new_this_page} new URLs (total: {len(unique)})")

        # if a full page added zero new URLs we've looped — stop
        if new_this_page == 0:
            break

        time.sleep(REQUEST_DELAY)

    return unique


def parse_article(html: str) -> str:
    parser = ArticleParser()
    parser.feed(html)
    return "\n".join(parser.paragraphs)


def main() -> None:
    print("Collecting World History Encyclopedia article URLs...")
    urls = collect_article_urls()
    print(f"\n  {len(urls)} article URLs found\n")

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
