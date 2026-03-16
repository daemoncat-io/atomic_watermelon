from html.parser import HTMLParser
import urllib.request
import urllib.error
import time

# ── Config ────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research corpus builder; respectful crawl; delay=2s)"
}
INDEX_URL = "https://iep.utm.edu/"
IEP_BASE = "https://iep.utm.edu"
OUTPUT_FILE = "datasets/iep_corpus.txt"
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


class EntryIndexParser(HTMLParser):
    """Extracts entry slugs from the IEP index page."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[str] = []
        self._all_hrefs: list[str] = []  # debug

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for attr, val in attrs:
            if attr == "href" and val:
                self._all_hrefs.append(val)
                # IEP entries are https://iep.utm.edu/{slug}/
                # exclude meta pages: /, /about/, /submissions/, etc.
                if (
                    val.startswith(IEP_BASE + "/")
                    and val.rstrip("/") != IEP_BASE
                    and "?" not in val
                    and "#" not in val
                    and "/about" not in val
                    and "/contact" not in val
                    and "/submissions" not in val
                    and "/category/" not in val
                    and "/tag/" not in val
                    and "/page/" not in val
                    and "/author/" not in val
                ):
                    slug = val.rstrip("/").split("/")[-1]
                    if slug:
                        self.entries.append(slug)


class ArticleParser(HTMLParser):
    """
    Extracts clean prose from an IEP entry page.
    IEP is WordPress — body lives in <div class="entry-content">.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_content = False
        self._depth_content = 0
        self._in_skip = False
        self._depth_skip = 0
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
            self._depth_content = 1
            return

        if not self._in_content:
            return

        if tag == "div":
            self._depth_content += 1

        if self._is_skip(tag, attr_dict) and not self._in_skip:
            self._in_skip = True
            self._depth_skip = 1
            return

        if self._in_skip:
            if tag == "div":
                self._depth_skip += 1
            return

        if tag in self.PARA_TAGS:
            self._in_collect = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if not self._in_content:
            return

        if self._in_skip:
            if tag == "div":
                self._depth_skip -= 1
                if self._depth_skip == 0:
                    self._in_skip = False
            return

        if tag == "div":
            self._depth_content -= 1
            if self._depth_content == 0:
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


def parse_entry_index(html: str) -> list[str]:
    parser = EntryIndexParser()
    parser.feed(html)

    if not parser.entries:
        print("  DEBUG — no entries matched. Sample hrefs:")
        for href in parser._all_hrefs[:30]:
            print(f"    {href}")

    seen: set[str] = set()
    unique: list[str] = []
    for slug in parser.entries:
        if slug not in seen:
            seen.add(slug)
            unique.append(slug)
    return unique


def parse_article(html: str) -> str:
    parser = ArticleParser()
    parser.feed(html)
    return "\n".join(parser.paragraphs)


def main() -> None:
    print("Fetching IEP entry index...")
    index_html = fetch(INDEX_URL)
    if not index_html:
        print("Failed to fetch index page.")
        return

    entries = parse_entry_index(index_html)
    print(f"  {len(entries)} entries found\n")

    if not entries:
        return

    collected = 0
    written = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, slug in enumerate(entries):
            url = f"{IEP_BASE}/{slug}/"
            print(f"[{i+1}/{len(entries)}] {slug}", end=" ... ", flush=True)

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
        f"\nDone: {collected / (1024**2):.1f} MB | {written} entries | {skipped} skipped"
    )


if __name__ == "__main__":
    main()
