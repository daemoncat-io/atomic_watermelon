from html.parser import HTMLParser
import urllib.request
import urllib.error
import time

# ── Config ────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research corpus builder; respectful crawl; delay=2s)"
}
CONTENTS_URL = "https://plato.stanford.edu/contents.html"
SEP_BASE = "https://plato.stanford.edu"
OUTPUT_FILE = "datasets/sep_corpus.txt"
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
    """Extracts entry slugs from the SEP contents page."""

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
                # match any href that contains "entries/" — catches
                # /entries/slug/, entries/slug/, ../entries/slug/, etc.
                if "entries/" in val:
                    # extract the slug — last non-empty path segment
                    slug = val.rstrip("/").split("/")[-1]
                    if slug and not slug.startswith("?") and not slug.startswith("#"):
                        self.entries.append(slug)


class ArticleParser(HTMLParser):
    """
    Extracts clean prose from a SEP entry page.
    Article body lives in <div id="main-text">.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_main = False
        self._in_skip = False
        self._in_collect = False
        self._depth_main = 0
        self._depth_skip = 0
        self.paragraphs: list[str] = []
        self._buf: list[str] = []

    SKIP_IDS = {
        "bibliography",
        "notes",
        "related-entries",
        "academic-tools",
        "other-internet-resources",
    }
    SKIP_TAGS = {"nav", "footer", "header", "script", "style"}
    PARA_TAGS = {"p", "h2", "h3", "h4"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        el_id = attr_dict.get("id", "")

        if tag == "div" and el_id == "main-text":
            self._in_main = True
            self._depth_main = 1
            return

        if not self._in_main:
            return

        if tag == "div":
            self._depth_main += 1

        if (tag in self.SKIP_TAGS or el_id in self.SKIP_IDS) and not self._in_skip:
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
        if not self._in_main:
            return

        if self._in_skip:
            if tag == "div":
                self._depth_skip -= 1
                if self._depth_skip == 0:
                    self._in_skip = False
            return

        if tag == "div":
            self._depth_main -= 1
            if self._depth_main == 0:
                self._in_main = False
            return

        if tag in self.PARA_TAGS and self._in_collect:
            text = " ".join(self._buf).strip()
            if text:
                self.paragraphs.append(text)
            self._in_collect = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_main and self._in_collect and not self._in_skip:
            cleaned = data.strip()
            if cleaned:
                self._buf.append(cleaned)


def parse_entry_index(html: str) -> list[str]:
    parser = EntryIndexParser()
    parser.feed(html)

    # debug: if nothing found, show a sample of what hrefs actually exist
    if not parser.entries:
        print("  DEBUG — no entries matched. Sample of hrefs found on page:")
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
    print("Fetching SEP entry index...")
    index_html = fetch(CONTENTS_URL)
    if not index_html:
        print("Failed to fetch contents page.")
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
            url = f"{SEP_BASE}/entries/{slug}/"
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
