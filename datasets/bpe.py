"""
Byte Pair Encoding tokenizer.

A BPE is NOT ML nor should it be construed as one. Its explicit purpose is to
produce a 'legend' for human language the machine can parse — the BPE legend
is what an ML model learns over.

Memory model:
    Streams corpus in chunks → normalize → pre-tokenize → count word frequencies.
    Never holds the full corpus in RAM. Merge learning operates entirely in
    integer-ID space with incremental pair counting.

Usage:
    tok = BPETokenizer(vocab_size=4096)
    tok.train("path/to/corpus.txt")
    tok.save("tokenizer.json")

    ids = tok.encode("Hello world")
    text = tok.decode(ids)
"""

from typing import IO, Iterator
from collections import Counter
from pathlib import Path
import unicodedata
import json
import sys
import re


class BPETokenizer:
    # Special tokens occupy the first slots.
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    # Chunk size for streaming reads. A line longer than this will inflate
    # carryover memory until a '\n' is seen — fine for natural-language
    # corpora, dangerous for pathological inputs.
    CHUNK_SIZE = 8 * 1024 * 1024
    MIN_CHUNK_SIZE = 256

    # 4 specials + 256 bytes.
    BASE_VOCAB_SIZE = len(SPECIAL_TOKENS) + 256
    # Upper bound from the packed-pair representation: pk = (a << 20) | b.
    MAX_VOCAB = 1 << 20

    def __init__(self, vocab_size: int = 4096):
        if vocab_size < self.BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size={vocab_size} below base vocab "
                f"({self.BASE_VOCAB_SIZE} = {len(self.SPECIAL_TOKENS)} specials "
                f"+ 256 bytes)"
            )
        if vocab_size > self.MAX_VOCAB:
            raise ValueError(
                f"vocab_size={vocab_size} exceeds packed-pair limit "
                f"({self.MAX_VOCAB})"
            )

        self.target_vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self._merge_ranks: dict[tuple[str, str], int] = {}

        # GPT-2 style pre-tokenization. `\w+` already covers digits, so no
        # separate `\d+` branch.
        self.split_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+""",
            re.UNICODE,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    # ================================================================
    # NORMALIZATION
    # ================================================================

    _TYPOGRAPHIC_MAP = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "--",
            "\u2026": "...",
            "\u00a0": " ",
            "\u200b": "",
            "\u200c": "",
            "\u200d": "",
            "\ufeff": "",
            "\u00ad": "",
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2015": "--",
            "\u2032": "'",
            "\u2033": '"',
            "\u02bc": "'",
            "\u2060": "",
        }
    )

    _CAP_NEWLINES = re.compile(r"\n{3,}")
    _COLLAPSE_WS = re.compile(r"[^\S\n]+")
    _TRAILING_SPACES = re.compile(r" +\n")

    @staticmethod
    def _is_valid_char(c: str) -> bool:
        if c == "\n" or c == "\t":
            return True
        return not unicodedata.category(c).startswith("C")

    @classmethod
    def _normalize_base(cls, text: str) -> str:
        """
        Per-chunk normalization. Newline-run capping and edge stripping happen
        elsewhere (cross-chunk state / document boundaries).
        """
        text = unicodedata.normalize("NFC", text)
        text = text.translate(cls._TYPOGRAPHIC_MAP)
        text = "".join(c for c in text if cls._is_valid_char(c))
        # \r is already dropped by _is_valid_char, but keep these for clarity.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = cls._COLLAPSE_WS.sub(" ", text)
        text = cls._TRAILING_SPACES.sub("\n", text)
        return text

    @classmethod
    def normalize(cls, text: str) -> str:
        """Full document normalization."""
        text = cls._normalize_base(text)
        text = cls._CAP_NEWLINES.sub("\n\n", text)
        return text.strip()

    # ================================================================
    # STREAMING CORPUS READER
    #
    # Concatenating all yielded chunks is bit-exact with normalize(full_text).
    # Chunks are split on the last newline so the pre-tokenizer never sees a
    # split word. Cross-chunk \n{3,} runs are capped via a running count of
    # trailing newlines in the already-emitted stream.
    # ================================================================

    @classmethod
    def _stream_normalized_chunks(
        cls,
        fh: IO[str],
        chunk_size: int,
    ) -> Iterator[str]:
        if chunk_size < cls.MIN_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size={chunk_size} below minimum {cls.MIN_CHUNK_SIZE}"
            )

        def raw_chunks() -> Iterator[str]:
            carryover = ""
            while True:
                raw = fh.read(chunk_size)
                if not raw:
                    if carryover:
                        yield cls._normalize_base(carryover)
                    return
                buf = carryover + raw
                split = buf.rfind("\n")
                if split == -1:
                    # No newline in this read — accumulate. Memory grows
                    # until a newline shows up.
                    carryover = buf
                    continue
                yield cls._normalize_base(buf[: split + 1])
                carryover = buf[split + 1 :]

        is_first = True
        # Trailing \n count of the already-emitted stream (0, 1, or 2 after capping).
        trailing_nl = 0
        # One-chunk lookahead so we know which chunk is the last (for rstrip).
        pending: str | None = None

        def cap_boundary(chunk: str) -> str:
            """Cap newlines at the chunk boundary and internally."""
            nonlocal trailing_nl
            if not chunk:
                return chunk
            if trailing_nl > 0:
                leading = len(chunk) - len(chunk.lstrip("\n"))
                excess = trailing_nl + leading - 2
                if excess > 0:
                    chunk = chunk[excess:]
            chunk = cls._CAP_NEWLINES.sub("\n\n", chunk)
            # Only refresh trailing_nl if we actually emit something. If the
            # chunk was reduced to "" by the boundary cap, the previously-
            # emitted trailing newlines are still the live boundary state.
            if chunk:
                trailing_nl = min(2, len(chunk) - len(chunk.rstrip("\n")))
            return chunk

        for chunk in raw_chunks():
            if is_first:
                chunk = chunk.lstrip()
                is_first = False
            chunk = cap_boundary(chunk)

            if pending:
                yield pending
            pending = chunk

        if pending:
            tail = pending.rstrip()
            if tail:
                yield tail

    # ================================================================
    # VOCABULARY
    # ================================================================

    def _build_base_vocab(self) -> dict[str, int]:
        """Base vocab: special tokens + all 256 byte values."""
        vocab = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        offset = len(self.SPECIAL_TOKENS)
        for b in range(256):
            vocab[bytes([b]).decode("latin-1")] = b + offset
        return vocab

    # ================================================================
    # PRE-TOKENIZATION
    # ================================================================

    def _pre_tokenize(self, text: str) -> list[str]:
        return self.split_pattern.findall(text)

    @staticmethod
    def _text_to_byte_tokens(word: str) -> list[str]:
        return [bytes([b]).decode("latin-1") for b in word.encode("utf-8")]

    # ================================================================
    # PHASE 1: STREAMING WORD-FREQUENCY ACCUMULATION
    # ================================================================

    def _count_words_streaming(self, fh: IO[str]) -> tuple[Counter[str], int]:
        word_counts: Counter[str] = Counter()
        norm_chars = 0
        for chunk in self._stream_normalized_chunks(fh, self.CHUNK_SIZE):
            norm_chars += len(chunk)
            word_counts.update(self._pre_tokenize(chunk))
        return word_counts, norm_chars

    # ================================================================
    # PHASE 2: MERGE LEARNING — incremental pair counting
    #
    # Operates entirely in integer-ID space. Key invariants:
    #   1. pair_counts[pk] is the total freq-weighted count of pair `pk`
    #      across all current word forms.
    #   2. pair_words[pk] contains every word index that has ever held pair
    #      `pk`. Stale entries (word no longer contains the pair) are
    #      tolerated — they're skipped via the has_pair guard.
    #   3. On each merge, only words in pair_words[best_pk] are rescanned.
    # ================================================================

    @staticmethod
    def _pack_pair(a: int, b: int) -> int:
        """Encode a token pair as a single int. Supports IDs up to 2^20."""
        return (a << 20) | b

    @staticmethod
    def _unpack_pair(p: int) -> tuple[int, int]:
        return (p >> 20, p & 0xFFFFF)

    def _learn_merges(
        self,
        word_counts: Counter[str],
        verbose: bool = True,
        min_frequency: int = 2,
    ) -> None:
        """
        Learn BPE merges from word frequency counts.

        min_frequency: drop word types appearing fewer than this many times
            before merge learning. Default 2 — singletons can never produce a
            pair with count >= 2 (our stop condition), so they're dead weight.
            Typically prunes 50-70% of unique types on large corpora. Set to
            1 for exact equivalence with naive BPE.
        """
        # Reset state.
        self.vocab = self._build_base_vocab()
        self.merges = []

        str_to_id: dict[str, int] = dict(self.vocab)
        id_to_str: dict[int, str] = {v: k for k, v in self.vocab.items()}

        def get_or_add(tok: str) -> int:
            tid = str_to_id.get(tok)
            if tid is None:
                tid = len(str_to_id)
                str_to_id[tok] = tid
                id_to_str[tid] = tok
            return tid

        # Convert word types → int sequences. Drop rare types.
        words: list[list[int]] = []
        freqs: list[int] = []
        pruned_types = 0

        for word_str, count in word_counts.items():
            if count < min_frequency:
                pruned_types += 1
                continue
            byte_tokens = self._text_to_byte_tokens(word_str)
            if not byte_tokens:
                continue
            words.append([str_to_id[t] for t in byte_tokens])
            freqs.append(count)

        # Release the string-based Counter — all data is now in int space.
        word_counts.clear()

        n_words = len(words)
        if verbose:
            print(f"  Word types kept (freq >= {min_frequency}): {n_words:,}")
            print(f"  Word types pruned: {pruned_types:,}")

        # Initial pair counts + reverse index.
        pair_counts: dict[int, int] = {}
        pair_words: dict[int, set[int]] = {}
        for wi, word in enumerate(words):
            freq = freqs[wi]
            for j in range(len(word) - 1):
                pk = self._pack_pair(word[j], word[j + 1])
                pair_counts[pk] = pair_counts.get(pk, 0) + freq
                pair_words.setdefault(pk, set()).add(wi)

        if verbose:
            print(f"  Initial unique pairs: {len(pair_counts):,}")

        n_merges = self.target_vocab_size - len(self.vocab)
        if verbose:
            print(f"  Base vocab: {len(self.vocab)}")
            print(f"  Target merges: {n_merges}")

        # --- Merge loop ---
        for merge_i in range(n_merges):
            if not pair_counts:
                if verbose:
                    print(f"  No more pairs at merge {merge_i}")
                break

            # Best pair: max count, then lex-min (a_str, b_str) as a
            # deterministic tie-break.
            best_pk = -1
            best_count = -1
            best_key: tuple[str, str] = ("", "")
            for pk, cnt in pair_counts.items():
                if cnt < best_count:
                    continue
                a, b = self._unpack_pair(pk)
                key = (id_to_str[a], id_to_str[b])
                if cnt > best_count or key < best_key:
                    best_count = cnt
                    best_pk = pk
                    best_key = key

            if best_count < 2:
                if verbose:
                    print(
                        f"  Stopping at merge {merge_i}: "
                        f"best pair count = {best_count}"
                    )
                break

            a_id, b_id = self._unpack_pair(best_pk)
            a_str, b_str = best_key
            merged_str = a_str + b_str
            merged_id = get_or_add(merged_str)

            self.merges.append((a_str, b_str))
            self.vocab[merged_str] = merged_id

            # Remove the merged pair globally.
            affected = pair_words.pop(best_pk, set())
            del pair_counts[best_pk]

            for wi in affected:
                word = words[wi]
                freq = freqs[wi]

                # Stale pair_words entries are possible (an earlier merge
                # eliminated this pair from this word). Skip cheaply.
                if not any(
                    word[j] == a_id and word[j + 1] == b_id
                    for j in range(len(word) - 1)
                ):
                    continue

                # Decrement counts for every adjacent pair in the OLD word,
                # except the merge pair itself (already removed globally).
                # Correctness over cleverness: we'll re-add new pairs below.
                for j in range(len(word) - 1):
                    pk = self._pack_pair(word[j], word[j + 1])
                    if pk == best_pk:
                        continue
                    new_count = pair_counts[pk] - freq
                    if new_count <= 0:
                        del pair_counts[pk]
                        pw = pair_words.get(pk)
                        if pw is not None:
                            pw.discard(wi)
                            if not pw:
                                del pair_words[pk]
                    else:
                        pair_counts[pk] = new_count

                # Apply merge in-place, left-to-right (non-overlapping).
                new_word: list[int] = []
                j = 0
                while j < len(word):
                    if j + 1 < len(word) and word[j] == a_id and word[j + 1] == b_id:
                        new_word.append(merged_id)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                words[wi] = new_word

                # Re-increment pairs for the NEW word.
                for j in range(len(new_word) - 1):
                    pk = self._pack_pair(new_word[j], new_word[j + 1])
                    pair_counts[pk] = pair_counts.get(pk, 0) + freq
                    pair_words.setdefault(pk, set()).add(wi)

            if verbose and (merge_i + 1) % 500 == 0:
                print(
                    f"  merge {merge_i + 1}/{n_merges}: "
                    f"{repr(a_str)} + {repr(b_str)} -> "
                    f"{repr(merged_str)} (count: {best_count:,}, "
                    f"pairs: {len(pair_counts):,})"
                )

        # Finalize lookups.
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if verbose:
            print(f"\n  Final vocab size: {self.vocab_size}")
            print(f"  Learned {len(self.merges)} merges")

    # ================================================================
    # TRAIN
    # ================================================================

    def train(
        self,
        corpus_path: str,
        verbose: bool = True,
        min_frequency: int = 2,
    ) -> None:
        """
        Learn BPE merges from a text corpus.

        Phase 1: stream → normalize → pre-tokenize → count word frequencies.
        Phase 2: learn merges with incremental pair counting.
        """
        path = Path(corpus_path)
        file_size = path.stat().st_size

        if verbose:
            print(f"Reading: {corpus_path} ({file_size / 1024**2:.1f} MiB)")
            print(f"Chunk size: {self.CHUNK_SIZE / 1024**2:.0f} MiB")
            print("\nPhase 1: Streaming word frequencies...")

        with open(corpus_path, "r", encoding="utf-8") as f:
            word_counts, norm_chars = self._count_words_streaming(f)

        if verbose:
            print(f"  Normalized chars streamed: {norm_chars:,}")
            print(f"  Unique pre-tokens: {len(word_counts):,}")
            total_tokens = sum(word_counts.values())
            print(f"  Total pre-token occurrences: {total_tokens:,}")
            print("\nPhase 2: Learning merges...")

        self._learn_merges(word_counts, verbose=verbose, min_frequency=min_frequency)

    # ================================================================
    # ENCODE / DECODE
    # ================================================================

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """
        Apply learned merges to a single pre-token's byte sequence.

        O(n^2) — at each step we scan for the lowest-rank applicable merge.
        Fine for natural-language word lengths; would need a priority queue
        for very long pre-tokens.
        """
        if len(tokens) <= 1:
            return tokens
        sentinel = len(self.merges)
        while True:
            best_rank = sentinel
            best_i = -1
            for i in range(len(tokens) - 1):
                rank = self._merge_ranks.get((tokens[i], tokens[i + 1]), sentinel)
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break
            tokens = (
                tokens[:best_i]
                + [tokens[best_i] + tokens[best_i + 1]]
                + tokens[best_i + 2 :]
            )
        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token ids. Input is assumed already-normalized."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for chunk in self._pre_tokenize(text):
            for token in self._apply_merges(self._text_to_byte_tokens(chunk)):
                ids.append(self.vocab.get(token, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text. Special tokens are silently dropped."""
        n_specials = len(self.SPECIAL_TOKENS)
        byte_parts: list[bytes] = []
        for token_id in ids:
            if token_id < n_specials:
                continue
            token = self.inverse_vocab.get(token_id)
            if token is None:
                continue
            byte_parts.append(token.encode("latin-1"))
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    # ================================================================
    # PERSISTENCE
    # ================================================================

    def save(self, path: str) -> None:
        data = {
            "target_vocab_size": self.target_vocab_size,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["target_vocab_size"])
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = data["vocab"]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok._merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        return tok

    # ================================================================
    # DIAGNOSTICS
    # ================================================================

    def summary(self) -> None:
        print("BPE Tokenizer")
        print(f"  Vocab size:  {self.vocab_size}")
        print(f"  Merges:      {len(self.merges)}")
        print(f"  Special:     {self.SPECIAL_TOKENS}")

    def compression_ratio(self, text: str) -> float:
        """Bytes per token — lower means more compression."""
        ids = self.encode(text)
        return len(text.encode("utf-8")) / len(ids) if ids else 0.0


if __name__ == "__main__":
    import io

    CORPUS = "datasets/sep_corpus.txt"
    VOCAB_SIZE = 4096
    OUT_PATH = "datasets/tokenizer.json"

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else CORPUS
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else VOCAB_SIZE
    out_path = sys.argv[3] if len(sys.argv) > 3 else OUT_PATH

    if not Path(corpus_path).exists():
        print(f"Corpus not found: {corpus_path}")
        print("Usage: python bpe.py [corpus_path] [vocab_size] [out_path]")
        sys.exit(1)

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus_path)
    tokenizer.summary()

    tokenizer.save(out_path)
    print(f"\nSaved to: {out_path}")

    # --- Validation ---
    with open(corpus_path, "r", encoding="utf-8") as f:
        sample = f.read(2000)

    sample_norm = BPETokenizer.normalize(sample)

    # Roundtrip: decode(encode(x)) == x for normalized text.
    ids = tokenizer.encode(sample_norm)
    decoded = tokenizer.decode(ids)
    roundtrip = decoded == sample_norm

    # Stream-vs-monolithic consistency: streaming the same sample with a
    # small chunk size should produce the same text as normalize() on the
    # whole sample. Forces multiple chunks for the 2000-char sample.
    streamed = "".join(
        BPETokenizer._stream_normalized_chunks(
            io.StringIO(sample), BPETokenizer.MIN_CHUNK_SIZE
        )
    )
    stream_match = streamed == sample_norm

    print("\nValidation (first 2K chars):")
    print(f"  Roundtrip:           {'✓' if roundtrip else '✗'}")
    print(f"  Stream == normalize: {'✓' if stream_match else '✗'}")
    print(f"  Compression: {tokenizer.compression_ratio(sample_norm):.2f} bytes/token")
    print(f"  {len(sample_norm)} chars -> {len(ids)} tokens")

    if not roundtrip:
        for i, (a, b) in enumerate(zip(sample_norm, decoded)):
            if a != b:
                print(f"  First mismatch at char {i}: {repr(a)} vs {repr(b)}")
                print(f"  Context: ...{repr(sample_norm[max(0, i - 20):i + 20])}...")
                break

    if not stream_match:
        for i, (a, b) in enumerate(zip(sample_norm, streamed)):
            if a != b:
                print(f"  Stream mismatch at char {i}: {repr(a)} vs {repr(b)}")
                print(f"  Context: ...{repr(sample_norm[max(0, i - 20):i + 20])}...")
                break
