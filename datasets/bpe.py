"""
Byte Pair Encoding tokenizer — built from scratch.

Learns merge rules from a corpus, encodes text to token ids,
decodes token ids back to text. No dependencies beyond stdlib.

Memory-efficient design:
  Phase 1 — Stream corpus in chunks, normalize, pre-tokenize, count word frequencies.
             Never holds full corpus in RAM.
  Phase 2 — Learn merges using integer token IDs, incremental pair counting, and
             in-place word mutation. Singleton words (freq=1) are dropped before
             merge learning since they can never produce a pair with count >= 2.

Usage:
    tokenizer = BPETokenizer(vocab_size=4096)
    tokenizer.train("path/to/corpus.txt")
    tokenizer.save("tokenizer.json")

    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)

Integration with bridge transformer:
    model = CrossAttentionBridgeTransformer(vocab_size=tokenizer.vocab_size)
"""

from typing import IO, Iterator
from collections import Counter
from pathlib import Path
import unicodedata
import json
import sys
import re


class BPETokenizer:
    # Special tokens occupy the first slots
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    # Chunk size for streaming reads.
    # Must be larger than the longest line in the corpus for bit-exact
    # equivalence with monolithic normalization + pre-tokenization.
    # 8 MiB is safe for any reasonable natural language corpus.
    CHUNK_SIZE = 8 * 1024 * 1024
    MIN_CHUNK_SIZE = 4096

    def __init__(self, vocab_size: int = 4096):
        self.target_vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

        # Pre-tokenization pattern
        self.split_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d"""
            r"""| ?\w+"""
            r"""| ?\d+"""
            r"""| ?[^\s\w]+"""
            r"""|\s+""",
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
    # CORPUS NORMALIZATION
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

    @staticmethod
    def _is_valid_char(c: str) -> bool:
        if c == "\n" or c == "\t":
            return True
        if unicodedata.category(c).startswith("C"):
            return False
        return True

    @classmethod
    def normalize(cls, text: str) -> str:
        """Full document normalization."""
        text = unicodedata.normalize("NFC", text)
        text = text.translate(cls._TYPOGRAPHIC_MAP)
        text = "".join(c for c in text if cls._is_valid_char(c))
        text = re.sub(r"[^\S\n]+", " ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +\n", "\n", text)
        text = text.strip()
        return text

    @classmethod
    def _normalize_chunk_inner(cls, text: str) -> str:
        """Per-chunk normalization — no newline capping (cross-chunk state)."""
        text = unicodedata.normalize("NFC", text)
        text = text.translate(cls._TYPOGRAPHIC_MAP)
        text = "".join(c for c in text if cls._is_valid_char(c))
        text = re.sub(r"[^\S\n]+", " ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r" +\n", "\n", text)
        return text

    # ================================================================
    # STREAMING CORPUS READER
    # ================================================================

    @classmethod
    def _stream_normalized_chunks(
        cls,
        fh: IO[str],
        chunk_size: int,
    ) -> Iterator[str]:
        """
        Yield normalized text chunks from a file handle.

        Splits on newlines so normalization stays line-local.
        Accumulates into carryover when no newline in buffer (line > chunk).
        Newline-run capping is stateful across chunks.
        First/last chunks are stripped to match document-level normalize().
        """
        if chunk_size < cls.MIN_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size={chunk_size} below minimum {cls.MIN_CHUNK_SIZE}. "
                f"Chunk must be larger than the longest line in the corpus "
                f"for correct pre-tokenization."
            )

        carryover = ""
        is_first = True
        prev_chunk: str | None = None
        trailing_newlines = 0

        _cap_re = re.compile(r"\n{3,}")

        def _prepare(text: str) -> str:
            nonlocal trailing_newlines
            if not text:
                return text

            if trailing_newlines > 0:
                leading = 0
                while leading < len(text) and text[leading] == "\n":
                    leading += 1
                combined = trailing_newlines + leading
                if combined > 2:
                    keep = max(0, 2 - trailing_newlines)
                    text = text[leading - keep :] if keep > 0 else text[leading:]

            text = _cap_re.sub("\n\n", text)

            if text:
                trailing_newlines = len(text) - len(text.rstrip("\n"))
            else:
                trailing_newlines = 0
            return text

        while True:
            raw = fh.read(chunk_size)

            if not raw:
                final = ""
                if carryover:
                    final = cls._normalize_chunk_inner(carryover)
                    if is_first:
                        final = final.lstrip()
                    final = _prepare(final)

                if prev_chunk is not None and final:
                    yield prev_chunk
                    final = final.rstrip()
                    if final:
                        yield final
                elif prev_chunk is not None:
                    stripped = prev_chunk.rstrip()
                    if stripped:
                        yield stripped
                elif final:
                    final = final.rstrip()
                    if final:
                        yield final
                break

            buf = carryover + raw
            split_at = buf.rfind("\n")

            if split_at == -1:
                carryover = buf
                continue
            else:
                chunk = cls._normalize_chunk_inner(buf[: split_at + 1])
                carryover = buf[split_at + 1 :]

            if is_first:
                chunk = chunk.lstrip()
                is_first = False

            chunk = _prepare(chunk)

            if prev_chunk is not None:
                yield prev_chunk
            prev_chunk = chunk

    # ================================================================
    # VOCABULARY
    # ================================================================

    def _build_base_vocab(self) -> dict[str, int]:
        """Base vocab: special tokens + all 256 byte values."""
        vocab = {}
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            vocab[tok] = i

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
    def _text_to_byte_tokens(word: str) -> tuple[str, ...]:
        return tuple(bytes([b]).decode("latin-1") for b in word.encode("utf-8"))

    # ================================================================
    # STREAMING WORD FREQUENCY ACCUMULATION
    # ================================================================

    def _count_words_streaming(
        self,
        fh: IO[str],
        verbose: bool = False,
    ) -> tuple[Counter[str], int, int]:
        word_counts: Counter[str] = Counter()
        raw_chars = 0
        norm_chars = 0

        for chunk in self._stream_normalized_chunks(fh, self.CHUNK_SIZE):
            norm_chars += len(chunk)
            tokens = self._pre_tokenize(chunk)
            word_counts.update(tokens)

        return word_counts, raw_chars, norm_chars

    # ================================================================
    # MERGE LEARNING — MEMORY-EFFICIENT
    #
    # Key optimizations over naive approach:
    #   1. Integer token IDs — Python caches ints 0..256 as singletons,
    #      so tuple[int,...] costs ~(56 + 8*len) bytes vs
    #      tuple[str,...] at ~(56 + 8*len + 50*len) bytes.
    #   2. Singleton pruning — words with freq=1 can never produce a
    #      pair with count >= 2 (our stop condition). Dropping them
    #      typically removes 50-70% of unique word types.
    #   3. Incremental pair counting — on each merge, only the words
    #      containing the merged pair are updated. Pair counts are
    #      adjusted in O(affected_words * avg_word_len) instead of
    #      O(total_words * avg_word_len).
    #   4. In-place word mutation — no dict copies per merge.
    #   5. Packed pair keys — (a << 20) | b encodes a pair as a single
    #      int for cheaper hashing and comparison.
    # ================================================================

    @staticmethod
    def _pack_pair(a: int, b: int) -> int:
        """Encode a token pair as a single int. Supports IDs up to 2^20 (~1M)."""
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
        Phase 2: learn BPE merges from word frequency counts.

        Operates entirely in integer-ID space. Maintains pair counts
        incrementally — only words affected by each merge are rescanned.

        min_frequency: drop word types with frequency below this threshold.
            Default 2 — singleton words (freq=1) can never produce a pair
            with count >= 2, so they're dead weight. On a 1GB corpus this
            typically prunes 50-70% of unique types. Set to 1 to include
            all words (matches naive BPE exactly but uses more memory).
        """
        # --- Build base vocab and string↔id mappings ---
        self.vocab = self._build_base_vocab()
        self.merges = []

        str_to_id: dict[str, int] = dict(self.vocab)
        id_to_str: dict[int, str] = {v: k for k, v in self.vocab.items()}

        def get_or_add_id(token_str: str) -> int:
            """Get existing ID or assign next sequential ID."""
            tid = str_to_id.get(token_str)
            if tid is None:
                tid = len(str_to_id)
                str_to_id[token_str] = tid
                id_to_str[tid] = token_str
            return tid

        # --- Convert words to int sequences, drop singletons ---
        # Parallel arrays: words[i] is the token sequence, freqs[i] is its count
        words: list[list[int]] = []
        freqs: list[int] = []
        singleton_count = 0
        singleton_tokens = 0

        for word_str, count in word_counts.items():
            byte_tokens = self._text_to_byte_tokens(word_str)
            if not byte_tokens:
                continue
            if count < min_frequency:
                singleton_count += 1
                singleton_tokens += len(byte_tokens)
                continue
            int_tokens = [str_to_id[t] for t in byte_tokens]
            words.append(int_tokens)
            freqs.append(count)

        n_words = len(words)

        # Release the string-based Counter — all data is now in int-space
        word_counts.clear()

        if verbose:
            print(f"  Word types after pruning (freq < {min_frequency}): {n_words:,}")
            print(
                f"  Pruned {singleton_count:,} low-frequency types "
                f"({singleton_tokens:,} byte tokens)"
            )

        # --- Build initial pair counts and reverse index ---
        # pair_counts: packed_pair → total weighted count
        # pair_words:  packed_pair → set of word indices containing this pair
        pair_counts: dict[int, int] = {}
        pair_words: dict[int, set[int]] = {}

        for wi in range(n_words):
            word = words[wi]
            freq = freqs[wi]
            for j in range(len(word) - 1):
                pk = self._pack_pair(word[j], word[j + 1])
                pair_counts[pk] = pair_counts.get(pk, 0) + freq
                if pk not in pair_words:
                    pair_words[pk] = set()
                pair_words[pk].add(wi)

        if verbose:
            print(f"  Initial unique pairs: {len(pair_counts):,}")

        # --- Merge loop ---
        n_merges = self.target_vocab_size - len(self.vocab)

        if verbose:
            print(f"  Base vocab: {len(self.vocab)}")
            print(f"  Target merges: {n_merges}")

        for merge_i in range(n_merges):
            # Find best pair
            if not pair_counts:
                if verbose:
                    print(f"  No more pairs at merge {merge_i}")
                break

            # Find best pair — deterministic tie-breaking:
            # max count first, then lexicographic on (a_str, b_str) for ties.
            # This matches Counter.most_common()'s insertion-order bias when
            # we insert in the same order as the naive approach.
            best_pk = -1
            best_count = -1
            best_key: tuple[str, str] = ("", "")

            for pk, cnt in pair_counts.items():
                if cnt > best_count:
                    best_count = cnt
                    best_pk = pk
                    a, b = self._unpack_pair(pk)
                    best_key = (id_to_str[a], id_to_str[b])
                elif cnt == best_count:
                    a, b = self._unpack_pair(pk)
                    key = (id_to_str[a], id_to_str[b])
                    if key < best_key:
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
            a_str = id_to_str[a_id]
            b_str = id_to_str[b_id]
            merged_str = a_str + b_str
            merged_id = get_or_add_id(merged_str)

            # Record merge (in string form for public API)
            self.merges.append((a_str, b_str))
            self.vocab[merged_str] = merged_id

            # --- Update affected words ---
            affected = pair_words.pop(best_pk, set())
            del pair_counts[best_pk]

            for wi in affected:
                word = words[wi]
                freq = freqs[wi]

                # Decrement pair counts for current adjacencies involving
                # any position where the merge will fire, plus their neighbors.
                # Then apply merge in-place, then increment new adjacencies.

                # Find all positions where merge fires
                merge_positions: list[int] = []
                j = 0
                while j < len(word) - 1:
                    if word[j] == a_id and word[j + 1] == b_id:
                        merge_positions.append(j)
                        j += 2  # skip — can't overlap
                    else:
                        j += 1

                if not merge_positions:
                    # Pair was in this word before but earlier merges in this
                    # batch already eliminated it. Skip.
                    continue

                # Collect positions whose pair counts need adjustment.
                # For a merge at position p, affected pairs are:
                #   (word[p-1], word[p])   — left neighbor, if p > 0
                #   (word[p], word[p+1])   — the pair itself (already removed above)
                #   (word[p+1], word[p+2]) — right neighbor, if p+2 < len
                # But we also need to handle consecutive merges carefully.

                # Decrement ALL current pairs in this word from counts/index.
                # This is simpler and still O(word_len). Correctness > cleverness.
                for j in range(len(word) - 1):
                    pk = self._pack_pair(word[j], word[j + 1])
                    if pk == best_pk:
                        continue  # already removed globally
                    pc = pair_counts.get(pk)
                    if pc is not None:
                        new_count = pc - freq
                        if new_count <= 0:
                            del pair_counts[pk]
                            pw = pair_words.get(pk)
                            if pw is not None:
                                pw.discard(wi)
                                if not pw:
                                    del pair_words[pk]
                        else:
                            pair_counts[pk] = new_count
                            # Keep word in pair_words — will re-add if still present

                # Apply merge in-place
                new_word: list[int] = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == a_id and word[j + 1] == b_id:
                        new_word.append(merged_id)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                words[wi] = new_word

                # Re-increment pairs for the new word
                for j in range(len(new_word) - 1):
                    pk = self._pack_pair(new_word[j], new_word[j + 1])
                    pair_counts[pk] = pair_counts.get(pk, 0) + freq
                    if pk not in pair_words:
                        pair_words[pk] = set()
                    pair_words[pk].add(wi)

            if verbose and (merge_i + 1) % 500 == 0:
                print(
                    f"  merge {merge_i + 1}/{n_merges}: "
                    f"{repr(a_str)} + {repr(b_str)} -> "
                    f"{repr(merged_str)} (count: {best_count:,}, "
                    f"pairs: {len(pair_counts):,})"
                )

        # Finalize
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

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

        Phase 1: Stream → normalize → pre-tokenize → count word frequencies.
        Phase 2: Learn merges with incremental pair counting.

        min_frequency: word types appearing fewer than this many times are
            dropped before merge learning. Default 2 saves ~50-70% of memory
            on large corpora. Set to 1 for exact equivalence with naive BPE.
        """
        path = Path(corpus_path)
        file_size = path.stat().st_size

        if verbose:
            print(f"Reading: {corpus_path} ({file_size / 1024**2:.1f} MiB)")
            print(f"Chunk size: {self.CHUNK_SIZE / 1024**2:.0f} MiB")

        # Phase 1
        if verbose:
            print("\nPhase 1: Streaming word frequencies...")

        with open(corpus_path, "r", encoding="utf-8") as f:
            word_counts, _, norm_chars = self._count_words_streaming(f, verbose)

        if verbose:
            print(f"  Normalized chars streamed: {norm_chars:,}")
            print(f"  Unique pre-tokens: {len(word_counts):,}")
            total_tokens = sum(word_counts.values())
            print(f"  Total pre-token occurrences: {total_tokens:,}")

        # Phase 2
        if verbose:
            print("\nPhase 2: Learning merges...")

        self._learn_merges(word_counts, verbose=verbose, min_frequency=min_frequency)

    # ================================================================
    # ENCODE / DECODE
    # ================================================================

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """Apply learned merges to a list of byte tokens."""
        for a, b in self.merges:
            merged = a + b
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            if len(tokens) == 1:
                break
        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token ids."""
        ids: list[int] = []

        if add_bos:
            ids.append(self.bos_id)

        chunks = self._pre_tokenize(text)
        for chunk in chunks:
            byte_tokens = list(
                bytes([b]).decode("latin-1") for b in chunk.encode("utf-8")
            )
            merged = self._apply_merges(byte_tokens)
            for token in merged:
                ids.append(self.vocab.get(token, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        byte_parts: list[bytes] = []
        for token_id in ids:
            token = self.inverse_vocab.get(token_id, "")
            if token in self.SPECIAL_TOKENS:
                continue
            byte_parts.append(token.encode("latin-1"))
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    # ================================================================
    # PERSISTENCE
    # ================================================================

    def save(self, path: str) -> None:
        """Save tokenizer state to JSON."""
        data = {
            "target_vocab_size": self.target_vocab_size,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(vocab_size=data["target_vocab_size"])
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = data["vocab"]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        return tok

    # ================================================================
    # DIAGNOSTICS
    # ================================================================

    def summary(self) -> None:
        print(f"BPE Tokenizer")
        print(f"  Vocab size:  {self.vocab_size}")
        print(f"  Merges:      {len(self.merges)}")
        print(f"  Special:     {self.SPECIAL_TOKENS}")

    def compression_ratio(self, text: str) -> float:
        """Bytes per token — lower means more compression."""
        ids = self.encode(text)
        return len(text.encode("utf-8")) / len(ids) if ids else 0.0


if __name__ == "__main__":
    CORPUS = "sep_corpus.txt"
    VOCAB_SIZE = 4096
    OUT_PATH = "datasets/tokenizer.json"

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else CORPUS
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else VOCAB_SIZE
    out_path = sys.argv[3] if len(sys.argv) > 3 else OUT_PATH

    if not Path(corpus_path).exists():
        print(f"Corpus not found: {corpus_path}")
        print(f"Usage: python bpe.py [corpus_path] [vocab_size] [out_path]")
        sys.exit(1)

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(corpus_path)
    tokenizer.summary()

    tokenizer.save(out_path)
    print(f"\nSaved to: {out_path}")

    # Roundtrip validation
    with open(corpus_path, "r") as f:
        sample = f.read(2000)

    sample_norm = BPETokenizer.normalize(sample)
    ids = tokenizer.encode(sample_norm)
    decoded = tokenizer.decode(ids)
    roundtrip = decoded == sample_norm

    print(f"\nValidation (first 2K chars, normalized):")
    print(f"  Roundtrip:   {'✓' if roundtrip else '✗'}")
    print(f"  Compression: {tokenizer.compression_ratio(sample_norm):.2f} bytes/token")
    print(f"  {len(sample_norm)} chars -> {len(ids)} tokens")

    if not roundtrip:
        for i, (a, b) in enumerate(zip(sample_norm, decoded)):
            if a != b:
                print(f"  First mismatch at char {i}: {repr(a)} vs {repr(b)}")
                print(f"  Context: ...{repr(sample_norm[max(0,i-20):i+20])}...")
                break
