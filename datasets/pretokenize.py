# datasets/pretokenize.py — run once

from pathlib import Path
from tqdm import tqdm
import numpy as np

from bpe import BPETokenizer


def pretokenize(
    corpus_path: str,
    out_path: str,
    tok: BPETokenizer,
) -> None:
    path = Path(corpus_path)
    print(f"Corpus:    {corpus_path} ({path.stat().st_size / 1024**2:.1f} MB)")
    print(f"Tokenizer: vocab={tok.vocab_size}, merges={len(tok.merges)}")

    text = path.read_text(encoding="utf-8", errors="ignore")

    with tqdm(total=1, desc="Normalizing", unit="corpus") as bar:
        text = BPETokenizer.normalize(text)
        bar.update(1)

    print(f"Normalized: {len(text):,} chars ({len(text) / 1024**2:.1f} MB)")

    pretokens = tok._pre_tokenize(text)
    ids: list[int] = []

    for word in tqdm(pretokens, desc="Encoding", unit="tok", unit_scale=True):
        byte_tokens = [bytes([b]).decode("latin-1") for b in word.encode("utf-8")]
        merged = tok._apply_merges(byte_tokens)
        for token in merged:
            ids.append(tok.vocab.get(token, tok.unk_id))

    print(f"Total tokens: {len(ids):,} ({len(ids) / len(text):.4f} t/c)")

    max_id = max(ids)
    dtype = np.uint16 if max_id <= 65535 else np.int32
    print(f"Max token ID: {max_id} → dtype: {dtype.__name__}")

    if dtype == np.int32:
        print("WARNING: uint16 overflow — update TokenDataset dtype to np.int32")

    arr = np.array(ids, dtype=dtype)
    arr.tofile(out_path)
    print(f"Written: {out_path} ({arr.nbytes / 1024**2:.1f} MB)")


if __name__ == "__main__":
    tok = BPETokenizer.load("datasets/tokenizer.json")
    pretokenize(
        corpus_path="datasets/sep_corpus.txt",
        out_path="datasets/sep_corpus.bin",
        tok=tok,
    )
