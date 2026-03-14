"""
Training Loop for Cross Attention Bridge Transformer
Shared encoder weights via adapters. Cross-attention (no skip gates). Compressive memory.
BPE tokenized input via datasets/tokenizer.json.
"""

from torch.utils.data import DataLoader, Dataset
from typing import Any
from tqdm import tqdm
import numpy as np
import hashlib
import random
import torch
import mmap
import time

from models.atomic_watermelon import AtomicWatermelon
from datasets.bpe import BPETokenizer
from models.logger import TrainingLog


# Load tokenizer once at module level
tokenizer = BPETokenizer.load("datasets/tokenizer.json")

with open("datasets/sample_prompts.txt") as f:
    PROMPT_POOL = [line.strip() for line in f if line.strip()]

FIXED_PROMPTS = {
    "recall_lovelace": "Lovelace completed her paper. Sixty-six pages. ",
    "prediction_sequence": "If one then two then three then ",
    "recall_reality": "Reality is frequently ",
}


config: dict[str, Any] = {
    # Data
    "dataset": "datasets/sep_corpus.txt",
    "vocab_size": tokenizer.vocab_size,
    "context_length": 4096,
    # Model
    "model": "cross_attention",
    "adapter_bottleneck": 128,
    "compress_chunk": 128,
    "memory_slots": 32,
    "d_model": 512,
    "dropout": 0.2,
    "n_layers": 6,
    "n_heads": 4,
    "d_ff": 2048,
    # Training
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "learning_rate": 3e-4,
    "mx_divergence": 1.5,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "batch_size": 1,
    "patience": 25,
    "epochs": 500,
    # Memory handling
    "detach_memory_grad": True,  # False = compress expectation + delta into memory
    "memory_shards": 1,
}


class TextDataset(Dataset):
    """
    Memory-maps the raw corpus. Encodes each chunk at __getitem__ time.
    """

    def __init__(
        self,
        corpus_path: str,
        tok: BPETokenizer,
        context_length: int,
        n_segments: int,
    ):
        self.tok = tok
        self.vocab_size = tok.vocab_size
        self.n_segments = n_segments
        self.segment_length = context_length // n_segments
        self.tokens_needed = self.segment_length * n_segments + 1
        self.chunk_bytes = self.tokens_needed * 6

        self._fh = open(corpus_path, "rb")
        self.mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mm)
        self.n_chunks = self.file_size // self.chunk_bytes
        self.hash = hashlib.sha256(self.mm[:1000]).hexdigest()[:16]

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_bytes
        raw_bytes = self.mm[start : start + self.chunk_bytes]
        text = raw_bytes.decode("utf-8", errors="ignore")
        text = BPETokenizer.normalize(text)
        ids = self.tok.encode(text)
        ids = ids[: self.tokens_needed]

        if len(ids) < self.tokens_needed:
            ids = ids + [self.tok.pad_id] * (self.tokens_needed - len(ids))

        t = torch.tensor(ids, dtype=torch.long)

        x_segs, y_segs = [], []
        for i in range(self.n_segments):
            s = i * self.segment_length
            x_segs.append(t[s : s + self.segment_length])
            y_segs.append(t[s + 1 : s + self.segment_length + 1])

        return torch.stack(x_segs), torch.stack(y_segs)

    def __del__(self):
        self.mm.close()
        self._fh.close()


def get_grad_norm(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def get_epoch_prompts(epoch: int, n_random: int = 5) -> dict[str, str]:
    prompts = dict(FIXED_PROMPTS)
    rng = random.Random(epoch)
    picks = rng.sample(PROMPT_POOL, n_random)
    for i, p in enumerate(picks):
        prompts[f"random_{i}"] = p
    return prompts


def train():
    torch.manual_seed(42)
    device = torch.device(config["device"])
    print(f"Device: {device}")
    print(
        f"Tokenizer: vocab_size={tokenizer.vocab_size}, merges={len(tokenizer.merges)}"
    )

    dataset = TextDataset(
        config["dataset"],
        tokenizer,
        config["context_length"],
        config["memory_shards"],
    )

    print(f"Corpus: {dataset.file_size / 1024**3:.2f} GiB raw text")
    print(f"Chunks: {len(dataset):,} × {dataset.chunk_bytes:,} bytes")

    train_size = int(0.75 * len(dataset))
    val_size = int(0.25 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    model = AtomicWatermelon(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config["context_length"],
        dropout=config["dropout"],
        memory_slots=config["memory_slots"],
        compress_chunk=config["compress_chunk"],
        adapter_bottleneck=config["adapter_bottleneck"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    log = TrainingLog(config, model)
    log.set_data_info(dataset, train_size, val_size, train_loader, val_loader)

    print(f"Run ID: {log.run_id}")
    print(f"Log: {log.log_file}")
    print(f"Dataset hash: {dataset.hash}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Parameters: {log.data['model_architecture']['param_count']:,}")
    print(f"Memory shards: {config['memory_shards']}")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        last_grad_norm = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):

            x_segments, y_segments = batch
            x_segments = x_segments.to(device)
            y_segments = y_segments.to(device)

            B, n_seg, T = x_segments.shape

            batch_loss = 0.0
            memory = None

            for seg_idx in range(n_seg):
                x = x_segments[:, seg_idx, :].contiguous()
                y = y_segments[:, seg_idx, :].contiguous()

                _, loss, memory = model(x, targets=y, memory=memory)
                batch_loss += loss

            batch_loss = batch_loss / n_seg
            batch_loss.backward()

            last_grad_norm = get_grad_norm(model)

            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )

            optimizer.step()
            optimizer.zero_grad()

            train_loss += batch_loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                x_segments, y_segments = batch
                x_segments = x_segments.to(device)
                y_segments = y_segments.to(device)

                B, n_seg, T = x_segments.shape
                batch_loss = 0.0
                memory = None

                for seg_idx in range(n_seg):
                    x = x_segments[:, seg_idx, :].contiguous()
                    y = y_segments[:, seg_idx, :].contiguous()
                    _, loss, memory = model(x, targets=y, memory=memory)
                    batch_loss += loss

                batch_loss = batch_loss / n_seg
                val_loss += batch_loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        is_best = val_loss < best_loss

        samples = None
        if True:
            prompts = get_epoch_prompts(epoch)
            samples = {}

            for name, prompt_text in prompts.items():
                prompt_ids = tokenizer.encode(prompt_text)
                prompt = torch.tensor([prompt_ids]).to(device)
                out, _ = model.generate(
                    prompt, max_tokens=144, temperature=0.8, top_k=40
                )
                decoded = tokenizer.decode(out[0].tolist())
                samples[name] = decoded
                print(f"Sample ({name}): {decoded[:200]}...[END]")

        log.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            epoch_time_sec=epoch_time,
            lr=current_lr,
            grad_norm=last_grad_norm,
            is_best=is_best,
            sample=samples,
        )

        if is_best:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "patience_counter": patience_counter,
                    "config": config,
                    "dataset_hash": dataset.hash,
                    "rng_states": {
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    },
                },
                log.checkpoint_path,
            )

            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print("Early stopping.")
                log.stop("early_stopping")
                break
    else:
        log.stop("completed")

    print(f"Best val loss: {best_loss:.4f}")

    print(f"\n{'=' * 55}")
    print(f"Final Test Evaluation (untouched during training)")
    print(f"{'=' * 55}")

    best_ckpt = torch.load(log.checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            x_segments, y_segments = batch
            x_segments = x_segments.to(device)
            y_segments = y_segments.to(device)

            B, n_seg, T = x_segments.shape
            batch_loss = 0.0
            memory = None

            for seg_idx in range(n_seg):
                x = x_segments[:, seg_idx, :].contiguous()
                y = y_segments[:, seg_idx, :].contiguous()
                _, loss, memory = model(x, targets=y, memory=memory)
                batch_loss += loss

            batch_loss = batch_loss / n_seg
            test_loss += batch_loss.item()

    test_loss /= len(test_loader)

    print(f"  Best val loss:  {best_loss:.6f} (epoch {best_ckpt['epoch']})")
    print(f"  Test loss:      {test_loss:.6f}")
    print(f"  Gap (test-val): {test_loss - best_loss:.6f}")

    log.data["test"] = {
        "test_loss": test_loss,
        "best_val_loss": best_loss,
        "gap": test_loss - best_loss,
        "test_samples": test_size,
        "test_batches": len(test_loader),
        "evaluated_at_epoch": best_ckpt["epoch"],
    }

    import json

    with open(log.log_file, "w") as f:
        json.dump(log.data, f, indent=2, default=str)

    print(f"Log saved: {log.log_file}")


if __name__ == "__main__":
    train()
