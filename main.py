"""
Training Loop for Atomic Watermelon — Substrate Ablation
Shared encoder weights. Dual mask. Nothing else.
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
from trainers.logger_aw import TrainingLog
from datasets.bpe import BPETokenizer


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
    "model": "atomic_watermelon",
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
    ):
        self.tok = tok
        self.vocab_size = tok.vocab_size
        self.context_length = context_length
        self.tokens_needed = context_length + 1
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
        x = t[: self.context_length]
        y = t[1 : self.context_length + 1]

        return x, y

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

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        last_grad_norm = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            _, loss, _ = model(x, targets=y)

            loss.backward()

            last_grad_norm = get_grad_norm(model)

            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                _, loss, _ = model(x, targets=y)
                val_loss += loss.item()

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
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            _, loss, _ = model(x, targets=y)
            test_loss += loss.item()

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
