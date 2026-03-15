"""
Training Loop for Atomic Watermelon — Substrate Ablation
Shared encoder weights. Dual mask. Nothing else.
"""

from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Any
from tqdm import tqdm
import numpy as np
import subprocess
import hashlib
import random
import torch
import json
import mmap
import time
import sys
import os

from models.atomic_watermelon import AtomicWatermelon
from trainers.logger_aw import TrainingLog
from datasets.bpe import BPETokenizer

_PIPELINE: list[tuple[Path, str, str]] = [
    # (required_output,         script,                  label)
    (Path("datasets/sep_corpus.txt"), "datasets/sep.py", "corpus builder"),
    (Path("datasets/tokenizer.json"), "datasets/bpe.py", "BPE trainer"),
    (Path("datasets/sep_corpus.bin"), "datasets/pretokenize.py", "pre-tokenizer"),
]


def _run_pipeline_if_needed() -> None:
    """
    Check each artifact in dependency order.
    If missing, run its producer script and verify the output appeared.
    Aborts the process on any failure — no point training on bad data.
    """
    any_ran = False

    for artifact, script, label in _PIPELINE:
        if artifact.exists():
            print(f"[setup] ✓ {artifact}")
            continue

        print(f"[setup] ✗ {artifact} not found — running {label} ({script})")
        any_ran = True

        result = subprocess.run(
            [sys.executable, script],
            check=False,
        )

        if result.returncode != 0:
            print(f"[setup] FATAL: {script} exited with code {result.returncode}")
            sys.exit(result.returncode)

        if not artifact.exists():
            print(f"[setup] FATAL: {script} succeeded but {artifact} was not created")
            sys.exit(1)

        print(f"[setup] ✓ {artifact} created")

    if any_ran:
        print("[setup] Pipeline complete\n")


if os.environ.get("_AW_WORKER") != "1":
    _run_pipeline_if_needed()

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
    "dataset": "datasets/sep_corpus.bin",  # ← binary, not .txt
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
    "accum_steps": 8,
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "patience": 25,
    "epochs": 500,
}


class TokenDataset(Dataset):
    def __init__(self, bin_path: str, context_length: int, vocab_size: int):
        self.bin_path = bin_path
        self.context_length = context_length
        self.vocab_size = vocab_size

        # Open once to get metadata and hash, then close
        with open(bin_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.n_tokens = len(mm) // 2
            self.n_chunks = (self.n_tokens - 1) // context_length
            self.hash = hashlib.sha256(mm[:2000]).hexdigest()[:16]
            mm.close()

        # These are opened lazily per-worker in __getitem__
        self._fh = None
        self._mm = None

    def _open(self):
        if self._mm is None:
            self._fh = open(self.bin_path, "rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        start = idx * self.context_length * 2
        end = start + (self.context_length + 1) * 2
        raw = self._mm[start:end]
        ids = np.frombuffer(raw, dtype=np.uint16).astype(np.int64)

        if len(ids) < self.context_length + 1:
            pad = np.zeros(self.context_length + 1 - len(ids), dtype=np.int64)
            ids = np.concatenate([ids, pad])

        t = torch.from_numpy(ids)
        return t[: self.context_length], t[1 : self.context_length + 1]

    def __getstate__(self):
        # exclude unpickleable handles — workers reopen via _open()
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_mm"] = None
        return state

    def __del__(self):
        if self._mm is not None:
            self._mm.close()
        if self._fh is not None:
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

    dataset = TokenDataset(
        config["dataset"], config["context_length"], config["vocab_size"]
    )

    print(f"Tokens: {dataset.n_tokens:,}")
    print(f"Chunks: {dataset.n_chunks:,} × {config['context_length']} tokens")
    print(f"Dataset hash: {dataset.hash}")

    train_size = int(0.75 * len(dataset))
    val_size = int(0.25 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
    )

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, **loader_kwargs)
    test_loader = DataLoader(test_set, **loader_kwargs)

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
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Parameters: {log.data['model_architecture']['param_count']:,}")

    best_loss = float("inf")
    patience_counter = 0
    accum_steps = config["accum_steps"]

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        last_grad_norm = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            x, y = batch
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)

            (loss / accum_steps).backward()
            train_loss += loss.item()

            if (i + 1) % accum_steps == 0:
                last_grad_norm = get_grad_norm(model)
                if config["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["max_grad_norm"]
                    )
                optimizer.step()
                optimizer.zero_grad()

        # flush any remainder
        if (i + 1) % accum_steps != 0:
            last_grad_norm = get_grad_norm(model)
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    _, loss, _ = model(x, targets=y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | Grad: {last_grad_norm:.3f} | Time: {epoch_time:.1f}s"
        )

        is_best = val_loss < best_loss

        prompts = get_epoch_prompts(epoch)
        samples = {}
        for name, prompt_text in prompts.items():
            prompt_ids = tokenizer.encode(prompt_text)
            prompt = torch.tensor([prompt_ids]).to(device)
            out, _ = model.generate(prompt, max_tokens=144, temperature=0.8, top_k=40)
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
            patience_counter = 0
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
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
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

    with open(log.log_file, "w") as f:
        json.dump(log.data, f, indent=2, default=str)

    print(f"Log saved: {log.log_file}")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    os.environ["_AW_WORKER"] = "1"
    mp.set_start_method("spawn", force=True)
    train()
