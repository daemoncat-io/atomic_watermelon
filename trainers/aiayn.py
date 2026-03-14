"""
Trainer for Encoder-Decoder Transformer (AIAYN)
Aligned with AW trainer patterns.

Training specifications from Attention Is All You Need:
- Optimizer: Adam with β1=0.9, β2=0.98, ε=10^-9
- LR schedule: d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
- Warmup steps: 4000
- Label smoothing: ε_ls = 0.1
"""

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import Any
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import hashlib
import random
import torch
import time

from trainers.logger_aw import TrainingLog
from models.aiayn import Transformer


config: dict[str, Any] = {
    # Data
    "dataset": "datasets/gen0_language_structure.txt",
    "context_length": 512,
    "dataset_repeat": 500,
    # Model
    "share_embeddings": True,
    "model": "aiayn",
    "d_model": 512,
    "dropout": 0.1,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
    # Training
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "label_smoothing": 0.1,
    "max_grad_norm": 1.0,
    "learning_rate": 1.0,  # Scaled by schedule
    "warmup_steps": 4000,
    "batch_size": 1,
    "epsilon": 1e-9,
    "patience": 10,
    "epochs": 110,
    "beta2": 0.98,
    "beta1": 0.9,
}


# =============================================================================
# LR Schedule (paper Section 5.3)
# =============================================================================


class TransformerLRScheduler:
    """
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Linear warmup for warmup_steps, then inverse sqrt decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        scale: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self.step_num = 0
        self._last_lr = 0.0

    def step(self):
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self._last_lr = lr

    def get_lr(self) -> float:
        step = max(1, self.step_num)
        return self.scale * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def get_last_lr(self) -> float:
        return self._last_lr


# =============================================================================
# Label Smoothing Loss (paper Section 5.4)
# =============================================================================


class LabelSmoothingLoss(nn.Module):
    """
    Soft targets: true class gets (1 - ε), others share ε uniformly.
    Hurts perplexity, improves generalization.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.contiguous().view(-1, logits.size(-1))
        if target.dim() == 2:
            target = target.contiguous().view(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.sum(dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        loss = (
            self.confidence * nll_loss
            + (self.smoothing / self.vocab_size) * smooth_loss
        )

        pad_mask = target != self.pad_idx
        loss = loss * pad_mask

        return loss.sum() / pad_mask.sum().clamp(min=1)


# =============================================================================
# Dataset
# =============================================================================


class ASCIISeq2SeqDataset(Dataset):
    """
    ASCII autoencoding dataset. Source and target are identical.
    BOS/EOS tokens prepended/appended for encoder-decoder training.
    """

    def __init__(self, text: str, context_length: int):
        self.tokens = [ord(c) for c in text]
        self.context_length = context_length
        self.hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        self.vocab_size = 256

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        chunk = self.tokens[start : start + self.context_length]

        src = [self.bos_idx] + chunk + [self.eos_idx]
        tgt = [self.bos_idx] + chunk + [self.eos_idx]

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
        }


def collate_seq2seq(batch: list[dict], pad_idx: int = 0) -> dict:
    """Collate with dynamic padding."""
    src_padded = nn.utils.rnn.pad_sequence(
        [item["src"] for item in batch], batch_first=True, padding_value=pad_idx
    )
    tgt_padded = nn.utils.rnn.pad_sequence(
        [item["tgt"] for item in batch], batch_first=True, padding_value=pad_idx
    )
    return {"src": src_padded, "tgt": tgt_padded}


# =============================================================================
# Utilities
# =============================================================================


def get_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def create_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source padding mask and target causal+padding mask."""
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2).to(device)

    tgt_len = tgt.size(1)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    tgt_causal_mask = (
        torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1) == 0
    )
    tgt_mask = tgt_pad_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)

    return src_mask, tgt_mask


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int = 100,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
) -> torch.Tensor:
    """Greedy decoding for sampling."""
    model.eval()
    device = next(model.parameters()).device
    src = src.to(device)

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    encoder_output = model.encode(src, src_mask)

    generated = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_len = generated.size(1)
        tgt_mask = (
            (torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1) == 0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        decoder_output = model.decode(generated, encoder_output, src_mask, tgt_mask)
        next_token = model.output_projection(decoder_output)[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == eos_idx:
            break

    return generated


# =============================================================================
# Training
# =============================================================================


def train():
    torch.manual_seed(42)
    device = torch.device(config["device"])
    print(f"Device: {device}")

    text = open(config["dataset"]).read() * config["dataset_repeat"]
    dataset = ASCIISeq2SeqDataset(text, config["context_length"])

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    collate = lambda b: collate_seq2seq(b, pad_idx=dataset.pad_idx)
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], collate_fn=collate
    )

    model = Transformer(
        src_vocab_size=dataset.vocab_size,
        tgt_vocab_size=dataset.vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        h=config["n_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config["context_length"] + 10,
        dropout=config["dropout"],
        share_embeddings=config["share_embeddings"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["epsilon"],
    )

    scheduler = TransformerLRScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
        scale=config["learning_rate"],
    )

    criterion = LabelSmoothingLoss(
        vocab_size=dataset.vocab_size,
        smoothing=config["label_smoothing"],
        pad_idx=dataset.pad_idx,
    )

    log = TrainingLog(config, model)
    log.set_data_info(dataset, train_size, val_size, train_loader, val_loader)

    print(f"Run ID: {log.run_id}")
    print(f"Log: {log.log_file}")
    print(f"Dataset hash: {dataset.hash}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Parameters: {log.data['model_architecture']['param_count']:,}")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        train_tokens = 0
        last_grad_norm = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input, dataset.pad_idx, device)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits, tgt_labels)

            loss.backward()
            last_grad_norm = get_grad_norm(model)

            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            n_tokens = (tgt_labels != dataset.pad_idx).sum().item()
            train_loss += loss.item() * n_tokens
            train_tokens += n_tokens

        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)

                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]

                src_mask, tgt_mask = create_masks(
                    src, tgt_input, dataset.pad_idx, device
                )

                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits, tgt_labels)

                n_tokens = (tgt_labels != dataset.pad_idx).sum().item()
                val_loss += loss.item() * n_tokens
                val_tokens += n_tokens

        train_loss /= max(1, train_tokens)
        val_loss /= max(1, val_tokens)
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()

        print(
            f"[{epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        is_best = val_loss < best_loss

        samples = None
        if (epoch + 1) % 5 == 0:
            prompts = {
                "To decode means to?": "To decode means to? ",
                "How are you feeling?": "How are you feeling? ",
                "What is your purpose?": "What is your purpose? ",
                "sup.": "sup. ",
            }
            samples = {}
            for name, prompt_text in prompts.items():
                src_tokens = (
                    [dataset.bos_idx]
                    + [ord(c) for c in prompt_text]
                    + [dataset.eos_idx]
                )
                src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
                out = greedy_decode(
                    model,
                    src_tensor,
                    max_len=100,
                    bos_idx=dataset.bos_idx,
                    eos_idx=dataset.eos_idx,
                    pad_idx=dataset.pad_idx,
                )
                decoded = "".join(
                    chr(t)
                    for t in out[0, 1:].tolist()
                    if 32 <= t < 127 or t in (10, 13, 9)
                )
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
                    "scheduler_step": scheduler.step_num,
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
    print(f"Log saved: {log.log_file}")


if __name__ == "__main__":
    train()
