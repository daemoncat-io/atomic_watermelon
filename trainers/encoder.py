"""
Minimal Training Loop
encoder-only
"""

from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Any
from tqdm import tqdm
import hashlib
import torch
import time
import sys


sys.path.insert(0, str(Path(__file__).parent.parent))

from trainers.logger_single_mode import TrainingLog
from models.encoder import Encoder

config: dict[str, Any] = {
    # Data
    "dataset": "datasets/gen0_language_structure.txt",
    "context_length": 512,
    "dataset_repeat": 500,
    # Model
    "model": "encoder",
    # Training
    "learning_rate": 2.93e-4,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "batch_size": 1,
    "device": "mps",
    "patience": 10,
    "epochs": 110,
}


class ASCIIDataset(Dataset):
    def __init__(self, text: str, context_length: int):
        self.tokens = [ord(c) for c in text]
        self.context_length = context_length
        self.hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        chunk = self.tokens[start : start + self.context_length + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])


def get_grad_norm(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def train():
    torch.manual_seed(42)
    print(f"Device: {config['device']}")

    text = open(config["dataset"]).read() * config["dataset_repeat"]
    dataset = ASCIIDataset(text, config["context_length"])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])

    model = Encoder().to(config["device"])
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
    print(f"Parameters: {log.data['model_architecture']['param_count']:,}")

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        last_grad_norm = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            x, y = x.to(config["device"]), y.to(config["device"])
            _, loss = model(x, y)
            loss.backward()
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
            last_grad_norm = get_grad_norm(model)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                x, y = x.to(config["device"]), y.to(config["device"])
                _, loss = model(x, y)
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
        if (epoch + 1) % 5 == 0:
            prompts = {
                "To decode means to?": "To decode means to? ",
                "How are you feeling?": "How are you feeling? ",
                "What is your purpose?": "What is your purpose? ",
                "sup.": "sup. ",
            }
            samples = {}
            for name, prompt_text in prompts.items():
                masked = torch.tensor([[ord(c) for c in prompt_text]]).to(
                    config["device"]
                )
                filled = model.fill_masks(masked)
                samples[name] = "".join(chr(t) for t in filled[0].tolist())
                print(f"Sample ({name}): {samples[name]}")

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
            torch.save(model.state_dict(), log.checkpoint_path)
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
