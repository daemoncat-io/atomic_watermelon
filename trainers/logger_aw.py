"""
Training Logger for Atomic Watermelon
"""

from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
import subprocess
import hashlib
import random
import psutil
import torch
import json
import time
import os


class TrainingLog:
    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        log_dir: str = "checkpoints",
    ):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{config['model']}_{self.run_id}.json"

        # Prime CPU percent measurement
        psutil.cpu_percent(percpu=True)

        self.data = {
            "run_id": self.run_id,
            "config": config,
            "model_architecture": self._get_model_architecture(model),
            "system": self._get_system_info(),
            "randomness": self._get_random_state(),
            "started_at": datetime.now().isoformat(),
            "epochs": [],
            "best": None,
            "stopped_at": None,
            "stop_reason": None,
            "total_training_time_sec": None,
        }
        self._save()
        self._start_time = time.time()

    def _get_random_state(self) -> dict:
        state = {
            "torch_initial_seed": torch.initial_seed(),
            "torch_rng_state_hash": hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest()[:16],
            "python_random_sample": random.random(),
        }
        if torch.cuda.is_available():
            state["cuda_initial_seed"] = torch.cuda.initial_seed()
        return state

    def _get_model_architecture(self, model: torch.nn.Module) -> dict:
        arch = {
            "param_count": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
        introspect_attrs = [
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "vocab_size",
            "context_length",
            "max_seq_len",
            "memory_slots",
            "compress_chunk",
            "adapter_bottleneck",
        ]
        for attr in introspect_attrs:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if isinstance(val, (int, float, str, bool, type(None))):
                    arch[attr] = val
        return arch

    def _get_system_info(self) -> dict:
        info = {
            "python_version": os.popen("python3 --version").read().strip(),
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cpu_model": self._get_cpu_model(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": self._get_cpu_freq(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "hostname": os.uname().nodename,
            "os": f"{os.uname().sysname} {os.uname().release}",
        }

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
        elif torch.backends.mps.is_available():
            info["gpu"] = "Apple Silicon (MPS)"
            info["gpu_cores"] = self._get_mps_gpu_cores()
            try:
                info["mps_recommended_max_gb"] = round(
                    torch.mps.recommended_max_memory() / (1024**3), 2
                )
            except Exception:
                pass

        return info

    def _get_cpu_model(self) -> str | None:
        try:
            if os.uname().sysname == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip():
                    return result.stdout.strip()
                result = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout.strip()
            else:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
        except Exception:
            pass
        return None

    def _get_cpu_freq(self) -> dict | None:
        freq = psutil.cpu_freq()
        if freq:
            return {
                "current": round(freq.current, 0),
                "min": round(freq.min, 0) if freq.min else None,
                "max": round(freq.max, 0) if freq.max else None,
            }
        return None

    def _get_mps_gpu_cores(self) -> int | None:
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split("\n"):
                if "Total Number of Cores" in line:
                    return int(line.split(":")[1].strip())
        except Exception:
            pass
        return None

    def _get_gpu_stats(self) -> dict | None:
        if torch.cuda.is_available():
            stats = {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
                "max_allocated_gb": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                ),
            }
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 4:
                    stats["util_percent"] = int(parts[0])
                    stats["memory_util_percent"] = int(parts[1])
                    stats["temperature_c"] = int(parts[2])
                    stats["power_watts"] = float(parts[3])
            except Exception:
                pass
            return stats

        elif torch.backends.mps.is_available():
            return {
                "allocated_gb": round(
                    torch.mps.current_allocated_memory() / (1024**3), 3
                ),
                "driver_allocated_gb": round(
                    torch.mps.driver_allocated_memory() / (1024**3), 3
                ),
            }

        return None

    def _get_utilization_stats(self) -> dict:
        per_core = psutil.cpu_percent(percpu=True)
        active_cores = sum(1 for c in per_core if c > 5.0)
        process = psutil.Process()

        stats = {
            "cpu": {
                "total_percent": psutil.cpu_percent(),
                "active_cores": active_cores,
                "process_percent": process.cpu_percent(),
                "process_threads": process.num_threads(),
            },
            "memory": {
                "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 3),
                "ram_available_gb": round(
                    psutil.virtual_memory().available / (1024**3), 3
                ),
                "ram_percent": psutil.virtual_memory().percent,
                "process_ram_gb": round(process.memory_info().rss / (1024**3), 3),
                "process_ram_percent": round(process.memory_percent(), 2),
            },
        }

        gpu_stats = self._get_gpu_stats()
        if gpu_stats:
            stats["gpu"] = gpu_stats

        return stats

    def set_data_info(
        self,
        dataset,
        train_size: int,
        val_size: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """
        Set dataset information from the trainer.

        Args:
            dataset: The full dataset (must have .hash and .vocab_size attributes)
            train_size: Number of training samples
            val_size: Number of validation samples
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        """
        context_length = self.config["context_length"]

        self.data["data"] = {
            "dataset_hash": dataset.hash,
            "vocab_size": dataset.vocab_size,
            "context_length": context_length,
            "train_samples": train_size,
            "val_samples": val_size,
            "train_tokens": train_size * context_length,
            "val_tokens": val_size * context_length,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "batch_size": self.config["batch_size"],
            "tokens_per_batch": self.config["batch_size"] * context_length,
        }
        self._save()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time_sec: float,
        lr: float,
        grad_norm: float,
        is_best: bool = False,
        sample: dict[str, str] | None = None,
    ):
        """
        Log metrics for a completed epoch.

        Args:
            epoch: Epoch number (0-indexed)
            train_loss: Average training loss for the epoch
            val_loss: Average validation loss for the epoch
            epoch_time_sec: Time taken for the epoch in seconds
            lr: Current learning rate
            grad_norm: Gradient norm (pre-clipping, from last batch)
            is_best: Whether this is the best validation loss so far
            sample: Optional dict of sample generations
        """
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "epoch_time_sec": round(epoch_time_sec, 2),
            "lr": lr,
            "grad_norm": round(grad_norm, 4),
            "is_best": is_best,
            "elapsed_sec": round(time.time() - self._start_time, 2),
            "utilization": self._get_utilization_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        if sample:
            entry["samples"] = {k: v[:500] for k, v in sample.items()}

        self.data["epochs"].append(entry)

        if is_best:
            self.data["best"] = {
                "epoch": epoch,
                "val_loss": round(val_loss, 6),
                "train_loss": round(train_loss, 6),
                "checkpoint": str(self.checkpoint_path.name),
            }

        self._save()

    def stop(self, reason: str):
        """
        Finalize the training log with summary statistics.

        Args:
            reason: Stop reason ('early_stopping', 'completed', 'interrupted', etc.)
        """
        self.data["stopped_at"] = datetime.now().isoformat()
        self.data["stop_reason"] = reason
        self.data["total_training_time_sec"] = round(time.time() - self._start_time, 2)

        if self.data["epochs"]:
            self._compute_summary()

        self._save()

    def _compute_summary(self):
        epochs = self.data["epochs"]

        train_losses = [e["train_loss"] for e in epochs]
        val_losses = [e["val_loss"] for e in epochs]
        grad_norms = [e["grad_norm"] for e in epochs]
        epoch_times = [e["epoch_time_sec"] for e in epochs]
        cpu_totals = [e["utilization"]["cpu"]["total_percent"] for e in epochs]
        active_cores = [e["utilization"]["cpu"]["active_cores"] for e in epochs]
        process_threads = [e["utilization"]["cpu"]["process_threads"] for e in epochs]
        ram_usage = [e["utilization"]["memory"]["process_ram_gb"] for e in epochs]

        self.data["summary"] = {
            "total_epochs": len(epochs),
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_train_loss": min(train_losses),
            "best_val_loss": min(val_losses),
            "avg_epoch_time_sec": round(sum(epoch_times) / len(epoch_times), 2),
            "total_time_min": round(self.data["total_training_time_sec"] / 60, 2),
            "avg_grad_norm": round(sum(grad_norms) / len(grad_norms), 4),
            "max_grad_norm": round(max(grad_norms), 4),
            "avg_cpu_percent": round(sum(cpu_totals) / len(cpu_totals), 2),
            "avg_active_cores": round(sum(active_cores) / len(active_cores), 1),
            "max_active_cores": max(active_cores),
            "avg_process_threads": round(
                sum(process_threads) / len(process_threads), 1
            ),
            "avg_ram_gb": round(sum(ram_usage) / len(ram_usage), 3),
            "max_ram_gb": round(max(ram_usage), 3),
        }

        # GPU summary - CUDA
        gpu_utils = [
            e["utilization"].get("gpu", {}).get("util_percent") for e in epochs
        ]
        gpu_utils = [g for g in gpu_utils if g is not None]
        if gpu_utils:
            gpu_mem = [e["utilization"]["gpu"]["allocated_gb"] for e in epochs]
            gpu_temps = [e["utilization"]["gpu"].get("temperature_c") for e in epochs]
            gpu_temps = [t for t in gpu_temps if t is not None]
            gpu_power = [e["utilization"]["gpu"].get("power_watts") for e in epochs]
            gpu_power = [p for p in gpu_power if p is not None]

            self.data["summary"]["avg_gpu_util_percent"] = round(
                sum(gpu_utils) / len(gpu_utils), 2
            )
            self.data["summary"]["max_gpu_util_percent"] = max(gpu_utils)
            self.data["summary"]["avg_gpu_mem_gb"] = round(
                sum(gpu_mem) / len(gpu_mem), 3
            )
            self.data["summary"]["max_gpu_mem_gb"] = round(max(gpu_mem), 3)
            if gpu_temps:
                self.data["summary"]["avg_gpu_temp_c"] = round(
                    sum(gpu_temps) / len(gpu_temps), 1
                )
                self.data["summary"]["max_gpu_temp_c"] = max(gpu_temps)
            if gpu_power:
                self.data["summary"]["avg_gpu_power_watts"] = round(
                    sum(gpu_power) / len(gpu_power), 1
                )

        # GPU summary - MPS
        mps_alloc = [
            e["utilization"].get("gpu", {}).get("allocated_gb") for e in epochs
        ]
        mps_alloc = [m for m in mps_alloc if m is not None]
        if mps_alloc and torch.backends.mps.is_available():
            mps_driver = [
                e["utilization"]["gpu"]["driver_allocated_gb"] for e in epochs
            ]
            self.data["summary"]["avg_mps_allocated_gb"] = round(
                sum(mps_alloc) / len(mps_alloc), 3
            )
            self.data["summary"]["max_mps_allocated_gb"] = round(max(mps_alloc), 3)
            self.data["summary"]["avg_mps_driver_gb"] = round(
                sum(mps_driver) / len(mps_driver), 3
            )
            self.data["summary"]["max_mps_driver_gb"] = round(max(mps_driver), 3)

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def checkpoint_path(self) -> Path:
        return self.log_dir / f"{self.config['model']}_{self.run_id}_best.pth"
