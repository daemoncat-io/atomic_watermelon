"""
heatmap_bridge.py

Visualize training activity across every component of every block.
Shows which subsystems are alive (moved from init) and which are dormant.

For AtomicWatermelon: encoder and decoder share weights — a single set of
attention + FF parameters per block, used in both bidirectional (encoder)
and causal (decoder) modes. So there is one "subsystem" per block, not three.

Metric: "drift from initialization"
  - LayerNorm weights: std of weight tensor (init = all 1s, so std = 0 means untrained)
  - LayerNorm bias:    std of bias tensor (init = all 0s)
  - Linear weights:    |current_std - xavier_expected_std| / xavier_expected_std
(xavier_uniform_ init has std = sqrt(2 / (fan_in + fan_out)) * sqrt(3))

Usage:
  python heatmap_bridge.py
  python heatmap_bridge.py -c checkpoints/atomic_watermelon_20260225_164548_best.pth
  python heatmap_bridge.py -c checkpoints/my_run_best.pth -o my_heatmap.png
"""

import torch.nn as nn
import matplotlib
import torch

from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
import json

from models.atomic_watermelon import AtomicWatermelon

matplotlib.use("Agg")

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cpu"  # CPU is fine for weight inspection
CHECKPOINT_PATH = "checkpoints/atomic_watermelon_20260513_225950_best.pth"
TOKENIZER_PATH = "datasets/tokenizer.json"
OUTPUT_PATH = "visualizations/heatmap_bridge.png"


# ============================================================
# METRICS
# ============================================================


def xavier_expected_std(shape: tuple[int, ...]) -> float:
    """Expected std for xavier_uniform_ initialized tensor."""
    if len(shape) < 2:
        return 0.0
    fan_in, fan_out = shape[1], shape[0]
    # xavier_uniform_ range: sqrt(6 / (fan_in + fan_out))
    # std of uniform(-a, a) = a / sqrt(3)
    a = math.sqrt(6.0 / (fan_in + fan_out))
    return a / math.sqrt(3)


def ln_drift(module: nn.LayerNorm | None) -> float:
    """
    How far has a LayerNorm drifted from init (weight=1, bias=0)?
    Returns a scalar in [0, ~1] range.
    """
    if module is None:
        return 0.0

    w = module.weight.detach().float()
    b = module.bias.detach().float() if module.bias is not None else torch.zeros(1)

    # Weight drift: init is all 1s, so std = 0 at init
    w_std = w.std().item()
    w_mean_drift = abs(w.mean().item() - 1.0)

    # Bias drift: init is all 0s
    b_std = b.std().item()
    b_mean_drift = abs(b.mean().item())

    return w_std + w_mean_drift + b_std + b_mean_drift


def linear_drift(module: nn.Linear | None) -> float:
    """
    How far has a Linear layer drifted from xavier init?
    Returns relative change in std from expected.
    """
    if module is None:
        return 0.0

    w = module.weight.detach().float()
    current_std = w.std().item()
    expected_std = xavier_expected_std(tuple(w.shape))

    if expected_std < 1e-8:
        return 0.0

    return abs(current_std - expected_std) / expected_std


# ============================================================
# DATA EXTRACTION
# ============================================================


def extract_block_metrics(
    model: AtomicWatermelon,
) -> list[dict[str, float]]:
    """
    For each block, compute drift metric for every component.
    BridgeBlock components: enc_ln1, enc_attn (w_qkv, w_o), enc_ln2, enc_ff (net[0], net[3])
    """
    all_metrics = []

    for block in model.blocks:
        m = {}
        m["ln1"] = ln_drift(block.enc_ln1)
        m["qkv"] = linear_drift(block.enc_attn.w_qkv)
        m["wo"] = linear_drift(block.enc_attn.w_o)
        m["ln2"] = ln_drift(block.enc_ln2)
        m["ff1"] = linear_drift(block.enc_ff.net[0])
        m["ff2"] = linear_drift(block.enc_ff.net[3])
        all_metrics.append(m)

    return all_metrics


def extract_global_metrics(model: AtomicWatermelon) -> dict[str, float]:
    """
    Extract drift metrics for non-block components.
    Note: lm_head.weight is tied to tok_emb.weight, so it's covered by tok_emb.
    """
    m = {}
    m["tok_emb"] = model.tok_emb.weight.detach().float().std().item()
    m["pos_emb"] = model.pos_emb.weight.detach().float().std().item()
    m["ln_f"] = ln_drift(model.ln_f)
    return m


# ============================================================
# VISUALIZATION
# ============================================================

# One subsystem per block — encoder and decoder share these weights.
BLOCK_COLS = [
    ("ln1", "LN₁"),
    ("qkv", "QKV"),
    ("wo", "Wₒ"),
    ("ln2", "LN₂"),
    ("ff1", "FF₁"),
    ("ff2", "FF₂"),
]


def build_heatmap_data(
    block_metrics: list[dict[str, float]],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build the matrix for the heatmap.

    Returns:
        data: [n_blocks, n_components] array of drift values
        col_labels: column labels
        row_labels: row labels
    """
    n_blocks = len(block_metrics)
    n_cols = len(BLOCK_COLS)
    data = np.zeros((n_blocks, n_cols))

    for row, metrics in enumerate(block_metrics):
        for col, (key, _) in enumerate(BLOCK_COLS):
            data[row, col] = metrics.get(key, 0.0)

    col_labels = [label for _, label in BLOCK_COLS]
    row_labels = [f"Block {i}" for i in range(n_blocks)]

    return data, col_labels, row_labels


def render_heatmap(
    block_metrics: list[dict[str, float]],
    global_metrics: dict[str, float],
    config: dict,
    output_path: str,
    checkpoint_path: str,
):
    data, col_labels, row_labels = build_heatmap_data(block_metrics)
    n_blocks, n_cols = data.shape

    # --- Color mapping ---
    DORMANT_THRESHOLD = 0.005

    colors_list = [
        (0.0, "#0a0a0a"),  # dormant — near black
        (0.01, "#1a0a2e"),  # threshold edge
        (0.15, "#2d1b69"),  # waking up
        (0.3, "#1b6ca0"),  # trained
        (0.5, "#1ba08c"),  # well trained
        (0.7, "#7acc29"),  # heavily trained
        (0.85, "#e6c820"),  # very active
        (1.0, "#f5f5dc"),  # max drift
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bridge_drift",
        [(pos, color) for pos, color in colors_list],
        N=256,
    )

    # log1p scaling to handle dynamic range
    data_log = np.log1p(data * 100)
    vmax = data_log.max() if data_log.max() > 0 else 1.0

    # --- Figure layout ---
    fig = plt.figure(figsize=(12, 9), facecolor="#0d0d0d")

    gs = GridSpec(
        3,
        1,
        height_ratios=[0.14, 1.0, 0.16],
        hspace=0.08,
        left=0.09,
        right=0.85,
        top=0.92,
        bottom=0.04,
    )

    # --- Title area ---
    ax_title = fig.add_subplot(gs[0])
    ax_title.set_facecolor("#0d0d0d")
    ax_title.axis("off")

    epoch = config.get("_best_epoch", "?")
    val_loss = config.get("_best_val_loss", "?")
    ax_title.text(
        0.0,
        0.85,
        "ATOMIC WATERMELON — TRAINING ACTIVITY HEATMAP",
        transform=ax_title.transAxes,
        fontsize=15,
        fontweight="bold",
        color="#e0e0e0",
        fontfamily="monospace",
    )
    ax_title.text(
        0.0,
        0.40,
        f"checkpoint: {Path(checkpoint_path).name}    "
        f"best epoch: {epoch}    val_loss: {val_loss}    "
        f"params: {config.get('_total_params', '?'):,}    "
        f"d_model: {config.get('d_model', '?')}    "
        f"n_layers: {config.get('n_layers', '?')}    "
        f"n_heads: {config.get('n_heads', '?')}    "
        f"d_ff: {config.get('d_ff', '?')}",
        transform=ax_title.transAxes,
        fontsize=8,
        color="#888888",
        fontfamily="monospace",
    )
    ax_title.text(
        0.0,
        0.05,
        "dual-mode shared weights: encoder (bidirectional) and decoder (causal) "
        "share parameters per block  │  "
        "dark = dormant (at init)  │  bright = active (trained)",
        transform=ax_title.transAxes,
        fontsize=7,
        color="#666666",
        fontfamily="monospace",
    )

    # --- Main heatmap ---
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor("#0d0d0d")

    im = ax.imshow(
        data_log,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    # Row labels
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels(row_labels, fontsize=10, fontfamily="monospace", color="#cccccc")

    # Column labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        col_labels,
        fontsize=10,
        fontfamily="monospace",
        color="#aaaaaa",
        rotation=0,
        ha="center",
    )

    # Single subsystem label at top
    ax.text(
        (n_cols - 1) / 2,
        -0.85,
        "SHARED ENCODER / DECODER BLOCK",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#cccccc",
        fontfamily="monospace",
    )
    ax.plot(
        [-0.3, n_cols - 0.7],
        [-0.55, -0.55],
        color="#555555",
        linewidth=1,
        clip_on=False,
    )

    # Annotate cells with raw drift values
    for row in range(n_blocks):
        for col in range(n_cols):
            val = data[row, col]
            if val < DORMANT_THRESHOLD:
                text = "—"
                color = "#333333"
            elif val < 0.05:
                text = f"{val:.3f}"
                color = "#888888"
            else:
                text = f"{val:.2f}"
                color = "#000000" if data_log[row, col] > vmax * 0.7 else "#cccccc"

            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontfamily="monospace",
                fontweight="bold" if val >= 0.05 else "normal",
            )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_blocks - 0.5, -0.5)
    ax.tick_params(axis="both", which="both", length=0)

    # Colorbar
    cbar_ax = fig.add_axes([0.87, 0.18, 0.018, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7, colors="#888888")
    cbar.set_label(
        "log₁₊(drift × 100)", fontsize=8, color="#888888", fontfamily="monospace"
    )
    cbar.outline.set_edgecolor("#333333")

    # --- Global components (bottom) ---
    ax_global = fig.add_subplot(gs[2])
    ax_global.set_facecolor("#0d0d0d")
    ax_global.axis("off")

    global_text = "GLOBAL:  "
    for key, val in global_metrics.items():
        status = "●" if val > DORMANT_THRESHOLD else "○"
        global_text += f"  {key}={val:.4f} [{status}]    "

    ax_global.text(
        0.0,
        0.7,
        global_text,
        transform=ax_global.transAxes,
        fontsize=8,
        color="#999999",
        fontfamily="monospace",
    )

    # Verdict — bucket by component family
    all_attn, all_ff, all_ln = [], [], []
    for metrics in block_metrics:
        all_attn.extend([metrics["qkv"], metrics["wo"]])
        all_ff.extend([metrics["ff1"], metrics["ff2"]])
        all_ln.extend([metrics["ln1"], metrics["ln2"]])

    attn_mean = float(np.mean(all_attn)) if all_attn else 0.0
    ff_mean = float(np.mean(all_ff)) if all_ff else 0.0
    ln_mean = float(np.mean(all_ln)) if all_ln else 0.0

    all_components = all_attn + all_ff + all_ln
    dormant = sum(1 for v in all_components if v < DORMANT_THRESHOLD)
    total = len(all_components)

    verdict_color = "#ff4444" if dormant > total * 0.5 else "#44ff44"
    ax_global.text(
        0.0,
        0.15,
        f"ATTN avg drift: {attn_mean:.4f}    "
        f"FF avg drift: {ff_mean:.4f}    "
        f"LN avg drift: {ln_mean:.4f}    "
        f"({dormant}/{total} block components dormant)",
        transform=ax_global.transAxes,
        fontsize=8,
        color=verdict_color,
        fontfamily="monospace",
        fontweight="bold",
    )

    plt.savefig(output_path, dpi=200, facecolor="#0d0d0d", bbox_inches="tight")
    plt.close()
    print(f"✅ Heatmap saved: {output_path}")


# ============================================================
# MAIN
# ============================================================


def main(
    checkpoint_path: str = CHECKPOINT_PATH,
    output_path: str = OUTPUT_PATH,
):
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]

    model = AtomicWatermelon(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["context_length"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Inject metadata for display
    cfg["_total_params"] = total_params
    cfg["_best_epoch"] = checkpoint.get("epoch", "?")
    cfg["_best_val_loss"] = f"{checkpoint.get('best_loss', 0):.6f}"

    # Extract metrics
    block_metrics = extract_block_metrics(model)
    global_metrics = extract_global_metrics(model)

    # Print summary
    print(f"\n{'='*60}")
    print("DRIFT FROM INITIALIZATION — PER BLOCK")
    print(f"{'='*60}")

    for i, metrics in enumerate(block_metrics):
        all_vals = list(metrics.values())
        status = "ACTIVE" if np.mean(all_vals) > 0.005 else "DORMANT"
        print(f"\n  Block {i}: {status}")
        print(f"    Attn: qkv={metrics['qkv']:.4f}  wo={metrics['wo']:.4f}")
        print(f"    FF:   ff1={metrics['ff1']:.4f}  ff2={metrics['ff2']:.4f}")
        print(f"    LN:   ln1={metrics['ln1']:.4f}  ln2={metrics['ln2']:.4f}")

    print(f"\n  Global:")
    for k, v in global_metrics.items():
        status = "ACTIVE" if v > 0.005 else "DORMANT"
        print(f"    {k:20s} {status:8s}  ({v:.4f})")

    # Render
    render_heatmap(block_metrics, global_metrics, cfg, output_path, checkpoint_path)

    # Also save raw metrics as JSON
    json_path = Path(output_path).with_suffix(".json")
    json_data = {
        "checkpoint": checkpoint_path,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "block_metrics": block_metrics,
        "global_metrics": global_metrics,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ Metrics JSON: {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Heatmap of atomic watermelon training activity"
    )
    parser.add_argument("--checkpoint", "-c", default=CHECKPOINT_PATH)
    parser.add_argument("--output", "-o", default=OUTPUT_PATH)
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
