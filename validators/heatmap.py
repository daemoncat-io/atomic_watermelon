"""
heatmap_bridge.py

Visualize training activity across every component of every block.
Shows which subsystems are alive (moved from init) and which are dormant.

Metric: "drift from initialization"
  - LayerNorm weights: std of weight tensor (init = all 1s, so std = 0 means untrained)
  - LayerNorm bias:    std of bias tensor (init = all 0s)
  - Linear weights:    |current_std - xavier_expected_std| / xavier_expected_std
                       (xavier_uniform_ init has std = sqrt(2 / (fan_in + fan_out)) * sqrt(3))
  - Combined per-component: max of weight and bias drift signals

Usage:
    python heatmap_bridge.py
    python heatmap_bridge.py -c checkpoints/cross_attention_20260225_164548_best.pth
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

from model.atomic_watermelon import AtomicWatermelon

matplotlib.use("Agg")

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cpu"  # CPU is fine for weight inspection
CHECKPOINT_PATH = "cross_attention_20260225_164548_best.pth"
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
    # Also check mean drift from 1.0
    w_std = w.std().item()
    w_mean_drift = abs(w.mean().item() - 1.0)

    # Bias drift: init is all 0s
    b_std = b.std().item()
    b_mean_drift = abs(b.mean().item())

    # Combined signal — weight std is the strongest indicator
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

    # Relative drift from expected
    return abs(current_std - expected_std) / expected_std


def adapter_drift(adapter: nn.Module | None) -> dict[str, float]:
    """Compute drift for each sub-component of an Adapter."""
    if adapter is None:
        return {"down": 0.0, "up": 0.0, "ln": 0.0}

    return {
        "down": linear_drift(getattr(adapter, "down", None)),
        "up": linear_drift(getattr(adapter, "up", None)),
        "ln": ln_drift(getattr(adapter, "ln", None)),
    }


# ============================================================
# DATA EXTRACTION
# ============================================================


def extract_block_metrics(
    model: AtomicWatermelon,
) -> list[dict[str, float]]:
    """
    For each block, compute drift metric for every component.
    Returns list of dicts, one per block.
    """
    blocks = model.blocks
    all_metrics = []

    for i, block in enumerate(blocks):
        m = {}

        # --- Encoder pathway ---
        m["enc_ln1"] = ln_drift(block.enc_ln1)
        m["enc_attn_qkv"] = linear_drift(block.enc_attn.w_qkv)
        m["enc_attn_wo"] = linear_drift(block.enc_attn.w_o)
        m["enc_ln2"] = ln_drift(block.enc_ln2)

        enc_ff_net = block.enc_ff.net
        m["enc_ff_w1"] = linear_drift(enc_ff_net[0])
        m["enc_ff_w2"] = linear_drift(enc_ff_net[3])

        # --- Decoder adapters ---
        for prefix in ["pre_attn", "post_attn", "pre_ff", "post_ff"]:
            adapter = getattr(block, f"dec_adapt_{prefix}", None)
            ad = adapter_drift(adapter)
            m[f"adapt_{prefix}_down"] = ad["down"]
            m[f"adapt_{prefix}_up"] = ad["up"]
            m[f"adapt_{prefix}_ln"] = ad["ln"]

        # --- Cross-attention ---
        m["cross_wq"] = linear_drift(block.cross_attn.w_q)
        m["cross_wkv"] = linear_drift(block.cross_attn.w_kv)
        m["cross_wo"] = linear_drift(block.cross_attn.w_o)

        # --- Cross LN + adapter ---
        m["cross_ln"] = ln_drift(block.cross_ln)

        cross_adapt = getattr(block, "cross_adapt", None)
        ca = adapter_drift(cross_adapt)
        m["cross_adapt_down"] = ca["down"]
        m["cross_adapt_up"] = ca["up"]
        m["cross_adapt_ln"] = ca["ln"]

        all_metrics.append(m)

    return all_metrics


def extract_global_metrics(model: AtomicWatermelon) -> dict[str, float]:
    """Extract drift metrics for non-block components."""
    m = {}
    m["tok_emb"] = model.tok_emb.weight.detach().float().std().item()
    m["pos_emb"] = model.pos_emb.weight.detach().float().std().item()
    m["mem_pos_emb"] = model.mem_pos_emb.weight.detach().float().std().item()
    m["compress_proj"] = linear_drift(model.compress_proj)
    m["compress_gate"] = linear_drift(model.compress_gate)
    m["ln_f"] = ln_drift(model.ln_f)
    return m


# ============================================================
# VISUALIZATION
# ============================================================

# Component layout: grouped by subsystem
# Each tuple: (key_in_metrics, display_label)
ENCODER_COLS = [
    ("enc_ln1", "LN₁"),
    ("enc_attn_qkv", "QKV"),
    ("enc_attn_wo", "Wₒ"),
    ("enc_ln2", "LN₂"),
    ("enc_ff_w1", "FF₁"),
    ("enc_ff_w2", "FF₂"),
]

ADAPTER_COLS = [
    ("adapt_pre_attn_down", "↓"),
    ("adapt_pre_attn_up", "↑"),
    ("adapt_pre_attn_ln", "LN"),
    ("adapt_post_attn_down", "↓"),
    ("adapt_post_attn_up", "↑"),
    ("adapt_post_attn_ln", "LN"),
    ("adapt_pre_ff_down", "↓"),
    ("adapt_pre_ff_up", "↑"),
    ("adapt_pre_ff_ln", "LN"),
    ("adapt_post_ff_down", "↓"),
    ("adapt_post_ff_up", "↑"),
    ("adapt_post_ff_ln", "LN"),
]

CROSS_COLS = [
    ("cross_wq", "Wq"),
    ("cross_wkv", "Wkv"),
    ("cross_wo", "Wₒ"),
    ("cross_ln", "LN"),
    ("cross_adapt_down", "↓"),
    ("cross_adapt_up", "↑"),
    ("cross_adapt_ln", "LN"),
]

SUBSYSTEMS = [
    ("ENCODER", ENCODER_COLS),
    ("DECODER ADAPTERS", ADAPTER_COLS),
    ("CROSS-ATTENTION + MEMORY BRIDGE", CROSS_COLS),
]


def build_heatmap_data(
    block_metrics: list[dict[str, float]],
) -> tuple[np.ndarray, list[str], list[str], list[tuple[int, int, str]]]:
    """
    Build the matrix for the heatmap.

    Returns:
        data: [n_blocks, n_components] array of drift values
        col_labels: column labels
        row_labels: row labels
        group_spans: [(start_col, end_col, group_name), ...]
    """
    n_blocks = len(block_metrics)

    all_cols = []
    group_spans = []
    col_offset = 0

    for group_name, cols in SUBSYSTEMS:
        start = col_offset
        for key, label in cols:
            all_cols.append((key, label))
            col_offset += 1
        group_spans.append((start, col_offset, group_name))

    n_cols = len(all_cols)
    data = np.zeros((n_blocks, n_cols))

    for row, metrics in enumerate(block_metrics):
        for col, (key, _) in enumerate(all_cols):
            data[row, col] = metrics.get(key, 0.0)

    col_labels = [label for _, label in all_cols]
    row_labels = [f"Block {i}" for i in range(n_blocks)]

    return data, col_labels, row_labels, group_spans


def render_heatmap(
    block_metrics: list[dict[str, float]],
    global_metrics: dict[str, float],
    config: dict,
    output_path: str,
    checkpoint_path: str,
):
    data, col_labels, row_labels, group_spans = build_heatmap_data(block_metrics)
    n_blocks, n_cols = data.shape

    # --- Color mapping ---
    # Threshold: anything below 0.005 drift is "dormant"
    # Log scale above that to show variation in trained components
    DORMANT_THRESHOLD = 0.005

    # Custom colormap: black (dead) -> deep blue (barely trained) -> cyan -> yellow -> white (heavily trained)
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

    # Normalize data for colormap
    # Use log1p scaling to handle the huge dynamic range
    data_log = np.log1p(data * 100)  # scale up then log
    vmax = data_log.max() if data_log.max() > 0 else 1.0

    # --- Figure layout ---
    fig = plt.figure(figsize=(22, 10), facecolor="#0d0d0d")

    gs = GridSpec(
        3,
        1,
        height_ratios=[0.12, 1.0, 0.15],
        hspace=0.08,
        left=0.06,
        right=0.88,
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
        "CROSS-ATTENTION BRIDGE TRANSFORMER — TRAINING ACTIVITY HEATMAP",
        transform=ax_title.transAxes,
        fontsize=16,
        fontweight="bold",
        color="#e0e0e0",
        fontfamily="monospace",
    )
    ax_title.text(
        0.0,
        0.35,
        f"checkpoint: {Path(checkpoint_path).name}    "
        f"best epoch: {epoch}    val_loss: {val_loss}    "
        f"params: {config.get('_total_params', '?'):,}    "
        f"d_model: {config.get('d_model', '?')}    "
        f"n_layers: {config.get('n_layers', '?')}    "
        f"memory_slots: {config.get('memory_slots', '?')}    "
        f"compress_chunk: {config.get('compress_chunk', '?')}",
        transform=ax_title.transAxes,
        fontsize=8,
        color="#888888",
        fontfamily="monospace",
    )
    ax_title.text(
        0.0,
        0.0,
        "metric: drift from initialization  │  "
        "LN: |μ-1| + σ(w) + σ(b)  │  "
        "Linear: |σ_current - σ_xavier| / σ_xavier  │  "
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

    # Grid lines between subsystems
    for start, end, name in group_spans:
        if start > 0:
            ax.axvline(x=start - 0.5, color="#333333", linewidth=1.5)

    # Row labels
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels(row_labels, fontsize=10, fontfamily="monospace", color="#cccccc")

    # Column labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        col_labels,
        fontsize=7,
        fontfamily="monospace",
        color="#aaaaaa",
        rotation=0,
        ha="center",
    )

    # Subsystem group labels (top)
    for start, end, name in group_spans:
        mid = (start + end - 1) / 2
        ax.text(
            mid,
            -0.9,
            name,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#cccccc",
            fontfamily="monospace",
        )
        # Bracket
        ax.plot(
            [start - 0.3, end - 0.7],
            [-0.6, -0.6],
            color="#555555",
            linewidth=1,
            clip_on=False,
        )

    # Adapter sub-group labels
    adapter_start = group_spans[1][0]
    adapter_subgroups = [
        (adapter_start, adapter_start + 3, "pre_attn"),
        (adapter_start + 3, adapter_start + 6, "post_attn"),
        (adapter_start + 6, adapter_start + 9, "pre_ff"),
        (adapter_start + 9, adapter_start + 12, "post_ff"),
    ]
    for s, e, label in adapter_subgroups:
        mid = (s + e - 1) / 2
        ax.text(
            mid,
            n_blocks + 0.1,
            label.replace("_", " "),
            ha="center",
            va="top",
            fontsize=6,
            color="#777777",
            fontfamily="monospace",
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
                fontsize=5.5,
                color=color,
                fontfamily="monospace",
                fontweight="bold" if val >= 0.05 else "normal",
            )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_blocks - 0.5, -0.5)
    ax.tick_params(axis="both", which="both", length=0)

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.65])
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
        color_tag = "active" if val > DORMANT_THRESHOLD else "dormant"
        global_text += f"  {key}={val:.4f} [{status}]    "

    ax_global.text(
        0.0,
        0.7,
        global_text,
        transform=ax_global.transAxes,
        fontsize=7,
        color="#999999",
        fontfamily="monospace",
    )

    # Summary verdict
    all_cross = []
    all_encoder = []
    all_adapter = []
    for metrics in block_metrics:
        for k, v in metrics.items():
            if k.startswith("cross_"):
                all_cross.append(v)
            elif k.startswith("enc_"):
                all_encoder.append(v)
            elif k.startswith("adapt_"):
                all_adapter.append(v)

    enc_mean = np.mean(all_encoder) if all_encoder else 0
    adapt_mean = np.mean(all_adapter) if all_adapter else 0
    cross_mean = np.mean(all_cross) if all_cross else 0
    cross_dormant = sum(1 for v in all_cross if v < DORMANT_THRESHOLD)
    cross_total = len(all_cross)

    verdict_color = "#ff4444" if cross_mean < DORMANT_THRESHOLD else "#44ff44"
    ax_global.text(
        0.0,
        0.15,
        f"ENCODER avg drift: {enc_mean:.4f}    "
        f"ADAPTER avg drift: {adapt_mean:.4f}    "
        f"CROSS-ATTN avg drift: {cross_mean:.4f}  "
        f"({cross_dormant}/{cross_total} components dormant)",
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
        memory_slots=cfg["memory_slots"],
        compress_chunk=cfg["compress_chunk"],
        adapter_bottleneck=cfg["adapter_bottleneck"],
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
    print("DRIFT FROM INITIALIZATION — PER SUBSYSTEM")
    print(f"{'='*60}")

    for i, metrics in enumerate(block_metrics):
        enc_vals = [v for k, v in metrics.items() if k.startswith("enc_")]
        adapt_vals = [v for k, v in metrics.items() if k.startswith("adapt_")]
        cross_vals = [v for k, v in metrics.items() if k.startswith("cross_")]

        enc_status = "ACTIVE" if np.mean(enc_vals) > 0.005 else "DORMANT"
        adapt_status = "ACTIVE" if np.mean(adapt_vals) > 0.005 else "DORMANT"
        cross_status = "ACTIVE" if np.mean(cross_vals) > 0.005 else "DORMANT"

        print(f"\n  Block {i}:")
        print(f"    Encoder:    {enc_status:8s}  (avg drift: {np.mean(enc_vals):.4f})")
        print(
            f"    Adapters:   {adapt_status:8s}  (avg drift: {np.mean(adapt_vals):.4f})"
        )
        print(
            f"    Cross-attn: {cross_status:8s}  (avg drift: {np.mean(cross_vals):.4f})"
        )

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
        description="Heatmap of bridge transformer training activity"
    )
    parser.add_argument("--checkpoint", "-c", default=CHECKPOINT_PATH)
    parser.add_argument("--output", "-o", default=OUTPUT_PATH)
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
