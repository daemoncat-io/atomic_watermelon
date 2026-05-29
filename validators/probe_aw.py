"""
probe_aw.py

Introspect AtomicWatermelon models. Extract attention patterns,
embedding topology, weight statistics, generation behavior. Output to CLI and JSON.

NOTE: AtomicWatermelon is a dual-stream shared-weight transformer:
    - blocks live in `model.blocks` (BridgeBlock instances)
    - each block shares ONE set of weights between the encoder (bidirectional)
      and decoder (causal) streams: enc_attn / enc_ln1 / enc_ln2 / enc_ff
    - there is NO memory, NO cross-attention, NO adapters, NO compression.
    - forward signature is forward(x, targets=None) -> (logits, loss, None)

MultiHeadAttention does not expose attention weights, so this probe
monkeypatches its forward at runtime to stash them for visualization.
"""

import torch.nn.functional as F
import matplotlib
import torch

from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Any
import math
import json

from models.atomic_watermelon import AtomicWatermelon, MultiHeadAttention
from datasets.bpe import BPETokenizer

matplotlib.use("Agg")

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOKENIZER_PATH = "datasets/tokenizer.json"
OUTPUT_DIR = Path("probe_results")
CHECKPOINT_PATH = "checkpoints/atomic_watermelon_20260513_225950_best.pth"


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class WeightStats:
    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float
    near_zero_count: int
    near_zero_pct: float
    total_params: int


@dataclass
class EmbeddingNeighbor:
    token: str
    token_id: int
    similarity: float


@dataclass
class EmbeddingProbe:
    query_token: str
    query_id: int
    neighbors: list[EmbeddingNeighbor]


@dataclass
class AttentionPattern:
    block_idx: int
    stream: str  # "enc" (bidirectional) | "dec" (causal)
    head_idx: int | None
    input_text: str
    pattern_shape: tuple[int, ...]
    # pattern matrices excluded from JSON by default (huge), saved as .pt


@dataclass
class GenerationSample:
    prompt: str
    output: str
    tokens_generated: int


@dataclass
class LayerStats:
    block_idx: int
    enc_ln1_weight_mean: float
    enc_ln2_weight_mean: float
    attn_qkv_norm: float | None
    attn_o_norm: float | None
    ff_w1_norm: float | None
    ff_w2_norm: float | None


@dataclass
class ProbeResults:
    """Complete probe output."""

    timestamp: str
    device: str
    checkpoint: str
    config: dict[str, Any]
    total_params: int
    trainable_params: int
    weight_stats: list[WeightStats] = field(default_factory=list)
    embedding_probes: list[EmbeddingProbe] = field(default_factory=list)
    attention_patterns: list[AttentionPattern] = field(default_factory=list)
    generation_samples: list[GenerationSample] = field(default_factory=list)
    layer_stats: list[LayerStats] = field(default_factory=list)
    attention_visualizations: list[str] = field(default_factory=list)


# ============================================================
# MODEL LOADING
# ============================================================


def load_model(checkpoint_path: str, device: str) -> tuple[AtomicWatermelon, dict]:
    """
    Load model from checkpoint. Config lives in the checkpoint.
    Returns (model, config).

    Only reads keys that AtomicWatermelon actually accepts. `max_seq_len`
    is read with a `context_length` fallback for older checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = AtomicWatermelon(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg.get("max_seq_len", cfg.get("context_length", 2048)),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device).eval(), cfg


# ============================================================
# WEIGHT INSPECTION
# ============================================================


def inspect_weights(model: torch.nn.Module) -> list[WeightStats]:
    """Extract statistics from all model parameters."""
    results = []

    for name, param in model.named_parameters():
        data = param.detach()
        near_zero = (data.abs() < 1e-6).sum().item()
        total = data.numel()

        stats = WeightStats(
            name=name,
            shape=tuple(data.shape),
            mean=data.mean().item(),
            std=data.std().item(),
            min=data.min().item(),
            max=data.max().item(),
            near_zero_count=int(near_zero),
            near_zero_pct=100 * near_zero / total if total > 0 else 0.0,
            total_params=total,
        )
        results.append(stats)

    return results


def print_weight_stats(stats: list[WeightStats]):
    """CLI output for weight inspection."""
    print("\n" + "=" * 70)
    print("WEIGHT INSPECTION")
    print("=" * 70)

    # Summary: group by component
    components: dict[str, list[WeightStats]] = {}
    for s in stats:
        prefix = s.name.split(".")[0]
        if "blocks" in s.name:
            # Extract block component: blocks.0.enc_attn -> blocks.*.enc_attn
            parts = s.name.split(".")
            if len(parts) >= 3:
                prefix = f"blocks.*.{parts[2]}"
        components.setdefault(prefix, []).append(s)

    for component, entries in components.items():
        total = sum(e.total_params for e in entries)
        mean_std = sum(e.std * e.total_params for e in entries) / max(total, 1)
        near_zero = sum(e.near_zero_count for e in entries)
        nz_pct = 100 * near_zero / max(total, 1)
        print(f"\n{component}: {total:,} params, avg_std={mean_std:.4f}", end="")
        if nz_pct > 1.0:
            print(f" ⚠️ {nz_pct:.1f}% near-zero", end="")
        print()

    # Full detail
    print(f"\n--- All {len(stats)} parameter tensors ---")
    for s in stats:
        print(
            f"  {s.name:60s} {str(s.shape):20s} μ={s.mean:+.4f} σ={s.std:.4f} [{s.min:.4f}, {s.max:.4f}]"
        )


# ============================================================
# EMBEDDING SPACE
# ============================================================


def probe_embeddings(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    query_tokens: list[str],
    top_k: int = 10,
) -> list[EmbeddingProbe]:
    """Find nearest neighbors in embedding space for given tokens."""
    results = []

    # Locate embedding weight
    emb_weight = None
    for name in ["tok_emb.weight", "embedding.weight", "embed.weight", "wte.weight"]:
        try:
            obj = model
            for part in name.split("."):
                obj = getattr(obj, part)
            emb_weight = obj.detach()
            break
        except AttributeError:
            continue

    if emb_weight is None:
        print("  ⚠️  Could not locate embedding weights")
        return results

    # Build reverse vocab: id -> token string
    # BPETokenizer stores vocab as str->int, invert it
    id_to_token: dict[int, str] = {}
    if hasattr(tokenizer, "vocab"):
        id_to_token = {v: k for k, v in tokenizer.vocab.items()}

    for query in query_tokens:
        token_ids = tokenizer.encode(query)
        if not token_ids:
            continue

        # Use the first token of the encoded query
        token_id = token_ids[0]
        if token_id >= emb_weight.shape[0]:
            continue

        token_emb = emb_weight[token_id]
        similarities = F.cosine_similarity(token_emb.unsqueeze(0), emb_weight, dim=1)
        values, indices = torch.topk(similarities, min(top_k, len(similarities)))

        neighbors = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            neighbor_str = id_to_token.get(idx, f"[{idx}]")
            neighbors.append(
                EmbeddingNeighbor(
                    token=repr(neighbor_str),
                    token_id=idx,
                    similarity=val,
                )
            )

        results.append(
            EmbeddingProbe(
                query_token=repr(query),
                query_id=token_id,
                neighbors=neighbors,
            )
        )

    return results


def print_embedding_probes(probes: list[EmbeddingProbe]):
    """CLI output for embedding probes."""
    print("\n" + "=" * 70)
    print("EMBEDDING SPACE")
    print("=" * 70)

    for probe in probes:
        print(f"\nNearest to {probe.query_token} (id={probe.query_id}):")
        for n in probe.neighbors:
            print(f"  {n.token:20s} id={n.token_id:5d}  sim={n.similarity:.4f}")


# ============================================================
# ATTENTION EXTRACTION
# ============================================================
#
# MultiHeadAttention computes softmax weights internally and discards them.
# We monkeypatch its forward at probe time so each call stashes the most
# recent attention weights on the module as `.attn_weights`. The dual-stream
# BridgeBlock calls the SAME enc_attn module twice per forward (once for the
# encoder/bidirectional stream, once for the decoder/causal stream), so we
# capture both invocations and tell them apart by whether a mask was passed.


def _patched_attention_forward(self, x, mask=None):
    """Drop-in replacement for MultiHeadAttention.forward that stashes weights."""
    B, T, C = x.shape

    qkv = self.w_qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)

    # Stash a detached copy. mask is None for the encoder (bidirectional)
    # stream, present for the decoder (causal) stream.
    self._captured_attn = attn.detach().cpu()
    self._captured_causal = mask is not None

    attn = self.dropout(attn)
    out = (attn @ v).transpose(1, 2).reshape(B, T, C)
    return self.w_o(out)


def extract_attention_patterns(
    model: AtomicWatermelon,
    input_ids: torch.Tensor,
    block_indices: list[int],
) -> tuple[list[AttentionPattern], dict[str, torch.Tensor]]:
    """
    Run one forward pass and capture self-attention weights from the encoder
    (bidirectional) and decoder (causal) streams of the requested blocks.

    Returns (patterns metadata, raw tensors keyed by "B{block}_{stream}").
    """
    patterns: list[AttentionPattern] = []
    raw_tensors: dict[str, torch.Tensor] = {}

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        print("  ⚠️  Model has no `blocks` attribute")
        return patterns, raw_tensors

    # Monkeypatch MultiHeadAttention.forward globally for the duration of the pass.
    original_forward = MultiHeadAttention.forward
    MultiHeadAttention.forward = _patched_attention_forward

    # Per-block capture: enc_attn is shared, so we need to intercept each of
    # its two calls. We do that by wrapping the block's enc_attn with a small
    # recording shim via a forward hook that reads the freshly-stashed weights.
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(block_idx: int):
        def hook_fn(mod, inp, out):
            attn = getattr(mod, "_captured_attn", None)
            if attn is None:
                return
            stream = "dec" if getattr(mod, "_captured_causal", False) else "enc"
            key = f"B{block_idx}_{stream}"
            # squeeze batch dim -> [n_heads, q, kv]
            captured[key] = attn.squeeze(0)

        return hook_fn

    target = set(i for i in block_indices if 0 <= i < len(blocks))
    for idx in target:
        h = blocks[idx].enc_attn.register_forward_hook(make_hook(idx))
        hooks.append(h)

    try:
        with torch.no_grad():
            model(input_ids)  # forward(x, targets=None) -> (logits, loss, None)
    except Exception as e:
        print(f"  ⚠️  Forward pass failed: {e}")
    finally:
        for h in hooks:
            h.remove()
        MultiHeadAttention.forward = original_forward
        # Clean up stashed attrs
        for idx in target:
            mod = blocks[idx].enc_attn
            for attr in ("_captured_attn", "_captured_causal"):
                if hasattr(mod, attr):
                    delattr(mod, attr)

    for key, attn_weights in captured.items():
        raw_tensors[key] = attn_weights
        block_idx = int(key.split("_", 1)[0][1:])
        stream = key.split("_", 1)[1]
        patterns.append(
            AttentionPattern(
                block_idx=block_idx,
                stream=stream,
                head_idx=None,
                input_text="",  # filled by caller
                pattern_shape=tuple(attn_weights.shape),
            )
        )

    return patterns, raw_tensors


def visualize_attention(
    attn: torch.Tensor,
    input_text: str,
    output_path: str,
    label: str = "",
    tokenizer: BPETokenizer | None = None,
) -> str:
    """
    Visualize attention pattern and save to file.
    Handles [n_heads, q_len, kv_len] or [q_len, kv_len].
    """
    attn_np = attn.cpu().float().numpy()

    # Token labels for axes
    token_labels = None
    if tokenizer is not None and len(input_text) > 0:
        token_ids = tokenizer.encode(input_text)
        id_to_tok = (
            {v: k for k, v in tokenizer.vocab.items()}
            if hasattr(tokenizer, "vocab")
            else {}
        )
        token_labels = [id_to_tok.get(tid, f"[{tid}]")[:6] for tid in token_ids]

    if len(attn_np.shape) == 3:
        n_heads = attn_np.shape[0]
        cols = min(n_heads, 4)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        if n_heads == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            im = ax.imshow(attn_np[head_idx], cmap="viridis", aspect="auto")
            ax.set_title(f"Head {head_idx}", fontsize=9)
            ax.set_xlabel("Key", fontsize=7)
            ax.set_ylabel("Query", fontsize=7)

            if token_labels and attn_np.shape[1] <= 40:
                q_labels = token_labels[: attn_np.shape[1]]
                k_labels = (
                    token_labels[: attn_np.shape[2]]
                    if attn_np.shape[2] <= len(token_labels)
                    else None
                )
                ax.set_yticks(range(len(q_labels)))
                ax.set_yticklabels(q_labels, fontsize=5)
                if k_labels:
                    ax.set_xticks(range(len(k_labels)))
                    ax.set_xticklabels(k_labels, fontsize=5, rotation=45, ha="right")

        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].set_visible(False)

        plt.colorbar(im, ax=axes[:n_heads], shrink=0.6)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_np, cmap="viridis", aspect="auto")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.colorbar(im, ax=ax)

    plt.suptitle(label, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


# ============================================================
# GENERATION
# ============================================================


def test_generation(
    model: AtomicWatermelon,
    tokenizer: BPETokenizer,
    prompts: list[str],
    device: str,
    max_tokens: int = 144,
    top_k: int = 40,
    temperature: float = 0.8,
) -> list[GenerationSample]:
    """Test model generation. Matches model.generate() signature."""
    results = []
    model.eval()

    for prompt_text in prompts:
        prompt_ids = tokenizer.encode(prompt_text)
        prompt = torch.tensor([prompt_ids]).to(device)

        try:
            out, _ = model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            decoded = tokenizer.decode(out[0].tolist())
            tokens_generated = len(out[0]) - len(prompt_ids)
        except Exception as e:
            decoded = f"[Generation failed: {e}]"
            tokens_generated = 0

        results.append(
            GenerationSample(
                prompt=prompt_text,
                output=decoded,
                tokens_generated=tokens_generated,
            )
        )

    return results


def print_generation_samples(samples: list[GenerationSample]):
    """CLI output for generation tests."""
    print("\n" + "=" * 70)
    print("GENERATION TEST")
    print("=" * 70)

    for sample in samples:
        print(f"\nPrompt: {sample.prompt!r}")
        print(f"Output ({sample.tokens_generated} tokens):")
        print(f"  {sample.output[:300]}")
        if len(sample.output) > 300:
            print(f"  ...[truncated, {len(sample.output)} chars total]")


# ============================================================
# LAYER ANALYSIS
# ============================================================


def analyze_layers(model: AtomicWatermelon) -> list[LayerStats]:
    """
    Extract per-block statistics. Each BridgeBlock shares one set of weights
    between the encoder and decoder streams: enc_ln1, enc_attn, enc_ln2, enc_ff.
    """
    results = []

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return results

    for i, block in enumerate(blocks):
        ln1 = getattr(block, "enc_ln1", None)
        ln2 = getattr(block, "enc_ln2", None)

        ln1_mean = (
            ln1.weight.mean().item()
            if ln1 is not None and hasattr(ln1, "weight")
            else 0.0
        )
        ln2_mean = (
            ln2.weight.mean().item()
            if ln2 is not None and hasattr(ln2, "weight")
            else 0.0
        )

        attn = getattr(block, "enc_attn", None)
        qkv_norm = _param_norm(attn, ["w_qkv"])
        o_norm = _param_norm(attn, ["w_o"])

        # FeedForward stores layers in a Sequential `net`: [Linear, GELU, Dropout, Linear]
        ff = getattr(block, "enc_ff", None)
        ff_w1_norm = _seq_linear_norm(ff, 0)
        ff_w2_norm = _seq_linear_norm(ff, -1)

        results.append(
            LayerStats(
                block_idx=i,
                enc_ln1_weight_mean=ln1_mean,
                enc_ln2_weight_mean=ln2_mean,
                attn_qkv_norm=qkv_norm,
                attn_o_norm=o_norm,
                ff_w1_norm=ff_w1_norm,
                ff_w2_norm=ff_w2_norm,
            )
        )

    return results


def _param_norm(module: Any, names: list[str]) -> float | None:
    """Get weight norm from first matching submodule/parameter name in module."""
    if module is None:
        return None
    for name in names:
        param = getattr(module, name, None)
        if param is not None:
            if hasattr(param, "weight"):
                return param.weight.norm().item()
            elif isinstance(param, torch.nn.Parameter):
                return param.norm().item()
            elif isinstance(param, torch.Tensor):
                return param.norm().item()
    return None


def _seq_linear_norm(ff: Any, which: int) -> float | None:
    """Weight norm of the `which`-th Linear inside FeedForward.net."""
    if ff is None:
        return None
    net = getattr(ff, "net", None)
    if net is None:
        return None
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    if not linears:
        return None
    try:
        return linears[which].weight.norm().item()
    except IndexError:
        return None


def print_layer_stats(stats: list[LayerStats]):
    """CLI output for layer analysis."""
    print("\n" + "=" * 70)
    print("LAYER ANALYSIS (shared enc/dec weights per block)")
    print("=" * 70)

    for s in stats:
        print(f"\nBlock {s.block_idx}:")
        print(f"  enc_ln1 weight mean: {s.enc_ln1_weight_mean:.4f}")
        print(f"  enc_ln2 weight mean: {s.enc_ln2_weight_mean:.4f}")
        if s.attn_qkv_norm is not None:
            print(f"  attn w_qkv norm:     {s.attn_qkv_norm:.4f}")
        if s.attn_o_norm is not None:
            print(f"  attn w_o norm:       {s.attn_o_norm:.4f}")
        if s.ff_w1_norm is not None:
            print(f"  ff linear1 norm:     {s.ff_w1_norm:.4f}")
        if s.ff_w2_norm is not None:
            print(f"  ff linear2 norm:     {s.ff_w2_norm:.4f}")


# ============================================================
# DUAL-STREAM ANALYSIS
# ============================================================


def analyze_dual_stream(
    model: AtomicWatermelon,
    tokenizer: BPETokenizer,
    device: str,
) -> dict[str, Any]:
    """
    AtomicWatermelon has no memory. Instead, characterize the dual-stream
    design: run one forward pass and report how divergent the encoder
    (bidirectional) and decoder (causal) attention patterns are at block 0,
    using the shared enc_attn module.
    """
    test_text = "The mind is not the brain. Experience is something else."
    token_ids = tokenizer.encode(test_text)
    x = torch.tensor([token_ids]).to(device)

    info: dict[str, Any] = {
        "n_blocks": len(getattr(model, "blocks", [])),
        "shared_weights": True,
        "encoder_attention": "bidirectional",
        "decoder_attention": "causal",
    }

    patterns, raw = extract_attention_patterns(model, x, block_indices=[0])
    enc = raw.get("B0_enc")
    dec = raw.get("B0_dec")
    if enc is not None and dec is not None and enc.shape == dec.shape:
        diff = (enc - dec).norm().item()
        info["block0_enc_dec_attn_l2_diff"] = diff
        info["block0_attn_shape"] = tuple(enc.shape)

    return info


def print_dual_stream(info: dict[str, Any]):
    """CLI output for dual-stream analysis."""
    print("\n" + "=" * 70)
    print("DUAL-STREAM ANALYSIS")
    print("=" * 70)
    print(f"  Blocks:             {info.get('n_blocks')}")
    print(f"  Shared weights:     {info.get('shared_weights')}")
    print(f"  Encoder attention:  {info.get('encoder_attention')}")
    print(f"  Decoder attention:  {info.get('decoder_attention')}")
    if "block0_enc_dec_attn_l2_diff" in info:
        print(f"  Block0 attn shape:  {info['block0_attn_shape']}")
        print(f"  Block0 enc/dec L2:  {info['block0_enc_dec_attn_l2_diff']:.4f}")
    else:
        print("  ⚠️  Could not compare enc/dec attention at block 0")


# ============================================================
# JSON SERIALIZATION
# ============================================================


def results_to_dict(results: ProbeResults) -> dict[str, Any]:
    """Convert ProbeResults to JSON-serializable dict."""

    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    return convert(results)


def save_json(results: ProbeResults, output_path: Path):
    """Save results to JSON file."""
    data = results_to_dict(results)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ JSON saved: {output_path}")


# ============================================================
# MAIN
# ============================================================


def probe(
    checkpoint_path: str = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
    tokenizer_path: str = TOKENIZER_PATH,
    embedding_queries: list[str] | None = None,
    test_prompts: list[str] | None = None,
    attention_blocks: list[int] | None = None,
    attention_text: str = "The mind is not the brain. Experience is something else entirely.",
    max_gen_tokens: int = 144,
) -> ProbeResults:
    """
    Run complete probe on an AtomicWatermelon.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Defaults
    if embedding_queries is None:
        embedding_queries = [
            "The",
            "the",
            " ",
            ".",
            "?",
            "!",
            "\n",
            "mind",
            "brain",
            "pattern",
            "truth",
            "knowledge",
            "Lovelace",
            "engine",
            "neuron",
        ]
    if test_prompts is None:
        test_prompts = [
            "Lovelace completed her paper. Sixty-six pages. ",
            "If one then two then three then ",
            "Reality is frequently ",
            "What is existence?",
            "The Analytical Engine has no pretensions whatever to ",
            "Something it is like to feel ",
        ]
    if attention_blocks is None:
        attention_blocks = [0, 2, 5]

    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Load model
    print(f"Device: {DEVICE}")
    print(f"Loading: {checkpoint_path}")
    model, cfg = load_model(checkpoint_path, DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")

    # Initialize results
    timestamp = datetime.now().isoformat()
    results = ProbeResults(
        timestamp=timestamp,
        device=DEVICE,
        checkpoint=str(checkpoint_path),
        config=cfg,
        total_params=total_params,
        trainable_params=trainable_params,
    )

    # 1. Weight inspection
    results.weight_stats = inspect_weights(model)
    print_weight_stats(results.weight_stats)

    # 2. Layer analysis
    results.layer_stats = analyze_layers(model)
    print_layer_stats(results.layer_stats)

    # 3. Dual-stream analysis (replaces the memory analysis; model has no memory)
    dual = analyze_dual_stream(model, tokenizer, DEVICE)
    print_dual_stream(dual)

    # 4. Embedding space
    results.embedding_probes = probe_embeddings(model, tokenizer, embedding_queries)
    print_embedding_probes(results.embedding_probes)

    # 5. Attention patterns
    print("\n" + "=" * 70)
    print("ATTENTION PATTERNS")
    print("=" * 70)

    token_ids = tokenizer.encode(attention_text)
    input_ids = torch.tensor([token_ids]).to(DEVICE)
    print(f"  Input: {attention_text!r}")
    print(f"  Tokens: {len(token_ids)}")

    attn_patterns, raw_tensors = extract_attention_patterns(
        model,
        input_ids,
        block_indices=attention_blocks,
    )

    for pattern in attn_patterns:
        pattern.input_text = attention_text

    results.attention_patterns = attn_patterns

    # Visualize and save raw tensors
    for key, tensor in raw_tensors.items():
        # Save raw attention tensor
        tensor_path = output_dir / f"attn_{key}.pt"
        torch.save(tensor, tensor_path)
        print(f"  {key}: shape={tuple(tensor.shape)} -> {tensor_path}")

        # Visualization
        viz_path = output_dir / f"attn_{key}.png"
        saved_path = visualize_attention(
            tensor,
            attention_text,
            str(viz_path),
            label=f"{key} | {attention_text[:50]}",
            tokenizer=tokenizer,
        )
        results.attention_visualizations.append(saved_path)
        print(f"    Visualization -> {saved_path}")

    if not raw_tensors:
        print("  ⚠️  No attention weights captured.")

    # 6. Generation tests
    results.generation_samples = test_generation(
        model,
        tokenizer,
        test_prompts,
        DEVICE,
        max_tokens=max_gen_tokens,
    )
    print_generation_samples(results.generation_samples)

    # Save JSON
    json_path = output_dir / f"probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json(results, json_path)

    print("\n" + "=" * 70)
    print("✅ PROBE COMPLETE")
    print("=" * 70)

    return results


# ============================================================
# CLI ENTRY POINT
# ============================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe AtomicWatermelon model")
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=CHECKPOINT_PATH,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        default=TOKENIZER_PATH,
        help="Path to BPE tokenizer JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--text",
        "-t",
        default="The mind is not the brain. Experience is something else entirely.",
        help="Text for attention analysis",
    )
    parser.add_argument(
        "--blocks",
        "-b",
        type=int,
        nargs="+",
        default=[0, 2, 5],
        help="Block indices for attention extraction",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=144,
        help="Max tokens for generation",
    )

    args = parser.parse_args()

    probe(
        checkpoint_path=args.checkpoint,
        output_dir=Path(args.output),
        tokenizer_path=args.tokenizer,
        attention_text=args.text,
        attention_blocks=args.blocks,
        max_gen_tokens=args.max_tokens,
    )
