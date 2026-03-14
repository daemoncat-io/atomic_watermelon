"""
probe_bridge_cross_attention.py

Introspect AtomicWatermelon models. Extract attention patterns,
embedding topology, weight statistics, generation behavior. Output to CLI and JSON.

No frameworks. No dependencies beyond torch and matplotlib.
"""

import torch.nn.functional as F
import matplotlib
import torch

from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from model.atomic_watermelon import AtomicWatermelon
from datasets.bpe import BPETokenizer

matplotlib.use("Agg")

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "mps"
CHECKPOINT_PATH = "cross_attention_20260225_164548_best.pth"
OUTPUT_DIR = Path("probe_results")
TOKENIZER_PATH = "datasets/tokenizer.json"


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
    layer_idx: int
    attn_type: str  # "self" | "cross" | "memory_compress"
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
    layer_idx: int
    ln1_weight_mean: float
    ln2_weight_mean: float
    self_attn_qkv_norm: float | None
    cross_attn_q_norm: float | None
    cross_attn_kv_norm: float | None
    ff_w1_norm: float | None
    adapter_down_norm: float | None
    adapter_up_norm: float | None


@dataclass
class MemoryStats:
    memory_slots: int
    compress_chunk: int
    adapter_bottleneck: int
    memory_state_shape: tuple[int, ...] | None
    memory_norm: float | None


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
    memory_stats: MemoryStats | None = None
    attention_visualizations: list[str] = field(default_factory=list)


# ============================================================
# MODEL LOADING
# ============================================================


def load_model(checkpoint_path: str, device: str) -> tuple[AtomicWatermelon, dict]:
    """
    Load model from checkpoint. Config lives in the checkpoint.
    Returns (model, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        if "layers" in s.name:
            # Extract layer component: layers.0.self_attn -> self_attn
            parts = s.name.split(".")
            if len(parts) >= 3:
                prefix = f"layers.*.{parts[2]}"
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


def extract_attention_patterns(
    model: AtomicWatermelon,
    input_ids: torch.Tensor,
    memory: torch.Tensor | None,
    layer_indices: list[int],
) -> tuple[list[AttentionPattern], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Extract self-attention and cross-attention weights from specified layers.

    Hooks into the forward pass. Returns patterns metadata, raw tensors keyed
    by "L{layer}_{type}", and updated memory state.
    """
    patterns: list[AttentionPattern] = []
    raw_tensors: dict[str, torch.Tensor] = {}
    captured: dict[str, list[torch.Tensor]] = {}

    hooks = []

    # Walk the model to find attention modules in target layers
    if hasattr(model, "layers"):
        for layer_idx in layer_indices:
            if layer_idx >= len(model.layers):
                continue

            layer = model.layers[layer_idx]

            # Hook self-attention
            self_attn = _get_attr_chain(layer, ["self_attn", "attn", "attention"])
            if self_attn is not None:
                key = f"L{layer_idx}_self"
                hooks.append(_register_attn_hook(self_attn, key, captured))

            # Hook cross-attention (memory attention)
            cross_attn = _get_attr_chain(
                layer, ["cross_attn", "memory_attn", "mem_attn"]
            )
            if cross_attn is not None:
                key = f"L{layer_idx}_cross"
                hooks.append(_register_attn_hook(cross_attn, key, captured))

    # Forward pass with memory
    with torch.no_grad():
        try:
            logits, _, new_memory = model(input_ids, memory=memory)
        except Exception as e:
            print(f"  ⚠️  Forward pass failed: {e}")
            new_memory = memory

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Package results
    seq_len = input_ids.shape[1]
    for key, tensors in captured.items():
        if not tensors:
            continue

        attn_weights = tensors[0].squeeze(0)  # Remove batch dim
        raw_tensors[key] = attn_weights

        parts = key.split("_", 1)
        layer_idx = int(parts[0][1:])
        attn_type = parts[1] if len(parts) > 1 else "self"

        patterns.append(
            AttentionPattern(
                layer_idx=layer_idx,
                attn_type=attn_type,
                head_idx=None,
                input_text="",  # filled by caller
                pattern_shape=tuple(attn_weights.shape),
            )
        )

    return patterns, raw_tensors, new_memory


def _get_attr_chain(obj: Any, names: list[str]) -> Any:
    """Try multiple attribute names, return first that exists."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return None


def _register_attn_hook(
    module: torch.nn.Module,
    key: str,
    captured: dict[str, list[torch.Tensor]],
) -> torch.utils.hooks.RemovableHook:
    """Register a forward hook that captures attention weights."""
    captured[key] = []

    def hook_fn(mod, inp, out):
        # Convention 1: module stores .attn_weights after forward
        if hasattr(mod, "attn_weights") and mod.attn_weights is not None:
            captured[key].append(mod.attn_weights.detach().cpu())
            return

        # Convention 2: output is (output, attn_weights) tuple
        if isinstance(out, tuple) and len(out) > 1:
            candidate = out[1]
            if isinstance(candidate, torch.Tensor) and candidate.dim() >= 2:
                captured[key].append(candidate.detach().cpu())
                return

    return module.register_forward_hook(hook_fn)


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
    """Test model generation. Matches training loop generate() signature."""
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
    """Extract per-layer statistics including cross-attention and adapter norms."""
    results = []

    layers = getattr(model, "layers", None)
    if layers is None:
        return results

    for i, layer in enumerate(layers):
        # Layer norms
        ln1 = _get_attr_chain(layer, ["ln1", "norm1", "ln_self"])
        ln2 = _get_attr_chain(layer, ["ln2", "norm2", "ln_ff"])

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

        # Self-attention
        self_attn = _get_attr_chain(layer, ["self_attn", "attn"])
        self_qkv_norm = _param_norm(self_attn, ["qkv", "in_proj", "q_proj", "W_q"])

        # Cross-attention
        cross_attn = _get_attr_chain(layer, ["cross_attn", "memory_attn", "mem_attn"])
        cross_q_norm = _param_norm(cross_attn, ["q_proj", "W_q", "query"])
        cross_kv_norm = _param_norm(cross_attn, ["kv_proj", "W_kv", "key"])

        # Feedforward
        ff = _get_attr_chain(layer, ["ff", "mlp", "feedforward"])
        ff_w1_norm = _param_norm(ff, ["w1", "fc1", "c_fc", "linear1"])

        # Adapter
        adapter = _get_attr_chain(layer, ["adapter", "bottleneck"])
        adapter_down = _param_norm(adapter, ["down", "down_proj", "W_down"])
        adapter_up = _param_norm(adapter, ["up", "up_proj", "W_up"])

        results.append(
            LayerStats(
                layer_idx=i,
                ln1_weight_mean=ln1_mean,
                ln2_weight_mean=ln2_mean,
                self_attn_qkv_norm=self_qkv_norm,
                cross_attn_q_norm=cross_q_norm,
                cross_attn_kv_norm=cross_kv_norm,
                ff_w1_norm=ff_w1_norm,
                adapter_down_norm=adapter_down,
                adapter_up_norm=adapter_up,
            )
        )

    return results


def _param_norm(module: Any, names: list[str]) -> float | None:
    """Get weight norm from first matching parameter name in module."""
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


def print_layer_stats(stats: list[LayerStats]):
    """CLI output for layer analysis."""
    print("\n" + "=" * 70)
    print("LAYER ANALYSIS")
    print("=" * 70)

    for s in stats:
        print(f"\nLayer {s.layer_idx}:")
        print(f"  ln1 weight mean:    {s.ln1_weight_mean:.4f}")
        print(f"  ln2 weight mean:    {s.ln2_weight_mean:.4f}")
        if s.self_attn_qkv_norm is not None:
            print(f"  self-attn qkv norm: {s.self_attn_qkv_norm:.4f}")
        if s.cross_attn_q_norm is not None:
            print(f"  cross-attn Q norm:  {s.cross_attn_q_norm:.4f}")
        if s.cross_attn_kv_norm is not None:
            print(f"  cross-attn KV norm: {s.cross_attn_kv_norm:.4f}")
        if s.ff_w1_norm is not None:
            print(f"  ff w1 norm:         {s.ff_w1_norm:.4f}")
        if s.adapter_down_norm is not None:
            print(f"  adapter down norm:  {s.adapter_down_norm:.4f}")
        if s.adapter_up_norm is not None:
            print(f"  adapter up norm:    {s.adapter_up_norm:.4f}")


# ============================================================
# MEMORY ANALYSIS
# ============================================================


def analyze_memory(
    model: AtomicWatermelon,
    tokenizer: BPETokenizer,
    device: str,
    cfg: dict,
) -> MemoryStats:
    """
    Run a forward pass and inspect the resulting memory state.
    """
    # Encode a representative chunk to prime memory
    test_text = "The mind is not the brain. Experience is something else."
    token_ids = tokenizer.encode(test_text)
    x = torch.tensor([token_ids]).to(device)

    memory = None
    with torch.no_grad():
        try:
            _, _, memory = model(x, memory=memory)
        except Exception as e:
            print(f"  ⚠️  Memory extraction failed: {e}")

    mem_shape = None
    mem_norm = None
    if memory is not None:
        mem_shape = tuple(memory.shape)
        mem_norm = memory.norm().item()

    return MemoryStats(
        memory_slots=cfg.get("memory_slots", 0),
        compress_chunk=cfg.get("compress_chunk", 0),
        adapter_bottleneck=cfg.get("adapter_bottleneck", 0),
        memory_state_shape=mem_shape,
        memory_norm=mem_norm,
    )


def print_memory_stats(stats: MemoryStats):
    """CLI output for memory analysis."""
    print("\n" + "=" * 70)
    print("MEMORY ANALYSIS")
    print("=" * 70)
    print(f"  Slots:              {stats.memory_slots}")
    print(f"  Compress chunk:     {stats.compress_chunk}")
    print(f"  Adapter bottleneck: {stats.adapter_bottleneck}")
    if stats.memory_state_shape is not None:
        print(f"  Memory shape:       {stats.memory_state_shape}")
        print(f"  Memory L2 norm:     {stats.memory_norm:.4f}")
    else:
        print("  ⚠️  No memory state produced")


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
    attention_layers: list[int] | None = None,
    attention_text: str = "The mind is not the brain. Experience is something else entirely.",
    max_gen_tokens: int = 144,
) -> ProbeResults:
    """
    Run complete probe on a AtomicWatermelon.
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
    if attention_layers is None:
        attention_layers = [0, 2, 5]

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

    # 3. Memory analysis
    results.memory_stats = analyze_memory(model, tokenizer, DEVICE, cfg)
    print_memory_stats(results.memory_stats)

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

    attn_patterns, raw_tensors, _ = extract_attention_patterns(
        model,
        input_ids,
        memory=None,
        layer_indices=attention_layers,
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
        print(
            "  ⚠️  No attention weights captured. Model may not expose them via hooks."
        )
        print(
            "      Check if attention modules store .attn_weights or return (out, weights) tuples."
        )

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
        "--layers",
        "-l",
        type=int,
        nargs="+",
        default=[0, 2, 5],
        help="Layer indices for attention extraction",
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
        attention_layers=args.layers,
        max_gen_tokens=args.max_tokens,
    )
