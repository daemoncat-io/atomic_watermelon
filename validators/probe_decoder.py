"""
Probe your Decoder.

Inspect attention patterns, embedding space, weight statistics.
Understand what the model actually learned.
"""

import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
import torch

from models.decoder import Decoder

matplotlib.use("Agg")

# Config
device = "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint_path = ""


class ASCIITokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def load_model(path: str) -> Decoder:
    """Load trained model."""
    checkpoint = torch.load(path, map_location=device)
    model = Decoder()
    model.load_state_dict(checkpoint)
    return model.to(device)


def inspect_weights(model: Decoder):
    """Print weight statistics for all layers."""
    print("\n" + "=" * 60)
    print("WEIGHT INSPECTION")
    print("=" * 60)

    for name, param in model.named_parameters():
        print(f"\n{name}")
        print(f"  Shape: {tuple(param.shape)}")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std:  {param.std().item():.6f}")
        print(f"  Min:  {param.min().item():.6f}")
        print(f"  Max:  {param.max().item():.6f}")

        zero_count = (param.abs() < 1e-6).sum().item()
        total = param.numel()
        if zero_count > 0:
            print(f"  ⚠️  Near-zero: {zero_count}/{total} ({100*zero_count/total:.2f}%)")


def extract_attention(
    model: Decoder, input_ids: torch.Tensor, layer_idx: int = 0
) -> torch.Tensor:
    """Extract attention weights from a specific layer."""
    model.eval()

    with torch.no_grad():
        x = model.tok_emb(input_ids) + model.pos_emb(
            torch.arange(input_ids.size(1), device=device)
        )

        for i, layer in enumerate(model.layers):
            if i == layer_idx:
                h = layer["ln1"](x)
                B, T, C = h.shape
                qkv = (
                    layer["attn"](h)
                    .reshape(B, T, 3, model.num_heads, model.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Causal mask
                attn = (q @ k.transpose(-2, -1)) / (model.head_dim**0.5)
                attn = attn.masked_fill(model.mask[:T, :T] == 0, float("-inf"))
                attn = F.softmax(attn, dim=-1)
                return attn

            # Continue forward
            h = layer["ln1"](x)
            qkv = (
                layer["attn"](h)
                .reshape(x.size(0), x.size(1), 3, model.num_heads, model.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) / (model.head_dim**0.5)
            attn = attn.masked_fill(
                model.mask[: x.size(1), : x.size(1)] == 0, float("-inf")
            )
            attn = F.softmax(attn, dim=-1)
            x = x + layer["proj"](
                (attn @ v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)
            )
            x = x + layer["ff"](layer["ln2"](x))

    return None


def visualize_attention(
    attention: torch.Tensor, tokens: str, save_path: str = "attention.png"
):
    """Save attention heatmaps for all heads."""
    attn = attention[0].cpu().numpy()
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(16, 8))
    axes = axes.flatten()

    for head in range(num_heads):
        ax = axes[head]
        ax.imshow(attn[head], cmap="viridis")
        ax.set_title(f"Head {head}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

        if len(tokens) <= 30:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(list(tokens), rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(list(tokens), fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def probe_embeddings(model: Decoder, tokenizer: ASCIITokenizer, chars: list[str]):
    """Find nearest neighbors in embedding space."""
    print("\n" + "=" * 60)
    print("EMBEDDING SPACE")
    print("=" * 60)

    embeddings = model.tok_emb.weight.detach()

    for char in chars:
        token_id = ord(char)
        token_emb = embeddings[token_id]

        similarities = F.cosine_similarity(token_emb.unsqueeze(0), embeddings, dim=1)
        values, indices = torch.topk(similarities, 10)

        print(f"\nNearest to '{char}' (ASCII {token_id}):")
        for val, idx in zip(values, indices):
            neighbor = chr(idx.item()) if 32 <= idx.item() < 127 else f"[{idx.item()}]"
            print(f"  {neighbor:10} sim: {val.item():.4f}")


def test_generation(model: Decoder, tokenizer: ASCIITokenizer, prompts: list[str]):
    """Test decoder generation — the defining decoder capability."""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    model.eval()
    for prompt in prompts:
        input_ids = torch.tensor([[ord(c) for c in prompt]]).to(device)
        output = model.generate(input_ids, max_tokens=150, top_k=1)
        text = "".join(chr(t) for t in output[0].tolist())
        print(f"\nPrompt: {prompt!r}")
        print(f"Output: {text}")


def analyze_layers(model: Decoder):
    """Check LayerNorm statistics across layers."""
    print("\n" + "=" * 60)
    print("LAYER ANALYSIS")
    print("=" * 60)

    for i, layer in enumerate(model.layers):
        ln1_mean = layer["ln1"].weight.mean().item()
        ln2_mean = layer["ln2"].weight.mean().item()
        print(f"Layer {i}: ln1={ln1_mean:.4f}, ln2={ln2_mean:.4f}")


if __name__ == "__main__":
    print(f"Device: {device}")

    tokenizer = ASCIITokenizer()

    print(f"Loading: {checkpoint_path}")
    model = load_model(checkpoint_path)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    inspect_weights(model)
    analyze_layers(model)
    probe_embeddings(model, tokenizer, ["a", "A", " ", ".", "?", "!", "e", "E"])

    test_text = "The dog ran."
    input_ids = torch.tensor([[ord(c) for c in test_text]]).to(device)

    for layer_idx in [0, 3, 7]:
        attn = extract_attention(model, input_ids, layer_idx=layer_idx)
        if attn is not None:
            visualize_attention(attn, test_text, f"attention_layer{layer_idx}.png")

    test_generation(
        model,
        tokenizer,
        [
            "To decode means to? ",
            "How are you feeling? ",
            "What is your purpose? ",
            "sup. ",
            "I am",
            "The dog",
        ],
    )

    print("\n✅ Probing complete")
