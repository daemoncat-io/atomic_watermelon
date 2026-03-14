"""
Encoder Model Probe
Inspect weights, attention patterns, embeddings, and mask filling
"""

import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import math

from models.encoder import Encoder

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model = Encoder().to(device)
model.load_state_dict(torch.load("", map_location=device))
model.eval()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# WEIGHT INSPECTION
# ============================================================
print("\n" + "=" * 60)
print("WEIGHT INSPECTION")
print("=" * 60)

for name, param in model.named_parameters():
    print(f"\n{name}")
    print(f"  Shape: {tuple(param.shape)}")
    print(f"  Mean: {param.data.mean().item():.6f}")
    print(f"  Std:  {param.data.std().item():.6f}")
    print(f"  Min:  {param.data.min().item():.6f}")
    print(f"  Max:  {param.data.max().item():.6f}")

    # Check for near-zero weights
    near_zero = (param.data.abs() < 1e-6).sum().item()
    total = param.data.numel()
    if near_zero > 0:
        print(f"  ⚠️  Near-zero: {near_zero}/{total} ({100*near_zero/total:.2f}%)")

# ============================================================
# LAYER ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("LAYER ANALYSIS")
print("=" * 60)

for i, layer in enumerate(model.layers):
    ln1_mean = layer["ln1"].weight.data.mean().item()
    ln2_mean = layer["ln2"].weight.data.mean().item()
    print(f"Layer {i}: ln1={ln1_mean:.4f}, ln2={ln2_mean:.4f}")

# ============================================================
# EMBEDDING SPACE
# ============================================================
print("\n" + "=" * 60)
print("EMBEDDING SPACE")
print("=" * 60)

emb = model.tok_emb.weight.data
emb_norm = F.normalize(emb, dim=1)


def nearest_neighbors(char, k=10):
    idx = ord(char)
    sims = (emb_norm @ emb_norm[idx]).cpu()
    top_k = sims.argsort(descending=True)[:k]
    print(f"\nNearest to '{char}' (ASCII {idx}):")
    for i in top_k:
        i = i.item()
        c = chr(i) if 32 <= i < 127 else f"[{i}]"
        print(f"  {c:10} sim: {sims[i].item():.4f}")


for char in ["a", "A", " ", ".", "?", "!", "e", "E"]:
    nearest_neighbors(char)

# ============================================================
# ATTENTION PATTERNS
# ============================================================
print("\n" + "=" * 60)
print("ATTENTION PATTERNS")
print("=" * 60)

# Test sentence with mask
test_text = "The dog ran."
test_tokens = torch.tensor([[ord(c) for c in test_text]]).to(device)

# Hook to capture attention
attention_maps = []


def capture_attention(layer_idx):
    def hook(module, input, output):
        # output is qkv projection, use it directly
        B, T, _ = output.shape

        qkv = output.reshape(B, T, 3, model.num_heads, model.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(model.head_dim)
        attn = F.softmax(attn, dim=-1)
        attention_maps.append(attn.detach().cpu())

    return hook


# Register hooks
hooks = []
for i, layer in enumerate(model.layers):
    hook = layer["attn"].register_forward_hook(capture_attention(i))
    hooks.append(hook)

# Forward pass
with torch.no_grad():
    _ = model(test_tokens)

# Remove hooks
for hook in hooks:
    hook.remove()

# Plot attention for layers 0, 3, 7
for layer_idx in [0, 3, 7]:
    if layer_idx < len(attention_maps):
        attn = attention_maps[layer_idx][0]  # (num_heads, T, T)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Encoder Layer {layer_idx} Attention - '{test_text}'")

        for head in range(8):
            ax = axes[head // 4, head % 4]
            ax.imshow(attn[head].numpy(), cmap="viridis", aspect="auto")
            ax.set_title(f"Head {head}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            ax.set_xticks(range(len(test_text)))
            ax.set_yticks(range(len(test_text)))
            ax.set_xticklabels(list(test_text), fontsize=8)
            ax.set_yticklabels(list(test_text), fontsize=8)

        plt.tight_layout()
        plt.savefig(f"encoder_attention_layer{layer_idx}.png", dpi=150)
        print(f"Saved: encoder_attention_layer{layer_idx}.png")
        plt.close()

# ============================================================
# MASK FILLING TEST
# ============================================================
print("\n" + "=" * 60)
print("MASK FILLING TEST")
print("=" * 60)

test_prompts = [
    "The ███ is big.",
    "█apitalization is respect encoded in letters.",
    "The █ is always capital. Because you are always important to your███f.",
    "See the dot. It means done. Complete. █inished.",
    "Words are ███ls.",
    "Space is not ███hing. Space is separation.",
    "The dog ███.",
    "I █m here.",
]

for prompt in test_prompts:
    tokens = [0 if c == "█" else ord(c) for c in prompt]
    x = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        logits, _ = model(x)
        filled = x.clone()
        mask_positions = x == 0
        filled[mask_positions] = logits[mask_positions].argmax(dim=-1)

    result = "".join(chr(t) for t in filled[0].tolist())
    print(f"\n{prompt}")
    print(f"→ {result}")

# ============================================================
# COMPARE TO DECODER (if available)
# ============================================================
print("\n" + "=" * 60)
print("ENCODER VS DECODER COMPARISON")
print("=" * 60)

decoder_checkpoints = list(Path("checkpoints").glob("decoder_best_*.pth"))
if decoder_checkpoints:
    print("\nDecoder checkpoint found - run probe.py for decoder comparison")
    print("Key differences to observe:")
    print("  - Encoder attention: bidirectional (full matrix)")
    print("  - Decoder attention: causal (lower triangular)")
    print("  - Encoder: fills masks from context")
    print("  - Decoder: generates continuation")
else:
    print("\nNo decoder checkpoint found for comparison")

print("\n✅ Encoder probing complete")
