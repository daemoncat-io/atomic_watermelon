"""
Minimal Transformer - 26M params
Encoder-only (MLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        embed_dim=512,
        num_layers=8,
        num_heads=8,
        context_length=512,
        dropout=0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(embed_dim),
                        "attn": nn.Linear(embed_dim, 3 * embed_dim),
                        "proj": nn.Linear(embed_dim, embed_dim),
                        "ln2": nn.LayerNorm(embed_dim),
                        "ff": nn.Sequential(
                            nn.Linear(embed_dim, 4 * embed_dim),
                            nn.GELU(),
                            nn.Linear(4 * embed_dim, embed_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.mlm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.mlm_head.weight = self.tok_emb.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, targets=None):
        B, T = x.shape

        x = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))

        for layer in self.layers:
            # Attention (bidirectional - no mask)
            h = layer["ln1"](x)
            qkv = (
                layer["attn"](h)
                .reshape(B, T, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = self.dropout(F.softmax(attn, dim=-1))

            x = x + layer["proj"]((attn @ v).transpose(1, 2).reshape(B, T, -1))

            # FFN
            x = x + layer["ff"](layer["ln2"](x))

        logits = self.mlm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        return logits, loss

    def fill_masks(self, x, mask_token=0):
        """Fill masked positions with predictions"""
        logits, _ = self(x)
        mask_positions = x == mask_token
        x = x.clone()
        x[mask_positions] = logits[mask_positions].argmax(dim=-1)
        return x


# Scoped validator. Only proves this file works. Nothing else.
if __name__ == "__main__":
    print("Instantiating Encoder (encoder-only, bidirectional)...")

    vocab_size = 256
    model = Encoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_layers=8,
        num_heads=8,
        context_length=512,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    batch_size = 2
    seq_len = 64

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nInput shapes:")
    print(f"  x:       {x.shape}")
    print(f"  targets: {targets.shape}")

    with torch.no_grad():
        logits, loss = model(x, targets=targets)

    print(f"\nOutput shape: {logits.shape}")
    print(f"Loss:         {loss.item():.4f}")

    # fill_masks: confirm bidirectional prediction works
    masked = x.clone()
    masked[0, 10:15] = 0
    filled = model.fill_masks(masked)
    print(f"fill_masks:   {filled.shape}")

    print("\n✓ Forward pass successful")
