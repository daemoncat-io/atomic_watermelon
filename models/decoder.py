"""
Minimal Transformer - 26M params
Decoder-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Decoder(nn.Module):
    def __init__(
        self,
        context_length=2048,
        vocab_size=256,
        embed_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.head_dim = embed_dim // num_heads
        self.context_length = context_length
        self.num_heads = num_heads

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.Linear(embed_dim, 3 * embed_dim),
                        "proj": nn.Linear(embed_dim, embed_dim),
                        "ln2": nn.LayerNorm(embed_dim),
                        "ln1": nn.LayerNorm(embed_dim),
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

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.ln_f = nn.LayerNorm(embed_dim)

        self.register_buffer(
            "mask", torch.tril(torch.ones(context_length, context_length))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, targets=None):
        B, T = x.shape

        x = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))

        for layer in self.layers:
            # Attention
            h = layer["ln1"](x)
            qkv = (
                layer["attn"](h)
                .reshape(B, T, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
            attn = self.dropout(F.softmax(attn, dim=-1))

            x = x + layer["proj"]((attn @ v).transpose(1, 2).reshape(B, T, -1))

            # FFN
            x = x + layer["ff"](layer["ln2"](x))

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, x, max_tokens=100, top_k=10):
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                logits, _ = self(x[:, -self.context_length :])
                if top_k == 1:
                    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits[:, -1] / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                x = torch.cat([x, next_token], dim=1)
        return x


# Scoped validator. Only proves this file works. Nothing else.
if __name__ == "__main__":
    print("Instantiating Decoder (decoder-only, causal)...")

    vocab_size = 256
    model = Decoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_layers=8,
        num_heads=8,
        context_length=2048,
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

    # generate: confirm causal generation works
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_tokens=20, top_k=10)
    print(f"\nGenerate:")
    print(f"  Prompt:    {prompt.shape}")
    print(f"  Generated: {generated.shape}")

    print("\n✓ Forward pass successful")
