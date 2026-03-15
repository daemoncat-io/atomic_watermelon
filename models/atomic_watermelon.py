"""
Atomic Watermelon — Substrate Ablation

Dual mode shared weight transformer. Nothing else.
No adapter. No cross-attention. No memory.

Encoder: bidirectional attention over current context
Decoder: borrows encoder weights directly, causal mask only

This is the floor. Two views of the same weights on the same input,
differentiated only by masking. If dual-mode behavior survives here,
shared weights ARE the bridge. Everything else is infrastructure.
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        qkv = (
            self.w_qkv(x)
            .reshape(B, T, 3, self.n_heads, self.d_k)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BridgeBlock(nn.Module):
    """
    Encoder: bidirectional attention
    Decoder: borrows encoder weights directly, causal mask only

    Data flow per block:
        enc_x  -> bidirectional self-attn -> FF -> enc_x'
        dec_x  -> causal self-attn (shared weights) -> FF (shared weights) -> dec_x'
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encoder (shared with decoder)
        self.enc_ln1 = nn.LayerNorm(d_model)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_ln2 = nn.LayerNorm(d_model)
        self.enc_ff = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        enc_x: torch.Tensor,  # [B, T, C]
        dec_x: torch.Tensor,  # [B, T, C]
        dec_causal_mask: torch.Tensor,  # [1, 1, T, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # --- Encoder: full bidirectional self-attention + FF ---
        enc_x = enc_x + self.dropout(self.enc_attn(self.enc_ln1(enc_x), mask=None))
        enc_x = enc_x + self.dropout(self.enc_ff(self.enc_ln2(enc_x)))

        # --- Decoder: causal self-attention (shared encoder weights) ---
        dec_attn_out = self.enc_attn(self.enc_ln1(dec_x), mask=dec_causal_mask)
        dec_x = dec_x + self.dropout(dec_attn_out)

        # --- Decoder: FF (shared encoder weights) ---
        dec_ff_out = self.enc_ff(self.enc_ln2(dec_x))
        dec_x = dec_x + self.dropout(dec_ff_out)

        return enc_x, dec_x


class AtomicWatermelon(nn.Module):
    """
    The floor.

    The encoder owns the weights.
    The decoder shares the encoder's layers directly.
    Same input, same weights, different masks.
    Nothing else.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Blocks
        self.blocks = nn.ModuleList(
            [BridgeBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
        )

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            x:       [B, T]    input token ids
            targets: [B, T]    target token ids for loss
            memory:  ignored, kept for API compatibility

        Returns:
            logits:     [B, T, V]
            loss:       scalar or None
            new_memory: always None
        """

        B, T = x.shape
        device = x.device

        # Embed current input
        tok = self.tok_emb(x) * math.sqrt(self.d_model)
        pos = self.pos_emb(torch.arange(T, device=device))
        x_emb = tok + pos  # [B, T, C]

        # Same input, two paths
        enc_x = x_emb
        dec_x = x_emb

        # Decoder causal mask
        dec_causal_mask = self.attention_mask[:, :, :T, :T]

        for block in self.blocks:
            enc_x, dec_x = block(enc_x, dec_x, dec_causal_mask)

        # Output from decoder stream
        logits = self.lm_head(self.ln_f(dec_x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, None

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_tokens: int = 100,
        top_k: int = 1,
        temperature: float = 1.0,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Generate tokens. No memory, just autoregressive generation.

        Returns:
            generated: [B, T+max_tokens]
            memory:    always None
        """
        self.eval()
        generated = x.clone()

        for _ in range(max_tokens):
            # Truncate to max_seq_len if needed
            ctx = generated[:, -self.max_seq_len :]

            logits, _, _ = self(ctx)
            logits = logits[:, -1, :] / temperature

            if top_k == 1 and temperature == 1.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_tok], dim=1)

        return generated, None
