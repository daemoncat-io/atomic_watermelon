"""
Atomic Watermelon

A dual mode mutual weight focused attention transformer with memory compression

Encoder: bidirectional, maintains compressed memory across calls
Decoder: borrows encoder weights, causal mask, pulls from memory

Memory structure: [compressed_slots | current_context]
- Compressed slots: fixed-size buffer of summarized past context
- Current context: full-resolution recent tokens
- When current overflows, oldest chunks compress into memory slots
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


class CrossAttention(nn.Module):
    """Decoder attends to encoder with separate Q projection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, C = q_input.shape
        T_kv = kv_input.shape[1]

        q = self.w_q(q_input).reshape(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        kv = (
            self.w_kv(kv_input)
            .reshape(B, T_kv, 2, self.n_heads, self.d_k)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T_q, C)
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


class Adapter(nn.Module):
    """
    Bottleneck adapter for distribution shift between encoder and decoder.

    Projects down to a smaller dimension, applies nonlinearity, projects back.
    This lets the decoder "translate" its activations into the encoder's
    expected distribution before borrowing encoder weights.
    """

    def __init__(self, d_model: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual adapter: x + adapter(x)
        residual = self.dropout(self.up(F.gelu(self.down(self.ln(x)))))
        return x + residual


class BridgeBlock(nn.Module):
    """
    Encoder: private weights, bidirectional attention
    Decoder: borrows encoder weights via adapters, causal mask

    The decoder uses lightweight adapters to transform its activations
    before passing through shared encoder layers. This handles the
    distribution mismatch while keeping most parameters shared.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        adapter_bottleneck: int = 64,
    ):
        super().__init__()

        # Encoder: full private weights (the "real" layers)
        self.enc_ln1 = nn.LayerNorm(d_model)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_ln2 = nn.LayerNorm(d_model)
        self.enc_ff = FeedForward(d_model, d_ff, dropout)

        # Decoder adapters: transform before using shared weights
        self.dec_adapt_pre_attn = Adapter(d_model, adapter_bottleneck, dropout)
        self.dec_adapt_post_attn = Adapter(d_model, adapter_bottleneck, dropout)
        self.dec_adapt_pre_ff = Adapter(d_model, adapter_bottleneck, dropout)
        self.dec_adapt_post_ff = Adapter(d_model, adapter_bottleneck, dropout)

        # Cross-attention (decoder queries encoder, can't be shared)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.cross_ln = nn.LayerNorm(d_model)
        self.cross_adapt = Adapter(d_model, adapter_bottleneck, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        enc_x: torch.Tensor,  # [B, M+T, C] memory + current
        dec_x: torch.Tensor,  # [B, T, C] current only
        dec_causal_mask: torch.Tensor,  # [1, 1, T, T]
        cross_mask: torch.Tensor,  # [1, 1, T, M+T]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # === ENCODER (owns the weights) ===
        enc_x = enc_x + self.dropout(self.enc_attn(self.enc_ln1(enc_x), mask=None))
        enc_x = enc_x + self.dropout(self.enc_ff(self.enc_ln2(enc_x)))

        # === DECODER (borrows via adapters) ===

        # Self-attention: adapt -> borrow enc_attn -> adapt
        dec_adapted = self.dec_adapt_pre_attn(dec_x)
        dec_attn_out = self.enc_attn(self.enc_ln1(dec_adapted), mask=dec_causal_mask)
        dec_x = dec_x + self.dropout(self.dec_adapt_post_attn(dec_attn_out))

        # Cross-attention to encoder (direct)
        cross_out = self.cross_attn(self.cross_ln(dec_x), enc_x, mask=cross_mask)

        # Feedforward: adapt -> borrow enc_ff -> adapt
        dec_adapted = self.dec_adapt_pre_ff(dec_x)
        dec_ff_out = self.enc_ff(self.enc_ln2(dec_adapted))
        dec_x = dec_x + self.dropout(self.dec_adapt_post_ff(dec_ff_out))

        return enc_x, dec_x


class AtomicWatermelon(nn.Module):
    """
    Transformer with compressive memory and weight-sharing via adapters.

    The encoder owns the heavy weights (attention, FFN).
    The decoder borrows these through lightweight adapters that handle distribution shift.
    The encoder maintains a fixed-size memory buffer of compressed past context.
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
        memory_slots: int = 32,
        compress_chunk: int = 64,
        adapter_bottleneck: int = 64,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.memory_slots = memory_slots
        self.compress_chunk = compress_chunk
        self.n_heads = n_heads

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Learned memory position embedding (memory slots don't use sequential positions)
        self.mem_pos_emb = nn.Embedding(memory_slots, d_model)

        # Blocks
        self.blocks = nn.ModuleList(
            [
                BridgeBlock(d_model, n_heads, d_ff, dropout, adapter_bottleneck)
                for _ in range(n_layers)
            ]
        )

        # Compression network
        self.compress_proj = nn.Linear(d_model, d_model)
        self.compress_gate = nn.Linear(d_model, 1)

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Causal mask (max size, will be sliced)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
        )

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _compress_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Compress a chunk of encoder states into a single summary vector."""
        # chunk: [B, chunk_size, C]
        weights = torch.softmax(self.compress_gate(chunk), dim=1)  # [B, chunk_size, 1]
        summary = (weights * self.compress_proj(chunk)).sum(
            dim=1, keepdim=True
        )  # [B, 1, C]
        return summary

    def _update_memory(
        self,
        enc_x: torch.Tensor,
        memory: torch.Tensor | None,
        n_memory: int,
    ) -> torch.Tensor:
        """
        Update memory with newly processed content.

        enc_x: [B, M+T, C] full encoder output (memory prefix + current)
        memory: [B, M, C] previous memory or None
        n_memory: number of memory positions in enc_x

        Returns: [B, M', C] updated memory
        """
        B = enc_x.shape[0]

        # Extract new content (everything after memory prefix)
        new_content = enc_x[:, n_memory:, :]  # [B, T, C]
        T = new_content.shape[1]

        if T < self.compress_chunk:
            # Not enough new content to compress
            return memory

        # Compress complete chunks
        n_chunks = T // self.compress_chunk
        summaries = []

        for i in range(n_chunks):
            start = i * self.compress_chunk
            end = start + self.compress_chunk
            chunk = new_content[:, start:end, :]
            summary = self._compress_chunk(chunk)
            summaries.append(summary)

        if not summaries:
            return memory

        new_summaries = torch.cat(summaries, dim=1)  # [B, n_chunks, C]

        if memory is None:
            # First compression
            if new_summaries.shape[1] <= self.memory_slots:
                return new_summaries
            # Too many summaries, keep most recent
            return new_summaries[:, -self.memory_slots :, :]

        # Combine old memory with new summaries
        combined = torch.cat([memory, new_summaries], dim=1)

        if combined.shape[1] <= self.memory_slots:
            return combined

        # Memory overflow: compress oldest slots to make room
        overflow = combined.shape[1] - self.memory_slots
        to_compress = combined[:, : overflow + 1, :]  # Compress overflow+1 into 1
        merged = self._compress_chunk(to_compress)

        return torch.cat([merged, combined[:, overflow + 1 :, :]], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward pass with optional memory.

        Args:
            x: [B, T] input token ids
            targets: [B, T] target token ids for loss
            memory: [B, M, C] compressed memory from previous calls

        Returns:
            logits: [B, T, V]
            loss: scalar or None
            new_memory: [B, M', C] updated memory
        """
        B, T = x.shape
        device = x.device

        # Embed current input
        tok = self.tok_emb(x) * math.sqrt(self.d_model)
        pos = self.pos_emb(torch.arange(T, device=device))
        x_emb = tok + pos  # [B, T, C]

        # Build encoder input: memory (if any) + current
        if memory is not None:
            M = memory.shape[1]
            # Add memory position embeddings
            mem_pos = self.mem_pos_emb(torch.arange(M, device=device))
            memory_positioned = memory + mem_pos
            enc_x = torch.cat([memory_positioned, x_emb], dim=1)  # [B, M+T, C]
            n_memory = M
        else:
            enc_x = x_emb
            n_memory = 0

        # Decoder input: just current segment
        dec_x = x_emb

        # Build masks
        # Decoder self-attention: causal over current segment
        dec_causal_mask = self.causal_mask[:, :, :T, :T]

        # Cross-attention mask:
        # - Full attention to all memory slots (positions 0..M-1)
        # - Causal attention to current segment (positions M..M+T-1)
        enc_len = enc_x.shape[1]
        cross_mask = torch.ones(1, 1, T, enc_len, device=device)

        if n_memory > 0:
            # Memory portion: fully visible
            # Current portion: causal
            current_causal = self.causal_mask[:, :, :T, :T]
            cross_mask[:, :, :, n_memory:] = current_causal
        else:
            # No memory, just causal over current
            cross_mask = dec_causal_mask

        # Process through blocks
        for block in self.blocks:
            enc_x, dec_x = block(enc_x, dec_x, dec_causal_mask, cross_mask)

        # Output from decoder
        logits = self.lm_head(self.ln_f(dec_x))

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Update memory
        new_memory = self._update_memory(enc_x, memory, n_memory)

        return logits, loss, new_memory

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
        Generate tokens with persistent memory.

        Processes prompt in chunks to build memory, then generates
        one token at a time while maintaining a sliding window buffer.

        Returns:
            generated: [B, T+max_tokens] full sequence
            memory: updated memory state
        """
        self.eval()
        B = x.shape[0]
        device = x.device
        generated = x.clone()

        # Process prompt in compress_chunk sized pieces to build memory
        prompt_len = x.shape[1]
        pos = 0
        while pos < prompt_len:
            end = min(pos + self.compress_chunk, prompt_len)
            chunk = x[:, pos:end]
            logits, _, memory = self(chunk, memory=memory)
            pos = end

        # Buffer holds uncompressed recent tokens (< compress_chunk)
        # Start with leftover from prompt that didn't form a complete chunk
        leftover_start = (prompt_len // self.compress_chunk) * self.compress_chunk
        buffer = x[:, leftover_start:]  # [B, leftover_len]

        # If prompt was exactly divisible, buffer is empty - need at least last token
        if buffer.shape[1] == 0:
            buffer = x[:, -1:]

        for _ in range(max_tokens):
            # Forward pass on buffer (memory has compressed history)
            logits, _, new_memory = self(buffer, memory=memory)
            logits = logits[:, -1, :] / temperature

            if top_k == 1 and temperature == 1.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_tok], dim=1)
            buffer = torch.cat([buffer, next_tok], dim=1)

            # If buffer reached compress_chunk, memory was updated
            # Trim buffer to just the new token for next iteration
            if new_memory is not None and (
                memory is None or new_memory.shape[1] > memory.shape[1]
            ):
                memory = new_memory
                # Keep only tokens after the last compressed chunk
                buffer = buffer[
                    :, -(buffer.shape[1] % self.compress_chunk or self.compress_chunk) :
                ]
                # If buffer is empty after mod, we just compressed exactly, keep last token
                if buffer.shape[1] == 0:
                    buffer = next_tok

        return generated, memory
