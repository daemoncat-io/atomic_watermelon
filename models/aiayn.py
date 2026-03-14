"""
Clinical Replication of "Attention Is All You Need" (Vaswani et al., 2017)
arXiv:1706.03762

Architecture specifications from paper:
- d_model = 512 (embedding dimension)
- d_ff = 2048 (feed-forward inner dimension)
- h = 8 (number of attention heads)
- d_k = d_v = 64 (key/value dimensions per head)
- N = 6 (number of encoder/decoder layers)
- P_drop = 0.1 (dropout rate)

Positional Encoding:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Attention:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Multi-Head:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
"""

import torch.nn.functional as F
from typing import Optional
import torch.nn as nn
import torch
import math


class ScaledDotProductAttention(nn.Module):
    """
    Equation 1 from paper:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    The scaling factor 1/sqrt(d_k) prevents the dot products from growing
    large in magnitude, which would push softmax into regions with extremely
    small gradients.
    """

    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape (batch, heads, seq_len, d_k)
            k: Key tensor of shape (batch, heads, seq_len, d_k)
            v: Value tensor of shape (batch, heads, seq_len, d_v)
            mask: Optional mask tensor. For padding mask: (batch, 1, 1, seq_len)
                  For causal mask: (1, 1, seq_len, seq_len)

        Returns:
            output: Attended values (batch, heads, seq_len, d_v)
            attention_weights: Attention distribution (batch, heads, seq_len, seq_len)
        """
        # QK^T / sqrt(d_k)
        # (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k) -> (batch, heads, seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask before softmax
        # Paper Section 3.2.3: "We implement this inside of scaled dot-product attention
        # by masking out (setting to -∞) all values in the input of the softmax"
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over key dimension
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights (paper Section 5.4)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Section 3.2.2: Multi-Head Attention

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Paper uses h=8 parallel attention heads with d_k = d_v = d_model/h = 64

    Parameter matrices:
    W_i^Q ∈ R^(d_model × d_k)
    W_i^K ∈ R^(d_model × d_k)
    W_i^V ∈ R^(d_model × d_v)
    W^O ∈ R^(h*d_v × d_model)
    """

    def __init__(self, d_model: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # 64 per paper
        self.d_v = d_model // h  # 64 per paper

        # Linear projections for Q, K, V
        # Implemented as single matrices that project to all heads simultaneously
        # then reshaped to (batch, heads, seq, d_k/d_v)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection W^O
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, heads, seq, d_k)
        # Project then split into h heads
        q = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)

        # Apply attention
        attended, _ = self.attention(q, k, v, mask)

        # Concatenate heads and apply output projection
        # (batch, heads, seq, d_v) -> (batch, seq, heads, d_v) -> (batch, seq, d_model)
        concat = (
            attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.w_o(concat)

        return output


class PositionWiseFeedForward(nn.Module):
    """
    Section 3.3: Position-wise Feed-Forward Networks

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Two linear transformations with ReLU activation in between.
    Applied to each position separately and identically.

    Paper uses d_model=512, d_ff=2048 (inner layer has 4x dimension)
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class PositionalEncoding(nn.Module):
    """
    Section 3.5: Positional Encoding

    Since the model contains no recurrence and no convolution, positional
    encodings are added to give the model information about sequence order.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The wavelengths form a geometric progression from 2π to 10000·2π.
    This encoding was chosen because it allows the model to learn to attend
    by relative positions, since PE(pos+k) can be represented as a linear
    function of PE(pos).
    """

    def __init__(
        self, d_model: int = 512, max_seq_len: int = 5000, dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the divisor term: 10000^(2i/d_model)
        # Using exp(log()) for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedded input (batch, seq_len, d_model)

        Returns:
            Positionally encoded input (batch, seq_len, d_model)
        """
        # Add positional encoding (broadcasts over batch)
        # Paper Section 3.4: "we multiply those weights by sqrt(d_model)"
        # The scaling is applied to embeddings before adding PE
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    Section 3.1: Encoder

    Each encoder layer has two sub-layers:
    1. Multi-head self-attention mechanism
    2. Position-wise fully connected feed-forward network

    Residual connection around each sub-layer, followed by layer normalization:
    LayerNorm(x + Sublayer(x))

    Paper applies dropout to output of each sub-layer before residual addition.
    """

    def __init__(
        self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Encoder input (batch, seq_len, d_model)
            src_mask: Source padding mask (batch, 1, 1, seq_len)

        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        # Self-attention with residual and layer norm
        attn_output = self.self_attention(x, x, x, src_mask)
        x = self.norm_1(x + self.dropout_1(attn_output))

        # Feed-forward with residual and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm_2(x + self.dropout_2(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Section 3.1: Decoder

    Each decoder layer has three sub-layers:
    1. Masked multi-head self-attention (prevents attending to future positions)
    2. Multi-head attention over encoder output (encoder-decoder attention)
    3. Position-wise fully connected feed-forward network

    Same residual + layer norm structure as encoder.

    The masking in self-attention ensures that predictions for position i
    can depend only on known outputs at positions less than i.
    """

    def __init__(
        self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, h, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            src_mask: Source padding mask for encoder-decoder attention
            tgt_mask: Target mask (causal + padding) for self-attention

        Returns:
            Decoder output (batch, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual and layer norm
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm_1(x + self.dropout_1(self_attn_output))

        # Encoder-decoder attention with residual and layer norm
        # Query from decoder, key/value from encoder
        enc_attn_output = self.encoder_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm_2(x + self.dropout_2(enc_attn_output))

        # Feed-forward with residual and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm_3(x + self.dropout_3(ff_output))

        return x


class Encoder(nn.Module):
    """
    Full encoder stack: N=6 identical layers.
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 512,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer norm (per paper implementation)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Embedded + positionally encoded input (batch, seq_len, d_model)
            src_mask: Source padding mask

        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


class Decoder(nn.Module):
    """
    Full decoder stack: N=6 identical layers.
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 512,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer norm (per paper implementation)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Embedded + positionally encoded target (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            Decoder output (batch, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class Transformer(nn.Module):
    """
    Complete Transformer model as described in "Attention Is All You Need"

    Architecture (Table 3 from paper - base model):
    - N = 6 (encoder and decoder layers)
    - d_model = 512
    - d_ff = 2048
    - h = 8
    - d_k = d_v = 64
    - P_drop = 0.1

    The model consists of:
    1. Source embedding + positional encoding
    2. Target embedding + positional encoding
    3. Encoder stack (N layers)
    4. Decoder stack (N layers)
    5. Final linear projection + softmax

    Paper Section 3.4: Embeddings are multiplied by sqrt(d_model)
    Paper Section 3.4: Source and target embeddings share weights with output projection
    """

    def __init__(
        self,
        src_vocab_size: int = 256,
        tgt_vocab_size: int = 256,
        d_model: int = 512,
        n_layers: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        share_embeddings: bool = True,
    ):
        """
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Model dimension (512 in paper)
            n_layers: Number of encoder/decoder layers (6 in paper)
            h: Number of attention heads (8 in paper)
            d_ff: Feed-forward inner dimension (2048 in paper)
            max_seq_len: Maximum sequence length for positional encoding
            dropout: Dropout rate (0.1 in paper)
            share_embeddings: Whether to share weights between embeddings and output projection
                              (True per paper Section 3.4)
        """
        super().__init__()

        self.d_model = d_model
        self.share_embeddings = share_embeddings

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding (shared between encoder and decoder)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and decoder stacks
        self.encoder = Encoder(n_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, h, d_ff, dropout)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying per paper Section 3.4
        # "we share the same weight matrix between the two embedding layers and
        # the pre-softmax linear transformation"
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.src_embedding.weight = self.tgt_embedding.weight
            self.output_projection.weight = self.tgt_embedding.weight

        # Initialize parameters per paper Section 5.3 (implied Xavier/Glorot)
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        Paper doesn't specify exact initialization but this is standard practice
        and matches the original implementation.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source token indices (batch, src_seq_len)
            src_mask: Source padding mask

        Returns:
            Encoder output (batch, src_seq_len, d_model)
        """
        # Embed and scale by sqrt(d_model) per paper Section 3.4
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_encoded = self.positional_encoding(src_embedded)

        return self.encoder(src_encoded, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder output.

        Args:
            tgt: Target token indices (batch, tgt_seq_len)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            Decoder output (batch, tgt_seq_len, d_model)
        """
        # Embed and scale by sqrt(d_model) per paper Section 3.4
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_encoded = self.positional_encoding(tgt_embedded)

        return self.decoder(tgt_encoded, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass through encoder-decoder.

        Args:
            src: Source token indices (batch, src_seq_len)
            tgt: Target token indices (batch, tgt_seq_len)
            src_mask: Source padding mask (batch, 1, 1, src_seq_len)
            tgt_mask: Target causal + padding mask (batch, 1, tgt_seq_len, tgt_seq_len)

        Returns:
            Output logits (batch, tgt_seq_len, tgt_vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary and return logits (softmax applied in loss function)
        return self.output_projection(decoder_output)

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate causal (look-ahead) mask for decoder self-attention.

        This prevents position i from attending to positions j > i,
        ensuring autoregressive generation.

        Paper Section 3.2.3: "We [...] mask out all values in the input of
        the softmax which correspond to illegal connections."

        Args:
            seq_len: Sequence length
            device: Target device

        Returns:
            Causal mask (1, 1, seq_len, seq_len) where 1 = attend, 0 = mask
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = (mask == 0).unsqueeze(0).unsqueeze(0)
        return mask

    @staticmethod
    def generate_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        Generate padding mask to prevent attention to padding tokens.

        Args:
            seq: Token indices (batch, seq_len)
            pad_idx: Padding token index

        Returns:
            Padding mask (batch, 1, 1, seq_len) where 1 = attend, 0 = mask
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_transformer_base(
    src_vocab_size: int, tgt_vocab_size: int, **kwargs
) -> Transformer:
    """
    Create Transformer with base model hyperparameters from paper Table 3.

    Base model: N=6, d_model=512, d_ff=2048, h=8
    65M parameters on EN-DE task
    """
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        n_layers=6,
        h=8,
        d_ff=2048,
        dropout=0.1,
        **kwargs,
    )


def create_transformer_big(
    src_vocab_size: int, tgt_vocab_size: int, **kwargs
) -> Transformer:
    """
    Create Transformer with big model hyperparameters from paper Table 3.

    Big model: N=6, d_model=1024, d_ff=4096, h=16
    213M parameters on EN-DE task
    Uses higher dropout (0.3) per paper
    """
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=1024,
        n_layers=6,
        h=16,
        d_ff=4096,
        dropout=0.3,
        **kwargs,
    )


# Scoped validator. Only proves this file works. Nothing else.
if __name__ == "__main__":
    # Verification: instantiate and test forward pass
    print("Instantiating Transformer (base model)...")

    # Typical vocab sizes for BPE tokenization as used in paper
    src_vocab = 37000
    tgt_vocab = 37000

    model = create_transformer_base(src_vocab, tgt_vocab)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 2
    src_len = 20
    tgt_len = 15

    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

    # Generate masks
    src_mask = Transformer.generate_padding_mask(src)
    tgt_mask = Transformer.generate_causal_mask(
        tgt_len
    ) & Transformer.generate_padding_mask(tgt)

    print(f"\nInput shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")
    print(f"  src_mask: {src_mask.shape}")
    print(f"  tgt_mask: {tgt_mask.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, tgt_len={tgt_len}, vocab={tgt_vocab})")

    print("\n✓ Forward pass successful")
