# Atomic Watermelon

A shared-weight dual-stream transformer. Bidirectional encoder over memory and current context. Causal decoder over current segment only. Encoder and decoder share attention and feedforward weights; decoder-side adapters handle the distribution shift. Decoder retrieves from encoder state via cross-attention. Older context compresses into fixed memory slots that persist across calls.

This is a research framework, not a finished product. Its purpose is to make the architecture, training process, and evaluation pipeline inspectable.

---

## Architecture

```
                     ATOMIC WATERMELON

        memory slots                     current tokens
             |                                 |
             v                                 v
        +----------+                    +-------------+
        | mem emb  |                    | tok + pos   |
        +----------+                    +-------------+
             |                                 |
             +-----------+   +-----------------+
                         |   |
                         v   v
                 encoder input: [ memory | current ]
                 decoder input: [ current only ]


              ┌──────────────── BRIDGEBLOCK ───────────────────────────────────────┐
              │                                                                    │
              │  encoder stream                                                    │
              │    enc_x -> LN -> self-attn(bidir) -> +                            │
              │         -> LN -> FF -> +                                           │
              │                                                                    │
              │  decoder stream                                                    │
              │    dec_x -> adapter -> LN                                          │
              │         -> shared encoder self-attn weights + causal mask -> +     │
              │         -> LN -> cross-attn(encoder) -> +                          │
              │         -> adapter -> LN -> shared FF -> +                         │
              │                                                                    │
              └──────────────────────┬─────────────────────────────────────────────┘
                                     |
                          repeated for N layers
                                     |
                   +-----------------+------------------+
                   |                                    |
                   v                                    v
      encoder state [ memory | current ]      decoder state [ current ]
                   |                                    |
                   |                                    +-> LN -> LM head -> logits
                   |
                   +-> compress old context chunks -> summary vectors -> memory buffer
```

| Parameter          | Value      |
| ------------------ | ---------- |
| Parameters         | 33,656,577 |
| d_model            | 512        |
| n_layers           | 6          |
| n_heads            | 4          |
| d_ff               | 2048       |
| adapter_bottleneck | 128        |
| compress_chunk     | 128        |
| memory_slots       | 32         |
| context_length     | 4096       |
| dropout            | 0.2        |
| batch_size         | 1          |

`n_heads < n_layers` is an empirical constraint, not a mathematical one.

---

## BRIDGEBLOCK

### Encoder branch

```
enc_x
  -> LN -> bidirectional self-attention -> residual add
  -> LN -> feedforward -> residual add
  -> enc_x'
```

### Decoder branch

```
dec_x
  -> adapter (pre-attn)
  -> LN -> shared self-attention weights + causal mask -> adapter (post-attn) -> residual add
  -> LN -> cross-attention [Q=decoder, K/V=encoder state] -> adapter -> residual add
  -> adapter (pre-FF)
  -> LN -> shared feedforward weights -> adapter (post-FF) -> residual add
  -> dec_x'
```

---

## Masking

| Stream             | Attention type   | Visibility                                          |
| ------------------ | ---------------- | --------------------------------------------------- |
| Encoder self-attn  | Bidirectional    | Full: memory + current context                      |
| Decoder self-attn  | Causal           | Current context only                                |
| Decoder cross-attn | Causal (partial) | Memory slots: unrestricted. Current context: causal |

---

## Design Principles

### Shared Weights, Two Execution Modes

One weight set. Two execution modes. The encoder runs it bidirectional — no mask, full context. The decoder runs it causally — causal mask, this position only. The weights are shared unmolested. Neither mode owns them.

The only decoder-specific parameters are the adapters — lightweight bottleneck modules (dim 128) that handle the distribution shift between execution modes. Four per block: pre/post attention, pre/post feedforward. Everything else is shared.

### The Cross-Attention Bridge

After each encoder pass, the decoder queries the encoder's full-context representation through cross-attention. The decoder supplies the queries. The encoder's output — memory slots and current segment — becomes keys and values. The bridge is asymmetric. The decoder consults the encoder. Not the reverse.

### Why Two Streams

Decoder-only transformers fuse two distinct operations into one pass: building a full-context representation and generating the next token. The causal mask makes full-context representation structurally impossible — the model cannot see its own future tokens while building the representation it needs to generate them well.

This architecture separates those roles:

- **Encoder**: representation building via bidirectional attention over memory and full current context
- **Decoder**: token generation via causal attention, consulting the encoder's representation through the bridge

The hypothesis is that explicit separation, with most weights shared, produces better representations per parameter than fusing the two in a decoder-only model.

---

## Compressive Memory

```
memory buffer: [ compressed slots | current context ]
```

When current context overflows, the oldest chunks are compressed:

```
encoder output [ memory | current ]
  -> split current into fixed chunks (chunk size: 128)
  -> chunk -> gate weights -> weighted sum -> summary vector
  -> append summary vector to memory slot buffer
  -> if buffer full: compress oldest overflow region into one merged slot
```

The compression network is fully differentiable. Gradient flows end to end through the gate weights. The model learns what to store, not just how to store it.

Memory is an encoder concern. The decoder accesses it exclusively through the bridge.

---

## Training Corpus

185MB of curated text:

- Stanford Encyclopedia of Philosophy

BPE tokenized, vocab_size=4096. Tokenizer trained on the corpus itself — vocabulary reflects actual distribution. The corpus is not included in the repository. `datasets/sep.py` rebuilds it.

One small component, `gen0_language_structure.txt`, is explicitly pedagogical — plain-language encodings of simple language regularities used for early structural training.

---

## Framework Structure

```
models/
  atomic_watermelon.py    # Bridge Transformer
  encoder.py              # Encoder-only baseline
  decoder.py              # Decoder-only baseline

trainers/
  aiayn.py                # AIAYN baseline training loop
  encoder.py              # Encoder-only training loop
  decoder.py              # Decoder-only training loop
  logger_aw.py            # Epoch-based training log with system telemetry

datasets/
  bpe.py                  # BPE tokenizer — hand-rolled
  sep.py                  # Corpus scraper
  gen0_language_structure.txt
  sample_prompts.txt      # Fixed + random prompts sampled each epoch

validators/
  probe_aw.py             # Bridge Transformer introspection
  probe_encoder.py        # Encoder-only introspection
  probe_decoder.py        # Decoder-only introspection
  heatmap_aw.py           # Attention heatmaps
  chat_aw.py              # Interactive inference — Bridge Transformer
  chat_encoder.py         # Interactive inference — encoder-only
  chat_decoder.py         # Interactive inference — decoder-only
  dashboard_aw.py         # Live training monitor

main.py
```

---

## Setup

```bash
bash init_venv.sh
source atomic_watermelon_venv/bin/activate
python3 datasets/sep.py
```

This will automate the cold start and spin up the SEP scrapper.

Then:

```bash
python3 datasets/bpe.py
```

This will tokenize the corpus.

Then:

```bash
python3 main.py
```

The encoder/decoder single-mode baselines use the gen0 dataset. The AIAYN baseline uses the SEP corpus. All of this is configurable.

---

## Prior Work

This architecture is informed by encoder-decoder transformers, compressive memory mechanisms, adapter-based parameter sharing, and shared-parameter transformer variants.

<<<<<<< HEAD
The specific combination implemented here — bidirectional encoder and causal decoder sharing backbone attention/feedforward weights, decoder-side adapters, decoder-to-encoder cross-attention over compressed persistent memory — is, to the author's knowledge, novel. That is a research claim, not a definitive literature claim.
=======
The combination — dual execution modes on shared weights, adapters for distribution shift,
compressive memory with full gradient flow, cross-attention bridge, no skip gates — does not
appear in prior literature. The closest neighbors are Compressive Transformer (DeepMind, 2019),
Subformer (2021), and adapter methods generally.

> > > > > > > refs/remotes/origin/main

---

## License

Apache 2.0. Use it, build on it. The patent grant clause means you cannot take this architecture, file adjacent patent claims, and come back. Attribution stays on derivatives.

Steven Midgley / DaemonCat LLC
