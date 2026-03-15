# Atomic Watermelon

A shared-weight dual-stream transformer. Bidirectional encoder and causal decoder share attention and feedforward weights. Same input, same weights, different masks.

This is a research framework, not a finished product. Its purpose is to make the architecture, training process, and evaluation pipeline inspectable.

---

## Architecture

```
                     ATOMIC WATERMELON

                        current tokens
                              |
                              v
                       +-------------+
                       | tok + pos   |
                       +-------------+
                              |
                     +--------+--------+
                     |                 |
                     v                 v
              encoder input      decoder input
              (same embedding)   (same embedding)


              ┌──────────── BRIDGEBLOCK ──────────────────────────┐
              │                                                    │
              │  encoder stream                                    │
              │    enc_x -> LN -> self-attn(bidir) -> +            │
              │         -> LN -> FF -> +                           │
              │                                                    │
              │  decoder stream                                    │
              │    dec_x -> LN -> shared self-attn + causal mask   │
              │         -> +                                       │
              │         -> LN -> shared FF -> +                    │
              │                                                    │
              └─────────────────────┬─────────────────────────────┘
                                    |
                         repeated for N layers
                                    |
                  +-----------------+------------------+
                  |                                    |
                  v                                    v
         encoder state                       decoder state
         (not used for output)                      |
                                                    +-> LN -> LM head -> logits
```

| Parameter      | Value      |
| -------------- | ---------- |
| Parameters     | 23,097,344 |
| d_model        | 512        |
| n_layers       | 6          |
| n_heads        | 4          |
| d_ff           | 2048       |
| context_length | 4096       |
| dropout        | 0.2        |
| batch_size     | 1          |

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
  -> LN -> shared self-attention weights + causal mask -> residual add
  -> LN -> shared feedforward weights -> residual add
  -> dec_x'
```

---

## Masking

| Stream            | Attention type | Visibility       |
| ----------------- | -------------- | ---------------- |
| Encoder self-attn | Bidirectional  | Full context     |
| Decoder self-attn | Causal         | Current and left |

---

## Design Principles

### Shared Weights, Two Execution Modes

One weight set. Two execution modes. The encoder runs it bidirectional — no mask, full context. The decoder runs it causally — causal mask, this position and left only. The weights are shared unmolested. Neither mode owns them.

There are no decoder-specific parameters. Every trainable parameter in the model serves both modes simultaneously. The architecture is the bridge.

### Why Two Streams

Decoder-only transformers fuse two distinct operations into one pass: building a full-context representation and generating the next token. The causal mask makes full-context representation structurally impossible — the model cannot see its own future tokens while building the representation it needs to generate them well.

This architecture separates those roles:

- **Encoder**: representation building via bidirectional attention over full current context
- **Decoder**: token generation via causal attention over the same shared weights

The hypothesis is that explicit separation with fully shared weights produces better representations per parameter than fusing the two in a decoder-only model. Behavior is determined by architectural constraint, not scale.

---

## Training Corpus

185MB of curated text:

- https://www.worldhistory.org
- https://plato.stanford.edu
- https://smarthistory.org

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
python3 main.py
```

This will automate the cold start.

The encoder/decoder single-mode baselines use the gen0 dataset. The AIAYN baseline uses the SEP corpus. All of this is configurable.

---

## Prior Work

This architecture is informed by encoder-decoder transformers and shared-parameter transformer variants.

The specific combination implemented here — bidirectional encoder and causal decoder sharing all backbone weights with no decoder-specific parameters, differentiated purely by masking — is, to the author's knowledge, novel. That is a research claim, not a definitive literature claim.

---

## License

Apache 2.0. Use it, build on it. The patent grant clause means you cannot take this architecture, file adjacent patent claims, and come back. Attribution stays on derivatives.

Steven Midgley / DaemonCat LLC
