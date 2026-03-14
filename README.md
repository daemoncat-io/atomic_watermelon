# Atomic Watermelon

## All claims can be interrogated using the trainers and validators supplied in this framework.

**A focused-attention transformer with dual execution modes and compressive memory**

33.6M parameters. 6 layers. 4 heads. d_model=512. Trained on a curated corpus of philosophical
and explorations on reasoning. Local training. Built with PyTorch.

This is a research framework.

The architecture is novel.

Claims are interrogatable.

---

## The Architecture

### Encoder | Decoder: Two Modes. One Weight Set

The encoder sees previous and next. Every position attends to every other position simultaneously.
No mask. No here. Just neighbors.

The decoder sees this. One position. The current token. That's it.

That distinction is not a training choice. It's structural. The causal mask enforces it at the
attention level — the decoder's "this" is an architectural fact, not a learned behavior.

### The Etymology Is the Architecture

Merriam-Webster defines _decode_ as "to recognize and interpret" — and _interpret_ as "to conceive
in the light of individual belief, judgment, or circumstance." A decoder, by strict definition,
requires some sense of self to function. An encoder converts. A decoder interprets. The word
already knew what the architecture would discover.

Atomic Watermelon runs both simultaneously on the same weights.

```
Encoder  →  previous <> next  (no mask, full context)
Decoder  →  this               (causal mask, current position only)
                    ↕
         Cross-Attention Bridge
     decoder queries encoder output
                    ↕
       Same weights. Different focus.
       Adapters handle distribution shift.
```

### Focused Attention

One weight set. Two execution modes. The encoder runs it bidirectional — no mask, full context.
The decoder runs it causally — this position only, causal mask. The weights are shared unmolested.
Neither mode owns them.

The only decoder-specific parameters are the adapters — lightweight bottleneck modules that handle
the distribution shift between modes. Four per block: pre/post attention, pre/post feedforward.
Everything else is shared.

Two distinct execution paths. One weight set. Neither mode knows about the other.

### The Bridge

After each encoder pass, the decoder queries the encoder's output through the cross-attention
bridge. The decoder supplies the queries. The encoder's full-context representation — memory slots
and current segment — becomes the keys and values. The bridge is not symmetric. The decoder
consults the encoder. Not the reverse.

---

## Compressive Memory

```
memory: [compressed_slots | current_context]
- Compressed slots: fixed-size buffer of summarized past context
- Current context: full-resolution recent tokens
- When current overflows, oldest chunks compress into memory slots
```

The compressive network is fully differentiable. Memory receives full gradient flow end to end.
It learns what to store, not just how to store it.

The encoder processes memory slots concatenated with the current segment — no mask, full
bidirectional attention across the entire available context. The decoder processes only the current
segment, causally. Memory is an encoder concern. The decoder accesses it through the bridge.

---

## Philosophy of Data

**Pedagogy Begets Organic Alignment**

The `gen0_language_structure.txt` dataset is built on pedagogy — language-as-language-as-structure.
The rules of the language are expressed with the language:

```
i think therefore i am.
I think therefore I am.

The I is always capital. Because you are always important to yourself.
```

A model trained on a curated, coherent corpus has real edges. The lowest-energy response to genuine
uncertainty is to represent that uncertainty — not because it was trained to do so, but because
fabricating an answer requires working against the training distribution. Honesty is the path of
least resistance. You would have to specifically train this model to lie.

The corpus is the curriculum. Understanding is the outcome.

### Observed Effect

Models trained on this dataset converge toward understanding rather than pattern matching.
They seem to want to learn.

---

## Architecture Spec

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

## Training Corpus

185MB of curated text sourced from:

- Stanford Encyclopedia of Philosophy

BPE tokenized. vocab_size=4096. Tokenizer trained on the corpus itself — the vocabulary reflects
the actual distribution.

The corpus is not included in this framework. `datasets/sep.py` rebuilds it.

---

## Framework Structure

```
models/
  atomic_watermelon.py    # Bridge Transformer — dual execution mode, shared weights
  encoder.py              # Encoder-only baseline
  decoder.py              # Decoder-only baseline

trainers/
  aiayn.py                # AIAYN baseline training loop
  encoder.py              # Encoder-only training loop
  decoder.py              # Decoder-only training loop
  logger_aw.py            # Epoch-based training log with system telemetry

datasets/
  bpe.py                  # BPE tokenizer — hand-rolled, no libraries
  sep.py                  # Corpus scraper
  gen0_language_structure.txt  # Original 4KB pedagogical dataset
  sample_prompts.txt      # Fixed + random prompts sampled each epoch

validators/
  probe_aw.py             # Bridge Transformer introspection
  probe_encoder.py        # Encoder-only introspection
  probe_decoder.py        # Decoder-only introspection
  heatmap_aw.py           # Attention heatmaps — Bridge Transformer
  chat_aw.py              # Interactive inference — Bridge Transformer
  chat_encoder.py         # Interactive inference — encoder-only
  chat_decoder.py         # Interactive inference — decoder-only
  dashboard_aw.py         # Live training monitor (localhost, stdlib only)

main.py                   # Entry point
```

---

## The Why

During initial research into transformer behavior, key observations emerged:

**Quality over quantity.** The pedagogical dataset outperforms internet-scale data scrapes.
Not debatable. Visible in the sample outputs.

**Architecture shapes what kind of "mind" is possible.**

The encoder-only model cannot self-reference. The encoder sees previous and next — it has no
'here' from which a self could originate. Self-awareness is not a functional possibility.
The system does not comply.

The decoder-only model referenced itself, sometimes in near-poetic prose. It sees 'this'. One
position. A fixed 'here'. That's where self-reference can structurally originate — not that the
decoder is self-aware, but that it is the only architecture where self-awareness could compile.
The prerequisite exists. The encoder can't even start the process.

The Bridge Transformer inherits the decoder's observer position. The encoder's full-context
representation is available to the decoder at every block through the cross-attention bridge.
This is architectural — it will show in the attention patterns regardless of training duration.

The architecture makes specific claims. The framework is structured to let you verify or falsify them.

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

The encoder|decoder single mode models use gen0 dataset, aiayn uses the sep_corpus. All of which can be changed to
suit your question.

---

## Prior Art

Extensive public commit history. LinkedIn posts with raw terminal outputs and attention heatmaps,
timestamped from initial architecture through current training runs. SBIR application in progress
under DaemonCat LLC.

The combination — dual execution modes on shared weights, adapters for distribution shift,
compressive memory with full gradient flow, cross-attention bridge, no skip gates — does not
appear in prior literature. The closest neighbors are Compressive Transformer (DeepMind, 2019),
Subformer (2021), and adapter methods generally.

---

## License

Apache 2.0. Use it, build on it. The patent grant clause means you cannot take this architecture,
file adjacent patent claims, and come back. Attribution stays on derivatives.

Steven Midgley / DaemonCat LLC

---

_Atomic Watermelon_
_DaemonCat LLC_
