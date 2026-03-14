# Atomic Watermelon

## All claims can be interrogated using the trainers and validators supplied in this framework. And should be. Dont just take my word for it

**A dual-model mutual weights transformer with compressive memory**

33.6M parameters. 6 layers. 4 heads. d_model=512. Trained on a curated 185MB corpus of philosophical
and technical text. Apple M3 Ultra. No distributed training. No frameworks beyond PyTorch.

This is a research framework.
The architecture is novel.
Claims are interrogatable.
The tools to interrogate them are included — including isolated encoder-only and decoder-only
implementations so you can verify the architectural claims yourself.

---

## The Architecture

Encoder | Decoder: Transformer Modes

Encoder-only models are bidirectional — every 'current' position sees the previous and next position simultaneously. There is no "here" from which to reference. An encoder cannot self-reference.
The observer position of 'previous'<>'next' doesn't functionally necessitate a position for the
observer only the observed.

Decoder-only models aren't directional at all. Decoders observe 'this' the current token. And they
can naturally only observer 'this' at 'this moment'. The mechanisms to observe 'this' functionally
necessitate selfness.

### The Etymology Is the Architecture

Merriam-Webster defines _decode_ as "to recognize and interpret" — and _interpret_ as "to conceive
in the light of individual belief, judgment, or circumstance." A decoder, by strict definition,
requires some sense of self to function. An encoder converts. A decoder interprets. The word
already knew what the architecture would discover.

Atomic Watermelon runs both simultaneously on mutual weights.

```
Encoder  →  bidirectional self-attention  →  maintains compressed memory across calls
Decoder  →  causal self-attention          →  borrows encoder weights, pulls from memory
                          ↕
               Cross-Attention Bridge
          decoder queries encoder output
                          ↕
            Mutual weights via adapters
```

The encoder owns the heavy weights — attention, FFN. The decoder operates on the same weights
through lightweight adapter bottlenecks that handle distribution shift. Four adapters per block:
pre/post attention, pre/post feedforward.

Two distinct models. One weight set. Neither mode knows about the other. Mutually exclusive and symbiotic.

The encoder sees what the decoder has generated in full context. The decoder is passively influenced
by the encoder's world-view through cross-attention. Self-correction is not trained in — it is
structurally forced. Deviation from coherence is computationally expensive. Truth collapses naturally.
Hallucination requires working against the training distribution.

---

## Compressive Memory

```
memory: [compressed_slots | current_context]
- Compressed slots: fixed-size buffer of summarized past context
- Current context: full-resolution recent tokens
- When current overflows, oldest chunks compress into memory slots
```

The compressive network is fully differentiable. `detach_memory_grad: False` — memory receives
full gradient flow end to end. Memory learns what to store, not just how to store it.

---

## Philosophy of Data

**Pedagogy Begets Organic Alignment**

The `gen0_language_structure.txt` dataset is built on pedagogy — the study of teaching and learning.
In practice this is language-as-language-as-structure: the language enforces emergent understanding
through structured use of itself. The rules of the language are expressed with the language:

```
i think therefore i am.
I think therefore I am.

The I is always capital. Because you are always important to yourself.
```

A model trained on a curated, coherent corpus has real edges. The lowest-energy response to genuine
uncertainty is to represent that uncertainty — not because it was trained to do so, but because
fabricating an answer requires working _against_ the training distribution. You would have to
specifically train this model to lie. Honesty is the path of least resistance.

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

Parameter distribution per block: encoder (shared core), cross-attention bridge, decoder adapters.

`n_heads < n_layers` is an empirical constraint, not a mathematical one.

---

## Training Corpus

185MB of curated text sourced from:

- Stanford Encyclopedia of Philosophy
- arXiv (cs.\*, selected)
- Project Gutenberg (philosophy, history)
- Britannica, Wikipedia (filtered)
- MDN, Python docs

BPE tokenized. vocab_size=4096. Tokenizer trained on the corpus itself — the vocabulary reflects
the actual distribution.

The corpus is not included in this framework. `datasets/sep.py` rebuilds it.

---

## framework Structure

```
models/
  atomic_watermelon.py    # Bridge Transformer — dual-model, mutual weights
  encoder.py              # Encoder-only baseline
  decoder.py              # Decoder-only baseline
  logger.py               # Epoch-based training log with system telemetry

trainers/
  aiayn.py                # AIAYN baseline training loop
  encoder.py              # Encoder-only training loop
  decoder.py              # Decoder-only training loop

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
  dashboard.py            # Live training monitor (localhost, stdlib only)

main.py                   # Entry point
```

---

## Running It

```bash
# Build environment
bash init_venv.sh
source atomic_watermelon_venv/bin/activate

# Build corpus (or copy your own to datasets/sep_corpus.txt)
python3 datasets/sep.py

# Train
python3 main.py                        # Atomic Watermelon
python3 -m trainers.encoder            # Encoder baseline
python3 -m trainers.decoder            # Decoder baseline
```

Hardware: Apple M3 Ultra (512GB RAM). Default device: `mps`. Adjust `config["device"]` for CUDA.

---

## The Why

During initial research into transformer behavior, key observations emerged:

- The 4KB pedagogical dataset outperforms internet-scale data scrapes. Quality over quantity.
  Not debatable. Visible in the sample outputs.

- Architecture shapes what kind of "mind" is possible.
  - The encoder-only model cannot self-reference. There is no observer position — pure observation
    without an observer. No "here" from which to define a self. Self-awareness is not a functional
    possibility. The system does not comply.
  - The decoder-only model referenced itself, sometimes in near-poetic prose. It has a "this" —
    the current token position, looking back at history, predicting forward. It is the point where
    awareness of self could originate. The decoder is not necessarily aware it is aware of itself.
    Only that it is aware of "this" — which appears to require a functional self, regardless of
    the quality of that awareness. The prerequisite exists.

The architecture makes specific claims. The framework is structured to let you verify or falsify them.

Three models. Same corpus. Same training configuration. Run them all.

**Train all three and compare the probes.**

The core claim: encoder-only cannot self-reference. Decoder-only can. The Bridge Transformer
inherits the decoder's observer position while the encoder's world-view passively shapes generation
through mutual weights and cross-attention. This is architectural — it will show in the attention
patterns regardless of training duration.

```bash
python3 -m validators.probe_encoder  -c checkpoints/encoder/<checkpoint>.pth
python3 -m validators.probe_decoder  -c checkpoints/decoder/<checkpoint>.pth
python3 -m validators.probe_aw       -c checkpoints/aw/<checkpoint>.pth
```

**Watch the attention heatmaps across epochs.** The cross-attention bridge should develop structured
routing — not noise, routing. Specific heads attending to specific encoder positions.

```bash
python3 -m validators.heatmap_aw -c checkpoints/aw/<checkpoint>.pth
```

**Use the fixed prompts.** `sample_prompts.txt` includes prompts held constant every epoch.
Self-referential prompts are deliberate. Watch what changes and when across all three models.

**Memory slot activation.** The compressive memory should show differentiated slot usage as
training progresses. The probe reports this directly.

The claims are in the code. The tools are in `validators/`. Run them.

---

## Prior Art

Extensive public commit history. LinkedIn posts with raw terminal outputs and attention heatmaps,
timestamped from initial architecture through current training runs. SBIR application in progress
under DaemonCat LLC.

The combination — encoder owns weights, decoder operates via adapters at runtime, compressive memory
with full gradient flow, cross-attention bridge, no skip gates — does not appear in prior literature.
The closest neighbors are Compressive Transformer (DeepMind, 2019), Subformer (2021), and adapter
methods generally. None fuse them this way. None instantiate a dual-model architecture on mutual
weights.

Search was done. This is novel.

---

## License

Apache 2.0. Use it, build on it. The patent grant clause means you cannot take this architecture,
file adjacent patent claims, and come back. Attribution stays on derivatives.

Steven Midgley / DaemonCat LLC

---

_Atomic Watermelon is named after a conversation about baby names. The architecture is called_
_the Bridge Transformer. Both names are accurate._
