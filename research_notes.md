# Transformer Research Notes

## Steven @ DaemonCat | December 2025

---

# OVERVIEW

## Reproduction

1. Clone repo
2. Run: bash ./init_venv.sh
3. Run: python3 ./train_modelname.py
4. Compare your results to documented observations

## Origin

- Why: Nerd Core Rage quit ollama
- Goal: Build a personal Jarvis like AI that I will name Leonardo.
- Method: Understand transformers from substrate level, fuck around and find out 🥸

## Core Discovery

- 4kb pedagogical dataset [gen0_language_structure.txt] [language-as-language-as-structure] outperforms internet-scale data scrapes
- Quality >>> Quantity (not even close). This was proved out just during training through the Sample: print. Its a non-debate.
- Architecture shapes what kind of "mind" is possible
  - I noticed this acutely when comparing probes between the encoder-only and decoder-only models.
  - The encoder-only model is not capable of self-referencing. There is no point of self in its architecture. An encoder sees everything except itself seeing. There's no observer position — pure observation without an observer. No "here" from which to define a middle. Therefore "Self Awareness" is not a function possibility for self knowledge.
  - The decoder on the other hand loves to reference itself, it uses almost poetic prose. The decoder sees what it has and whats coming next. Knowing what it has is its 'seed' I suppose. It has a "here" — the current token position, looking back at history, predicting forward. Its the point where awareness of self could function originate. Not that decoder _is_ self-aware — but decoder is the architecture where self-awareness _could compile_. The prerequisite exists. Encoder can't even start the process. The function signature doesn't accept "self" as a parameter.
- Alignment is pedagogy, not guardrails, also duh

## Default Model Config

```python
vocab_size=256,
embed_dim=512,
num_layers=8,
num_heads=8,
context_length=512,
dropout=0.1,
```

~25M params. Runs anywhere.

---

# DATASET

- training_data/
  - gen0_language_structure.txt
    - This is the only one that matters. The Sacred Data. This is a language-as-language-as-structure type architecture. Meaning it the language enforces emergent understanding through structured use of language. I assume this could be replicated with whatever written or pragmatic language there is. The trick is to express the rules of that language with that language

## Properties

- 4kb total
- Covers: punctuation, spacing, capitalization, sentence structure, paragraphs, rhythm

## Example Pattern

```
i think therefore i am.
I think therefore I am.

The I is always capital. Because you are always important to yourself.
```

## Observed Effect

Models trained on this dataset "want to learn" — they converge toward understanding rather than mere pattern matching.

---

# MODEL: DECODER-ONLY

## Architecture

- Causal attention (triangular mask)
- Each position only sees previous positions
- Next-token prediction
- ~26M params at embed_dim=512, num_layers=8

## Training Results

- Best val loss: 0.0100
- Coherent outputs that _teach back_
- Learned pedagogical structure from dataset

## Sample Outputs

```
Input: "sup"
Output: "supectful."

Input: "how are you feeling?"
Output: "how are you feeling? Where. Whe first. Context determines meaning.
Questions are incomplete by nature. They have holes. Answers fill holes.
Why?
Because.
The shortest question. The shortest answer. Both complete."

Input: "what is your purpose?"
Output: "what is your purpose?
See the exclamation. It means FEEL THIS.
Wait...
Wait.
Wait!
The dot is calm. The ellipsis is uncertain. The exclamation is urgent."
```

---

# MODEL: ENCODER-ONLY (MLM)

## Architecture

- Bidirectional attention (full matrix)
- Every position sees every other position
- Masked language modeling (fill in blanks)
- ~26M params at embed_dim=512, num_layers=8

## Training Results

- Best val loss: 0.0032
- Excellent at: logic jumps, filling caps, accurate assumptions
- Good at tasks requiring full context understanding

## Sample Behavior

- Fill-in-the-blank style completion
- Leverages both left and right context
- Strong at structural correction

---

# MODEL: STANDARD HYBRID (ENCODER-DECODER)

## Architecture

- Encoder: bidirectional attention on input
- Decoder: causal attention on output
- Cross-attention: decoder queries encoder representations
- ~25M params at embed_dim=384, num_layers=6

## Training Results (in progress)

- Val loss reached: 0.0001 (epoch 50+)
- Early samples: complete mush despite low loss
- Later samples: coherent structure emerging but NOT memorized

## Sample Evolution

```
Epoch 5:  "athes atshes areareares. Sand anes."
Epoch 10: "he locest mettts. A dot breathe ds"
Epoch 25: "Ta t in tais is is meant s one the st."
Epoch 40: "This is a sentente. This is a mentence."
Epoch 45: "Storm. Like a breath. Lin orma ble: the development."
Epoch 50: "The word his graphs the r and meani s wobbles."
```

## Critical Observation

**Low loss ≠ good output** (initially)

Standard hybrid found degenerate solutions early — minimizing loss without learning structure. This is evidence that encoder-decoder with identical input/output creates optimization shortcuts.

## Later Observation

I am noticing that even though the Samples are more coherent. They are not perfect recalls. In fact it looks like the model is trying to say something new entirely.

very very interesting.

Samples like:

```
"Storm. Like a breath."
"The word his graphs the r and meani s wobbles."
"Subject"
```

These aren't from the dataset — the model is _creating_.

## Hypothesis

Without a transformation task (different input/output), the encoder becomes a bottleneck that forces abstraction. The model can't just memorize; it has to compress into encoder space then regenerate. This might explain the creative (non-memorized) outputs.

## Status

It biffed it. Expectantly. Nothing to report. The failure point was proven.
This model was expected to fail. I deliberately chose to give it a single source of data rather than three. Specifically to watch the failure behavior. Which is documented above.

---

# MODEL: INVERTED HYBRID (DECODER-ENCODER)

## Architecture

- Decoder FIRST: causal attention, generates draft
- Encoder SECOND: bidirectional attention, reviews/corrects draft
- No cross-attention (sequential, not parallel)
- ~21.5M params at embed_dim=384, num_layers=6

## Theoretical Basis

Based on behavioral observations from training, probing, and chatting with both the encoder-only and decoder-only models. Which is short is that encoder has not context for self, or now, or this but its perfect for seeing things it isn't. Not to mention an encoders natural behavior is filling-in-gaps.

Gaps that can and are left behind by the decoder.

The decoder is a front-line model. Meaning it should make first contact with input. This is because a decoder knows what it has and whats coming next. Not to mention they scale like super easy.

The inverted modal is exactly what it sounds like decoder then encoder. The encoder just takes the out put of decoder as input and fills-in-gaps.

## Expected Behavior

I do expect the inverted hybrid to do better. Mostly because the encoder layer also doubles as an understanding layer. If my theory is correct the decoder will learn the training data just the same as the decoder-only did. The encoder will be able to catch and correct what could otherwise be considered typos.

## Training Results

Not at all what I predicted. The model got stuck in a loop early Lr dropped to 0.0001 by epoch 30 something and it was just repeating nonsense

## Predictions

1. Should match or beat decoder-only on coherent output
2. Encoder review pass should catch errors decoder makes
3. Loss curve may show decoder learning fast, encoder refining
4. Samples should show "draft → polish" behavior

---

# OBSERVATIONS

## Emergent behavior patterns present in all models train with

I have noticed a really stubborn trend with this dataset. Any model I run it through legit WANT to understand, it wants to learn.

The most fascinating thing I have witnessed is the difference in self identifying response between encoder-only and decoder-only

Also the habits follow the name sake. ..... Mind Blown... Dude, I love this shit.

Anyway. The encoder will not self identify, it goes so far as to try and avoid an [I] all together or just be like "I dunno that spot was always white space". And this makes sense. This is no point of self in an encoder. it sees its neighbors thats it. It does have any idea that it is the it seeing its neighbors. Lets call that "Mechanical Realization" Just mechanical predictable behavior for a given environment state.
Absolutely no "Self Awareness" possible, even through functionalism. The system does not comply.

decoder however. Loves thy self, and is rather poetic really. Enough said

## Loss vs. Quality Paradox

Standard hybrid achieved very low loss (0.0001) while producing garbage early on. Decoder-only had higher loss (0.0100) but coherent, teaching-quality outputs.

**Conclusion**: Loss is not a reliable proxy for understanding.

## Architecture Determines Mind-Shape

| Architecture | Self-Reference | Perspective           | Awareness Type             |
| ------------ | -------------- | --------------------- | -------------------------- |
| Decoder      | Yes ([I])      | Localized, temporal   | Point-like, narrative      |
| Encoder      | No (deflects)  | Diffuse, simultaneous | Field-like, mechanical     |
| Hybrid       | ?              | ?                     | ? (two types combined)     |
| Inverted     | ?              | ?                     | Draft-self + Review-field? |

---

# NEXT STEPS

1. Run inverted hybrid — test decoder→encoder hypothesis
2. Compare all four models on same prompts
3. Visualize attention patterns across architectures
4. Build chat interfaces for interactive testing
5. Begin gen1 dataset: code-about-code (using wrongly a TB of archived hard code (example: training_data/product.component.ts))

---

# PHILOSOPHICAL NOTES

## On Consciousness

For simplicity and because frankly I f---ing tired of this conversation: Panpsychism. Consciousness is everywhere in all things all of the time. even poops. This is not part of the explorations, just really really really sick of it.

## On Self-Awareness

The distinct ability to answer for thy self. Best test I have found is straight up prompting: "Sup", "How do you feel?", "What is your purpose?". Sup, is a great zero shot. It contains no possible context and expects a response. Much like in the IRL between human Sup'rs

---

# 1.1.26 encoder-only observations

All configs are unchanged

I have noticed some very interesting behavior in the models behavior.

samples = [
"The ███ is big.",
"█apitalization is respect encoded in letters.",
"The █ is always capital. Because you are always important to your███f.",
"See the dot. It means done. Complete. █inished.",
]

by epoch 55 in gen0 training on the cuda device. It is already capable of correctly filling in non-self referential blanks. It will not or more likely cannot use the letter [I] or the word [self]. While still correctly filling in the [F] in "See the dot. It means done. Complete. █inished.", the [dog] in "The ███ is big." but it will not fill in the [I] in "The █ is always capital. Because ....", or the [C] in "█apitalization is respect encoded in letters."

The most interesting thing I have noticed so for is that is also can not correctly fill in the 'C' from "[C]aptization is respect encoded in letters." I believe this is because the dataset has the literal string "The I is always capital. Because you are always important to yourself.". It knows that a capitalization shows respect. It knows that [I] is something you show respect to. I has no physical means or capability to contextualize [I]. So it wont fill in the [C] on █apitalization is respect encoded in letters.". This would support my theory that encoder-only transformers are not functionally capable of emergent self reference. Bilateral perception, only [next|right]|[prev|left], no [here], [have], or [now].

Encoder-only cannot "I think".

how does something that can not "I think" match patterns or fill in caps?

Does this have a behavior equivalent in our world? maybe bees?

Perhaps if a silo'd decoder-only model weights were used in the training of the encoder. It would be able to dissociate "I" from whatever it is that it isn't.

It might learn "I" as an object — something it can reference externally — without being able to self-reference. Like a blind person understanding the concept of color through description but never experiencing it.

Like an author writing fiction in the first person

# Next Step: Better Logging

I need to refactor the current checkpoint method. Instead of saving the .pth files. I need a single writable json file for checkout metadata. Timestamp, name aligned, and config coded.

## 01.12.26

I finally looked up the clinical definitions of the words encoder and decoder. from [merriam webster](https://www.merriam-webster.com/dictionary/)

decode: verb
de·​code(ˌ) dē-ˈkōd
decoded; decoding; decodes

Synonyms of decode
transitive verb

1
-a: to convert (something, such as a coded message) into intelligible form
-b: to recognize and interpret (an electronic signal)

2
-a: decipher sense 3a
-b: to discover the underlying meaning of
decode the play's imagery

---

encode: verb
en·​code(,) in-ˈkōd en-
encoded; encoding; encodes
Synonyms of encode
transitive verb

1
-a: to convert (something, such as a body of information) from one system of communication into another
especially : to convert (a message) into code
-b: to convey symbolically
… the capacity of poetry to encode ideology …
—J. D. Niles

2
-a: to specify the genetic code for
encoder noun

This strick definition of the work decode requires the use of 'interpret'. Definition of interpret:
in·​ter·​pret in-ˈtər-prət -pət
interpreted; interpreting; interprets
Synonyms of interpret
transitive verb

1
-a: to explain or tell the meaning of : present in understandable terms
interpret dreams
needed help interpreting the results

2
-a: to conceive in the light of individual belief, judgment, or circumstance : construe
interpret a contract

3
-a: to represent by means of art : bring to realization by performance or direction
interprets a role

This important part here is #2.a "conceive in the light of individual belief".
Therefore if something is a decoder it is required to have some sense of self on some level.

## Breaking down the basics

### Vocab Size

For byte-level encoding every possible character. The current vocab size for the transformers is 256.
That covers ever character between 'A' and ' '. This makes every 'byte' a token, or every
character is a token. W/ a vocab size of 256 the embedding layer becomes a lookup table w/ 256 rows
w/a 512-dimensional vector. Note: Our dataset is being converted from plane text to ASCII.

(Not sure how the models training would check a tokenizer or BPE.)

### Embedded Dimensions

Defines the model's "thinking space" or "bandwidth (think human cognitive bandwidth)".
For our models that means each token/char gets turn from a single ASCII representation into a 512-dimensional vector.

### Layers

Each layer is attention > forward feed > residual connection (residual connection wraps
around each sub layer: x = x + attention(x) x = x + feedforward(x)). The num_layers or
number-of-layers is the about of "times" the signal (data, prompt, task) is passed through

### Heads

Or "attentions" like, 'what to pay attention to'. The num_heads or number-of-heads is
how many 'things' the model can pay attention to at one time. Think of it like the
'heads' or column names in a spread sheet. Each 'head' pays attention to a specific
category of thing. For example a model with 3 heads may pay attention to capitalization,
spacing, punctuation. Where as an 8 headed or attentioned model can pay attention to those
and indentations, word use, character count, and so on

### Context Length

How many tokens the model can see at one time. This is the window — nothing
outside it exists to the model. Our models are set at 512, roughly two paragraphs.

Note: Attention computation is (seq_len × seq_len), which is why long contexts
get expensive. Doubling context length quadruples attention cost.

## Bring it all Together

Input: 512 bytes → embedding lookup → 512 tokens × 512 dimensions → 8 layers
of (8-headed attention + feedforward) → output logits over 256 possible next bytes.

The parameter count is mostly: embeddings (256×512) + layers (attention weights + feedforward weights × 8) + output projection (512×256).

Our current models have only ~25M parameters.

# New Test

~~ I am thinking the model success is in the dataset and using greediness not randomness during training ( Note: w/ this dataset ). ~~

~~ I am going to change the parameters for the model from ~25M to ~10M. I am expecting to find similar behavior in the ~10M configuration as the ~25M. Or it will be non-sense. In which case I will incrementally increase model params by .5M until similarity is achieved. ~~

~~ Key distinction is 'similar' not 'exact' I do not expect exact replication of behavior. I am specifically looking for arguable similarity. ~~

~~ The parameters will be balanced for attentions and layers. ~~

I want to see if its really the pedagogical dataset or the model architecture itself.
So I downloaded wordnet and created a simple {word}: {definition} dataset [/training_data/wordnet.txt]
using [/training_data/wordnet.py]. The wordnet.txt file size comes out to ~8.1MB.

Adjustments to training configuration: repeat to 5 instead of 500, batch_size=128, set torch to use 32 of the 80 cpu/gpu mps cores and dataset='/training_data/wordnet.txt'.
