# ByteBitGPT — Experiment 1: Ternary vs Binary vs Full-Precision at 10M Params

## Hypothesis

A byte-level (tokenizer-free), ternary-weight ("1.58-bit", BitNet-b1.58-style)
transformer was proposed to test four things at tiny (~10M parameter) scale,
where the published literature (BitNet b1.58, byte-level architectures like
BLT, TinyStories) has examined each piece separately but not this specific
combination:

1. Does ternary's extra "zero" state (vs. pure binary {-1,+1}) provide a
   real, measurable quality advantage at small scale, or only at the
   billion-parameter scale where it was originally demonstrated?
2. How large is the quality gap between ternary weights and full precision
   at 10M params?
3. Does the model's depth/width shape predict *which* capabilities degrade
   under quantization (per the TinyStories finding: width -> knowledge
   retention, depth -> contextual coherence)?
4. (Not yet tested — see Next Steps) Does a simple-to-complex data
   curriculum change any of the above?

## Architecture

| | |
|---|---|
| Input representation | Raw UTF-8 bytes, vocab size 256 — no tokenizer |
| Layers (depth) | 12 |
| d_model (width) | 256 |
| Attention heads | 8 (32 dims/head) |
| Context length | 512 bytes |
| Width:depth ratio | ~21 (narrower/deeper than GPT-2-small's ~64) |
| Parameters | 9.74M (fp) / 9.78M (ternary, binary — extra LayerNorm params) |
| Weight quantization | Ternary {-1,0,+1} or binary {-1,+1} via straight-through estimator, scale factor = mean(\|w\|), full-precision shadow weights updated by gradient descent |
| Output head | Kept full precision in all three runs (standard practice) |

## Data

- Dataset: TinyStories (`roneneldan/TinyStories`), streamed, capped at 200,000
  documents (~181MB of raw text).
- Split: 99% train / 1% validation, held out from the same distribution.
- No preprocessing beyond UTF-8 byte encoding — no tokenizer, no vocabulary
  file, no cleaning beyond what's already in the source dataset.

## Training

- Optimizer: AdamW, cosine LR schedule (peak 3e-4, 200-step warmup)
- Precision: fp16 mixed precision, T4 GPU (Colab free tier)
- Batch size 32, block size 512, 5000 steps, ~1 epoch over the loaded data
- Identical hyperparameters, data, and steps across all three weight modes
  — the only variable changed is weight precision

## Results

| Mode | Params | Best val_loss | Step reached | Wall-clock time |
|---|---|---|---|---|
| **Full precision (fp)** | 9.74M | **0.7870** | 4999 | 2530s (~42 min) |
| **Ternary** | 9.78M | **0.8665** | 4500 | 2702s (~45 min) |
| **Binary** | 9.78M | **1.1867** | 4999 | 2903s (~48 min) |

Relative to fp: ternary is **+10.1%** loss, binary is **+50.8%** loss.
Relative to ternary: binary is **+37.0%** loss.

### Sample text at step 4000 (matched checkpoint across runs)

**fp:**
> Once upon a time, there was a little girl named Lily. One day, Lily's mom
> told her that she wanted to dry the bed and she had played with her back.
> Mom saw a big colo[...]

**ternary:**
> Once upon a time, there was a little girl named Lily. They loved to play
> with her cows, but then one find search. One day, her mommy caught a
> said, "Here, it's my bes[...]

**binary:**
> Once upon a time, it was very loong, and said. You looked with the to
> make it bird not con the had oner to be minny. He cour to the greet to
> of it. He had it it was c[...]

Coherence length (roughly, how many words before the sentence stops making
grammatical sense) tracks the loss ordering exactly: fp > ternary > binary.

## Discussion

**Finding 1 — the "zero matters" claim replicates at 10M params, clearly.**
The BitNet b1.58 paper's central claim — that ternary's zero state (the
ability to switch a connection off entirely, not just flip its sign) gives
a large practical advantage over pure binary weights — was demonstrated
at 700M-3B parameter scale. This result shows the same effect holds, and
is large (+37% loss for binary vs ternary), at roughly 1/100,000th that
scale. This is the strongest and cleanest finding of this run.

**Finding 2 — the ternary-to-fp gap is smaller than might be expected at
this scale.** BitNet b1.58's own results show a larger fp-vs-ternary gap
at smaller model sizes (700M) that closes by 3B params. This run's gap
(+10%) is modest for a model this small. One plausible explanation: the
TinyStories domain has an unusually narrow, simple target distribution
(short sentences, ~1500-word vocabulary), which may require less raw
representational precision than open-domain text — a testable hypothesis
in itself (see Next Steps).

**Finding 3 — the architecture's shape likely explains the specific
*kind* of degradation observed.** This model is narrower and deeper than
a standard small transformer (width:depth ratio ~21 vs. GPT-2-small's
~64). Per the finding that width correlates with knowledge retention and
depth with contextual coherence, this shape predicts exactly what was
observed: coherent multi-sentence subject-tracking (e.g., "Lily" carried
correctly across two sentences) alongside narrow, repetitive vocabulary
and factual/logical breakdown mid-sentence. This suggests the
quantization penalty may interact with model shape — not yet tested
directly (see Next Steps).

**Finding 4 — no training-time speed advantage was observed for
low-bit modes; if anything the opposite.** fp finished fastest (2530s),
ternary next (2702s), binary slowest (2903s). This is expected and not a
contradiction of BitNet's efficiency claims: those claims are about
*inference-time* efficiency using specialized bit-packed kernels
(bitnet.cpp) and reduced weight storage, not about training with a
straight-through-estimator simulation running on top of ordinary fp16
tensor operations. This run measured a quality trade, not a speed trade.

## Limitations

- 10M parameters is far below any scale where "general capability" is
  meaningful. This is a controlled comparison of weight precision, not a
  step toward a usable assistant.
- Single dataset (TinyStories only), single architecture shape, single
  run per condition — no variance estimate (multiple seeds would
  strengthen the loss-gap numbers).
- Training-time overhead of the quantization ops was not optimized or
  profiled in detail; the wall-clock differences (Finding 4) are
  suggestive, not a rigorous efficiency benchmark.

## Next steps (not yet run)

1. **Depth-vs-width ablation**: hold ternary mode and param count fixed,
   vary shape (e.g. 6 layers x 384 width vs. 12 x 256 vs. 24 x 176).
   Tests whether the quantization penalty is shape-dependent (Finding 3,
   directly).
2. **Scale-up**: repeat fp/ternary/binary comparison at a larger size
   (e.g. ~50M params) to check whether the fp-ternary gap shrinks further
   with scale, as in BitNet b1.58's original results.
3. **Curriculum stage 2**: continue training (via checkpoint resume) on
   FineWeb-Edu, then Cosmopedia-style synthetic textbooks, to see how a
   capacity-constrained ternary model responds to broader, harder data
   compared to fp at the same size.
4. **Multi-seed replication** of the core fp/ternary/binary comparison to
   put error bars on the +10% / +50% figures above.
