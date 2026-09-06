# ByteBitGPT: a small experiment nobody's published yet

## The actual hypothesis being tested

A byte-level (tokenizer-free), ternary-weight ("1.58-bit", BitNet-b1.58-style)
transformer, at tiny (~10M parameter) scale, trained on a curated
simple-to-complex English curriculum. Specifically:

1. Ternary weights should beat pure binary weights, because the extra
   "zero" state lets a connection switch off entirely instead of being
   forced to always add or subtract (BitNet b1.58's core finding, at
   billion-parameter scale). Does that hold at 10M scale too?
2. The gap between ternary and full-precision should be larger at small
   scale and shrink as the model gets bigger (BitNet b1.58 showed this
   from 700M to 3B). Can you see the same trend from ~5M to ~50M?
3. Depth vs. width matters differently for different skills in normal
   models (width -> knowledge retention, depth -> contextual coherence,
   per the TinyStories paper). Does *ternary* weight-sharing change which
   one matters more, since each ternary weight carries less information?
4. Does a simple-to-complex data curriculum (TinyStories -> FineWeb-Edu ->
   Cosmopedia-style synthetic textbooks) help a capacity-starved ternary
   model more or less than it helps a full-precision model of the same
   size?

None of these four questions have a clear answer in the published
literature as far as available searches show -- the pieces (BitNet b1.58,
byte-level/tokenizer-free models, TinyStories, curriculum-filtered data)
have each been studied alone, but not stacked together at this scale.
That's what makes this worth actually running, not just reading about.

## What this project is NOT

- Not a path to a generally knowledgeable assistant. ~10M parameters
  cannot store broad world knowledge no matter how good the data or
  architecture is -- that's a hard capacity ceiling, not a training
  problem. Use an existing pretrained small model (e.g. via Ollama) for
  that need in parallel.
- Not multilingual -- byte input here is plain UTF-8, no MYTE. That's
  fine and simpler, since English-only was the stated goal.

## Files

- `model.py` -- ByteBitGPT: byte-level vocab (256, literally every byte
  value, no tokenizer file), with a `BitLinear` layer that can run in
  `fp` / `ternary` / `binary` mode via a straight-through estimator.
- `train.py` -- Colab-ready training loop with mixed precision, Drive
  checkpointing (Colab free tier disconnects -- don't lose your run),
  and the ablation table to fill in at the bottom of the file.

## Quickstart in Colab

```python
# Cell 1
from google.colab import drive
drive.mount('/content/drive')

!pip install -q datasets torch --upgrade

# Cell 2 -- upload model.py and train.py to /content/, or clone from a repo
%cd /content

# Cell 3 -- first run: small model, TinyStories, ternary
!python train.py --mode ternary --n_layer 12 --d_model 256 --steps 5000 \
    --stage tinystories --max_docs 200000

# Cell 4 -- baseline for comparison, same shape, full precision
!python train.py --mode fp --n_layer 12 --d_model 256 --steps 5000 \
    --stage tinystories --max_docs 200000

# Cell 5 -- binary, to test the "zero matters" claim yourself
!python train.py --mode binary --n_layer 12 --d_model 256 --steps 5000 \
    --stage tinystories --max_docs 200000
```

Each run logs `val_loss` every `--eval_every` steps to a `.log.csv` in
your checkpoint dir on Drive, and prints a text sample every
`--sample_every` steps so you can watch it go from byte-noise to
recognizable English in real time.

## Practical T4 notes

- `d_model=256, n_layer=12` is roughly 10M params and should comfortably
  fit on a T4 (16GB) with batch_size=32, block_size=512.
- fp16 autocast is used (not bf16 -- T4 doesn't support bf16 well).
- Free Colab sessions disconnect / time out. Checkpointing to Drive every
  `eval_every` steps plus `--resume` means a disconnect costs you at most
  that interval of training, not the whole run.
- `--max_docs` caps how much of a streamed dataset gets pulled and held
  in memory as raw bytes -- start small (the default) to make sure the
  whole pipeline works before committing a long run to a bigger slice.

## After you have results

Fill in the ablation table at the bottom of `train.py`. The real value of
this project isn't the model you end up with -- a 10M-parameter model
will not be smart. The value is the table: it's actual, first-hand
empirical evidence about how ternary weights, depth, and data curriculum
interact at small scale, which is exactly the "run the tests, look at the
evidence, form new hypotheses" loop from the start of this conversation.
