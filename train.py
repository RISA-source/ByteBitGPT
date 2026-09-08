"""
bitbyte_lm/train.py

Colab-ready training + ablation harness for ByteBitGPT.

WHAT THIS TESTS (the actual hypothesis, not just "let's train a model"):
  1. Does a ternary-weight ("1.58-bit") transformer close the gap with a
     full-precision transformer of the same shape, on byte-level input?
  2. Does that gap close *faster with depth* than with width, matching
     BitNet b1.58's finding that the gap shrinks with scale?
  3. Does ternary beat pure binary by a clear margin at this tiny scale
     (testing the "zero as a switch" claim yourself)?
  4. Does curriculum / data mixture change these answers?

MEMORY NOTES (read this if you hit an OOM / silent crash again):
  - Training uses get_batch() with torch.randint, NOT a shuffling
    DataLoader. A shuffling DataLoader's RandomSampler eagerly builds a
    full torch.randperm(N) -- for a several-hundred-million-position
    corpus that alone is multiple GB. get_batch() only ever allocates
    O(batch_size), regardless of corpus size.
  - Data loading builds ONE bytearray per stage and extends it in place,
    then folds it directly into ONE combined bytearray -- no
    list-of-chunks-then-join, no dict-of-bytes-then-join-again, no
    bytearray(bytes_obj) conversion at the end. Measured: the old
    list->join->dict->join->bytearray pipeline held up to 2x the logical
    corpus size in memory simultaneously from redundant copies alone,
    on top of the corpus itself. This version holds ~1x.

HOW TO USE IN COLAB:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload this whole bitbyte_lm/ folder
  3. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
  4. pip install -q datasets torch pyspellchecker --upgrade
  5. Run: python train.py --mode ternary --n_layer 12 --d_model 256 --steps 5000
"""

import argparse
import os
import time
import math
import torch
from torch.utils.data import Dataset, DataLoader

from model import ByteBitGPT


# ---------------------------------------------------------------------------
# Data: byte-level, no tokenizer. Every stage is folded into ONE bytearray
# for training; validation stays split per-stage for per-domain eval.
# ---------------------------------------------------------------------------

def load_curriculum_bytes(stage="tinystories", max_docs=None):
    """
    stage:
      "tinystories" -- simple synthetic children's stories (start here)
      "fineweb_edu" -- classifier-filtered educational web text (stage 2)
      "cosmopedia"  -- synthetic textbooks, broader "science/math flavor" (stage 3)
    Returns a bytearray, built by extending IN PLACE as documents stream in.
    (Not a list of chunks joined at the end -- that pattern holds both the
    list AND the joined result in memory simultaneously; measured 2x peak
    for the join step alone on a synthetic 500MB corpus.)
    """
    from datasets import load_dataset

    name_map = {
        "tinystories": ("roneneldan/TinyStories", None, "text"),
        "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        "cosmopedia": ("HuggingFaceTB/cosmopedia", "stories", "text"),
    }
    hf_name, config, field = name_map[stage]
    ds = load_dataset(hf_name, config, split="train", streaming=True)

    buf = bytearray()
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        buf.extend(row[field].encode("utf-8", errors="ignore"))
        buf.extend(b"\n\n")
        if i % 5000 == 0:
            print(f"  ...loaded {i} docs, {len(buf)/1e6:.1f}MB so far")
    return buf


def load_curriculum_bytes_cached(stage, max_docs, cache_dir=None):
    """Same as load_curriculum_bytes, but checks a cache file first (keyed
    by stage + doc count) and writes one after downloading. Without this,
    every --resume or retry re-streams the entire dataset from HuggingFace
    from scratch. Cache lives in cache_dir (point at Drive so it survives
    disconnects, not just re-runs within one session)."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{stage}_{max_docs}.bin")
        if os.path.exists(cache_path):
            print(f"[cache] found cached data for stage='{stage}' max_docs={max_docs}: "
                  f"{cache_path} -- skipping re-download")
            with open(cache_path, "rb") as f:
                return bytearray(f.read())
    buf = load_curriculum_bytes(stage, max_docs=max_docs)
    if cache_path:
        with open(cache_path, "wb") as f:
            f.write(buf)
        print(f"[cache] saved {len(buf)/1e6:.1f}MB to {cache_path} for future runs")
    return buf


def load_multi_stage_bytes(stage_weights, max_docs_total=200000, cache_dir=None):
    """
    stage_weights: dict like {"tinystories": 0.6, "fineweb_edu": 0.4}
    Approximates the mixture ratio via document COUNT per stage (weight *
    max_docs_total). Returns:
      combined_train (one bytearray, all stages folded in)
      val_by_stage   (dict, kept SEPARATE per stage for per-domain eval)

    Builds the combined train buffer incrementally: each stage's raw data
    is loaded, its val slice is copied out (small, ~1%), the rest is
    folded into the shared train bytearray via .extend(), and the stage's
    own buffer is freed immediately. Never holds two full-size copies of
    the same stage's data at once.
    """
    combined_train = bytearray()
    val_by_stage = {}
    for stage, weight in stage_weights.items():
        docs_for_stage = max(1, int(max_docs_total * weight))
        print(f"[mixture] loading stage='{stage}' weight={weight} -> {docs_for_stage} docs")
        raw = load_curriculum_bytes_cached(stage, docs_for_stage, cache_dir=cache_dir)
        split = int(0.99 * len(raw))
        # memoryview slicing does NOT copy (plain bytearray[:split] DOES --
        # measured that alone re-introducing a near-full-size temp copy).
        mv = memoryview(raw)
        val_by_stage[stage] = bytes(mv[split:])   # small slice, fine to copy
        combined_train.extend(mv[:split])          # fold in with no intermediate copy
        print(f"[mixture] stage='{stage}': {len(raw)/1e6:.2f}MB total, "
              f"{split/1e6:.2f}MB train, {len(raw)-split} bytes val")
        del raw, mv
    return combined_train, val_by_stage


def parse_stage_weights(stages_arg):
    """'tinystories:0.6,fineweb_edu:0.4' -> {'tinystories': 0.6, 'fineweb_edu': 0.4}
    normalized to sum to 1.0. A bare 'tinystories' (no weight) means weight 1.0."""
    result = {}
    for part in stages_arg.split(","):
        part = part.strip()
        if ":" in part:
            name, w = part.split(":")
            result[name.strip()] = float(w)
        else:
            result[part] = 1.0
    total = sum(result.values())
    return {k: v / total for k, v in result.items()}


class ByteDataset(Dataset):
    """Keeps the corpus as uint8 (1 byte/elem). Accepts bytes OR bytearray
    directly -- if it's already a bytearray (as load_multi_stage_bytes now
    returns), no extra copy is made; torch.frombuffer wraps it in place."""
    def __init__(self, data_bytes, block_size):
        if not isinstance(data_bytes, bytearray):
            data_bytes = bytearray(data_bytes)
        self.data = torch.frombuffer(data_bytes, dtype=torch.uint8)
        self._keep_alive = data_bytes  # frombuffer doesn't own the memory -- keep a ref
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size].long()
        y = self.data[idx + 1: idx + 1 + self.block_size].long()
        return x, y


# ---------------------------------------------------------------------------
# Training loop, T4-friendly: fp16 mixed precision, checkpoint to Drive
# ---------------------------------------------------------------------------

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    stage_weights = parse_stage_weights(args.stages)
    print(f"stage mixture: {stage_weights}")
    train_bytes, val_by_stage = load_multi_stage_bytes(
        stage_weights, args.max_docs, cache_dir=args.data_cache_dir)
    print(f"combined training bytes: {len(train_bytes)/1e6:.2f}MB across "
          f"{len(stage_weights)} stage(s)")

    train_ds = ByteDataset(train_bytes, args.block_size)
    del train_bytes  # ByteDataset now owns this buffer via self._keep_alive

    # No DataLoader for training: len(train_ds) can be hundreds of millions
    # of positions, and DataLoader(shuffle=True) would eagerly build a full
    # torch.randperm over that length (multiple GB) before the first batch.
    # Sampling random start offsets directly needs O(batch_size) memory and
    # gives the same effective "randomly windowed" training data.
    def get_batch():
        max_start = len(train_ds) - 1
        ix = torch.randint(0, max_start, (args.batch_size,))
        x = torch.stack([train_ds.data[i:i + args.block_size] for i in ix]).long()
        y = torch.stack([train_ds.data[i + 1:i + 1 + args.block_size] for i in ix]).long()
        return x, y

    # One SEPARATE validation loader per stage -- lets you see "did it get
    # worse on tinystories while learning fineweb_edu" instead of hiding it
    # in one blended number. shuffle=False -> SequentialSampler, cheap
    # regardless of corpus size (the randperm issue only bites shuffle=True).
    val_loaders = {}
    for stage, vbytes in val_by_stage.items():
        vds = ByteDataset(vbytes, args.block_size)
        val_loaders[stage] = DataLoader(vds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, drop_last=True)

    model = ByteBitGPT(
        d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
        block_size=args.block_size, mode=args.mode, n_experts=args.n_experts,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1,
                             betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    if len(stage_weights) == 1:
        # Single-stage run: keep the OLD naming convention, so --resume/
        # --init_from still find checkpoints from before multi-stage existed.
        stage_tag = next(iter(stage_weights))
    else:
        stage_tag = "+".join(f"{k}{int(v*100)}" for k, v in stage_weights.items())
    run_name = f"{args.mode}_L{args.n_layer}_D{args.d_model}_{stage_tag}"
    if args.n_experts > 1:
        run_name += f"_E{args.n_experts}"
    latest_path = os.path.join(ckpt_dir, f"{run_name}_latest.pt")
    best_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    log_path = os.path.join(ckpt_dir, f"{run_name}.log.csv")

    start_step = 0
    best_val = float("inf")
    log_header = ("step,train_loss," + ",".join(f"val_loss_{s}" for s in val_loaders)
                  + ",val_loss_avg,lr,elapsed_s\n")
    if os.path.exists(latest_path) and args.resume:
        print(f"resuming from {latest_path}")
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        best_val = state.get("best_val", float("inf"))
    elif args.init_from:
        print(f"initializing weights from {args.init_from} "
              f"(fresh optimizer, fresh step count -- curriculum continuation, "
              f"not a mid-run resume)")
        state = torch.load(args.init_from, map_location=device)
        model.load_state_dict(state["model"])
        with open(log_path, "w") as f:
            f.write(log_header)
    else:
        with open(log_path, "w") as f:
            f.write(log_header)

    t0 = time.time()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch()
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda', enabled=(device == "cuda"), dtype=torch.float16):
            _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.eval_every == 0 or step == args.steps - 1:
            per_stage_val = {s: evaluate(model, loader, device, n_batches=20)
                              for s, loader in val_loaders.items()}
            val_avg = sum(per_stage_val.values()) / len(per_stage_val)
            elapsed = time.time() - t0
            val_str = " ".join(f"val_{s} {v:.4f}" for s, v in per_stage_val.items())
            print(f"step {step:6d} | train_loss {loss.item():.4f} | "
                  f"{val_str} | val_avg {val_avg:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            with open(log_path, "a") as f:
                vals_csv = ",".join(f"{per_stage_val[s]:.4f}" for s in val_loaders)
                f.write(f"{step},{loss.item():.4f},{vals_csv},{val_avg:.4f},{lr:.6f},{elapsed:.1f}\n")

            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best_val": best_val,
                    "per_stage_val": per_stage_val, "args": vars(args)}
            torch.save(ckpt, latest_path)

            if val_avg < best_val:
                best_val = val_avg
                ckpt["best_val"] = best_val
                torch.save(ckpt, best_path)
                print(f"  -> new best val_avg {best_val:.4f}, saved to {best_path}")

            if step % args.snapshot_every == 0:
                snap_path = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
                torch.save(ckpt, snap_path)

        if step % args.sample_every == 0 and step > 0:
            sample_generation(model, device)

    print("done. latest checkpoint:", latest_path)
    print("best checkpoint:", best_path, f"(val_loss={best_val:.4f})")


@torch.no_grad()
def evaluate(model, loader, device, n_batches=20):
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', enabled=(device == "cuda"), dtype=torch.float16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def sample_generation(model, device, prompt="Once upon a time"):
    idx = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_bytes=150)
    print("--- sample ---")
    print(ByteBitGPT.decode(out[0]))
    print("--------------")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fp", "ternary", "binary"], default="ternary")
    p.add_argument("--stages", type=str, default="tinystories",
                    help="comma list of stage:weight, e.g. 'tinystories:0.7,fineweb_edu:0.3'. "
                         "A bare name means weight 1.0 -- old single-stage commands still work.")
    p.add_argument("--max_docs", type=int, default=200000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--snapshot_every", type=int, default=1000)
    p.add_argument("--ckpt_dir", type=str, default="/content/drive/MyDrive/bitbyte_lm_ckpts")
    p.add_argument("--data_cache_dir", type=str,
                    default="/content/drive/MyDrive/bitbyte_lm_ckpts/data_cache",
                    help="cache downloaded stage data here so repeated runs don't re-download. "
                         "Pass '' to disable.")
    p.add_argument("--n_experts", type=int, default=1,
                    help="1 = normal single MLP (default). >1 = MoE MLP (Branch-Train-MiX "
                         "'mix' step) -- use with --init_from a merge_experts.py output.")
    p.add_argument("--resume", action="store_true",
                    help="continue THIS exact run from its own checkpoint")
    p.add_argument("--init_from", type=str, default=None,
                    help="path to a checkpoint from a DIFFERENT run to initialize weights from "
                         "-- fresh optimizer/step count, for curriculum continuation or merges")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
