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
  4. Does curriculum (TinyStories -> more complex educational text) change
     these answers?

HOW TO USE IN COLAB:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload this whole bitbyte_lm/ folder (or clone if you push it to a repo)
  3. Mount Drive so checkpoints survive disconnects:
         from google.colab import drive
         drive.mount('/content/drive')
  4. pip install datasets torch --upgrade -q
  5. Run: python train.py --mode ternary --n_layer 12 --d_model 256 --steps 5000
  6. Repeat with --mode fp and --mode binary, and with different
     --n_layer/--d_model combos at MATCHED param count, to fill in the
     ablation table described at the bottom of this file.
"""

import argparse
import os
import time
import math
import torch
from torch.utils.data import Dataset, DataLoader

from model import ByteBitGPT


# ---------------------------------------------------------------------------
# Data: byte-level, no tokenizer. Curriculum = concatenate datasets in order.
# ---------------------------------------------------------------------------

def load_curriculum_bytes(stage="tinystories", max_docs=None):
    """
    stage:
      "tinystories" -- simple synthetic children's stories (start here)
      "fineweb_edu" -- classifier-filtered educational web text (stage 2)
      "cosmopedia"  -- synthetic textbooks, broader "science/math flavor"
                       written in simple prose (stage 3)
    Returns a single big bytes object.
    """
    from datasets import load_dataset

    name_map = {
        "tinystories": ("roneneldan/TinyStories", None, "text"),
        "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        "cosmopedia": ("HuggingFaceTB/cosmopedia", "stories", "text"),
    }
    hf_name, config, field = name_map[stage]
    ds = load_dataset(hf_name, config, split="train", streaming=True)

    chunks = []
    total_bytes = 0
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = row[field]
        chunks.append(text.encode("utf-8", errors="ignore"))
        total_bytes += len(chunks[-1])
        if i % 5000 == 0:
            print(f"  ...loaded {i} docs, {total_bytes/1e6:.1f}MB so far")
    return b"\n\n".join(chunks)


def load_curriculum_bytes_cached(stage, max_docs, cache_dir=None):
    """Same as load_curriculum_bytes, but checks a cache file first (keyed
    by stage + doc count) and writes one after downloading. This matters a
    lot in practice: without it, every --resume or retry re-streams the
    entire dataset from HuggingFace from scratch, which for large max_docs
    can be most of your session's wall-clock time before training even
    starts. Cache lives in cache_dir (point this at Drive so it survives
    disconnects too, not just re-runs within a session)."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{stage}_{max_docs}.bin")
        if os.path.exists(cache_path):
            print(f"[cache] found cached data for stage='{stage}' max_docs={max_docs}: "
                  f"{cache_path} -- skipping re-download")
            with open(cache_path, "rb") as f:
                return f.read()
    raw = load_curriculum_bytes(stage, max_docs=max_docs)
    if cache_dir:
        with open(cache_path, "wb") as f:
            f.write(raw)
        print(f"[cache] saved {len(raw)/1e6:.1f}MB to {cache_path} for future runs")
    return raw


def load_multi_stage_bytes(stage_weights, max_docs_total=200000, cache_dir=None):
    """
    stage_weights: dict like {"tinystories": 0.6, "fineweb_edu": 0.4}
    Approximates the mixture ratio via document COUNT per stage (weight *
    max_docs_total), not exact byte-matching -- simple, transparent, and
    good enough at this scale. Returns two dicts, each keyed by stage name:
      train_bytes_by_stage, val_bytes_by_stage
    Each stage keeps its OWN held-out validation split -- this is what lets
    you track per-domain loss separately (catch forgetting/interference
    between domains) instead of one blended number that hides it.
    """
    train_by_stage, val_by_stage = {}, {}
    for stage, weight in stage_weights.items():
        docs_for_stage = max(1, int(max_docs_total * weight))
        print(f"[mixture] loading stage='{stage}' weight={weight} -> {docs_for_stage} docs")
        raw = load_curriculum_bytes_cached(stage, docs_for_stage, cache_dir=cache_dir)
        split = int(0.99 * len(raw))
        train_by_stage[stage] = raw[:split]
        val_by_stage[stage] = raw[split:]
        print(f"[mixture] stage='{stage}': {len(raw)/1e6:.2f}MB total, "
              f"{len(train_by_stage[stage])/1e6:.2f}MB train / "
              f"{len(val_by_stage[stage])/1e6:.2f}MB val")
    return train_by_stage, val_by_stage


def parse_stage_weights(stages_arg):
    """'tinystories:0.6,fineweb_edu:0.4' -> {'tinystories': 0.6, 'fineweb_edu': 0.4}
    normalized to sum to 1.0. A bare 'tinystories' (no weight) is treated as
    weight 1.0 -- old single-stage runs still work with --stages tinystories."""
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
    def __init__(self, data_bytes, block_size):
        self.data = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).long()
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
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
    train_by_stage, val_by_stage = load_multi_stage_bytes(
        stage_weights, args.max_docs, cache_dir=args.data_cache_dir)

    # Combined training buffer: simple concatenation across stages. Random
    # windowing in ByteDataset means this behaves like a shuffled mixture
    # already -- no separate shuffle step needed.
    train_bytes = b"".join(train_by_stage.values())
    print(f"combined training bytes: {len(train_bytes)/1e6:.2f}MB across "
          f"{len(train_by_stage)} stage(s)")

    train_ds = ByteDataset(train_bytes, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)

    # One SEPARATE validation loader per stage -- this is what lets you see
    # "did it get worse on tinystories while learning fineweb_edu" instead
    # of a single blended number that would hide that.
    val_loaders = {}
    for stage, vbytes in val_by_stage.items():
        vds = ByteDataset(vbytes, args.block_size)
        val_loaders[stage] = DataLoader(vds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=2, drop_last=True)

    model = ByteBitGPT(
        d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
        block_size=args.block_size, mode=args.mode,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1,
                             betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    if len(stage_weights) == 1:
        # Single-stage run: keep the OLD naming convention unchanged, so
        # --resume/--init_from still find checkpoints from before this
        # multi-stage feature existed (e.g. your existing tinystories runs).
        stage_tag = next(iter(stage_weights))
    else:
        stage_tag = "+".join(f"{k}{int(v*100)}" for k, v in stage_weights.items())
    run_name = f"{args.mode}_L{args.n_layer}_D{args.d_model}_{stage_tag}"
    latest_path = os.path.join(ckpt_dir, f"{run_name}_latest.pt")
    best_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    log_path = os.path.join(ckpt_dir, f"{run_name}.log.csv")

    start_step = 0
    best_val = float("inf")
    log_header = "step,train_loss," + ",".join(f"val_loss_{s}" for s in val_loaders) + ",val_loss_avg,lr,elapsed_s\n"
    if os.path.exists(latest_path) and args.resume:
        print(f"resuming from {latest_path}")
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        best_val = state.get("best_val", float("inf"))
    elif args.init_from:
        print(f"initializing weights from {args.init_from} "
              f"(fresh optimizer, fresh step count -- this is curriculum "
              f"continuation, not a mid-run resume)")
        state = torch.load(args.init_from, map_location=device)
        model.load_state_dict(state["model"])
        with open(log_path, "w") as f:
            f.write(log_header)
    else:
        with open(log_path, "w") as f:
            f.write(log_header)

    data_iter = iter(train_loader)
    t0 = time.time()

    for step in range(start_step, args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
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

            # Always overwrite "latest" -- this is what --resume loads.
            torch.save(ckpt, latest_path)

            # "best" is judged on the AVERAGE across stages (equally
            # weighted regardless of mixture ratio, so a stage with a
            # small weight can't be silently ignored when picking the
            # best checkpoint). Per-stage numbers are always printed and
            # saved alongside it, so nothing is hidden.
            if val_avg < best_val:
                best_val = val_avg
                ckpt["best_val"] = best_val
                torch.save(ckpt, best_path)
                print(f"  -> new best val_avg {best_val:.4f}, saved to {best_path}")

            # Periodic dated snapshot so you can later compare the model's
            # behavior at different points in training, not just the end.
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
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
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
                         "A bare name (e.g. 'tinystories') means weight 1.0 -- old single-stage "
                         "commands still work unchanged.")
    p.add_argument("--max_docs", type=int, default=200000,
                    help="cap docs loaded so a single Colab session finishes; raise later")
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
    p.add_argument("--snapshot_every", type=int, default=1000,
                    help="save a dated, non-overwritten checkpoint every N steps")
    p.add_argument("--ckpt_dir", type=str, default="/content/drive/MyDrive/bitbyte_lm_ckpts")
    p.add_argument("--data_cache_dir", type=str,
                    default="/content/drive/MyDrive/bitbyte_lm_ckpts/data_cache",
                    help="cache downloaded stage data here (on Drive, survives sessions) "
                         "so --resume and repeated runs don't re-download from HuggingFace "
                         "every time. Pass '' to disable caching.")
    p.add_argument("--resume", action="store_true",
                    help="continue THIS exact run (same mode/shape/stage) from its own checkpoint")
    p.add_argument("--init_from", type=str, default=None,
                    help="path to a checkpoint from a DIFFERENT stage/run to initialize weights "
                         "from, keeping the model but starting a fresh optimizer and step count "
                         "-- this is how you move a model from tinystories into fineweb_edu etc.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)


# ---------------------------------------------------------------------------
# ABLATION TABLE TO FILL IN (this is the actual experiment / research output)
# ---------------------------------------------------------------------------
#
# Run each combo for the same number of steps and same data stage, record
# val_loss (lower = better) from the .log.csv files. Keep total param count
# roughly matched within each row by trading n_layer against d_model.
#
#   mode     | n_layer | d_model | params | val_loss @ step 5000
#   ---------|---------|---------|--------|----------------------
#   fp       |   6     |   384   |  ~11M  |
#   fp       |  12     |   256   |  ~10M  |
#   fp       |  24     |   176   |  ~10M  |
#   ternary  |   6     |   384   |  ~11M  |
#   ternary  |  12     |   256   |  ~10M  |
#   ternary  |  24     |   176   |  ~10M  |
#   binary   |   6     |   384   |  ~11M  |
#   binary   |  12     |   256   |  ~10M  |
#   binary   |  24     |   176   |  ~10M  |
#
# Questions this table answers directly:
#   - Row-wise (fp vs ternary vs binary at same shape): how big is the
#     precision penalty at THIS tiny scale? (literature says the gap should
#     be large at small scale and shrink with scale -- do you see the same
#     pattern going from 10M to whatever bigger size you can afford?)
#   - Column-wise within ternary (6x384 vs 12x256 vs 24x176): does depth
#     buy back quality lost to ternary weights, more than width does?
#     This is your original hypothesis, tested directly.
#   - Ternary vs binary gap: is it as large, at tiny scale, as BitNet's
#     paper found at billion-parameter scale? Or does it behave differently
#     down here? Nobody has published this specific comparison at 10M scale
#     with byte-level input as far as available literature shows.
#
# Then repeat the best row's config across stage="tinystories" ->
# "fineweb_edu" -> "cosmopedia" to see how curriculum interacts with
# bit-width -- e.g. does ternary need *more* curriculum help than fp to
# reach the same loss, because it has less raw capacity per weight?
