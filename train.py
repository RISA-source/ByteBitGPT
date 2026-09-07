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

    print(f"loading stage: {args.stage}")
    raw = load_curriculum_bytes(args.stage, max_docs=args.max_docs)
    print(f"total training bytes: {len(raw)/1e6:.2f}MB")

    split = int(0.99 * len(raw))
    train_bytes, val_bytes = raw[:split], raw[split:]
    train_ds = ByteDataset(train_bytes, args.block_size)
    val_ds = ByteDataset(val_bytes, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
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
    run_name = f"{args.mode}_L{args.n_layer}_D{args.d_model}_{args.stage}"
    latest_path = os.path.join(ckpt_dir, f"{run_name}_latest.pt")
    best_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    log_path = os.path.join(ckpt_dir, f"{run_name}.log.csv")

    start_step = 0
    best_val = float("inf")
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
            f.write("step,train_loss,val_loss,lr,elapsed_s\n")
    else:
        with open(log_path, "w") as f:
            f.write("step,train_loss,val_loss,lr,elapsed_s\n")

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
            val_loss = evaluate(model, val_loader, device, n_batches=20)
            elapsed = time.time() - t0
            print(f"step {step:6d} | train_loss {loss.item():.4f} | "
                  f"val_loss {val_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            with open(log_path, "a") as f:
                f.write(f"{step},{loss.item():.4f},{val_loss:.4f},{lr:.6f},{elapsed:.1f}\n")

            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best_val": best_val, "args": vars(args)}

            # Always overwrite "latest" -- this is what --resume loads.
            torch.save(ckpt, latest_path)

            # Keep a separate "best" checkpoint so a late-training overfit
            # or a bad run doesn't cost you your best result.
            if val_loss < best_val:
                best_val = val_loss
                ckpt["best_val"] = best_val
                torch.save(ckpt, best_path)
                print(f"  -> new best val_loss {best_val:.4f}, saved to {best_path}")

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
    p.add_argument("--stage", choices=["tinystories", "fineweb_edu", "cosmopedia"],
                    default="tinystories")
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
