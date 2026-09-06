"""
bitbyte_lm/load_and_use.py

This is what "future work" actually looks like: a brand new Colab session,
weeks later, no training loop running -- just load a saved checkpoint and
either generate text with it or keep training it further.

Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    !python load_and_use.py --ckpt /content/drive/MyDrive/bitbyte_lm_ckpts/ternary_L12_D256_tinystories_best.pt --prompt "Once upon a time"
"""

import argparse
import torch
from model import ByteBitGPT


def load_model(ckpt_path, device="cpu"):
    state = torch.load(ckpt_path, map_location=device)
    train_args = state["args"]
    model = ByteBitGPT(
        d_model=train_args["d_model"],
        n_layer=train_args["n_layer"],
        n_head=train_args["n_head"],
        block_size=train_args["block_size"],
        mode=train_args["mode"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded checkpoint from step {state['step']}, "
          f"best_val={state.get('best_val', 'n/a')}, "
          f"trained with args={train_args}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to a .pt checkpoint on Drive")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_bytes", type=int, default=300)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    idx = torch.tensor([list(args.prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_bytes=args.max_new_bytes)
    print(ByteBitGPT.decode(out[0]))


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# To keep training a saved model further (not just generate from it):
#
#   python train.py --mode ternary --n_layer 12 --d_model 256 \
#       --stage fineweb_edu --resume \
#       --ckpt_dir /content/drive/MyDrive/bitbyte_lm_ckpts
#
# --resume finds the *_latest.pt file matching this exact mode/n_layer/
# d_model/stage combo and continues from its saved step, optimizer state,
# and weights -- this is literally how you'd move a model from the
# TinyStories stage to the FineWeb-Edu stage of the curriculum without
# starting over.
# ---------------------------------------------------------------------------
