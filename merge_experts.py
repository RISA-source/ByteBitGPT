"""
merge_experts.py -- the "Mix" step of Branch-Train-MiX (BTX).

Usage (after you've run the "Branch" + "Train" steps -- see bottom of this
file for the full 3-phase recipe):

    python merge_experts.py \
        --checkpoints ternary_L12_D256_tinystories_best.pt \
                      ternary_L12_D256_fineweb_edu_best.pt \
                      ternary_L12_D256_cosmopedia_best.pt \
        --ckpt_dir /content/drive/MyDrive/bitbyte_lm_ckpts \
        --out ternary_L12_D256_moe_E3_merged.pt

What it does:
  - Loads N branch checkpoints (must be same mode/n_layer/d_model/n_head/
    block_size, each trained with --n_experts 1 on its OWN single stage,
    all branched from the SAME shared starting checkpoint via --init_from
    so they're already in the same "basin" -- this is why plain averaging
    of the non-expert parameters is safe here without permutation
    alignment, unlike merging independently-initialized models).
  - Builds a new ByteBitGPT with n_experts=N.
  - Each branch's MLP becomes ONE dedicated expert in every block (branch i
    -> expert i), copied over exactly, not averaged -- this is what
    preserves each branch's learned specialization.
  - Everything else (embeddings, attention, layer norms, output head) is
    averaged across the N branches.
  - The gate/router is left at its fresh random init -- it has not seen
    any data yet. It gets trained during the next phase (MoE-finetuning:
    run train.py again with --n_experts N --init_from <this file's output>
    on the COMBINED stage mixture).
"""
import argparse
import os
import torch

from model import ByteBitGPT, MLP


def merge(checkpoint_paths, mode, n_layer, d_model, n_head, block_size):
    n_experts = len(checkpoint_paths)
    states = [torch.load(p, map_location="cpu") for p in checkpoint_paths]
    sds = [s["model"] for s in states]

    merged = ByteBitGPT(d_model=d_model, n_layer=n_layer, n_head=n_head,
                         block_size=block_size, mode=mode, n_experts=n_experts)
    merged_sd = merged.state_dict()

    for key in merged_sd.keys():
        if ".mlp.experts." in key:
            # e.g. "blocks.3.mlp.experts.1.fc1.weight" -> branch 1's
            # "blocks.3.mlp.fc1.weight", copied exactly (not averaged).
            prefix, rest = key.split(".mlp.experts.")
            expert_idx_str, param_path = rest.split(".", 1)
            expert_idx = int(expert_idx_str)
            branch_key = f"{prefix}.mlp.{param_path}"
            merged_sd[key] = sds[expert_idx][branch_key].clone()
        elif key.startswith("blocks.") and ".mlp.gate." in key:
            # fresh router -- leave as randomly initialized, it hasn't
            # learned anything yet, that happens in the finetuning phase.
            continue
        else:
            # everything else (embeddings, attention, layer norms, head):
            # average across all N branches. Safe here specifically because
            # all branches started from the SAME shared checkpoint (see
            # module docstring) -- they're already in the same basin, no
            # permutation alignment needed, unlike merging independently
            # initialized models (which needs alignment -- see the
            # aligned_merge experiment this recipe was validated against).
            stacked = torch.stack([sd[key].float() for sd in sds], dim=0)
            merged_sd[key] = stacked.mean(dim=0).to(merged_sd[key].dtype)

    merged.load_state_dict(merged_sd)
    return merged


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                    help="filenames (relative to --ckpt_dir) of the N branch "
                         "checkpoints to merge, one per domain")
    p.add_argument("--ckpt_dir", type=str, default="/content/drive/MyDrive/bitbyte_lm_ckpts")
    p.add_argument("--out", type=str, required=True,
                    help="output filename (saved into --ckpt_dir), pass this to "
                         "train.py's --init_from for the MoE-finetuning phase")
    p.add_argument("--mode", choices=["fp", "ternary", "binary"], required=True)
    p.add_argument("--n_layer", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--n_head", type=int, required=True)
    p.add_argument("--block_size", type=int, required=True)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    paths = [os.path.join(args.ckpt_dir, f) for f in args.checkpoints]
    print(f"merging {len(paths)} branch checkpoints:")
    for p in paths:
        print(f"  - {p}")

    merged = merge(paths, args.mode, args.n_layer, args.d_model, args.n_head, args.block_size)

    out_path = os.path.join(args.ckpt_dir, args.out)
    torch.save({"model": merged.state_dict()}, out_path)
    print(f"saved merged {len(paths)}-expert model to {out_path}")
    print(f"next: python train.py --mode {args.mode} --n_layer {args.n_layer} "
          f"--d_model {args.d_model} --n_experts {len(paths)} --init_from {out_path} "
          f"--stages 'tinystories:0.33,fineweb_edu:0.33,cosmopedia:0.34' --steps <more steps>")


# ---------------------------------------------------------------------------
# FULL 3-PHASE RECIPE (Branch-Train-MiX, Sukhbaatar et al. 2024, adapted to
# ByteBitGPT). Run these in order:
#
# PHASE 0 -- shared seed (branch FROM the same starting point, not from
# independent random inits -- this is what makes plain averaging safe later):
#   python train.py --mode ternary --n_layer 12 --d_model 256 \
#       --stages tinystories --max_docs 5000 --steps 200
#   (a short, cheap run just to get one shared starting checkpoint;
#    ternary_L12_D256_tinystories_latest.pt is your seed)
#
# PHASE 1 -- BRANCH + TRAIN (embarrassingly parallel -- run these separately,
# each starting from the SAME seed checkpoint via --init_from):
#   python train.py --mode ternary --n_layer 12 --d_model 256 \
#       --stages tinystories --max_docs 500000 --steps 10000 \
#       --init_from <seed>_latest.pt
#   python train.py --mode ternary --n_layer 12 --d_model 256 \
#       --stages fineweb_edu --max_docs 500000 --steps 10000 \
#       --init_from <seed>_latest.pt
#   python train.py --mode ternary --n_layer 12 --d_model 256 \
#       --stages cosmopedia --max_docs 500000 --steps 10000 \
#       --init_from <seed>_latest.pt
#
# PHASE 2 -- MIX (this script):
#   python merge_experts.py \
#       --checkpoints ternary_L12_D256_tinystories_best.pt \
#                     ternary_L12_D256_fineweb_edu_best.pt \
#                     ternary_L12_D256_cosmopedia_best.pt \
#       --mode ternary --n_layer 12 --d_model 256 --n_head 8 --block_size 512 \
#       --out ternary_L12_D256_moe_E3_merged.pt
#
# PHASE 3 -- MOE-FINETUNE (continue training the merged model jointly on
# the COMBINED mixture -- this is what lets the router learn, and what
# recovers cross-domain sharpness that plain merging alone leaves behind):
#   python train.py --mode ternary --n_layer 12 --d_model 256 --n_experts 3 \
#       --init_from ternary_L12_D256_moe_E3_merged.pt \
#       --stages "tinystories:0.33,fineweb_edu:0.33,cosmopedia:0.34" \
#       --max_docs 500000 --steps 5000
#
# Compare this run's final per-stage val losses against your EXISTING
# single-shared-model baseline (n_experts=1, same combined --stages mixture,
# same total steps) -- that's the actual ablation result.
# ---------------------------------------------------------------------------
