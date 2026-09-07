"""
model_soup.py -- the OTHER answer to "combine multiple trained weight sets",
much simpler than merge_experts.py's MoE route.

Instead of keeping N experts and routing between them (more params, more
compute, a gate to train), this just averages the weights of N checkpoints
into ONE ordinary model -- same size, same inference cost as any single
input model, no gate, nothing to route.

This ONLY makes sense if all N checkpoints share a common ancestor (were
--init_from'd from the same seed checkpoint, like the BTX branches in
merge_experts.py's recipe). If they were trained from independent random
initializations, naive averaging usually produces a broken model -- the
internal neurons aren't in matching order across the two models, so
averaging position-by-position mixes together neurons that learned
unrelated things. (There's a fix for that case too -- permutation
alignment, aka "Git Re-Basin" -- but it's a separate, more involved
technique, not implemented here. Not needed for shared-seed branches.)

Usage:
    python model_soup.py \
        --checkpoints ternary_L12_D256_tinystories_best.pt \
                      ternary_L12_D256_fineweb_edu_best.pt \
                      ternary_L12_D256_cosmopedia_best.pt \
        --ckpt_dir /content/drive/MyDrive/bitbyte_lm_ckpts \
        --out ternary_L12_D256_soup_merged.pt

Then use the output exactly like any normal single-expert checkpoint:
    python train.py --mode ternary --n_layer 12 --d_model 256 \
        --init_from ternary_L12_D256_soup_merged.pt \
        --stages "tinystories:0.33,fineweb_edu:0.33,cosmopedia:0.34" \
        --steps 5000

Compare THIS run's per-stage val losses against the MoE (merge_experts.py)
run's, at the same total step count -- that's the real answer to "does
converging to one shared weight work as well as keeping several and
routing between them," with your own numbers instead of a guess.
"""
import argparse
import os
import torch


def soup(checkpoint_paths):
    states = [torch.load(p, map_location="cpu") for p in checkpoint_paths]
    sds = [s["model"] for s in states]
    keys = sds[0].keys()
    for sd in sds[1:]:
        assert sd.keys() == keys, "checkpoints must all be the same architecture/shape"

    merged_sd = {}
    for key in keys:
        stacked = torch.stack([sd[key].float() for sd in sds], dim=0)
        merged_sd[key] = stacked.mean(dim=0).to(sds[0][key].dtype)
    return merged_sd


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                    help="filenames (relative to --ckpt_dir) of the N checkpoints to average -- "
                         "must share a common ancestor checkpoint (same --init_from seed)")
    p.add_argument("--ckpt_dir", type=str, default="/content/drive/MyDrive/bitbyte_lm_ckpts")
    p.add_argument("--out", type=str, required=True)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    paths = [os.path.join(args.ckpt_dir, f) for f in args.checkpoints]
    print(f"averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p}")

    merged_sd = soup(paths)
    out_path = os.path.join(args.ckpt_dir, args.out)
    torch.save({"model": merged_sd}, out_path)
    print(f"saved averaged (souped) model to {out_path}")
    print(f"this is an ORDINARY single model -- same size/cost as any one input "
          f"checkpoint. Use it exactly like any --init_from checkpoint, no --n_experts needed.")
