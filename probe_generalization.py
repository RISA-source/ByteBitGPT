"""
bitbyte_lm/probe_generalization.py

Val_loss tells you how surprised the model is by held-out TEXT. It doesn't
directly tell you whether the model has learned the general PATTERN that
makes English words (spelling/orthography rules) versus just memorizing a
fixed set of frequent word-shapes. This script tests that more directly.

Method (a rough, automatable version of a "wug test"):
  1. Generate a large amount of text UNCONDITIONALLY (many independent
     samples, low-to-medium temperature so it's not just parroting the
     single most likely continuation).
  2. Split into words (split on anything that isn't a letter).
  3. For each generated word, check:
       - Is it a real, correctly-spelled English word? (dictionary lookup)
       - If real, how RARE is it in general English? (frequency lookup)
       - If not real, does it still look phonotactically English-like?
         (crude proxy: fraction of its letter-bigrams that are common
         English bigrams, vs random noise)
  4. Report:
       - % of generated words that are valid real words
       - % of unique words (vocabulary diversity -- pure memorization of a
         short list would show low uniqueness)
       - median/mean frequency-rank of the valid words produced (LOWER
         average frequency = model is reproducing rarer, less-memorized
         words correctly = stronger evidence of a learned general pattern
         rather than rote memorization of the handful of most common words)
       - % of invalid "words" that are still plausible English-shaped
         non-words (spelling pattern learned even where the specific word
         wasn't)

This is a heuristic, not a rigorous linguistic evaluation -- treat it as a
comparable SCORE across your fp/ternary/binary checkpoints and across
training durations, not an absolute truth about what the model "knows".
"""

import argparse
import re
import statistics
from collections import Counter

import torch
from spellchecker import SpellChecker

from model import ByteBitGPT


COMMON_ENGLISH_BIGRAMS = set([
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es",
    "or", "te", "of", "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng",
    "se", "ha", "as", "ou", "io", "le", "ve", "co", "me", "de", "hi", "ri",
    "ro", "ic", "ne", "ea", "ra", "ce", "li", "ch", "ll", "be", "ma", "si",
    "om", "ur",
])


def bigram_plausibility(word):
    word = word.lower()
    if len(word) < 2:
        return 1.0  # too short to judge, don't penalize
    bigrams = [word[i:i+2] for i in range(len(word) - 1)]
    hits = sum(1 for bg in bigrams if bg in COMMON_ENGLISH_BIGRAMS)
    return hits / len(bigrams)


@torch.no_grad()
def generate_corpus(model, device, n_samples=40, bytes_per_sample=300, temperature=0.9):
    texts = []
    for _ in range(n_samples):
        # Seed with a single space so generation isn't conditioned on any
        # specific prompt content -- as close to "unconditional" as this
        # architecture allows without a true BOS token.
        idx = torch.tensor([[ord(" ")]], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_bytes=bytes_per_sample, temperature=temperature, top_k=40)
        texts.append(ByteBitGPT.decode(out[0]))
    return texts


def run_probe(ckpt_path, n_samples=40, bytes_per_sample=300, temperature=0.9):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt_path, map_location=device)
    train_args = state["args"]
    model = ByteBitGPT(
        d_model=train_args["d_model"], n_layer=train_args["n_layer"],
        n_head=train_args["n_head"], block_size=train_args["block_size"],
        mode=train_args["mode"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    texts = generate_corpus(model, device, n_samples, bytes_per_sample, temperature)
    all_words = []
    for t in texts:
        all_words.extend(re.findall(r"[A-Za-z]+", t))

    sc = SpellChecker()
    valid_words = []
    invalid_words = []
    for w in all_words:
        wl = w.lower()
        if len(wl) < 2:
            continue
        if wl in sc:
            valid_words.append(wl)
        else:
            invalid_words.append(wl)

    total = len(valid_words) + len(invalid_words)
    unique_valid = set(valid_words)
    unique_all = set(valid_words + invalid_words)

    freqs = [sc.word_frequency[w] for w in valid_words if sc.word_frequency[w] > 0]
    plausibility_scores = [bigram_plausibility(w) for w in invalid_words]

    print(f"\n=== Generalization probe: {ckpt_path} ===")
    print(f"mode={train_args['mode']}  n_layer={train_args['n_layer']}  "
          f"d_model={train_args['d_model']}  trained_step={state['step']}  "
          f"best_val={state.get('best_val', 'n/a')}")
    print(f"total words generated: {total}")
    if total == 0:
        print("(no words extracted -- try more samples or check generation)")
        return

    print(f"valid real-word rate: {len(valid_words)/total:.1%}")
    print(f"vocabulary diversity (unique/total, ALL words): {len(unique_all)/total:.1%}")
    print(f"vocabulary diversity (unique/total, VALID words only): "
          f"{len(unique_valid)/max(1,len(valid_words)):.1%}")
    if freqs:
        print(f"median frequency of valid words produced: {statistics.median(freqs):,.0f}")
        print(f"  (lower = rarer words correctly spelled = stronger evidence "
              f"of learned pattern, not just memorized common words)")
    if plausibility_scores:
        print(f"mean bigram-plausibility of INVALID non-words: "
              f"{statistics.mean(plausibility_scores):.2f}  (1.0 = fully "
              f"English-shaped despite not being a real word)")

    print("\nsample generated words:")
    print("  valid  :", ", ".join(list(unique_valid)[:15]) if unique_valid else "(none)")
    print("  invalid:", ", ".join(list(set(invalid_words))[:15]) if invalid_words else "(none)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_samples", type=int, default=40)
    p.add_argument("--bytes_per_sample", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    args = p.parse_args()
    run_probe(args.ckpt, args.n_samples, args.bytes_per_sample, args.temperature)
