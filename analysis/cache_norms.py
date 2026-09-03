#!/usr/bin/env python3
"""Cache per-input-channel activation norms ||X^f||, ||X^r|| for the TOFU models.

FRAG needs one forward pass over forget and retain calibration data per (model, split).
Caching those norms once lets analysis/compute_predictors.py score any number of
checkpoints afterwards on CPU, with no further forward passes.

Writes {FRAG_NORMS}/{scale}_{split}_{fn,rn}.pt. Needs a GPU; roughly 10 minutes for
1B and 3B across the three splits.

Environment:
    FRAG_NORMS    output directory                              (default ./norms_cache)
    HF_HUB_CACHE  HuggingFace hub cache with the base models    (default ~/.cache/huggingface/hub)

Usage:
    python analysis/cache_norms.py                 # 1B and 3B, all three splits
    python analysis/cache_norms.py --scales 1B
"""
import argparse
import glob
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "frp"))
sys.path.insert(0, os.path.join(ROOT, "frag"))
from data import load_tofu  # noqa: E402
from predictor import collect_input_norms, get_target_modules  # noqa: E402

OUT = os.environ.get("FRAG_NORMS", "norms_cache")
HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

# paper defaults (Appendix A.1): 128 calibration sequences of 256 tokens
NUM_SAMPLES, MAX_LENGTH = 128, 256
SPLITS = [("forget01", "retain99"), ("forget05", "retain95"), ("forget10", "retain90")]


def hub_snapshot(repo_dirname):
    matches = glob.glob(os.path.join(HUB, repo_dirname, "snapshots", "*", ""))
    if not matches:
        raise FileNotFoundError(f"no snapshot for {repo_dirname} under {HUB}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", nargs="+", default=["1B", "3B"])
    ap.add_argument("--target", default="all", choices=["all", "mlp", "attention", "linear"])
    ap.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for scale in args.scales:
        if all(os.path.exists(f"{OUT}/{scale}_{fs}_{k}.pt")
               for fs, _ in SPLITS for k in ("fn", "rn")):
            print(f"[{scale}] already cached, skipping", flush=True)
            continue

        ref = hub_snapshot(f"models--open-unlearning--tofu_Llama-3.2-{scale}-Instruct_full")
        print(f"[{scale}] loading {ref}", flush=True)
        tok = AutoTokenizer.from_pretrained(ref)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ref, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device).eval()
        targets = get_target_modules(model, args.target)

        for forget_split, retain_split in SPLITS:
            if os.path.exists(f"{OUT}/{scale}_{forget_split}_fn.pt"):
                continue
            fn = collect_input_norms(
                model, load_tofu(forget_split, tok, args.max_length, args.num_samples),
                args.device, targets)
            rn = collect_input_norms(
                model, load_tofu(retain_split, tok, args.max_length, args.num_samples),
                args.device, targets)
            torch.save({k: v.cpu() for k, v in fn.items()}, f"{OUT}/{scale}_{forget_split}_fn.pt")
            torch.save({k: v.cpu() for k, v in rn.items()}, f"{OUT}/{scale}_{forget_split}_rn.pt")
            print(f"  [{scale}] {forget_split} cached", flush=True)

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
