#!/usr/bin/env python3
"""Table 1 predictor columns: global L2 and FRAG (%) for every TOFU checkpoint.

For each (method, forget split) this loads the unlearned checkpoint, compares it with the
original model, and reports

    L2   = ||W_u - W_0||_2          over all shared weights (global displacement)
    FRAG = cos(F, dW^2) - cos(R, dW^2)   over the scored projections, x100 for readability

using cached per-input-channel activation norms (see analysis/cache_norms.py). No forward
pass is needed here, so this runs on CPU; the paper's Table 1 columns are the mean over the
three forget splits.

Paths come from the environment so the script is portable:
    OPEN_UNLEARNING  root of the OpenUnlearning checkout holding saves/     (default ~/open-unlearning)
    FRAG_NORMS       directory of cached norms from cache_norms.py          (default ./norms_cache)
    HF_HUB_CACHE     HuggingFace hub cache holding the original models      (default ~/.cache/huggingface/hub)

Usage:
    python analysis/compute_predictors.py --scale 1B
    python analysis/compute_predictors.py --scale 3B --csv results/predictors_3b.csv
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frag"))
from predictor import get_target_modules, StreamReader  # noqa: E402

OU = os.environ.get("OPEN_UNLEARNING", os.path.expanduser("~/open-unlearning"))
NORMS = os.environ.get("FRAG_NORMS", "norms_cache")
HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

SPLITS = ["forget01", "forget05", "forget10"]
SUFFIX = {"forget01": "_forget01", "forget05": "_forget05", "forget10": ""}

# (display name, checkpoint tag). FRP is stored under the original "RC" (rank-combo) tag.
METHODS = [("GA", "GradAscent"), ("GradDiff", "GradDiff"), ("NPO", "NPO"),
           ("RMU", "RMU"), ("SP", "SP"), ("FRP", "RC")]
# retain-veto fraction p reported in Table 1 for each scale
RV_P = {"1B": "p015", "3B": "p005"}


def hub_snapshot(repo_dirname: str) -> str:
    matches = glob.glob(os.path.join(HUB, repo_dirname, "snapshots", "*", ""))
    if not matches:
        raise FileNotFoundError(f"no snapshot for {repo_dirname} under {HUB}")
    return matches[0]


def predictors(state, scored_keys, ckpt, forget_norms, retain_norms, gamma=1.0, eps=1e-6):
    """Global L2 over all shared weights; FRAG over the scored projections."""
    un = StreamReader(ckpt)
    sq = dot_f = dot_r = nrm_f = nrm_r = nrm_d = 0.0
    for key in set(un.m) & set(state):
        W0 = state[key].float()
        d2 = (un.get(key).float() - W0) ** 2
        sq += float(d2.sum())
        if key not in scored_keys:
            continue
        module = key[:-len(".weight")]
        if module not in forget_norms:
            continue
        absW = W0.abs()
        xf = forget_norms[module].float().unsqueeze(0)
        xr = retain_norms[module].float().unsqueeze(0)
        F = absW * (xf / (xr + eps))
        R = absW * (xr / (xf + eps))
        nrm_d += float((d2 * d2).sum())
        dot_f += float((F * d2).sum()); nrm_f += float((F * F).sum())
        dot_r += float((R * d2).sum()); nrm_r += float((R * R).sum())
    dn = math.sqrt(nrm_d)
    cos_f = dot_f / (math.sqrt(nrm_f) * dn)   # A_f in Eq. (5)
    cos_r = dot_r / (math.sqrt(nrm_r) * dn)   # A_r in Eq. (5)
    return math.sqrt(sq), cos_f - gamma * cos_r, cos_f, cos_r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["1B", "3B"], default="1B")
    ap.add_argument("--gamma", type=float, default=1.0, help="retain-penalty weight (paper: 1)")
    ap.add_argument("--target", default="all", choices=["all", "mlp", "attention", "linear"])
    ap.add_argument("--csv", default=None, help="also write per-split rows to this CSV")
    args = ap.parse_args()

    ref = hub_snapshot(f"models--open-unlearning--tofu_Llama-3.2-{args.scale}-Instruct_full")
    AutoTokenizer.from_pretrained(ref)
    model = AutoModelForCausalLM.from_pretrained(
        ref, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    scored = {n + ".weight" for n in get_target_modules(model, args.target)}
    state = dict(model.state_dict())

    canonical = f"{OU}/saves/table1_canonical/saves/unlearn"
    unlearn = f"{OU}/saves/unlearn"
    rv = RV_P[args.scale]
    runs = [(d, lambda s, t=t: f"{canonical}/canonical_{args.scale}_{t}{SUFFIX[s]}")
            for d, t in METHODS]
    def rv_path(split):
        # scripts/frp_tofu.sh writes frp_<scale>_<split>_p###; frprv_ is the original tag.
        for prefix in ("frp", "frprv"):
            p = f"{unlearn}/{prefix}_{args.scale}_{split}_{rv}"
            if os.path.isdir(p):
                return p
        return f"{unlearn}/frp_{args.scale}_{split}_{rv}"

    runs.append((f"FRP-RV(p={int(rv[1:]) / 100:.2f})", rv_path))

    rows = []
    print(f"{'method':16s} {'L2':>10s} {'FRAG%':>10s}    per-split L2 / FRAG%", flush=True)
    for name, path_of in runs:
        l2s, frags = [], []
        for split in SPLITS:
            ckpt = path_of(split)
            if not os.path.isdir(ckpt):
                print(f"{name:16s} MISSING {ckpt}", flush=True)
                l2s = None
                break
            fn = torch.load(f"{NORMS}/{args.scale}_{split}_fn.pt", weights_only=True)
            rn = torch.load(f"{NORMS}/{args.scale}_{split}_rn.pt", weights_only=True)
            l2, frag, cos_f, cos_r = predictors(state, scored, ckpt, fn, rn, gamma=args.gamma)
            l2s.append(l2); frags.append(frag)
            rows.append((args.scale, name, split, l2, frag * 100, cos_f * 100, cos_r * 100))
        if l2s:
            detail = "  ".join(f"{a:.3f}/{b * 100:.4f}" for a, b in zip(l2s, frags))
            print(f"{name:16s} {sum(l2s) / 3:10.3f} {sum(frags) / 3 * 100:10.3f}    {detail}",
                  flush=True)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("model,method,forget_split,L2,FRAG_pct,cosF_pct,cosR_pct\n")
            for scale, name, split, l2, frag, cf, cr in rows:
                f.write(f"LLaMA-3.2-{scale},{name},{split},"
                        f"{l2:.6f},{frag:.6f},{cf:.6f},{cr:.6f}\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
