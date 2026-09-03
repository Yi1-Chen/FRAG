#!/usr/bin/env python3
"""Inject isotropic Gaussian noise into model weights.

Counterfactual for the Siddiqui-style L2-displacement hypothesis: produces
checkpoints with matched L2 to RC pruning but random (forget-blind) direction.

Usage:
  python inject_noise.py \
    --input_dir <BASE_CKPT_DIR> \
    --output_dir <OUT_DIR> \
    --sigma 0.004 --scope mlp --seed 42 --dtype bfloat16

Scope:
  mlp  -- only {gate,up,down}_proj weights (matches RC's edit scope)
  all  -- all parameter tensors except layernorm/norm

Saves a noise_injection.json summary alongside the HF checkpoint with the
realized total/per-bucket L2 displacement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MLP_TOKENS = ("gate_proj", "up_proj", "down_proj")
ATTN_TOKENS = ("q_proj", "k_proj", "v_proj", "o_proj")


def classify(name: str) -> str:
    n = name.lower()
    if any(t in n for t in MLP_TOKENS):
        return "mlp"
    if any(t in n for t in ATTN_TOKENS):
        return "attention"
    if "embed" in n or "lm_head" in n:
        return "embedding"
    if "norm" in n:
        return "layernorm"
    return "other"


def should_perturb(name: str, scope: str) -> bool:
    bucket = classify(name)
    if scope == "mlp":
        return bucket == "mlp"
    if scope == "attn":
        return bucket == "attention"
    if scope == "embed":
        return bucket == "embedding"
    if scope == "all":
        # Skip layernorm even under "all" — perturbing scale params usually breaks the model
        # and Siddiqui's Fig.5 also excludes batch-norm statistics.
        return bucket != "layernorm"
    raise ValueError(f"unknown scope: {scope}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="HF checkpoint dir to load")
    ap.add_argument("--output_dir", required=True, help="HF checkpoint dir to write")
    ap.add_argument("--mode", default="gaussian", choices=("gaussian", "const"),
                    help="gaussian: add N(0, sigma) noise; const: add a fixed scalar to every weight")
    ap.add_argument("--sigma", type=float, default=None, help="Gaussian std (required when mode=gaussian)")
    ap.add_argument("--const_value", type=float, default=1.0, help="Additive scalar when mode=const")
    ap.add_argument("--scope", default="mlp", choices=("mlp", "attn", "embed", "all"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    args = ap.parse_args()
    if args.mode == "gaussian" and args.sigma is None:
        ap.error("--sigma is required when --mode=gaussian")

    torch.manual_seed(args.seed)
    torch_dtype = getattr(torch, args.dtype)

    print(f"[inject_noise] loading model from {args.input_dir} dtype={args.dtype}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(args.input_dir, torch_dtype=torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.input_dir)
    model.eval()

    per_bucket_sq: Dict[str, float] = {"mlp": 0.0, "attention": 0.0, "embedding": 0.0, "layernorm": 0.0, "other": 0.0}
    total_sq = 0.0
    n_modified = 0
    n_total = 0

    with torch.no_grad():
        for name, p in model.named_parameters():
            n_total += 1
            if not should_perturb(name, args.scope):
                continue
            # Generate perturbation in float32 for numerical accuracy of the L2 measurement,
            # then cast to the parameter's dtype before adding in-place.
            if args.mode == "gaussian":
                pert = torch.randn(p.shape, dtype=torch.float32, device="cpu") * args.sigma
            else:  # const
                pert = torch.full(p.shape, float(args.const_value), dtype=torch.float32, device="cpu")
            sq = float((pert * pert).sum().item())
            bucket = classify(name)
            per_bucket_sq[bucket] += sq
            total_sq += sq
            p.data.add_(pert.to(p.dtype).to(p.device))
            n_modified += 1

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[inject_noise] saving to {args.output_dir}", file=sys.stderr)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "scope": args.scope,
        "mode": args.mode,
        "sigma": args.sigma,
        "const_value": args.const_value if args.mode == "const" else None,
        "seed": args.seed,
        "dtype": args.dtype,
        "n_modified_tensors": n_modified,
        "n_total_tensors": n_total,
        "total_l2": math.sqrt(total_sq),
        "mlp_l2": math.sqrt(per_bucket_sq["mlp"]),
        "attn_l2": math.sqrt(per_bucket_sq["attention"]),
        "embed_l2": math.sqrt(per_bucket_sq["embedding"]),
        "layernorm_l2": math.sqrt(per_bucket_sq["layernorm"]),
        "other_l2": math.sqrt(per_bucket_sq["other"]),
    }
    with open(os.path.join(args.output_dir, "noise_injection.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
