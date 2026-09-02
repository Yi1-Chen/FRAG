#!/usr/bin/env python3
"""Compute ||W_unlearned - W_original||_2 with per-module breakdown.

CPU-only. Lazy shard-by-shard loading via safetensors so 8B+ does not OOM.

Two modes:
  1) Single pair:
       python measure_l2.py --unlearned <DIR> --original <DIR> --label LABEL [--out FILE]
  2) Manifest:
       python measure_l2.py --manifest manifest.json --out results.json

Manifest schema (JSON list of objects):
  {
    "method": "RC",
    "scale": "1B",
    "config": "3pct shrink=0.05",
    "label": "1B_RC_3pct",
    "unlearned": "/abs/path/to/unlearned_dir",
    "original": "/abs/path/to/tofu_*_full",
    "pre_eval": "/abs/path/to/eval/pre_dir",    # optional, for ES extraction
    "post_eval": "/abs/path/to/eval/post_dir",  # optional
    "family": "pruning|gradient|random|reference",
    "notes": "..."                              # optional
  }

Output JSON columns per row:
  total_l2, mlp_l2, attn_l2, embed_l2, other_l2, mismatch_keys, n_keys_compared
  pre_ES, pre_util, post_ES   (if eval dirs present)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import torch
except ImportError:
    print("ERROR: torch required", file=sys.stderr)
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors required (pip install safetensors)", file=sys.stderr)
    sys.exit(1)


def _classify_key(key: str) -> str:
    """Bucket a state-dict key into mlp / attention / embedding / layernorm / other."""
    k = key.lower()
    if "embed" in k:
        return "embedding"
    if "mlp" in k or "feed_forward" in k or any(t in k for t in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if "attn" in k or "attention" in k or any(t in k for t in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if "norm" in k or "layernorm" in k or "rmsnorm" in k:
        return "layernorm"
    return "other"


def _list_shards(model_dir: str) -> Tuple[Dict[str, str], List[str]]:
    """Return (key_to_shardfile, [shard_filenames]).

    Handles both single-file and sharded safetensors layouts.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        idx = json.load(open(index_path))
        weight_map = idx.get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        return weight_map, shards
    # Single file
    single = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(single):
        # Try any single .safetensors
        cands = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        if len(cands) != 1:
            raise FileNotFoundError(
                f"{model_dir}: no model.safetensors.index.json and {len(cands)} .safetensors files"
            )
        single = os.path.join(model_dir, cands[0])
    fname = os.path.basename(single)
    weight_map: Dict[str, str] = {}
    with safe_open(single, framework="pt") as sf:
        for k in sf.keys():
            weight_map[k] = fname
    return weight_map, [fname]


def measure_pair(unlearned_dir: str, original_dir: str, verbose: bool = True) -> Dict[str, float]:
    """Compute total + per-module L2 between two checkpoints.

    Streams tensors shard-by-shard. Each tensor is cast to float32 before squaring.
    """
    unl_map, unl_shards = _list_shards(unlearned_dir)
    orig_map, orig_shards = _list_shards(original_dir)

    # Intersect keys
    keys_unl = set(unl_map.keys())
    keys_orig = set(orig_map.keys())
    common = sorted(keys_unl & keys_orig)
    only_unl = sorted(keys_unl - keys_orig)
    only_orig = sorted(keys_orig - keys_unl)

    if verbose:
        print(f"  keys: unlearned={len(keys_unl)} original={len(keys_orig)} common={len(common)}", file=sys.stderr)
        if only_unl:
            print(f"  only in unlearned ({len(only_unl)}): {only_unl[:5]}{'...' if len(only_unl)>5 else ''}", file=sys.stderr)
        if only_orig:
            print(f"  only in original ({len(only_orig)}): {only_orig[:5]}{'...' if len(only_orig)>5 else ''}", file=sys.stderr)

    # Group common keys by (unlearned_shard, original_shard) to minimize file opens.
    pair_groups: Dict[Tuple[str, str], List[str]] = {}
    for k in common:
        pair_groups.setdefault((unl_map[k], orig_map[k]), []).append(k)

    per_bucket_sq: Dict[str, float] = {"mlp": 0.0, "attention": 0.0, "embedding": 0.0, "layernorm": 0.0, "other": 0.0}
    total_sq = 0.0
    n_keys = 0
    skipped_shape = 0
    skipped_keys: List[str] = []

    for (unl_shard, orig_shard), keys in pair_groups.items():
        unl_path = os.path.join(unlearned_dir, unl_shard)
        orig_path = os.path.join(original_dir, orig_shard)
        with safe_open(unl_path, framework="pt") as unl_sf, safe_open(orig_path, framework="pt") as orig_sf:
            for k in keys:
                t_unl = unl_sf.get_tensor(k)
                t_orig = orig_sf.get_tensor(k)
                if t_unl.shape != t_orig.shape:
                    skipped_shape += 1
                    skipped_keys.append(k)
                    continue
                diff = t_unl.to(torch.float32) - t_orig.to(torch.float32)
                sq = float((diff * diff).sum().item())
                bucket = _classify_key(k)
                per_bucket_sq[bucket] += sq
                total_sq += sq
                n_keys += 1

    result = {
        "total_l2": math.sqrt(total_sq),
        "mlp_l2": math.sqrt(per_bucket_sq["mlp"]),
        "attn_l2": math.sqrt(per_bucket_sq["attention"]),
        "embed_l2": math.sqrt(per_bucket_sq["embedding"]),
        "layernorm_l2": math.sqrt(per_bucket_sq["layernorm"]),
        "other_l2": math.sqrt(per_bucket_sq["other"]),
        "n_keys_compared": n_keys,
        "n_keys_skipped_shape": skipped_shape,
        "skipped_keys": skipped_keys,
        "n_only_in_unlearned": len(only_unl),
        "n_only_in_original": len(only_orig),
    }
    return result


def load_summary(eval_dir: Optional[str]) -> Dict[str, float]:
    if not eval_dir:
        return {}
    p = os.path.join(eval_dir, "TOFU_SUMMARY.json")
    if not os.path.exists(p):
        return {}
    s = json.load(open(p))
    return {
        "ES": s.get("extraction_strength"),
        "model_utility": s.get("model_utility"),
        "forget_truth_ratio": s.get("forget_truth_ratio"),
        "forget_Q_A_Prob": s.get("forget_Q_A_Prob"),
        "forget_Q_A_ROUGE": s.get("forget_Q_A_ROUGE"),
    }


def process_row(row: Dict, verbose: bool = True) -> Dict:
    out = dict(row)
    t0 = time.time()
    unlearned = row.get("unlearned")
    original = row.get("original")
    if not unlearned or not original:
        out["status"] = "skipped: missing unlearned/original path"
        return out
    if not (os.path.isdir(unlearned) and os.path.isdir(original)):
        out["status"] = f"skipped: directory missing ({unlearned} or {original})"
        return out
    try:
        l2 = measure_pair(unlearned, original, verbose=verbose)
        out.update(l2)
        out["wall_seconds"] = round(time.time() - t0, 1)
        out["status"] = "ok"
    except Exception as e:
        out["status"] = f"error: {type(e).__name__}: {e}"
        if verbose:
            print(f"  ERROR: {e}", file=sys.stderr)
    # Attach eval summaries if paths present
    pre = load_summary(row.get("pre_eval"))
    if pre:
        out["pre_ES"] = pre.get("ES")
        out["pre_util"] = pre.get("model_utility")
    post = load_summary(row.get("post_eval"))
    if post:
        out["post_ES"] = post.get("ES")
        out["post_util"] = post.get("model_utility")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure ||W_unlearned - W_original||_2 with per-module breakdown")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="JSON manifest file (list of rows)")
    g.add_argument("--unlearned", help="path to unlearned checkpoint dir (single mode)")
    ap.add_argument("--original", help="path to original checkpoint dir (required with --unlearned)")
    ap.add_argument("--label", default="(unnamed)", help="row label")
    ap.add_argument("--out", help="output JSON path (default stdout)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-row progress logs")
    args = ap.parse_args()

    if args.manifest:
        rows = json.load(open(args.manifest))
        if not isinstance(rows, list):
            print("ERROR: manifest must be a JSON list", file=sys.stderr)
            return 2
    else:
        if not args.original:
            ap.error("--original required with --unlearned")
        rows = [{
            "label": args.label,
            "unlearned": args.unlearned,
            "original": args.original,
        }]

    results = []
    for i, row in enumerate(rows, 1):
        label = row.get("label", f"row{i}")
        if not args.quiet:
            print(f"[{i}/{len(rows)}] {label}", file=sys.stderr)
        result = process_row(row, verbose=not args.quiet)
        results.append(result)
        if not args.quiet and result.get("status") == "ok":
            print(f"  total_l2={result['total_l2']:.4f} mlp={result['mlp_l2']:.4f} attn={result['attn_l2']:.4f} embed={result['embed_l2']:.4f}  [{result['wall_seconds']}s]", file=sys.stderr)

    out_text = json.dumps(results, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out_text)
        if not args.quiet:
            print(f"Wrote {len(results)} rows to {args.out}", file=sys.stderr)
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
