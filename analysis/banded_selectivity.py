#!/usr/bin/env python3
"""L2-matched selectivity sweep — the experiment that makes FRAG beat global L2.

For each alpha in [0,1] we zero exactly 3% of the MLP weights, but the zeroed set is drawn
from the SAME magnitude distribution for every alpha (per-module magnitude bins, each bin
contributes the same fraction). This holds the update norm (L2) essentially CONSTANT across
the family. Only WHICH weights inside each bin are chosen changes with alpha:
  alpha = 1 -> the FRP forget/retain-selective weight in the bin
  alpha = 0 -> a random weight in the bin
So selectivity varies from blind (0) to fully forget-selective (1) at fixed L2.

Prediction: robustness (post-attack ES) improves with alpha; FRAG tracks it; global L2 is flat
and therefore cannot rank the family. Usage: --alpha 0.5 --out saves/unlearn/band_a050 [--sparsity 0.03]
"""
from __future__ import annotations
import argparse, glob, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
NC = os.environ.get("FRAG_NORMS", "norms_cache")
HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
REF = glob.glob(os.path.join(HUB, "models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full", "snapshots", "*", ""))[0]
MLP_KW = ["gate_proj", "up_proj", "down_proj"]

def is_mlp(n): return any(k in n for k in MLP_KW)

def ranks(x):  # ascending dense ranks along dim=1
    return x.argsort(dim=1).argsort(dim=1).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)   # 0 = random, 1 = FRP-selective
    ap.add_argument("--sparsity", type=float, default=0.03)
    ap.add_argument("--nbins", type=int, default=50)        # magnitude bins per row
    ap.add_argument("--beta", type=float, default=0.05)     # FRP magnitude prior (same as canonical)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(REF)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(REF, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev).eval()
    fn = torch.load(f"{NC}/1B_forget10_fn.pt"); rn = torch.load(f"{NC}/1B_forget10_rn.pt")
    mods = {n: m for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and is_mlp(n) and n in fn}

    total_pruned = 0; total = 0
    with torch.no_grad():
        for n, m in mods.items():
            W = m.weight.detach().float().cpu()
            absW = W.abs()
            rows, cols = absW.shape
            nz = int(a.sparsity * cols)              # weights to zero per row
            if nz == 0: continue

            # FRP rank_combo score (higher = more forget-selective): rank(Xf/Xr) + beta*rank(|W|)
            act_ratio = (fn[n].float() / (rn[n].float() + 1e-6)).unsqueeze(0).expand_as(absW)
            frp_score = ranks(act_ratio) + a.beta * ranks(absW)
            # random score (deterministic per seed)
            g = torch.Generator().manual_seed(a.seed + hash(n) % 100000)
            rand_score = torch.rand(absW.shape, generator=g)

            # blended selection score (higher = more likely to be zeroed): alpha mixes FRP vs random
            blended = a.alpha * ranks(frp_score) + (1 - a.alpha) * ranks(rand_score)

            # magnitude-banded, vectorized: sort columns by |W|, split into equal contiguous bins,
            # zero top-k by blended within each bin. Same per-bin magnitude range + same k for every
            # alpha -> the zeroed-weight magnitude distribution (hence L2) is matched across the family.
            g = cols // a.nbins                       # bin size
            usable = g * a.nbins
            k = max(1, round(a.sparsity * g))         # zeros per bin (constant across rows/bins/alpha)
            mag_order = absW.argsort(dim=1)           # ascending |W|; col idx by magnitude rank
            b_sorted = torch.gather(blended, 1, mag_order)[:, :usable]     # blended in magnitude order
            b_bins = b_sorted.view(rows, a.nbins, g)                      # (rows, nbins, g)
            topk_idx = b_bins.topk(k, dim=2).indices                     # top-k blended within each bin
            sel_sorted = torch.zeros(rows, a.nbins, g, dtype=torch.bool)
            sel_sorted.scatter_(2, topk_idx, True)
            sel_sorted = sel_sorted.view(rows, usable)
            mask = torch.zeros_like(absW, dtype=torch.bool)
            mask.scatter_(1, mag_order[:, :usable], sel_sorted)          # back to original columns
            m.weight.data[mask.to(dev)] = 0.0
            total_pruned += int(mask.sum().item()); total += absW.numel()

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"[band a={a.alpha}] pruned {total_pruned}/{total} ({100*total_pruned/total:.2f}%) -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
