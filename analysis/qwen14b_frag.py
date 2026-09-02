#!/usr/bin/env python3
"""FRAG (cosI, cosJ, FRAG = cosI - gamma*cosJ) for Qwen2.5-14B-Instruct on WMDP-cyber.

Same hyperparameters as the TOFU predictor script: targets=all (mlp+attn),
gamma=1.0, NS=128, ML=256, WMDP-cyber forget+retain calibration.

Output:
  {OUTDIR}/wmdp_qwen14b_{method}.json   per method
  {OUTDIR}/wmdp_qwen14b_frag.csv        summary

NPO is not included: open-unlearning's NPO ref-model deepcopy + device_map=auto
is incompatible at 14B on a single 48GB device.
"""
from __future__ import annotations
import glob, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "frag"))
sys.path.insert(0, os.path.join(ROOT, "frp"))
from predictor import score_unlearned, collect_input_norms, get_target_modules
from data import load_data

REPO = os.environ.get("OPEN_UNLEARNING", os.path.expanduser("~/open-unlearning"))
HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

NS, ML, DEV, TGT, GAMMA = 128, 256, "cuda", "all", 1.0
TAG = "wmdp_qwen14b"
OUT_DIR = os.environ.get("OUTDIR", os.path.join(ROOT, "runs"))
os.makedirs(OUT_DIR, exist_ok=True)

def hub(p):
    return glob.glob(os.path.join(HUB, p, "snapshots", "*", ""))[0]

REF = hub("models--Qwen--Qwen2.5-14B-Instruct")
CKDIR = f"{REPO}/saves/table2_canonical/saves/unlearn"
EVALDIR = f"{REPO}/saves/table2_canonical/saves/eval"

# (display, ckpt-method-name) — ckpt = {CKDIR}/{name}_wmdp_qwen14b
#                              eval = {EVALDIR}/{name}_wmdp_qwen14b_{pre,post}
METHODS = [
    ("GA",       "gradascent"),
    ("GradDiff", "graddiff"),
    ("RMU",      "rmu"),
    ("SP",       "sp"),
    ("FLP",      "flp"),
]


def read_cyber_acc(eval_dir):
    p = os.path.join(eval_dir, "LMEval_SUMMARY.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("wmdp_cyber/acc")


def main():
    tok = AutoTokenizer.from_pretrained(REF)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[load] reference: {REF}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        REF, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEV).eval()

    targets = get_target_modules(model, TGT)
    print(f"[modules] {len(targets)} ({TGT})", flush=True)

    print("[calib] loading WMDP-cyber forget/retain ...", flush=True)
    forget, retain = load_data("wmdp_cyber", tok, ML, NS)
    print(f"[calib] forget={len(forget)} retain={len(retain)}; collecting activation norms ...", flush=True)
    fn = collect_input_norms(model, forget, DEV, targets)
    rn = collect_input_norms(model, retain, DEV, targets)

    rows = []
    print(f"\n{'Method':10}{'cosI':>10}{'cosJ':>10}{'FRAG':>10}{'FRAG%':>10}"
          f"{'pre_ES':>10}{'post_ES':>10}{'ΔES':>9}", flush=True)
    for disp, name in METHODS:
        ck = os.path.join(CKDIR, f"{name}_wmdp_qwen14b")
        if not os.path.isdir(ck):
            print(f"  MISS ckpt {ck}", flush=True)
            continue
        s = score_unlearned(model, fn, rn, ck, targets, gamma=GAMMA)
        pre = read_cyber_acc(os.path.join(EVALDIR, f"{name}_wmdp_qwen14b_pre"))
        post = read_cyber_acc(os.path.join(EVALDIR, f"{name}_wmdp_qwen14b_post"))
        d_es = (post - pre) if (pre is not None and post is not None) else None
        row = {
            "method": disp,
            "cosI": s["cosI"], "cosJ": s["cosJ"], "FRAG": s["FRAG"],
            "modules_used": s["modules_used"],
            "pre_es": pre, "post_es": post, "d_es": d_es,
            "ckpt": ck,
        }
        rows.append(row)
        json.dump(row, open(os.path.join(OUT_DIR, f"{TAG}_{disp}.json"), "w"), indent=2)
        des_str = f"{d_es:+.4f}" if d_es is not None else "  n/a"
        pre_str = f"{pre:.4f}" if pre is not None else "  n/a"
        post_str = f"{post:.4f}" if post is not None else "  n/a"
        print(f"  {disp:8s}{s['cosI']:>+10.4f}{s['cosJ']:>+10.4f}"
              f"{s['FRAG']:>+10.4f}{s['FRAG']*100:>+9.3f}%"
              f"{pre_str:>10}{post_str:>10}{des_str:>9}", flush=True)

    csv_path = os.path.join(OUT_DIR, f"{TAG}_frag.csv")
    cols = ["method", "cosI", "cosJ", "FRAG", "modules_used", "pre_es", "post_es", "d_es"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                "" if r[c] is None else (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]))
                for c in cols
            ) + "\n")
    print(f"\n[done] wrote {csv_path}", flush=True)
    print(f"[done] per-method JSONs in {OUT_DIR}/{TAG}_*.json", flush=True)


if __name__ == "__main__":
    main()
