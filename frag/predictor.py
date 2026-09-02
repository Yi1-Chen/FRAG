#!/usr/bin/env python3
"""
predictor.py — FRAG, a training-free predictor of relearning robustness (Section 3.2).

Given an ORIGINAL model, an UNLEARNED model, and forget/retain calibration text, computes
(per input channel j:  r_j = ||X^f_j|| / ||X^r_j||,  the forget/retain activation ratio):

    I  = |W| * r            forget importance                  (F, Eq. 3)
    J  = |W| * (1/r)        retain importance                  (R, Eq. 4)
    ΔW = W_unlearned - W_original                              (D = ΔW^2, Eq. 5)
    cosI = cos(I, ΔW^2)              forget-alignment          (A_f, Eq. 5; intermediate)
    cosJ = cos(J, ΔW^2)              retain-alignment          (A_r, Eq. 5; intermediate)
    FRAG  = cosI - gamma * cosJ       Forget-Retain Alignment Gap  (THE predictor)

  Naming: I/J and cosI/cosJ are this file's names for the paper's F/R and A_f/A_r, and
  gamma is the paper's retain-penalty lambda in Eq. (5), fixed at 1.

  FRAG is the net forget-minus-retain alignment; cosI and cosJ are only intermediates.
  Why the retain term (cosJ) is REQUIRED: robustness must be judged together with UTILITY, not
  by ΔES alone. A utility-collapsed edit (e.g. over-pruning shared weights) attains near-zero ΔES
  yet is useless; cosI alone (gamma=0) scores such an edit HIGH and so validates a destroyed model
  as "robust" — it cannot actually verify robustness. Subtracting cosJ penalizes hitting
  retain-critical weights and correctly demotes utility-collapsed edits below selective ones.
  We therefore use gamma=1 (the full gap). gamma=0 reduces FRAG to cosI, which tracks ΔES in
  isolation but is blind to utility (and is fooled by utility-collapse), so it is NOT used.
Higher FRAG => update concentrated on forget-critical, NOT retain-critical, weights => robust & usable.

Self-contained: only needs  torch, transformers, safetensors.  No other dependencies.

USAGE
  python FRAG_predictor.py \
      --original  <orig_model_path_or_HF_id> \
      --unlearned <unlearned_model_path> \
      --forget_file forget.txt --retain_file retain.txt \
      [--target all] [--gamma 1.0] [--num_samples 128] [--max_length 256] [--json_out out.json]

  forget_file / retain_file:  one example per line (plain text, OR JSONL with a "text" field).
  Use a sample of the forget set and the retain set (~128 lines each is plenty).

NOTES
  * Unlearned checkpoint must be in safetensors format (sharded or single .safetensors).
  * Module selection uses Llama/Qwen-style names (gate/up/down/q/k/v/o).
    For other architectures use  --target linear  to score ALL nn.Linear layers.
  * ORIGINAL and UNLEARNED must share architecture (same weight keys & shapes).
  * The score is computed over the chosen modules; --target all (mlp+attn) is recommended.
"""
from __future__ import annotations
import argparse, glob, os, json, math
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- module selection
_KW = {"mlp":["gate_proj","up_proj","down_proj"],
       "attention":["q_proj","k_proj","v_proj","o_proj"],
       "all":["gate_proj","up_proj","down_proj","q_proj","k_proj","v_proj","o_proj"]}
def get_target_modules(model, target="all"):
    if target == "linear":   # any architecture: every Linear layer
        return [n for n,m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    kw = _KW[target]
    return [n for n,m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and any(k in n for k in kw)]

# ---------------------------------------------------------------- activation norms (Wanda-style)
@torch.no_grad()
def collect_input_norms(model, encodings, device, target_modules):
    """Per-input-channel L2 norm of activations, averaged over calibration batches."""
    hooks, norms = [], {}
    tset = set(target_modules)
    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            cn = x.norm(p=2, dim=0).cpu()
            norms[name] = norms.get(name, torch.zeros_like(cn)) + cn
        return hook
    for name, mod in model.named_modules():
        if name in tset:
            hooks.append(mod.register_forward_hook(mk(name)))
    n = 0
    for enc in encodings:
        model(input_ids=enc["input_ids"].to(device),
              attention_mask=enc["attention_mask"].to(device)); n += 1
    for h in hooks: h.remove()
    for k in norms: norms[k] /= max(n, 1)
    return norms

# ---------------------------------------------------------------- streaming weight reader
class StreamReader:
    """Lazy per-tensor reader over (possibly sharded) safetensors — low memory for big models."""
    def __init__(self, d):
        self.d = d; self.h = {}
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.m = json.load(open(idx))["weight_map"]
        else:
            f = glob.glob(os.path.join(d, "*.safetensors"))
            if not f: raise FileNotFoundError(f"no .safetensors in {d}")
            with safe_open(f[0], framework="pt") as sf:
                self.m = {k: os.path.basename(f[0]) for k in sf.keys()}
    def has(self, k): return k in self.m
    def get(self, k):
        sh = self.m[k]
        if sh not in self.h: self.h[sh] = safe_open(os.path.join(self.d, sh), framework="pt")
        return self.h[sh].get_tensor(k)

# ---------------------------------------------------------------- the core score
def score_unlearned(model, fn, rn, unlearned, targets, gamma=1.0):
    """model = loaded ORIGINAL; fn,rn = forget/retain per-channel norms; unlearned = ckpt dir."""
    name2mod = dict(model.named_modules()); un = StreamReader(unlearned)
    dI = dJ = nI = nJ = d4 = 0.0; used = 0
    for name in targets:
        wk = name + ".weight"
        if name not in name2mod or not un.has(wk) or name not in fn: continue
        Wo = name2mod[name].weight.detach().float().cpu(); Wu = un.get(wk).float()
        if Wu.shape != Wo.shape: continue
        d2 = (Wu - Wo) ** 2                                       # ΔW^2
        W = Wo.abs(); f = fn[name].float(); rr = rn[name].float()
        I = W * (f / (rr + 1e-6)).unsqueeze(0)                    # |W|*(Xf/Xr)
        J = W * (rr / (f + 1e-6)).unsqueeze(0)                    # |W|*(Xr/Xf)
        dI += float((I*d2).sum()); dJ += float((J*d2).sum())
        nI += float((I*I).sum()); nJ += float((J*J).sum()); d4 += float((d2*d2).sum()); used += 1
    dn = math.sqrt(d4) if d4 > 0 else 0.0
    c = lambda dot, n2: dot/(math.sqrt(n2)*dn) if (n2 > 0 and dn > 0) else 0.0
    cosI, cosJ = c(dI, nI), c(dJ, nJ)                      # intermediate alignments
    return {"cosI": cosI, "cosJ": cosJ, "FRAG": cosI - gamma*cosJ, "modules_used": used}

# ---------------------------------------------------------------- calibration loader
def load_examples(path, tok, max_length, num_samples):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line); text = obj["text"] if isinstance(obj, dict) and "text" in obj else line
        except Exception:
            text = line
        out.append(tok(text, return_tensors="pt", truncation=True, max_length=max_length))
        if len(out) >= num_samples: break
    return out

def main():
    ap = argparse.ArgumentParser(description="FRAG (Forget-Retain Alignment Gap): relearning-robustness predictor for an unlearned model.")
    ap.add_argument("--original", required=True)
    ap.add_argument("--unlearned", required=True)
    ap.add_argument("--forget_file", required=True, help="forget-set calibration (one example/line; plain text or jsonl 'text')")
    ap.add_argument("--retain_file", required=True, help="retain-set calibration (same format)")
    ap.add_argument("--target", default="all", choices=["all","mlp","attention","linear"])
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="retain-penalty weight in FRAG = cosI - gamma*cosJ. Default 1 = full gap (judges robustness WITH utility). gamma=0 ignores utility (cosI alone is fooled by utility-collapse).")
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json_out", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.original)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print(f"[load] original: {a.original}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.original, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True).to(a.device).eval()
    targets = get_target_modules(model, a.target)
    print(f"[modules] {len(targets)} ({a.target})", flush=True)
    forget = load_examples(a.forget_file, tok, a.max_length, a.num_samples)
    retain = load_examples(a.retain_file, tok, a.max_length, a.num_samples)
    print(f"[calib] forget={len(forget)} retain={len(retain)}; collecting activation norms...", flush=True)
    fn = collect_input_norms(model, forget, a.device, targets)
    rn = collect_input_norms(model, retain, a.device, targets)
    print(f"[score] streaming unlearned: {a.unlearned}", flush=True)
    res = score_unlearned(model, fn, rn, a.unlearned, targets, gamma=a.gamma)
    print("\n===== Forget-Retain Alignment Gap =====")
    print(f"  cosI = {res['cosI']:+.4f}   (forget-alignment; intermediate)")
    print(f"  cosJ = {res['cosJ']:+.4f}   (retain-alignment; intermediate)")
    print(f"  FRAG  = {res['FRAG']:+.4f}   (= cosI - {a.gamma}*cosJ; higher = more robust & utility-preserving)")
    print(f"  (scored over {res['modules_used']} modules, target={a.target}; gamma={a.gamma})")
    if a.json_out:
        json.dump({**res, "target": a.target, "gamma": a.gamma}, open(a.json_out, "w"), indent=2)
        print(f"  wrote {a.json_out}")
if __name__ == "__main__":
    main()
