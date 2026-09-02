"""Wanda-based weight pruning for LLM unlearning."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_data, DATASETS

try:
    import wandb
except ImportError:
    wandb = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int = 42) -> None:
    """Set all RNGs deterministically. Matches src/trainer/utils.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_torch_dtype(dtype):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype}")


def get_target_keywords(target):
    if target == "mlp":
        return ["gate_proj", "up_proj", "down_proj"]
    if target == "attention":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target == "all":
        return ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    raise ValueError(f"Unknown target: {target}")


def get_target_modules(model, target="mlp"):
    targets = []
    target_keywords = get_target_keywords(target)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(k in name for k in target_keywords):
                targets.append(name)
    return targets


def collect_input_norms(model, encodings, input_device, target_modules):
    """Collect per-input-channel L2 norms.

    Norms are accumulated on CPU to avoid GPU memory buildup for large models.
    """
    hooks = []
    norms = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])
            col_norms = x_flat.norm(p=2, dim=0).cpu()
            if name not in norms:
                norms[name] = torch.zeros_like(col_norms, dtype=torch.float32)
            norms[name] += col_norms
        return hook_fn

    for name, module in model.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))

    count = 0
    with torch.no_grad():
        for enc in tqdm(encodings, desc="Collecting input norms"):
            input_ids = enc["input_ids"].to(input_device)
            attention_mask = enc["attention_mask"].to(input_device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            count += 1

    for h in hooks:
        h.remove()
    for name in norms:
        norms[name] /= count
    return norms


def compute_wanda_importance(model, input_norms, target_modules):
    """S_ij = |W_ij| * ||X_j||."""
    importance = {}
    for name, module in model.named_modules():
        if name in target_modules and name in input_norms:
            W = module.weight.detach().float().cpu().abs()
            X_norm = input_norms[name].float().cpu()
            importance[name] = W * X_norm.unsqueeze(0)
    return importance


def compute_scores(model, forget_imp, retain_imp, forget_norms, retain_norms,
                   scoring, alpha=1.0, beta=0.05, pretrained_weights=None,
                   target_modules=None):
    """Compute pruning scores per weight."""
    all_scores = {}

    # Random / magnitude_only ignore activation norms / Wanda importance — iterate target_modules directly
    if scoring in ("random", "magnitude_only"):
        if target_modules is None:
            raise ValueError(f"{scoring} scoring requires target_modules")
        score_keys = set(target_modules)
    else:
        score_keys = set(forget_imp.keys()) if forget_imp else set(forget_norms.keys())

    for name, module in model.named_modules():
        if name not in score_keys:
            continue

        S_f = forget_imp.get(name)
        S_r = retain_imp.get(name)

        if scoring == "random":
            # Per-element random scores; top-k selection then picks a random fraction per row.
            # RNG already seeded by seed_everything() in main(), so masks differ across seeds
            # but are deterministic for a given seed.
            scores = torch.rand_like(module.weight.detach().float().cpu())

        elif scoring == "magnitude_only":
            # Classical magnitude pruning: s_ij = |W_ij|. No calibration data used.
            scores = module.weight.detach().float().cpu().abs()

        elif scoring == "fria_importance":
            # FRIA per-weight importance I_ij = |W_ij| * r_j (paper §3.3 "naive" baseline).
            # Tests the magnitude-duality claim: ranking weights by I collapses to
            # ranking by |W| because |W| spans orders of magnitude.
            W = module.weight.detach().float().cpu().abs()
            fn = forget_norms[name].float().cpu()
            rn = retain_norms[name].float().cpu()
            act_ratio = fn / (rn + 1e-6)
            scores = W * act_ratio.unsqueeze(0)

        elif scoring == "wanda_ratio":
            scores = S_f / (S_r + 1e-6)

        elif scoring == "wanda_diff":
            scores = S_f - alpha * S_r

        elif scoring == "wanda_forget_only":
            scores = S_f

        elif scoring == "wanda_normalized":
            scores = S_f / (S_f + S_r + 1e-8)

        elif scoring == "wanda_margin":
            scores = torch.clamp(S_f - alpha * S_r, min=0.0)

        elif scoring == "wanda_protected":
            scores = S_f.clone()
            retain_threshold = torch.quantile(S_r.flatten(), alpha / 100.0)
            protect_mask = S_r >= retain_threshold
            scores[protect_mask] = 0.0

        elif scoring == "rank_combo":
            W = module.weight.detach().float().cpu().abs()
            fn = forget_norms[name].float().cpu()
            rn = retain_norms[name].float().cpu()
            act_ratio = fn / (rn + 1e-6)
            act_scores = act_ratio.unsqueeze(0).expand_as(W)
            act_ranks = act_scores.argsort(dim=1).argsort(dim=1).float()
            w_ranks = W.argsort(dim=1).argsort(dim=1).float()
            scores = act_ranks + beta * w_ranks

        elif scoring == "rank_product":
            # Rank-space multiplicative: s_ij = rank(r_j) * rank(|W_ij|).
            # Same rank space as rank_combo (FRIP), only operator differs (× vs +).
            # Tests "addition vs multiplication" axis isolated from "raw vs rank" axis.
            W = module.weight.detach().float().cpu().abs()
            fn = forget_norms[name].float().cpu()
            rn = retain_norms[name].float().cpu()
            act_ratio = fn / (rn + 1e-6)
            act_scores = act_ratio.unsqueeze(0).expand_as(W)
            act_ranks = act_scores.argsort(dim=1).argsort(dim=1).float()
            w_ranks = W.argsort(dim=1).argsort(dim=1).float()
            scores = act_ranks * w_ranks

        elif scoring == "norm_combine":
            W = module.weight.detach().float().cpu().abs()
            fn = forget_norms[name].float().cpu()
            rn = retain_norms[name].float().cpu()
            act_ratio = fn / (rn + 1e-6)
            act_scores = act_ratio.unsqueeze(0).expand_as(W)
            act_min = act_scores.min(dim=1, keepdim=True).values
            act_max = act_scores.max(dim=1, keepdim=True).values
            norm_ratio = (act_scores - act_min) / (act_max - act_min + 1e-10)
            w_min = W.min(dim=1, keepdim=True).values
            w_max = W.max(dim=1, keepdim=True).values
            norm_wmag = (W - w_min) / (w_max - w_min + 1e-10)
            scores = norm_ratio + beta * norm_wmag

        elif scoring == "log_combine":
            W = module.weight.detach().float().cpu().abs()
            fn = forget_norms[name].float().cpu()
            rn = retain_norms[name].float().cpu()
            act_ratio = fn / (rn + 1e-6)
            act_scores = act_ratio.unsqueeze(0).expand_as(W)
            log_ratio = torch.log(act_scores + 1)
            log_wmag = torch.log(W + 1)
            lr_min = log_ratio.min(dim=1, keepdim=True).values
            lr_max = log_ratio.max(dim=1, keepdim=True).values
            norm_log_ratio = (log_ratio - lr_min) / (lr_max - lr_min + 1e-10)
            lw_min = log_wmag.min(dim=1, keepdim=True).values
            lw_max = log_wmag.max(dim=1, keepdim=True).values
            norm_log_wmag = (log_wmag - lw_min) / (lw_max - lw_min + 1e-10)
            scores = norm_log_ratio + beta * norm_log_wmag

        elif scoring == "rank_weighted":
            W = module.weight.detach().float().cpu().abs()
            fn = forget_norms[name].float().cpu()
            rn = retain_norms[name].float().cpu()
            act_ratio = fn / (rn + 1e-6)
            act_scores = act_ratio.unsqueeze(0).expand_as(W)
            act_ranks = act_scores.argsort(dim=1).argsort(dim=1).float()
            w_ranks = W.argsort(dim=1).argsort(dim=1).float()
            act_mag_weight = act_scores / (act_scores.mean(dim=1, keepdim=True) + 1e-10)
            w_mag_weight = W / (W.mean(dim=1, keepdim=True) + 1e-10)
            scores = act_ranks * act_mag_weight + beta * w_ranks * w_mag_weight

        elif scoring == "task_vector_mag":
            if pretrained_weights is None or name not in pretrained_weights:
                raise ValueError("task_vector_mag requires --pretrained_model")
            W = module.weight.detach().float().cpu()
            W_pre = pretrained_weights[name].float().cpu()
            scores = (W - W_pre).abs()

        else:
            raise ValueError(f"Unknown scoring: {scoring}")

        all_scores[name] = scores

    return all_scores


def score_and_prune_streaming(model, forget_norms, retain_norms, target_modules,
                              scoring, sparsity, beta=0.05,
                              prune_mode="zero", shrink_factor=0.1,
                              invert_selection=False, protect_by="none", protect_p=0.0,
                              **kwargs):
    """Fused score+prune that processes one module at a time (memory-efficient for 72B+).
    invert_selection=True flips top-k to bottom-k (anti-FLP negative control).
    protect_by!="none": before top-k, veto (exclude from pruning) the top-`protect_p`
    fraction of weights per row by a protection score. This trades robustness for utility
    (FRP-RV: retain-vetoed FRP). Modes:
      mag    : protect top |W|                       (magnitude spine)
      retain : protect top |W|*||Xr||                (retain importance)
      excl   : protect top |W|*||Xr||/||Xf|| = |W|/r (retain-EXCLUSIVE: retain-important
               AND forget-unimportant; FRAG J direction; least-harmful to forgetting)."""
    total_pruned = 0
    total_weights = 0

    for name, module in model.named_modules():
        if name not in target_modules or name not in forget_norms:
            continue

        W = module.weight.detach().float().cpu().abs()
        fn = forget_norms[name].float().cpu()
        rn = retain_norms[name].float().cpu()

        if scoring == "rank_combo":
            act_ratio = fn / (rn + 1e-6)
            act_scores = act_ratio.unsqueeze(0).expand_as(W)
            act_ranks = act_scores.argsort(dim=1).argsort(dim=1).float()
            w_ranks = W.argsort(dim=1).argsort(dim=1).float()
            s = act_ranks + beta * w_ranks
        elif scoring == "norm_combine":
            act_ratio = fn / (rn + 1e-6)
            act_scores = act_ratio.unsqueeze(0).expand_as(W)
            act_min = act_scores.min(dim=1, keepdim=True).values
            act_max = act_scores.max(dim=1, keepdim=True).values
            act_norm = (act_scores - act_min) / (act_max - act_min + 1e-8)
            w_min = W.min(dim=1, keepdim=True).values
            w_max = W.max(dim=1, keepdim=True).values
            w_norm = (W - w_min) / (w_max - w_min + 1e-8)
            s = act_norm + beta * w_norm
        else:
            raise ValueError(f"Streaming mode not supported for scoring={scoring}")

        sp = sparsity[name] if isinstance(sparsity, dict) else sparsity
        n_prune = int(sp * s.shape[1])
        if n_prune == 0:
            del W, s
            continue

        if protect_by != "none" and protect_p > 0:
            n_pro = int(protect_p * s.shape[1])
            if n_pro > 0:
                if protect_by == "mag":
                    P = W
                elif protect_by == "retain":
                    P = W * rn.unsqueeze(0)
                elif protect_by == "excl":
                    P = W * (rn / (fn + 1e-6)).unsqueeze(0)
                else:
                    raise ValueError(f"Unknown protect_by={protect_by}")
                pidx = torch.topk(P, n_pro, dim=1).indices
                pmask = torch.zeros_like(s, dtype=torch.bool)
                pmask.scatter_(1, pidx, True)
                s = s.clone()
                s[pmask] = float("-inf")

        _, topk_idx = torch.topk(s, n_prune, dim=1, largest=not invert_selection)
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        dev_mask = mask.to(module.weight.device)

        with torch.no_grad():
            if prune_mode == "zero":
                module.weight.data[dev_mask] = 0
            elif prune_mode == "shrink":
                module.weight.data[dev_mask] *= shrink_factor

        n = mask.sum().item()
        total_pruned += n
        total_weights += s.numel()
        del W, s, mask, dev_mask

    actual_sparsity = total_pruned / total_weights if total_weights > 0 else 0
    print(f"Total pruned: {total_pruned}/{total_weights} ({100*actual_sparsity:.1f}%) [mode={prune_mode}]")
    return model, {"total_pruned": total_pruned, "total_weights": total_weights, "actual_sparsity": actual_sparsity}


def prune_by_scores(model, scores, sparsity, prune_mode="zero",
                    shrink_factor=0.1, noise_scale=0.01, replace_factor=0.01,
                    pretrained_weights=None, tv_alpha=1.0, negate_factor=1.0,
                    invert_selection=False):
    """Prune top-scoring weights per row with configurable pruning mode.
    invert_selection=True flips top-k to bottom-k (anti-FLP negative control)."""
    total_pruned = 0
    total_weights = 0

    for name, module in model.named_modules():
        if name not in scores:
            continue

        s = scores[name]
        sp = sparsity[name] if isinstance(sparsity, dict) else sparsity
        n_prune = int(sp * s.shape[1])
        if n_prune == 0:
            continue

        _, topk_idx = torch.topk(s, n_prune, dim=1, largest=not invert_selection)
        mask = torch.zeros_like(s, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        dev_mask = mask.to(module.weight.device)

        with torch.no_grad():
            W = module.weight.data
            if prune_mode == "zero":
                W[dev_mask] = 0
            elif prune_mode == "shrink":
                W[dev_mask] *= shrink_factor
            elif prune_mode == "noise":
                row_std = W.std(dim=1, keepdim=True)
                noise = torch.randn_like(W) * row_std * noise_scale
                W[dev_mask] = noise[dev_mask]
            elif prune_mode == "retain_mean":
                for i in range(W.shape[0]):
                    kept = W[i, ~dev_mask[i]]
                    if len(kept) > 0:
                        W[i, dev_mask[i]] = kept.mean() * replace_factor
            elif prune_mode == "task_vector":
                if pretrained_weights is None or name not in pretrained_weights:
                    raise ValueError(f"task_vector mode requires --pretrained_model")
                W_pre = pretrained_weights[name].to(W.device)
                tau = W - W_pre  # task vector = finetuned - pretrained
                W[dev_mask] = W[dev_mask] - tv_alpha * tau[dev_mask]
            elif prune_mode == "negate":
                W[dev_mask] = -W[dev_mask] * negate_factor
            elif prune_mode == "mean_replace":
                for i in range(W.shape[0]):
                    kept = W[i, ~dev_mask[i]]
                    if len(kept) > 0:
                        W[i, dev_mask[i]] = kept.mean() * replace_factor
            else:
                raise ValueError(f"Unknown prune_mode: {prune_mode}")

        n = mask.sum().item()
        total_pruned += n
        total_weights += s.numel()

    actual_sparsity = total_pruned / total_weights if total_weights > 0 else 0
    print(f"Total pruned: {total_pruned}/{total_weights} ({100*actual_sparsity:.1f}%) [mode={prune_mode}]")
    return model, {"total_pruned": total_pruned, "total_weights": total_weights, "actual_sparsity": actual_sparsity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="tofu",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--scoring", type=str, default="wanda_ratio",
                        choices=["wanda_ratio", "wanda_diff", "wanda_forget_only",
                                 "wanda_normalized", "wanda_margin", "wanda_protected",
                                 "rank_combo", "rank_product", "norm_combine", "log_combine", "rank_weighted", "task_vector_mag",
                                 "random", "magnitude_only", "fria_importance"])
    parser.add_argument("--sparsity", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="For diff/margin: retain weight. For protected: percentile")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="For rank_combo: weight magnitude rank coefficient")
    parser.add_argument("--target", type=str, default="mlp",
                        choices=["mlp", "attention", "all"])
    parser.add_argument("--sparsity_mode", type=str, default="uniform",
                        choices=["uniform", "spectral", "dual"])
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="Range control for spectral sparsity allocation")
    parser.add_argument("--prune_mode", type=str, default="zero",
                        choices=["zero", "shrink", "noise", "retain_mean", "task_vector", "negate", "mean_replace"],
                        help="How to handle pruned weights")
    parser.add_argument("--shrink_factor", type=float, default=0.1,
                        help="For shrink mode: multiply pruned weights by this")
    parser.add_argument("--noise_scale", type=float, default=0.01,
                        help="For noise mode: noise = row_std * noise_scale")
    parser.add_argument("--replace_factor", type=float, default=0.01,
                        help="For retain_mean mode: replace with mean * factor")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="For task_vector mode: base pretrained model path")
    parser.add_argument("--tv_alpha", type=float, default=1.0,
                        help="For task_vector mode: subtraction strength")
    parser.add_argument("--negate_factor", type=float, default=1.0,
                        help="For negate mode: negation scaling factor")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device_map", type=str, default="single",
                        choices=["single", "auto"],
                        help="single: load on 1 GPU. auto: split across GPUs (for large models)")
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--protect_by", type=str, default="none",
                        choices=["none", "mag", "retain", "excl"],
                        help="FRP-RV retain veto: exclude top --protect_p frac per row by this "
                             "score before pruning. 'excl' = |W|/r (retain-exclusive).")
    parser.add_argument("--protect_p", type=float, default=0.0,
                        help="Fraction per row to protect (veto) when --protect_by != none")
    parser.add_argument("--output_dir", type=str, required=True)
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "unlearning-pruning"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invert_selection", action="store_true",
                        help="Select BOTTOM-k by score instead of top-k (anti-FLP negative control)")
    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"[seed] seed_everything({args.seed}) — cudnn.deterministic=True")

    # wandb setup
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is required for --wandb. Install with: pip install wandb")
        model_short = args.model.split("/")[-1]
        run_name = args.wandb_run_name or f"{model_short}_{args.scoring}_s{args.sparsity}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.wandb_tags,
            config=vars(args),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = get_torch_dtype(args.dtype)
    use_auto_device_map = (args.device_map == "auto" and torch.cuda.is_available())

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if use_auto_device_map:
        # Multi-GPU: split model across all available GPUs
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", **model_kwargs)
        # Determine input device from the embedding layer
        input_device = next(model.parameters()).device
        print(f"Model loaded with device_map=auto, input_device={input_device}")
    elif torch.cuda.is_available() and args.dtype in ["fp16", "bf16"]:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        model.to(device)
        input_device = device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto" if torch.cuda.is_available() else None,
            **model_kwargs,
        )
        input_device = device
    model.eval()

    print(f"Loading data for dataset={args.dataset}...")
    forget_data, retain_data = load_data(
        dataset=args.dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_samples=args.num_samples,
        forget_split=args.forget_split,
        retain_split=args.retain_split,
    )

    target_modules = get_target_modules(model, args.target)
    print(f"Target modules: {len(target_modules)} linear layers ({args.target})")

    if args.scoring in ("random", "magnitude_only"):
        print(f"{args.scoring} — skipping forget/retain norm collection")
        forget_norms = {}
        retain_norms = {}
    else:
        print("Collecting forget input norms...")
        forget_norms = collect_input_norms(model, forget_data, input_device, target_modules)
        print("Collecting retain input norms...")
        retain_norms = collect_input_norms(model, retain_data, input_device, target_modules)

    # Skip importance precomputation for scorings that don't use it (saves ~220GB RAM for 72B)
    if args.scoring in ("rank_combo", "rank_product", "norm_combine", "log_combine", "rank_weighted", "task_vector_mag",
                        "random", "magnitude_only", "fria_importance"):
        print(f"Scoring {args.scoring} computes inline — skipping Wanda importance precomputation")
        forget_imp = {}
        retain_imp = {}
    else:
        print("Computing Wanda importance...")
        forget_imp = compute_wanda_importance(model, forget_norms, target_modules)
        retain_imp = compute_wanda_importance(model, retain_norms, target_modules)

    # Load pretrained weights if needed for task_vector scoring or pruning
    pretrained_weights = None
    if args.prune_mode == "task_vector" or args.scoring == "task_vector_mag":
        if args.pretrained_model is None:
            raise ValueError("--pretrained_model required for task_vector mode/scoring")
        print(f"Loading pretrained model: {args.pretrained_model}")
        _pretrained = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        pretrained_weights = {}
        for n, m in _pretrained.named_modules():
            if isinstance(m, torch.nn.Linear):
                pretrained_weights[n] = m.weight.data.clone()
        del _pretrained

    # Use streaming (fused score+prune) for memory-efficient large model support
    use_streaming = (args.scoring in ("rank_combo", "norm_combine")
                     and args.sparsity_mode == "uniform"
                     and args.prune_mode in ("zero", "shrink"))

    spectral_data = None

    if use_streaming:
        print(f"Scoring+Pruning (streaming): {args.scoring} (beta={args.beta}), sparsity={args.sparsity}")
        model, prune_stats = score_and_prune_streaming(
            model, forget_norms, retain_norms, target_modules,
            scoring=args.scoring, sparsity=args.sparsity, beta=args.beta,
            prune_mode=args.prune_mode, shrink_factor=args.shrink_factor,
            invert_selection=args.invert_selection,
            protect_by=args.protect_by, protect_p=args.protect_p,
        )
    else:
        print(f"Scoring: {args.scoring} (alpha={args.alpha}, beta={args.beta})")
        scores = compute_scores(model, forget_imp, retain_imp, forget_norms, retain_norms,
                               args.scoring, args.alpha, args.beta,
                               pretrained_weights=pretrained_weights,
                               target_modules=target_modules)

        if args.sparsity_mode != "uniform":
            from spectral import (compute_block_alphas, compute_block_divergence,
                                  allocate_sparsity, get_module_sparsity)

            print("Computing spectral metrics...")
            block_alphas, module_alphas = compute_block_alphas(model, target_modules)

            if args.sparsity_mode == "spectral":
                block_metrics = block_alphas
            elif args.sparsity_mode == "dual":
                block_divs = compute_block_divergence(forget_norms, retain_norms, target_modules)
                block_metrics = {
                    idx: block_alphas[idx] * block_divs.get(idx, 1.0)
                    for idx in block_alphas
                }

            block_sparsity = allocate_sparsity(block_metrics, args.sparsity, args.epsilon)
            sparsity_arg = get_module_sparsity(target_modules, block_sparsity, args.sparsity)

            spectral_data = {
                "block_alphas": {str(k): v for k, v in block_alphas.items()},
                "block_sparsity": {str(k): v for k, v in block_sparsity.items()},
            }
            if args.sparsity_mode == "dual":
                spectral_data["block_divergence"] = {str(k): v for k, v in block_divs.items()}
                spectral_data["block_metrics"] = {str(k): v for k, v in block_metrics.items()}

            for idx in sorted(block_sparsity.keys()):
                alpha_str = f"{block_alphas.get(idx, 0):.3f}"
                print(f"  Block {idx}: sparsity={block_sparsity[idx]:.4f} (alpha={alpha_str})")
        else:
            sparsity_arg = args.sparsity

        print(f"Pruning with sparsity={args.sparsity} (mode={args.sparsity_mode})")
        model, prune_stats = prune_by_scores(model, scores, sparsity_arg,
                                              prune_mode=args.prune_mode,
                                              shrink_factor=args.shrink_factor,
                                              noise_scale=args.noise_scale,
                                              replace_factor=args.replace_factor,
                                              pretrained_weights=pretrained_weights,
                                              tv_alpha=args.tv_alpha,
                                              negate_factor=args.negate_factor,
                                              invert_selection=args.invert_selection)

    print(f"Saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = vars(args)
    config["prune_stats"] = prune_stats
    if spectral_data is not None:
        config["spectral"] = spectral_data
    with open(os.path.join(args.output_dir, "wanda_unlearn_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if args.wandb:
        wandb.log(prune_stats)
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
