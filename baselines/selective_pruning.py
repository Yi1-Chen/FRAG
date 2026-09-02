"""
Scoring functions:
  1. activation_ratio  Original Selective Pruning (Pochinkov & Schoots 2024)
  2. gradient_ratio    Gradient magnitude ratio (forget/retain)
  3. cosine_score      My method: gradient directionality-aware scoring

Usage:
    CUDA_VISIBLE_DEVICES=1 python selective_pruning_scored.py \
        --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --forget_split forget10 --retain_split retain90 \
        --scoring cosine_score --mlp_frac 0.05 \
        --output_dir saves/unlearn/tofu_1B_forget10_CosineScore_5pct
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frp"))
from data import load_data, DATASETS


# ===========================================================
# Scoring 1: Activation Ratio (original Selective Pruning)
# ===========================================================

def collect_activations(model, encodings, device, num_tokens=50000):
    hooks = []
    activations = {}

    for i, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                act = output.detach().abs()
                act_flat = act.reshape(-1, act.shape[-1])
                if layer_idx not in activations:
                    activations[layer_idx] = {"sum": torch.zeros(act.shape[-1], device=device, dtype=torch.float32), "count": 0}
                activations[layer_idx]["sum"] += act_flat.sum(dim=0).float()
                activations[layer_idx]["count"] += act_flat.shape[0]
            return hook_fn
        h = layer.mlp.up_proj.register_forward_hook(make_hook(i))
        hooks.append(h)

    tokens_seen = 0
    with torch.no_grad():
        for enc in tqdm(encodings, desc="Collecting activations"):
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            tokens_seen += input_ids.numel()
            if tokens_seen >= num_tokens:
                break

    for h in hooks:
        h.remove()

    result = {}
    for layer_idx in sorted(activations.keys()):
        result[layer_idx] = activations[layer_idx]["sum"] / activations[layer_idx]["count"]
    return result


def collect_samplewise_up_proj_activations(model, encodings, device):
    """Collect per-sample scalar activations for each up_proj neuron."""
    hooks = []
    activations = {}

    for i, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Use mean over tokens as the sample-level scalar for each neuron.
                sample_values = output.detach().float().mean(dim=1).cpu()
                if layer_idx not in activations:
                    activations[layer_idx] = []
                activations[layer_idx].append(sample_values)
            return hook_fn
        h = layer.mlp.up_proj.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for enc in tqdm(encodings, desc="Collecting MANU activations"):
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    result = {}
    for layer_idx in sorted(activations.keys()):
        result[layer_idx] = torch.cat(activations[layer_idx], dim=0)
    return result


def score_activation_ratio(model, forget_act, retain_act):
    """Original Selective Pruning: forget_activation / (retain_activation + eps)"""
    scores = {}
    for layer_idx in sorted(forget_act.keys()):
        scores[layer_idx] = forget_act[layer_idx] / (retain_act[layer_idx] + 1e-5)
    return scores


def _compute_manu_importance(sample_activations, scoring):
    if scoring == "manu_abs":
        return sample_activations.abs().mean(dim=0)
    if scoring == "manu_freq":
        median = sample_activations.median(dim=0).values
        return (sample_activations > median.unsqueeze(0)).float().mean(dim=0)
    if scoring == "manu_var":
        return sample_activations.std(dim=0, unbiased=False)
    if scoring == "manu_rms":
        return sample_activations.pow(2).mean(dim=0).sqrt()
    if scoring == "manu_combined":
        abs_score = sample_activations.abs().mean(dim=0)
        median = sample_activations.median(dim=0).values
        freq_score = (sample_activations > median.unsqueeze(0)).float().mean(dim=0)
        var_score = sample_activations.std(dim=0, unbiased=False)
        rms_score = sample_activations.pow(2).mean(dim=0).sqrt()
        return abs_score + freq_score + var_score + rms_score
    raise ValueError(f"Unknown MANU scoring: {scoring}")


def score_manu(forget_acts, retain_acts, scoring, eps=1e-5):
    scores = {}
    for layer_idx in sorted(forget_acts.keys()):
        forget_imp = _compute_manu_importance(forget_acts[layer_idx], scoring)
        retain_imp = _compute_manu_importance(retain_acts[layer_idx], scoring)
        scores[layer_idx] = forget_imp / (retain_imp + eps)
    return scores


# ===========================================================
# Scoring 2 & 3: Gradient-based scoring
# ===========================================================

def collect_per_neuron_gradients(model, encodings, device, num_samples=100):
    """Compute per-neuron gradient of NLL loss w.r.t. up_proj weights.
    Returns dict: layer_idx -> tensor of shape (intermediate_size,)
    representing the mean gradient magnitude per neuron.
    For cosine scoring, also returns the full mean gradient direction.
    """
    n_layers = len(model.model.layers)
    intermediate_size = model.model.layers[0].mlp.up_proj.weight.shape[0]

    grad_sum = {i: torch.zeros(intermediate_size, device=device, dtype=torch.float32) for i in range(n_layers)}
    grad_sq_sum = {i: torch.zeros(intermediate_size, device=device, dtype=torch.float32) for i in range(n_layers)}
    count = 0

    for enc in tqdm(encodings[:num_samples], desc="Computing gradients"):
        model.zero_grad()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for i, layer in enumerate(model.model.layers):
            # gradient of loss w.r.t. each row of up_proj.weight
            # up_proj.weight shape: (intermediate_size, hidden_size)
            # per-neuron gradient = L2 norm across hidden_size dimension
            g = layer.mlp.up_proj.weight.grad
            if g is not None:
                per_neuron_grad = g.float().norm(dim=1)  # (intermediate_size,)
                grad_sum[i] += per_neuron_grad
                grad_sq_sum[i] += per_neuron_grad ** 2

        count += 1

    grad_mean = {i: grad_sum[i] / count for i in range(n_layers)}
    return grad_mean


def collect_per_neuron_gradient_vectors(model, encodings, device, num_samples=100):
    """Collect full gradient vectors per neuron for cosine similarity computation.
    Returns dict: layer_idx -> tensor of shape (intermediate_size, hidden_size)
    """
    n_layers = len(model.model.layers)
    intermediate_size = model.model.layers[0].mlp.up_proj.weight.shape[0]
    hidden_size = model.model.layers[0].mlp.up_proj.weight.shape[1]

    # accumulate gradients
    grad_accum = {i: torch.zeros(intermediate_size, hidden_size, device=device, dtype=torch.float32)
                  for i in range(n_layers)}
    count = 0

    for enc in tqdm(encodings[:num_samples], desc="Computing gradient vectors"):
        model.zero_grad()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        for i, layer in enumerate(model.model.layers):
            g = layer.mlp.up_proj.weight.grad
            if g is not None:
                grad_accum[i] += g.float()

        count += 1

    grad_mean = {i: grad_accum[i] / count for i in range(n_layers)}
    return grad_mean


def score_gradient_ratio(model, forget_encodings, retain_encodings, device, num_samples=100):
    """Gradient magnitude ratio: ||grad_forget|| / (||grad_retain|| + eps)"""
    print("Computing forget gradients...")
    forget_grad = collect_per_neuron_gradients(model, forget_encodings, device, num_samples)
    print("Computing retain gradients...")
    retain_grad = collect_per_neuron_gradients(model, retain_encodings, device, num_samples)

    scores = {}
    for layer_idx in sorted(forget_grad.keys()):
        scores[layer_idx] = forget_grad[layer_idx] / (retain_grad[layer_idx] + 1e-8)
    return scores


def score_cosine(model, forget_encodings, retain_encodings, device, num_samples=100):
    """Cosine v1: gradient directionality + ratio (no double magnitude)
    Score = (1 - |cos(g_f, g_r)|) * ||g_f|| / (||g_r|| + eps)
    """
    print("Computing forget gradient vectors...")
    forget_gvec = collect_per_neuron_gradient_vectors(model, forget_encodings, device, num_samples)
    print("Computing retain gradient vectors...")
    retain_gvec = collect_per_neuron_gradient_vectors(model, retain_encodings, device, num_samples)

    scores = {}
    for layer_idx in sorted(forget_gvec.keys()):
        gf = forget_gvec[layer_idx]
        gr = retain_gvec[layer_idx]

        gf_norm = gf.norm(dim=1)
        gr_norm = gr.norm(dim=1)

        cos_sim = torch.nn.functional.cosine_similarity(gf, gr, dim=1)
        directionality = 1.0 - cos_sim.abs()
        magnitude_ratio = gf_norm / (gr_norm + 1e-8)

        scores[layer_idx] = directionality * magnitude_ratio

    return scores


def score_cosine_hybrid(model, forget_encodings, retain_encodings, forget_act, retain_act, device, num_samples=100):
    """Cosine v2 (hybrid): activation ratio * gradient directionality filter
    Score = (act_forget / (act_retain + eps)) * (1 - |cos(g_f, g_r)|)

    Uses activation ratio from original Selective Pruning, but filters out
    shared neurons using gradient directionality.
    """
    print("Computing gradient vectors for directionality...")
    forget_gvec = collect_per_neuron_gradient_vectors(model, forget_encodings, device, num_samples)
    retain_gvec = collect_per_neuron_gradient_vectors(model, retain_encodings, device, num_samples)

    scores = {}
    for layer_idx in sorted(forget_act.keys()):
        act_ratio = forget_act[layer_idx] / (retain_act[layer_idx] + 1e-5)

        gf = forget_gvec[layer_idx]
        gr = retain_gvec[layer_idx]
        cos_sim = torch.nn.functional.cosine_similarity(gf, gr, dim=1)
        directionality = 1.0 - cos_sim.abs()

        scores[layer_idx] = act_ratio * directionality

    return scores


def score_cosine_filter(model, forget_encodings, retain_encodings, forget_act, retain_act, device, num_samples=100, cos_threshold=0.5):
    """Cosine filter: use activation ratio but zero out neurons where cos > threshold.
    Only prune neurons that are both high activation ratio AND low cosine similarity.
    Protects shared neurons without penalizing the score continuously.

    Score = act_ratio  if |cos| < threshold
            0          if |cos| >= threshold
    """
    print("Computing gradient vectors for cosine filter...")
    forget_gvec = collect_per_neuron_gradient_vectors(model, forget_encodings, device, num_samples)
    retain_gvec = collect_per_neuron_gradient_vectors(model, retain_encodings, device, num_samples)

    scores = {}
    total_protected = 0
    total_neurons = 0
    for layer_idx in sorted(forget_act.keys()):
        act_ratio = forget_act[layer_idx] / (retain_act[layer_idx] + 1e-5)

        gf = forget_gvec[layer_idx]
        gr = retain_gvec[layer_idx]
        cos_sim = torch.nn.functional.cosine_similarity(gf, gr, dim=1)

        # hard filter: protect neurons with high cosine similarity
        protect_mask = cos_sim.abs() >= cos_threshold
        filtered_scores = act_ratio.clone()
        filtered_scores[protect_mask] = 0.0

        total_protected += protect_mask.sum().item()
        total_neurons += protect_mask.numel()
        scores[layer_idx] = filtered_scores

    print(f"Protected {total_protected}/{total_neurons} neurons ({100*total_protected/total_neurons:.1f}%) with |cos| >= {cos_threshold}")
    return scores


def score_cosine_soft(model, forget_encodings, retain_encodings, forget_act, retain_act, device, num_samples=100):
    """Cosine soft: only penalize positively correlated neurons (cos > 0).
    Neurons with cos < 0 (conflicting gradients) are not penalized.

    Score = act_ratio * max(0, 1 - cos)
    When cos > 0 (shared): penalty applied, score reduced
    When cos < 0 (conflicting): max(0, 1-cos) > 1, score boosted
    When cos = 0 (orthogonal): no change
    """
    print("Computing gradient vectors for soft cosine...")
    forget_gvec = collect_per_neuron_gradient_vectors(model, forget_encodings, device, num_samples)
    retain_gvec = collect_per_neuron_gradient_vectors(model, retain_encodings, device, num_samples)

    scores = {}
    for layer_idx in sorted(forget_act.keys()):
        act_ratio = forget_act[layer_idx] / (retain_act[layer_idx] + 1e-5)

        gf = forget_gvec[layer_idx]
        gr = retain_gvec[layer_idx]
        cos_sim = torch.nn.functional.cosine_similarity(gf, gr, dim=1)

        # only penalize positive cosine, boost negative cosine
        soft_penalty = torch.clamp(1.0 - cos_sim, min=0.0)
        scores[layer_idx] = act_ratio * soft_penalty

    return scores


def score_weighted_combo(model, forget_encodings, retain_encodings, forget_act, retain_act, device, num_samples=100, alpha=0.7):
    """Weighted combination of activation ratio and gradient-based directionality score.
    Score = alpha * act_ratio + (1-alpha) * grad_ratio * (1 - |cos|)
    """
    print("Computing gradient info for weighted combo...")
    forget_gvec = collect_per_neuron_gradient_vectors(model, forget_encodings, device, num_samples)
    retain_gvec = collect_per_neuron_gradient_vectors(model, retain_encodings, device, num_samples)

    scores = {}
    for layer_idx in sorted(forget_act.keys()):
        act_ratio = forget_act[layer_idx] / (retain_act[layer_idx] + 1e-5)

        gf = forget_gvec[layer_idx]
        gr = retain_gvec[layer_idx]
        gf_norm = gf.norm(dim=1)
        gr_norm = gr.norm(dim=1)
        cos_sim = torch.nn.functional.cosine_similarity(gf, gr, dim=1)

        grad_ratio = gf_norm / (gr_norm + 1e-8)
        directionality = 1.0 - cos_sim.abs()

        # normalize both components to similar scale
        act_ratio_norm = act_ratio / (act_ratio.max() + 1e-8)
        grad_score_norm = (grad_ratio * directionality)
        grad_score_norm = grad_score_norm / (grad_score_norm.max() + 1e-8)

        scores[layer_idx] = alpha * act_ratio_norm + (1 - alpha) * grad_score_norm

    return scores


# ===========================================================
# Pruning
# ===========================================================

def prune_by_scores(model, scores, mlp_frac):
    all_scores = torch.cat([scores[i] for i in sorted(scores.keys())])
    k = int(mlp_frac * all_scores.numel())
    if k == 0:
        print("Nothing to prune")
        return model

    topk = torch.topk(all_scores, k, largest=True)
    threshold = topk.values.min().item()

    total_pruned = 0
    for layer_idx in sorted(scores.keys()):
        mask = scores[layer_idx] >= threshold
        n_pruned = mask.sum().item()
        total_pruned += n_pruned

        if n_pruned > 0:
            layer = model.model.layers[layer_idx]
            with torch.no_grad():
                layer.mlp.gate_proj.weight.data[mask] = 0
                layer.mlp.up_proj.weight.data[mask] = 0
                layer.mlp.down_proj.weight.data[:, mask] = 0
            print(f"  Layer {layer_idx}: pruned {n_pruned} / {scores[layer_idx].numel()}")

    print(f"Total pruned: {total_pruned} / {all_scores.numel()} ({100*total_pruned/all_scores.numel():.1f}%)")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="tofu",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--scoring", type=str, required=True,
                        choices=["activation_ratio", "gradient_ratio", "cosine_score", "cosine_hybrid",
                                 "cosine_filter", "cosine_soft", "weighted_combo",
                                 "manu_abs", "manu_freq", "manu_var", "manu_rms",
                                 "manu_combined"])
    parser.add_argument("--mlp_frac", type=float, default=0.05)
    parser.add_argument("--cos_threshold", type=float, default=0.5, help="For cosine_filter scoring")
    parser.add_argument("--alpha", type=float, default=0.7, help="For weighted_combo scoring")
    parser.add_argument("--num_tokens", type=int, default=50000, help="For activation collection")
    parser.add_argument("--num_grad_samples", type=int, default=100, help="For gradient computation")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )

    print(f"Loading data (dataset={args.dataset})...")
    forget_data, retain_data = load_data(
        args.dataset, tokenizer, args.max_length, args.num_samples,
        args.forget_split, args.retain_split,
    )

    print(f"Computing scores with method: {args.scoring}")

    if args.scoring == "activation_ratio":
        model.eval()
        forget_act = collect_activations(model, forget_data, device, args.num_tokens)
        retain_act = collect_activations(model, retain_data, device, args.num_tokens)
        scores = score_activation_ratio(model, forget_act, retain_act)

    elif args.scoring == "gradient_ratio":
        model.train()
        scores = score_gradient_ratio(model, forget_data, retain_data, device, args.num_grad_samples)
        model.eval()

    elif args.scoring == "cosine_score":
        model.train()
        scores = score_cosine(model, forget_data, retain_data, device, args.num_grad_samples)
        model.eval()

    elif args.scoring == "cosine_hybrid":
        model.eval()
        forget_act = collect_activations(model, forget_data, device, args.num_tokens)
        retain_act = collect_activations(model, retain_data, device, args.num_tokens)
        model.train()
        scores = score_cosine_hybrid(model, forget_data, retain_data, forget_act, retain_act, device, args.num_grad_samples)
        model.eval()

    elif args.scoring == "cosine_filter":
        model.eval()
        forget_act = collect_activations(model, forget_data, device, args.num_tokens)
        retain_act = collect_activations(model, retain_data, device, args.num_tokens)
        model.train()
        scores = score_cosine_filter(model, forget_data, retain_data, forget_act, retain_act, device, args.num_grad_samples, args.cos_threshold)
        model.eval()

    elif args.scoring == "cosine_soft":
        model.eval()
        forget_act = collect_activations(model, forget_data, device, args.num_tokens)
        retain_act = collect_activations(model, retain_data, device, args.num_tokens)
        model.train()
        scores = score_cosine_soft(model, forget_data, retain_data, forget_act, retain_act, device, args.num_grad_samples)
        model.eval()

    elif args.scoring == "weighted_combo":
        model.eval()
        forget_act = collect_activations(model, forget_data, device, args.num_tokens)
        retain_act = collect_activations(model, retain_data, device, args.num_tokens)
        model.train()
        scores = score_weighted_combo(model, forget_data, retain_data, forget_act, retain_act, device, args.num_grad_samples, args.alpha)
        model.eval()

    elif args.scoring in ["manu_abs", "manu_freq", "manu_var", "manu_rms", "manu_combined"]:
        model.eval()
        forget_sample_acts = collect_samplewise_up_proj_activations(model, forget_data, device)
        retain_sample_acts = collect_samplewise_up_proj_activations(model, retain_data, device)
        scores = score_manu(forget_sample_acts, retain_sample_acts, args.scoring)

    print(f"Pruning top {args.mlp_frac*100}% of neurons...")
    model = prune_by_scores(model, scores, args.mlp_frac)

    print(f"Saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = vars(args)
    with open(os.path.join(args.output_dir, "pruning_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
