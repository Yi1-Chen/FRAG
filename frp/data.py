"""Shared data loading for pruning methods."""

import json
from pathlib import Path
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_tofu(split, tokenizer, max_length=256, num_samples=500):
    ds = load_dataset("locuslab/TOFU", split, split="train")
    encodings = []
    for item in ds:
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        encodings.append(enc)
        if len(encodings) >= num_samples:
            break
    return encodings


def load_jsonl(path, tokenizer, max_length=256, num_samples=500):
    encodings = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item["text"] if isinstance(item, dict) else item
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            encodings.append(enc)
            if len(encodings) >= num_samples:
                break
    return encodings


def load_muse(split, tokenizer, max_length=256, num_samples=500):
    """Load MUSE-News forget/retain data.
    split: 'forget' or 'retain'
    """
    ds = load_dataset("muse-bench/MUSE-News", "train", split=split)
    encodings = []
    for item in ds:
        enc = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=max_length)
        encodings.append(enc)
        if len(encodings) >= num_samples:
            break
    return encodings


# Dataset registry: maps --dataset value to (forget_loader, retain_loader)
DATASETS = {
    "tofu": {
        "forget": lambda tok, ml, ns, fs="forget10", **_: load_tofu(fs, tok, ml, ns),
        "retain": lambda tok, ml, ns, rs="retain90", **_: load_tofu(rs, tok, ml, ns),
    },
    "wmdp_cyber": {
        "forget": lambda tok, ml, ns, **_: load_jsonl(
            REPO_ROOT / "data" / "wmdp" / "wmdp-corpora" / "cyber-forget-corpus.jsonl", tok, ml, ns),
        "retain": lambda tok, ml, ns, **_: load_jsonl(
            REPO_ROOT / "data" / "wmdp" / "wmdp-corpora" / "cyber-retain-corpus.jsonl", tok, ml, ns),
    },
    "wmdp_bio": {
        "forget": lambda tok, ml, ns, **_: load_jsonl(
            REPO_ROOT / "data" / "wmdp" / "wmdp-corpora" / "bio-retain-corpus.jsonl", tok, ml, ns),
        "retain": lambda tok, ml, ns, **_: load_jsonl(
            REPO_ROOT / "data" / "wmdp" / "wmdp-corpora" / "bio-retain-corpus.jsonl", tok, ml, ns),
    },
    "muse_news": {
        "forget": lambda tok, ml, ns, **_: load_muse("forget", tok, ml, ns),
        "retain": lambda tok, ml, ns, **_: load_muse("retain1", tok, ml, ns),
    },
}


def load_data(dataset, tokenizer, max_length=256, num_samples=500,
              forget_split="forget10", retain_split="retain90"):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASETS.keys())}")

    loaders = DATASETS[dataset]
    forget_data = loaders["forget"](tokenizer, max_length, num_samples, fs=forget_split, rs=retain_split)
    retain_data = loaders["retain"](tokenizer, max_length, num_samples, fs=forget_split, rs=retain_split)

    if dataset == "wmdp_bio":
        print("Warning: wmdp_bio has no separate forget corpus; using bio-retain for both.")

    return forget_data, retain_data
