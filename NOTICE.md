# Third-party code, data and methods

This repository is released under the MIT License (see `LICENSE`). It builds on the
following work, which keeps its own licence and should be cited alongside this one.

## OpenUnlearning

Unlearning training, the relearning attacks and TOFU/MUSE evaluation run inside
[OpenUnlearning](https://github.com/locuslab/open-unlearning) (MIT). It is not vendored
here: `scripts/` invokes an OpenUnlearning checkout that you install separately, and
the gradient-based baselines (GA, GradDiff, NPO, RMU) are its trainers at their default
settings. The Extraction Strength and Model Utility metrics are its implementations.

> Dorna, Mekala, Zhao, McCallum, Lipton, Kolter and Maini. *OpenUnlearning:
> Accelerating LLM unlearning via unified benchmarking of methods and metrics.*
> NeurIPS Datasets and Benchmarks, 2025.

## Selective Pruning

`baselines/selective_pruning.py` is our implementation of the Selective Pruning baseline
described in the paper below; the original authors' code is at
<https://github.com/nickypro/selective-pruning>.

> Pochinkov and Schoots. *Dissecting language models: Machine unlearning via selective
> pruning.* arXiv:2403.01267, 2024.

## Wanda importance

The per-input-channel activation-norm importance used by both FRAG and FRP follows
Wanda's calibration.

> Sun, Liu, Bair and Kolter. *A simple and effective pruning approach for large language
> models.* ICLR, 2024.

## Benchmarks and models

- TOFU — <https://huggingface.co/datasets/locuslab/TOFU> (Maini et al., COLM 2024).
  `calib/forget.txt` and `calib/retain.txt` are sampled from it.
- WMDP — <https://huggingface.co/datasets/cais/wmdp> (Li et al., ICML 2024). The cyber
  corpora are downloaded separately and are not redistributed here.
- MUSE-News — <https://huggingface.co/datasets/muse-bench/MUSE-News> (Shi et al., ICLR 2025).
- Base models: `open-unlearning/tofu_Llama-3.2-{1B,3B}-Instruct_full` (Llama 3.2 Community
  License) and `Qwen/Qwen2.5-14B-Instruct` (Apache 2.0).
