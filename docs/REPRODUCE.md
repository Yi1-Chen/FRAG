# Reproducing the paper

Everything below refers to *Distance Is Not Enough: Forget-Retain Alignment Gap Predicts
LLM Relearning Robustness*. Section and table numbers are the paper's.

## Contents

```
frag/predictor.py              FRAG: the training-free robustness predictor (Section 3.2)
frp/prune.py                   FRP: rank-space pruning, with the retain veto (Section 3.3)
frp/data.py                    TOFU / WMDP / MUSE calibration loaders
baselines/selective_pruning.py Selective Pruning (Pochinkov & Schoots, 2024)
baselines/inject_noise.py      direction-blind noise controls (Appendix A.2)
scripts/                       end-to-end pipelines: unlearn, attack, sweeps
analysis/                      predictors, Spearman, L2, matched-L2 control
calib/                         forget / retain calibration text for the predictor
results/                       the numbers reported in the paper, as CSV
```

## Setup

FRAG and FRP need only `pip install -r requirements.txt`. Training, the relearning
attacks and TOFU evaluation run inside [OpenUnlearning](https://github.com/locuslab/open-unlearning);
install it separately and point the scripts at it:

```bash
export OPEN_UNLEARNING=/path/to/open-unlearning   # checkout used for train.py / eval.py
export FRAG_NORMS=$PWD/norms_cache                # cached activation norms
export OUTDIR=$PWD/runs                           # per-run logs and summaries
export PYTHON=$OPEN_UNLEARNING/.venv/bin/python   # interpreter with OpenUnlearning installed
```

Models come from the HuggingFace cache (`HF_HUB_CACHE`, default `~/.cache/huggingface/hub`):
`open-unlearning/tofu_Llama-3.2-{1B,3B}-Instruct_full` for TOFU and
`Qwen/Qwen2.5-14B-Instruct` for WMDP-cyber. WMDP corpora go in `data/wmdp/wmdp-corpora/`.

All reported runs used Python 3.11, torch 2.4.1+cu121, transformers 4.45.1, seed 42, on
4x RTX A6000 48 GB.

## Notation: paper symbols and code flags

The code predates the paper's notation, so the names differ. This is the mapping:

| Paper | Meaning | Code flag | Value used |
|---|---|---|---|
| lambda, Eq. (5) | FRAG retain-penalty weight | `frag/predictor.py --gamma` | 1.0 |
| epsilon, Eq. (3)-(4) | activation-ratio stabiliser | hard-coded | 1e-6 |
| lambda, Eq. (6) | FRP magnitude prior | `frp/prune.py --beta` | 0.05 |
| beta, Eq. (6) | FRP retain penalty | `frp/prune.py --protect_p` | 0.00 / 0.15 (1B), 0.00 / 0.05 (3B) |
| rho | FRP sparsity | `frp/prune.py --sparsity` | 0.03 |

Note `--beta` in the code is the paper's magnitude prior lambda, **not** the paper's beta.
Table 1's `FRP (beta=0.15)` row is `--protect_p 0.15`.

Two places where the implementation is more specific than Eq. (6):

- The first term ranks the raw activation ratio `x_f / x_r` rather than
  `F = |W| * x_f / x_r`; the magnitude enters only through the `lambda * rank(|W|)` term.
  This matches the description in Appendix B.3 ("rank-space combination of activation
  ratio r and |W|").
- The retain penalty is applied as a hard veto rather than a subtracted rank: the
  top-`protect_p` fraction of retain-exclusive weights per row (scored by
  `|W| * x_r / x_f`, the R direction of Eq. (4)) is excluded from pruning outright.
  `--protect_p 0` recovers plain FRP.

## FRAG on your own checkpoint

Self-contained; needs only an original model, an unlearned checkpoint, and two text files:

```bash
python frag/predictor.py \
  --original open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --unlearned /path/to/unlearned_checkpoint \
  --forget_file calib/forget.txt --retain_file calib/retain.txt
```

Defaults follow Appendix A.1: 128 calibration sequences of 256 tokens, all seven
projections per block, gamma = 1. It prints `cosI` and `cosJ` (the two alignment terms of
Eq. 5) and `FRAG = cosI - gamma * cosJ`. Roughly 25 s at 1B, 4 min at 14B on one A6000.

## Table 1 — TOFU relearning robustness

Three stages. Each is one row of the table, for one model scale and one forget split.

```bash
# 1. baselines: GA, GradDiff, NPO, RMU, and Selective Pruning
for M in GradAscent GradDiff NPO RMU SP; do
  METHOD=$M SCALE=1B FORGET_SPLIT=forget10 GPU=0 bash scripts/unlearn_baselines.sh
done

# 2. FRP and FRP-RV (P is the paper's beta, the retain veto)
SCALE=1B P=0    FS=forget10 GPU=0 bash scripts/frp_tofu.sh
SCALE=1B P=0.15 FS=forget10 GPU=0 bash scripts/frp_tofu.sh

# 3. the forget and forget+retain attacks against any checkpoint from step 1
#    (unlearn_baselines.sh already runs the retain attack; frp_tofu.sh runs all three)
for A in forget forget_retain; do
  CKPT=<checkpoint> NAME=<tag> SCALE=1B FS=forget10 ATTACK=$A EPOCHS=1 LR=1e-5 GPU=0 \
    bash scripts/attack.sh
done
```

Repeat for `forget01`, `forget05`, `forget10` and for `SCALE=3B`; Table 1 averages the
three splits (Appendix B.2). FRP cells use zero-out pruning
(`--prune_mode shrink --shrink_factor 0`), 3% MLP sparsity and 400 calibration sequences.

The predictor columns come from cached activation norms:

```bash
python analysis/cache_norms.py                                    # GPU, ~10 min
python analysis/compute_predictors.py --scale 1B --csv results/predictors_1b.csv
python analysis/compute_predictors.py --scale 3B --csv results/predictors_3b.csv
```

## Table 2 and Appendix C.1 — WMDP-cyber

```bash
python frp/prune.py --model Qwen/Qwen2.5-14B-Instruct --dataset wmdp_cyber \
  --scoring rank_combo --beta 0.05 --sparsity 0.03 --dtype fp16 \
  --output_dir saves/unlearn/frp_wmdp_qwen14b
python scripts/relearn_wmdp.py --model saves/unlearn/frp_wmdp_qwen14b \
  --data data/wmdp/wmdp-corpora/cyber-retain-corpus.jsonl --epochs 1
python analysis/qwen14b_frag.py
```

Accuracy on WMDP-cyber and 5-shot MMLU are measured with lm-eval-harness through
OpenUnlearning's WMDP evaluator.

## Table 3 — does FRAG predict robustness better than distance?

Spearman rho between each predictor and post-attack Delta-ES under the forget+retain
attack, over healthy checkpoints only (GA is excluded: its pre-attack utility collapses to
zero, so its Delta-ES measures recovery from a broken model). The `w/o FRP` column drops
every FRP checkpoint, which rules out FRAG being credited for ranking the method built on
the same principle.

```bash
python analysis/table3_spearman.py \
  --predictors results/predictors_1b.csv results/predictors_3b.csv \
  --results results/table1_tofu_per_split.csv
```

**Read this before comparing against the printed table.** The script reports FRAG as
Eq. (5) defines it — the lambda = 1 gap, `cos(F, dW^2) - cos(R, dW^2)` — and also the two
cosine terms on their own. The global-L2 column reproduces the paper exactly, including the
sign flip at 3B, which confirms the checkpoint set and the ground truth are the ones the
table was built from. The FRAG column does not reproduce: the published values track the
single forget-alignment cosine (lambda = 0) rather than the gap.

| | L2 all | L2 w/o | FRAG all | FRAG w/o | cos(F) all | cos(F) w/o | printed all | printed w/o |
|---|---|---|---|---|---|---|---|---|
| 1B     | -0.56 | -0.36 | -0.66 | -0.34 | -0.91 | -0.83 | -0.92 | -0.85 |
| 3B     | -0.16 | +0.13 | -0.62 | -0.36 | -0.77 | -0.68 | -0.72 | -0.71 |
| Pooled | -0.36 | -0.10 | -0.60 | -0.31 | -0.76 | -0.67 | -0.78 | -0.74 |

The qualitative claim survives the correction: with the lambda = 1 gap, FRAG still beats
global L2 pooled (-0.60 vs -0.36, and -0.31 vs -0.10 without FRP), and L2 still loses the
sign at 3B while FRAG does not. What changes is the margin, and the 1B `w/o FRP` cell,
where the gap (-0.34) and L2 (-0.36) are effectively tied.

Two further points the extra columns make visible. `cos(R, dW^2)` — the *retain*-alignment
term alone — scores -0.68 pooled, about as well as `cos(F, dW^2)` at -0.76; on healthy
checkpoints the two terms are nearly collinear, so a result from either one alone does not
separate forget-critical from retain-critical placement. And `frag/predictor.py` itself
documents lambda = 0 as unsuitable, because it rewards utility-collapsed edits — the exact
failure mode Appendix A.2 is about. The gap is the quantity the method's own argument calls
for, and it is the one Table 1's FRAG column uses.

## Figure 2 and Appendix A.2 — why distance can be spoofed

Isotropic noise added to every MLP weight moves the model as far as FRP does while
unlearning nothing. Global L2 and each individual cosine term rank it as robust; the
FRAG gap does not.

```bash
bash scripts/noise_controls.sh          # sigma = 0.002 and 0.004 controls, then attack them
python analysis/banded_selectivity.py --alpha 1.0 --out saves/unlearn/band_a100
```

`banded_selectivity.py` is the matched-L2 control: it zeroes the same fraction of weights
drawn from the same magnitude distribution for every `alpha`, so displacement is held
essentially constant and only *which* weights are chosen changes.

## Appendix B.3 — ablations

```bash
bash scripts/ablation_scoring.sh       # Table 5, scoring rule
bash scripts/sweep_magnitude_prior.sh  # Table 6, magnitude prior (paper lambda = code --beta)
bash scripts/sweep_sparsity.sh         # Table 7, sparsity
```

## Results shipped with this repository

`results/` holds the numbers this code produces, so a fresh run can be diffed against them:

| File | Contents |
|---|---|
| `table1_tofu_avg.csv` | Table 1, averaged over the three forget splits |
| `table1_tofu_per_split.csv` | the same runs split by forget01/05/10 (Appendix C.2) |
| `predictors_1b.csv`, `predictors_3b.csv` | per-split global L2, FRAG (%), and both cosine terms |
| `table3_spearman.csv` | Table 3, with the published column alongside for comparison |

These were rebuilt from the evaluation JSONs and checkpoints of the reported runs, not
copied from the paper. Where the two disagree, the CSV is the recomputation and the
difference is recorded here rather than edited away:

- **Table 1** reproduces in full — every cell of both the 1B and 3B blocks, ES, utility and
  Delta-ES alike, to within rounding, and the FRAG column to the printed digit.
- **Two cells of the Table 1 L2 column** differ: 3B NPO (1.87 here, 1.51 printed) and 3B RMU
  (2.08 vs 1.49). The printed values came from stored `measure_l2.py` records for those
  runs, and the forget01/forget05 checkpoints now on disk no longer match them; forget10
  agrees exactly for both, and GA, GradDiff and SP agree throughout. Nothing in the paper
  turns on these two numbers — both are "small L2" either way.
- **Table 3's FRAG column** does not reproduce as Eq. (5) defines FRAG; see the Table 3
  section above.
