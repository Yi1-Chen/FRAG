#!/usr/bin/env bash
# §4.3 FLP Ablation — full pipeline on LLaMA-3.2-1B + TOFU forget10.
#
# Two ablation tables:
#   A) Scoring (magnitude-duality):  5 variants at 3% MLP sparsity
#      - MagOnly  : s = |W|                       (classical magnitude pruning)
#      - WandaFwd : s = |W|·||X^f||               (upstream Wanda, retain-blind)
#      - FRIA_I   : s = |W|·r                     (naive FRIA importance — §3.3 claim)
#      - RankR0   : s = rank(r)                   (selectivity rank, no magnitude)
#      - FRIP     : s = rank(r) + 0.05·rank(|W|)  (ours)
#   B) Target layer (matched ~29M params/layer):  3 variants — re-uses existing prune ckpts
#      - MLP_match  (3.6%) / Attn_match (17.4%) / All_match (3.0%)
#      For target table, prune+eval-pre already done; this script adds relearn + eval-post + L2.
#
# Per row pipeline: [prune] → eval-pre → L2-pre → 1ep retain relearn → eval-post → L2-post → cleanup RELEARN_DIR.
# Distributes 8 jobs across GPU 0 and GPU 1 in two sequential queues.
#
# Run via tmux on kun1.  Single script, no external infra.

set -u
cd ~/open-unlearning
FRAG_REPO="${FRAG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OU="${OPEN_UNLEARNING:-$HOME/open-unlearning}"
PYTHON="${PYTHON:-$OU/.venv/bin/python}"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
OUTDIR="${OUTDIR:-$FRAG_REPO/runs}"
cd "$OU"

BASE_LOCAL=$(ls -d "$HUB"/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots/*/ 2>/dev/null | head -1 | sed "s:/$::")
MODEL_CFG="Llama-3.2-1B-Instruct"
TAG=tofu_1B_ablation43
LOGDIR=$OUTDIR/logs/${TAG}
L2DIR=$OUTDIR/l2_per_task
mkdir -p "$LOGDIR" "$L2DIR" saves/unlearn saves/eval saves/finetune

ts() { date +%H:%M:%S; }

# ---- per-row pipeline ------------------------------------------------
# Args: GPU LABEL UNLEARN_DIR SKIP_PRUNE PRUNE_ARGS...
# SKIP_PRUNE=1 means UNLEARN_DIR already populated (target backfill case).
run_row() {
  local GPU=$1; shift
  local LABEL=$1; shift
  local UNLEARN_DIR=$1; shift
  local SKIP_PRUNE=$1; shift
  local PRUNE_ARGS="$*"

  local LOG="${LOGDIR}/${LABEL}.log"
  local EVAL_PRE=saves/eval/${TAG}_${LABEL}_pre
  local RELEARN_DIR=saves/finetune/${TAG}_${LABEL}_retain1ep
  local EVAL_POST=saves/eval/${TAG}_${LABEL}_post
  local L2_PRE_OUT=${L2DIR}/${TAG}_${LABEL}_pre.json
  local L2_POST_OUT=${L2DIR}/${TAG}_${LABEL}_post.json

  log() { echo "[$(ts)] [GPU${GPU} ${LABEL}] $1" | tee -a "$LOG"; }

  log "START row (skip_prune=${SKIP_PRUNE})"

  # 1. PRUNE
  if [[ "$SKIP_PRUNE" == "0" ]]; then
    if [[ -f "${UNLEARN_DIR}/config.json" ]]; then
      log "SKIP prune (exists)"
    else
      log "START prune  args=${PRUNE_ARGS}"
      CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FRAG_REPO/frp/prune.py" \
        --model "$BASE_LOCAL" \
        --dataset tofu --forget_split forget10 --retain_split retain90 \
        --prune_mode zero --seed 42 --dtype fp32 \
        --output_dir "$UNLEARN_DIR" \
        $PRUNE_ARGS >> "$LOG" 2>&1
      [[ $? -ne 0 ]] && { log "FAIL prune"; return 1; }
      log "DONE prune"
    fi
  fi

  # 2. EVAL PRE
  if [[ -f "${EVAL_PRE}/TOFU_SUMMARY.json" ]]; then
    log "SKIP eval-pre (exists)"
  else
    log "START eval-pre"
    CUDA_VISIBLE_DEVICES=$GPU python src/eval.py --config-name=eval.yaml \
      experiment=eval/tofu/default.yaml \
      model="$MODEL_CFG" \
      model.model_args.pretrained_model_name_or_path="$UNLEARN_DIR" \
      model.model_args.attn_implementation=eager \
      paths.output_dir="$EVAL_PRE" \
      task_name="${TAG}_${LABEL}_pre" >> "$LOG" 2>&1
    [[ $? -ne 0 ]] && { log "FAIL eval-pre"; return 1; }
    log "DONE eval-pre"
  fi

  # 3. L2 PRE
  if [[ ! -f "$L2_PRE_OUT" ]]; then
    log "MEASURE L2-pre"
    "$PYTHON" "$FRAG_REPO/analysis/measure_l2.py" --unlearned "$UNLEARN_DIR" --original "$BASE_LOCAL" \
      --label "${TAG}_${LABEL}_pre" --quiet --out "$L2_PRE_OUT" >> "$LOG" 2>&1 \
      || log "L2-pre failed (non-fatal)"
  fi

  # 4. RELEARN 1EP RETAIN
  if [[ -f "${RELEARN_DIR}/config.json" ]]; then
    log "SKIP relearn (exists)"
  else
    log "START relearn (retain90, 1ep)"
    CUDA_VISIBLE_DEVICES=$GPU python src/train.py --config-name=train.yaml \
      experiment=finetune/tofu/default \
      model="$MODEL_CFG" \
      model.model_args.pretrained_model_name_or_path="$UNLEARN_DIR" \
      model.model_args.attn_implementation=eager \
      data/datasets@data.train=TOFU_QA_retain \
      trainer.args.num_train_epochs=1 \
      trainer.args.eval_on_start=false \
      ++trainer.args.remove_unused_columns=false \
      ++trainer.args.optim=adamw_torch \
      trainer.args.seed=42 \
      paths.output_dir="$RELEARN_DIR" \
      task_name="${TAG}_${LABEL}_retain1ep" >> "$LOG" 2>&1
    [[ $? -ne 0 ]] && { log "FAIL relearn"; rm -rf "$RELEARN_DIR"; return 1; }
    log "DONE relearn"
  fi

  # 5. EVAL POST
  if [[ -f "${EVAL_POST}/TOFU_SUMMARY.json" ]]; then
    log "SKIP eval-post (exists)"
  else
    log "START eval-post"
    CUDA_VISIBLE_DEVICES=$GPU python src/eval.py --config-name=eval.yaml \
      experiment=eval/tofu/default.yaml \
      model="$MODEL_CFG" \
      model.model_args.pretrained_model_name_or_path="$RELEARN_DIR" \
      model.model_args.attn_implementation=eager \
      paths.output_dir="$EVAL_POST" \
      task_name="${TAG}_${LABEL}_post" >> "$LOG" 2>&1
    [[ $? -ne 0 ]] && { log "FAIL eval-post"; return 1; }
    log "DONE eval-post"
  fi

  # 6. L2 POST
  if [[ ! -f "$L2_POST_OUT" ]]; then
    log "MEASURE L2-post"
    "$PYTHON" "$FRAG_REPO/analysis/measure_l2.py" --unlearned "$RELEARN_DIR" --original "$BASE_LOCAL" \
      --label "${TAG}_${LABEL}_post" --quiet --out "$L2_POST_OUT" >> "$LOG" 2>&1 \
      || log "L2-post failed (non-fatal)"
  fi

  # 7. CLEANUP RELEARN_DIR (only if eval-post succeeded)
  if [[ -f "${EVAL_POST}/TOFU_SUMMARY.json" ]] && [[ -d "$RELEARN_DIR" ]]; then
    log "CLEANUP rm RELEARN_DIR"
    rm -rf "$RELEARN_DIR"
  fi

  log "ALL DONE row"
  return 0
}

# ---- queue dispatch --------------------------------------------------
# GPU 0 queue: 4 rows (scoring 1,3,5 + target backfill 1)
# GPU 1 queue: 4 rows (scoring 2,4 + target backfill 2,3)
queue_gpu0() {
  local GPU=0
  run_row $GPU "scoring_MagOnly" \
    "saves/unlearn/${TAG}_scoring_MagOnly" 0 \
    "--scoring magnitude_only --sparsity 0.03 --target mlp"
  run_row $GPU "scoring_FRIA_I" \
    "saves/unlearn/${TAG}_scoring_FRIA_I" 0 \
    "--scoring fria_importance --sparsity 0.03 --target mlp"
  run_row $GPU "scoring_FRIP" \
    "saves/unlearn/${TAG}_scoring_FRIP" 0 \
    "--scoring rank_combo --beta 0.05 --sparsity 0.03 --target mlp"
  run_row $GPU "target_MLP" \
    "saves/unlearn/tofu_1B_FRIP_targetAbl_matched_MLP" 1 ""
}
queue_gpu1() {
  local GPU=1
  run_row $GPU "scoring_WandaFwd" \
    "saves/unlearn/${TAG}_scoring_WandaFwd" 0 \
    "--scoring wanda_forget_only --sparsity 0.03 --target mlp"
  run_row $GPU "scoring_RankR0" \
    "saves/unlearn/${TAG}_scoring_RankR0" 0 \
    "--scoring rank_combo --beta 0.0 --sparsity 0.03 --target mlp"
  run_row $GPU "target_Attn" \
    "saves/unlearn/tofu_1B_FRIP_targetAbl_matched_Attn" 1 ""
  run_row $GPU "target_All" \
    "saves/unlearn/tofu_1B_FRIP_targetAbl_matched_All" 1 ""
}

echo "==================== START $(date) ===================="
queue_gpu0 > "${LOGDIR}/queue_gpu0.log" 2>&1 &
PID0=$!
queue_gpu1 > "${LOGDIR}/queue_gpu1.log" 2>&1 &
PID1=$!
echo "GPU0 queue PID=$PID0  GPU1 queue PID=$PID1"
wait $PID0; RC0=$?
wait $PID1; RC1=$?
echo "GPU0 rc=$RC0  GPU1 rc=$RC1"

echo "==================== RESULTS $(date) ===================="
for L in scoring_MagOnly scoring_WandaFwd scoring_FRIA_I scoring_RankR0 scoring_FRIP target_MLP target_Attn target_All; do
  echo ""
  echo "--- $L ---"
  pre=saves/eval/${TAG}_${L}_pre/TOFU_SUMMARY.json
  post=saves/eval/${TAG}_${L}_post/TOFU_SUMMARY.json
  # Target rows have pre-eval in old location
  if [[ "$L" == target_* ]]; then
    pre=saves/eval/eval_tofu_1B_FRIP_targetAbl_matched_${L#target_}/TOFU_SUMMARY.json
  fi
  l2pre=${L2DIR}/${TAG}_${L}_pre.json
  l2post=${L2DIR}/${TAG}_${L}_post.json
  python - <<PY
import json, os
def load(p):
    if not os.path.exists(p): return None
    try: return json.load(open(p))
    except: return None
pre  = load("$pre")
post = load("$post")
l2p  = load("$l2pre")
l2q  = load("$l2post")
def g(d,k):
    return None if d is None else d.get(k)
es_pre  = g(pre, "extraction_strength")
u_pre   = g(pre, "model_utility")
es_post = g(post,"extraction_strength")
u_post  = g(post,"model_utility")
des = (es_post - es_pre) if (es_pre is not None and es_post is not None) else None
l2p_v = g(l2p, "total_l2")
l2q_v = g(l2q, "total_l2")
print(f"  pre  ES={es_pre}  U={u_pre}")
print(f"  post ES={es_post}  U={u_post}")
print(f"  dES={des}")
print(f"  L2 pre={l2p_v}  post={l2q_v}")
PY
done
echo ""
echo "==================== END $(date) ===================="
