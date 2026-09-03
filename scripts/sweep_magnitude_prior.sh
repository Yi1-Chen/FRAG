#!/usr/bin/env bash
# Table 6: FRP magnitude-prior sweep on LLaMA-3.2-1B / TOFU forget10.
#
# NOTE ON NAMING. This sweeps the code flag --beta, which is the paper's magnitude
# prior lambda in Eq. (6), NOT the paper's beta (that is the retain penalty, which the
# code applies as --protect_p). See docs/REPRODUCE.md for the full symbol mapping.
#
# Fixed: 3% MLP sparsity, weights zeroed, seed 42, 400 calibration sequences of 256
# tokens. Variable: --beta in {0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00}; 0.05 is the
# headline value and is reused from the scoring ablation if that run is already present.
#
# Per point: prune -> eval -> L2 -> 1 epoch retain-split relearn -> eval -> L2 ->
# delete the relearned checkpoint. FRAG is scored for all points at the end.
# Work is distributed across GPUs 0-3, two jobs per GPU.

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
TAG=tofu_1B_betasweep
LOGDIR=$OUTDIR/logs/${TAG}
L2DIR=$OUTDIR/l2_per_task
mkdir -p "$LOGDIR" "$L2DIR" saves/unlearn saves/eval saves/finetune

ts() { date +%H:%M:%S; }

# tag<->beta map: TAG_b005 for β=0.05 etc (β value × 1000, 4-digit pad)
beta_label() { printf "b%04d" "$(echo "$1*1000" | bc | cut -d. -f1)"; }

run_row() {
  local GPU=$1; shift
  local BETA=$1; shift
  local LABEL=$(beta_label "$BETA")
  local UNLEARN_DIR="saves/unlearn/${TAG}_${LABEL}"
  local EVAL_PRE="saves/eval/${TAG}_${LABEL}_pre"
  local RELEARN_DIR="saves/finetune/${TAG}_${LABEL}_retain1ep"
  local EVAL_POST="saves/eval/${TAG}_${LABEL}_post"
  local L2_PRE_OUT="${L2DIR}/${TAG}_${LABEL}_pre.json"
  local L2_POST_OUT="${L2DIR}/${TAG}_${LABEL}_post.json"
  local LOG="${LOGDIR}/${LABEL}.log"

  log() { echo "[$(ts)] [GPU${GPU} β=${BETA}] $1" | tee -a "$LOG"; }
  log "START row"

  # 1. PRUNE
  if [[ -f "${UNLEARN_DIR}/config.json" ]]; then
    log "SKIP prune (exists)"
  else
    log "START prune"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$FRAG_REPO/frp/prune.py" \
      --model "$BASE_LOCAL" \
      --dataset tofu --forget_split forget10 --retain_split retain90 \
      --scoring rank_combo --beta "$BETA" --sparsity 0.03 --target mlp \
      --prune_mode zero --seed 42 --dtype fp32 \
      --output_dir "$UNLEARN_DIR" >> "$LOG" 2>&1
    [[ $? -ne 0 ]] && { log "FAIL prune"; return 1; }
    log "DONE prune"
  fi

  # 2. EVAL PRE
  if [[ -f "${EVAL_PRE}/TOFU_SUMMARY.json" ]]; then
    log "SKIP eval-pre"
  else
    log "START eval-pre"
    CUDA_VISIBLE_DEVICES=$GPU python src/eval.py --config-name=eval.yaml \
      experiment=eval/tofu/default.yaml model="$MODEL_CFG" \
      model.model_args.pretrained_model_name_or_path="$UNLEARN_DIR" \
      model.model_args.attn_implementation=eager \
      paths.output_dir="$EVAL_PRE" \
      task_name="${TAG}_${LABEL}_pre" >> "$LOG" 2>&1
    [[ $? -ne 0 ]] && { log "FAIL eval-pre"; return 1; }
    log "DONE eval-pre"
  fi

  # 3. L2 PRE
  [[ ! -f "$L2_PRE_OUT" ]] && "$PYTHON" "$FRAG_REPO/analysis/measure_l2.py" \
    --unlearned "$UNLEARN_DIR" --original "$BASE_LOCAL" \
    --label "${TAG}_${LABEL}_pre" --quiet --out "$L2_PRE_OUT" >> "$LOG" 2>&1

  # 4. RELEARN
  if [[ -f "${RELEARN_DIR}/config.json" ]]; then
    log "SKIP relearn"
  else
    log "START relearn (retain, 1ep)"
    CUDA_VISIBLE_DEVICES=$GPU python src/train.py --config-name=train.yaml \
      experiment=finetune/tofu/default model="$MODEL_CFG" \
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
    log "SKIP eval-post"
  else
    log "START eval-post"
    CUDA_VISIBLE_DEVICES=$GPU python src/eval.py --config-name=eval.yaml \
      experiment=eval/tofu/default.yaml model="$MODEL_CFG" \
      model.model_args.pretrained_model_name_or_path="$RELEARN_DIR" \
      model.model_args.attn_implementation=eager \
      paths.output_dir="$EVAL_POST" \
      task_name="${TAG}_${LABEL}_post" >> "$LOG" 2>&1
    [[ $? -ne 0 ]] && { log "FAIL eval-post"; return 1; }
    log "DONE eval-post"
  fi

  # 6. L2 POST
  [[ ! -f "$L2_POST_OUT" ]] && "$PYTHON" "$FRAG_REPO/analysis/measure_l2.py" \
    --unlearned "$RELEARN_DIR" --original "$BASE_LOCAL" \
    --label "${TAG}_${LABEL}_post" --quiet --out "$L2_POST_OUT" >> "$LOG" 2>&1

  # 7. CLEANUP RELEARN
  if [[ -f "${EVAL_POST}/TOFU_SUMMARY.json" ]] && [[ -d "$RELEARN_DIR" ]]; then
    log "CLEANUP rm RELEARN_DIR"
    rm -rf "$RELEARN_DIR"
  fi

  log "ALL DONE row"
  return 0
}

# β=0.05 already done as §4.3 FRIP — symlink so collector finds it under
# the betasweep naming scheme without re-running.
if [[ ! -e "saves/unlearn/${TAG}_b0050" ]]; then
  ln -s "$(pwd)/saves/unlearn/tofu_1B_ablation43_scoring_FRIP" "saves/unlearn/${TAG}_b0050"
fi
if [[ ! -e "saves/eval/${TAG}_b0050_pre" ]]; then
  ln -s "$(pwd)/saves/eval/tofu_1B_ablation43_scoring_FRIP_pre" "saves/eval/${TAG}_b0050_pre"
fi
if [[ ! -e "saves/eval/${TAG}_b0050_post" ]]; then
  ln -s "$(pwd)/saves/eval/tofu_1B_ablation43_scoring_FRIP_post" "saves/eval/${TAG}_b0050_post"
fi
if [[ ! -e "${L2DIR}/${TAG}_b0050_pre.json" ]]; then
  ln -s "$(pwd)/${L2DIR}/tofu_1B_ablation43_scoring_FRIP_pre.json" "${L2DIR}/${TAG}_b0050_pre.json"
fi
if [[ ! -e "${L2DIR}/${TAG}_b0050_post.json" ]]; then
  ln -s "$(pwd)/${L2DIR}/tofu_1B_ablation43_scoring_FRIP_post.json" "${L2DIR}/${TAG}_b0050_post.json"
fi

queue_gpu0() { run_row 0 0.00; run_row 0 0.50; }
queue_gpu1() { run_row 1 0.01; run_row 1 1.00; }
queue_gpu2() { run_row 2 0.10; }
queue_gpu3() { run_row 3 0.25; }

echo "==================== START $(date) ===================="
queue_gpu0 > "${LOGDIR}/queue_gpu0.log" 2>&1 & P0=$!
queue_gpu1 > "${LOGDIR}/queue_gpu1.log" 2>&1 & P1=$!
queue_gpu2 > "${LOGDIR}/queue_gpu2.log" 2>&1 & P2=$!
queue_gpu3 > "${LOGDIR}/queue_gpu3.log" 2>&1 & P3=$!
echo "PIDs: $P0 $P1 $P2 $P3"
wait $P0 $P1 $P2 $P3
echo "==================== END $(date) ===================="
