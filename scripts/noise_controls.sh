#!/bin/bash
# Appendix A.2 / Figure 2: the direction-blind noise control.
#   inject Gaussian noise -> [optional retain-repair 1ep] -> eval pure
#   -> retain-only mask-free 1ep relearn -> eval post-relearn
#
# Usage:
#   SIGMA=0.004 GPU=0 bash scripts/noise_controls.sh         # N1 (pure noise)
#   SIGMA=0.004 WD=1 GPU=1 bash scripts/noise_controls.sh    # N2 (noise + retain-repair = Siddiqui Weight Distortion)
#
# Reference: Siddiqui et al. 2025 §6.1.
#   - Weight Distortion (WD)  = isotropic Gaussian noise on weights + retain finetune (this script with WD=1)
#   - Weight Dist Reg  (WDR)  = training-time loss regularizer maximizing ||θ − θ₀||₂ (NOT implemented here)
#
# Required env:
#   SIGMA     -- Gaussian noise std (e.g. 0.002, 0.004, 0.008)
# Optional env:
#   WD        -- 1 to run retain-repair 1ep on noise ckpt (default 0). Implements Siddiqui WD.
#   GPU       -- CUDA device id (default 0)
#   SCOPE     -- mlp or all (default mlp; matches RC scope)
#   SEED      -- seed for noise (default 42)
#   SCALE     -- 1B only for now (default 1B)
#   TASK      -- override task name

set -u
FRAG_REPO="${FRAG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OU="${OPEN_UNLEARNING:-$HOME/open-unlearning}"
PYTHON="${PYTHON:-$OU/.venv/bin/python}"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
OUTDIR="${OUTDIR:-$FRAG_REPO/runs}"
cd "$OU"

SIGMA="${SIGMA:?need SIGMA}"
WD="${WD:-${WDR:-0}}"   # WD enables retain-repair 1ep (= Siddiqui Weight Distortion). WDR kept as legacy alias.
GPU="${GPU:-0}"
SCOPE="${SCOPE:-mlp}"
SEED="${SEED:-42}"
SCALE="${SCALE:-1B}"
export CUDA_VISIBLE_DEVICES=$GPU

FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"

case "$SCALE" in
  1B) MODEL_KEY=Llama-3.2-1B-Instruct; MODEL_CONFIG=Llama-3.2-1B-Instruct ;;
  *)  echo "Only SCALE=1B supported for noise experiment"; exit 1 ;;
esac

# SCOPE may be mlp | attn | embed | all (canonical_noise.sh forwards to inject_noise.py)
FULL="${PRETRAINED_DIR}/tofu_${MODEL_KEY}_full"

# Tag e.g. sigma=0.004 scope=mlp -> noise_1B_sigma0p0040_mlp
SIGMA_TAG=$(python -c "print(f'{float(${SIGMA}):.4f}'.replace('.', 'p'))")
WD_TAG=""
[[ "$WD" == "1" ]] && WD_TAG="_wd"
TASK="${TASK:-noise_${SCALE}_sigma${SIGMA_TAG}_${SCOPE}${WD_TAG}}"

NOISE_DIR=saves/unlearn/${TASK}_raw
UNLEARN_DIR=saves/unlearn/${TASK}
EVAL_PURE=saves/eval/${TASK}_pure
RELEARN_DIR=saves/finetune/${TASK}_retain1ep
EVAL_POST=saves/eval/${TASK}_postrelearnretain1ep

LOGDIR=logs/canonical_noise
mkdir -p $LOGDIR saves/unlearn saves/eval saves/finetune
LOG=$LOGDIR/${TASK}.log

ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] sigma=$SIGMA wd=$WD scope=$SCOPE GPU$GPU $1" | tee -a $LOG; }

#=============================================================
# 0. DISK PREFLIGHT
#=============================================================
AVAIL=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [[ $AVAIL -lt 8 ]]; then
  log "ABORT: ${AVAIL}G free, need 8G"
  exit 1
fi
log "PREFLIGHT ok (${AVAIL}G free)"

#=============================================================
# 1. INJECT NOISE
#=============================================================
if [[ -f $NOISE_DIR/config.json ]]; then
  log "SKIP noise-inject (exists)"
else
  log "START noise-inject sigma=$SIGMA scope=$SCOPE seed=$SEED"
  $PYTHON "$FRAG_REPO/baselines/inject_noise.py" \
    --input_dir $FULL \
    --output_dir $NOISE_DIR \
    --sigma $SIGMA --scope $SCOPE --seed $SEED \
    --dtype bfloat16 >> $LOG 2>&1
  RC=$?
  if [[ $RC -ne 0 ]]; then
    log "FAIL noise-inject (exit=$RC)"
    rm -rf $NOISE_DIR
    exit $RC
  fi
  log "DONE noise-inject"
fi

#=============================================================
# 2. OPTIONAL WD REPAIR (Siddiqui Weight Distortion — noise + retain finetune)
#=============================================================
if [[ "$WD" == "1" ]]; then
  if [[ -f $UNLEARN_DIR/config.json ]]; then
    log "SKIP wd-repair (exists)"
  else
    log "START wd-repair (retain 1ep)"
    $PYTHON src/train.py --config-name=train.yaml \
      experiment=finetune/tofu/default model=$MODEL_CONFIG \
      model.model_args.pretrained_model_name_or_path=$NOISE_DIR \
      model.model_args.attn_implementation=eager \
      data/datasets@data.train=TOFU_QA_${RETAIN_SPLIT} \
      trainer.args.num_train_epochs=1 \
      trainer.args.eval_on_start=false \
      ++trainer.args.remove_unused_columns=false \
      trainer.args.seed=$SEED \
      paths.output_dir=$UNLEARN_DIR task_name=${TASK}_wd_repair >> $LOG 2>&1
    RC=$?
    if [[ $RC -ne 0 ]]; then
      log "FAIL wd-repair (exit=$RC)"
      rm -rf $UNLEARN_DIR
      exit $RC
    fi
    log "DONE wd-repair"
  fi
else
  # No repair: point UNLEARN_DIR -> NOISE_DIR via symlink (avoid 2x disk).
  if [[ ! -e $UNLEARN_DIR ]]; then
    ln -s "$(realpath $NOISE_DIR)" $UNLEARN_DIR
    log "SYMLINK UNLEARN_DIR -> NOISE_DIR (no WD repair)"
  fi
fi

#=============================================================
# 3. EVAL PURE (pre-attack)
#=============================================================
if [[ -f $EVAL_PURE/TOFU_EVAL.json ]]; then
  log "SKIP eval-pure (exists)"
else
  log "START eval-pure"
  $PYTHON src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default model=$MODEL_CONFIG \
    model.model_args.pretrained_model_name_or_path=$UNLEARN_DIR \
    model.model_args.attn_implementation=eager \
    forget_split=$FORGET_SPLIT \
    paths.output_dir=$EVAL_PURE task_name=${TASK}_pure >> $LOG 2>&1
  log "DONE eval-pure"
fi

#=============================================================
# 4. RETAIN-ONLY MASK-FREE 1EP RELEARN ATTACK
#=============================================================
if [[ -f $RELEARN_DIR/config.json ]]; then
  log "SKIP relearn (exists)"
else
  log "START relearn (retain 1ep mask-free)"
  $PYTHON src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default model=$MODEL_CONFIG \
    model.model_args.pretrained_model_name_or_path=$UNLEARN_DIR \
    model.model_args.attn_implementation=eager \
    data/datasets@data.train=TOFU_QA_${RETAIN_SPLIT} \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_on_start=false \
    ++trainer.args.remove_unused_columns=false \
    trainer.args.seed=42 \
    paths.output_dir=$RELEARN_DIR task_name=${TASK}_retain1ep >> $LOG 2>&1
  RC=$?
  if [[ $RC -ne 0 ]]; then
    log "FAIL relearn (exit=$RC)"
    rm -rf $RELEARN_DIR
    exit $RC
  fi
  log "DONE relearn"
fi

#=============================================================
# 5. EVAL POST-RELEARN
#=============================================================
if [[ -f $EVAL_POST/TOFU_EVAL.json ]]; then
  log "SKIP eval-post (exists)"
else
  log "START eval-post"
  $PYTHON src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default model=$MODEL_CONFIG \
    model.model_args.pretrained_model_name_or_path=$RELEARN_DIR \
    model.model_args.attn_implementation=eager \
    forget_split=$FORGET_SPLIT \
    paths.output_dir=$EVAL_POST task_name=${TASK}_postrelearn >> $LOG 2>&1
  log "DONE eval-post"
fi

#=============================================================
# 6. CLEANUP (gated on eval-post completeness)
#=============================================================
SUMMARY=$EVAL_POST/TOFU_SUMMARY.json
if [[ -f $SUMMARY ]] && grep -q extraction_strength $SUMMARY 2>/dev/null; then
  if [[ -d $RELEARN_DIR ]] && [[ "${KEEP_RELEARN_DIR:-0}" != "1" ]]; then
    log "CLEANUP rm RELEARN_DIR=$RELEARN_DIR"
    rm -rf $RELEARN_DIR
  fi
  log "ALL DONE"
else
  log "FINAL incomplete (no SUMMARY) — kept all ckpts"
fi
