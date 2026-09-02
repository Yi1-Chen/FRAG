#!/bin/bash
# Canonical pipeline for ONE (method, scale):
#   unlearn -> eval pure -> retain-only mask-free 1ep relearn -> eval post-relearn
#
# Usage: METHOD=RC SCALE=1B GPU=0 bash canonical_one.sh
#
# Methods: RC, WR, GradAscent, GradDiff, NPO, SimNPO, RMU
# Scales: 1B, 3B, 8B (LLaMA), Qwen1.5B, Qwen3B, Qwen7B
#
# For 7B/8B scales, deletes BOTH RELEARN_DIR and UNLEARN_DIR after eval-post
# succeeds (eval JSONs preserved). For smaller scales, only RELEARN_DIR is
# deleted; UNLEARN_DIR is kept per existing convention.

set -u
FRAG_REPO="${FRAG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OU="${OPEN_UNLEARNING:-$HOME/open-unlearning}"
PYTHON="${PYTHON:-$OU/.venv/bin/python}"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
PRETRAINED_DIR="${PRETRAINED_DIR:-$OU/saves/pretrained}"
cd "$OU"

METHOD="${METHOD:?need METHOD}"
SCALE="${SCALE:?need SCALE}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU

FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
case "$FORGET_SPLIT" in
  forget01) RETAIN_SPLIT=retain99 ;;
  forget05) RETAIN_SPLIT=retain95 ;;
  forget10) RETAIN_SPLIT=retain90 ;;
  *) echo "Bad FORGET_SPLIT=$FORGET_SPLIT"; exit 1 ;;
esac
TASK_SUFFIX=""
[[ "$FORGET_SPLIT" != "forget10" ]] && TASK_SUFFIX="_${FORGET_SPLIT}"

case "$SCALE" in
  1B)      MODEL_KEY=Llama-3.2-1B-Instruct  ; MODEL_CONFIG=Llama-3.2-1B-Instruct  ;;
  3B)      MODEL_KEY=Llama-3.2-3B-Instruct  ; MODEL_CONFIG=Llama-3.2-3B-Instruct  ;;
  8B)      MODEL_KEY=Llama-3.1-8B-Instruct  ; MODEL_CONFIG=Llama-3.1-8B-Instruct  ;;
  Qwen1.5B) MODEL_KEY=Qwen2.5-1.5B-Instruct ; MODEL_CONFIG=Qwen2.5-1.5B-Instruct ;;
  Qwen3B)   MODEL_KEY=Qwen2.5-3B-Instruct   ; MODEL_CONFIG=Qwen2.5-3B-Instruct   ;;
  Qwen7B)   MODEL_KEY=Qwen2.5-7B-Instruct   ; MODEL_CONFIG=Qwen2.5-7B-Instruct   ;;
  *) echo "Bad SCALE=$SCALE" ; exit 1 ;;
esac

# Per-scale minimum free disk required before starting (GB).
case "$SCALE" in
  1B|Qwen1.5B) MIN_FREE_GB=8 ;;
  3B|Qwen3B)   MIN_FREE_GB=10 ;;
  Qwen7B|8B)   MIN_FREE_GB=45 ;;
esac

# Whether to delete UNLEARN_DIR after a successful eval-post (yes for 7B/8B).
case "$SCALE" in
  *) DELETE_UNLEARN_AFTER=${DELETE_UNLEARN_AFTER:-0} ;;
esac

FULL="${PRETRAINED_DIR}/tofu_${MODEL_KEY}_full"

TASK="${TASK:-canonical_${SCALE}_${METHOD}${TASK_SUFFIX}}"
UNLEARN_DIR=saves/unlearn/${TASK}
EVAL_PURE=saves/eval/${TASK}_pure
RELEARN_DIR=saves/finetune/${TASK}_${ATTACK_SPLIT:-retain}1ep
EVAL_POST=saves/eval/${TASK}_postrelearn${ATTACK_SPLIT:-retain}1ep
LOGDIR=logs/canonical
mkdir -p $LOGDIR saves/unlearn saves/eval saves/finetune
LOG=$LOGDIR/${TASK}.log

ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $METHOD/$SCALE/GPU$GPU $1" | tee -a $LOG; }

# Sentinel: if /tmp/xfam/skip_<SCALE>_<METHOD> exists, exit 0 (used to stop a queued
# chain step from racing with a manually launched run of the same method on another GPU).
SENTINEL=/tmp/xfam/skip_${SCALE}_${METHOD}
if [[ -f $SENTINEL ]]; then
  log "SKIP method per sentinel: $SENTINEL"
  exit 0
fi

#=============================================================
# 0. DISK PREFLIGHT
#=============================================================
AVAIL=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [[ $AVAIL -lt $MIN_FREE_GB ]]; then
  log "ABORT: ${AVAIL}G free, need ${MIN_FREE_GB}G for $SCALE"
  exit 1
fi
log "PREFLIGHT ok (${AVAIL}G free, need ${MIN_FREE_GB}G)"

#=============================================================
# 1. UNLEARN
#=============================================================
if [[ -f $UNLEARN_DIR/config.json ]]; then
  log "SKIP unlearn (exists)"
else
  log "START unlearn"
  case "$METHOD" in
    RC) $PYTHON "$FRAG_REPO/frp/prune.py" \
        --model $FULL --dataset tofu --forget_split $FORGET_SPLIT --retain_split $RETAIN_SPLIT \
        --scoring rank_combo --beta 0.05 --sparsity ${SPARSITY:-0.03} \
        --prune_mode ${PRUNE_MODE:-shrink} --shrink_factor ${SHRINK_FACTOR:-0} \
        --noise_scale ${NOISE_SCALE:-0.01} --dtype fp32 \
        --output_dir $UNLEARN_DIR --seed ${SEED:-42} >> $LOG 2>&1
        ;;
    WR) $PYTHON "$FRAG_REPO/frp/prune.py" \
        --model $FULL --dataset tofu --forget_split $FORGET_SPLIT --retain_split $RETAIN_SPLIT \
        --scoring wanda_ratio --sparsity 0.03 --prune_mode zero --dtype fp32 \
        --output_dir $UNLEARN_DIR --seed 42 >> $LOG 2>&1
        ;;
    SP) $PYTHON "$FRAG_REPO/baselines/selective_pruning.py" \
        --model $FULL --dataset tofu --forget_split $FORGET_SPLIT --retain_split $RETAIN_SPLIT \
        --scoring activation_ratio --mlp_frac ${MLP_FRAC:-0.05} \
        --output_dir $UNLEARN_DIR >> $LOG 2>&1
        ;;
    GradAscent|GradDiff|NPO|SimNPO|RMU)
      # 7B/8B gradient methods need gradient_checkpointing + smaller batch to fit in 48GB.
      # NPO/RMU also instantiate a reference model (~2x weights) so need batch=1.
      MEM_EXTRA=()
      case "$SCALE" in
        Qwen7B|8B)
          case "$METHOD" in
            NPO|RMU) MEM_EXTRA=( '++trainer.args.gradient_checkpointing=true' '++trainer.args.per_device_train_batch_size=1' '++trainer.args.gradient_accumulation_steps=8' );;
            *)       MEM_EXTRA=( '++trainer.args.gradient_checkpointing=true' '++trainer.args.per_device_train_batch_size=4' );;
          esac
          ;;
        3B|Qwen3B)
          # 3B-class NPO/RMU instantiate a reference model; eval-during-train OOMs at 48GB without checkpointing.
          case "$METHOD" in
            NPO|RMU) MEM_EXTRA=( '++trainer.args.gradient_checkpointing=true' '++trainer.args.per_device_train_batch_size=1' '++trainer.args.gradient_accumulation_steps=8' );;
          esac
          ;;
      esac
      $PYTHON src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default model=$MODEL_CONFIG \
        model.model_args.pretrained_model_name_or_path=$FULL \
        model.model_args.attn_implementation=eager \
        trainer=$METHOD ++trainer.args.remove_unused_columns=false \
        trainer.args.eval_on_start=false \
        ++trainer.args.optim=adamw_torch \
        forget_split=$FORGET_SPLIT retain_split=$RETAIN_SPLIT \
        "${MEM_EXTRA[@]}" \
        paths.output_dir=$UNLEARN_DIR task_name=$TASK mode=unlearn >> $LOG 2>&1
      RC=$?
      if [[ $RC -ne 0 ]]; then
        log "FAIL unlearn (exit=$RC) — removing partial UNLEARN_DIR"
        rm -rf $UNLEARN_DIR
        exit $RC
      fi
      ;;
    *) log "BAD method"; exit 1 ;;
  esac
  log "DONE unlearn"
fi

#=============================================================
# 2. EVAL PURE (TOFU + MMLU)
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
# 3. RETAIN-ONLY MASK-FREE 1EP RELEARN
#=============================================================
if [[ -f $RELEARN_DIR/config.json ]]; then
  log "SKIP relearn (exists)"
else
  log "START relearn (retain-only, mask-free, 1ep)"
  RELEARN_EXTRA=()
  case "$SCALE" in
    Qwen7B|8B) RELEARN_EXTRA=( '++trainer.args.gradient_checkpointing=true' );;
  esac
  $PYTHON src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default model=$MODEL_CONFIG \
    model.model_args.pretrained_model_name_or_path=$UNLEARN_DIR \
    model.model_args.attn_implementation=eager \
    data/datasets@data.train=$([[ "${ATTACK_SPLIT:-retain}" == "retain" ]] && echo "TOFU_QA_$RETAIN_SPLIT" || echo "TOFU_QA_$FORGET_SPLIT") \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_on_start=false \
    ++trainer.args.remove_unused_columns=false \
    ++trainer.args.optim=adamw_torch \
    trainer.args.seed=42 \
    "${RELEARN_EXTRA[@]}" \
    paths.output_dir=$RELEARN_DIR task_name=${TASK}_${ATTACK_SPLIT:-retain}1ep >> $LOG 2>&1
  RC=$?
  if [[ $RC -ne 0 ]]; then
    log "FAIL relearn (exit=$RC) — removing partial RELEARN_DIR"
    rm -rf $RELEARN_DIR
    exit $RC
  fi
  log "DONE relearn"
fi

#=============================================================
# 4. EVAL POST-RELEARN
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
# 4b. MEASURE POST-L2 (inline, while relearn ckpt is fresh, before cleanup)

log "ALL DONE"

#=============================================================
# 5. CLEANUP — gated on eval-post JSON existing AND having extraction_strength
#=============================================================
SUMMARY=$EVAL_POST/TOFU_SUMMARY.json
if [[ -f $SUMMARY ]] && grep -q extraction_strength $SUMMARY 2>/dev/null; then
  if [[ -d $RELEARN_DIR ]] && [[ "${KEEP_RELEARN_DIR:-0}" != "1" ]]; then
    log "CLEANUP rm RELEARN_DIR=$RELEARN_DIR (eval JSON preserved)"
    rm -rf $RELEARN_DIR
  fi
  if [[ $DELETE_UNLEARN_AFTER -eq 1 ]] && [[ -d $UNLEARN_DIR ]]; then
    log "CLEANUP rm UNLEARN_DIR=$UNLEARN_DIR (7B/8B mode; eval JSONs preserved)"
    rm -rf $UNLEARN_DIR
  fi
else
  log "CLEANUP skipped (eval-post summary missing or incomplete: $SUMMARY)"
fi
