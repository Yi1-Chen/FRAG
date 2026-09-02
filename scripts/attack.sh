#!/bin/bash
# Parameterized relearning attack + post-eval for ONE existing unlearned checkpoint.
# Mirrors canonical_one.sh relearn/eval stages; adds ATTACK/EPOCHS/LR/OPTIM knobs.
#   ATTACK=retain        -> TOFU_QA_retain90       (canonical, Hu et al.)
#   ATTACK=forget        -> TOFU_QA_forget         (= forget10, direct leak)
#   ATTACK=forget_retain -> TOFU_QA_full           (union, full white-box)
# Usage: CKPT=<path> NAME=<tag> ATTACK=retain EPOCHS=1 LR=1e-5 [OPTIM=sgd] GPU=0 bash attack_one.sh
set -u
FRAG_REPO="${FRAG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OU="${OPEN_UNLEARNING:-$HOME/open-unlearning}"
PYTHON="${PYTHON:-$OU/.venv/bin/python}"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
OUTDIR="${OUTDIR:-$FRAG_REPO/runs}"
cd "$OU"

CKPT="${CKPT:?need CKPT}"
NAME="${NAME:?need NAME}"
ATTACK="${ATTACK:-retain}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
OPTIM="${OPTIM:-adamw_torch}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU

case "$ATTACK" in
  retain)        DS=TOFU_QA_retain90 ;;
  forget)        DS=TOFU_QA_forget ;;
  forget_retain) DS=TOFU_QA_full ;;
  *) echo "bad ATTACK=$ATTACK"; exit 1 ;;
esac

MODEL_CONFIG=Llama-3.2-1B-Instruct
TAG="atk_${NAME}_${ATTACK}${EPOCHS}ep_lr${LR}"
[[ "$OPTIM" != "adamw_torch" ]] && TAG="${TAG}_${OPTIM}"
[[ "$SEED" != "42" ]] && TAG="${TAG}_s${SEED}"
RELEARN_DIR=saves/finetune/${TAG}
EVAL_POST=saves/eval/${TAG}
LOGDIR=logs/atk
mkdir -p $LOGDIR saves/finetune saves/eval
LOG=$LOGDIR/${TAG}.log

ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $TAG/GPU$GPU $1" | tee -a $LOG; }

if [[ ! -f $CKPT/config.json ]]; then log "ABORT: no ckpt at $CKPT"; exit 1; fi

if [[ -f $RELEARN_DIR/config.json ]]; then
  log "SKIP relearn (exists)"
else
  log "START relearn ($ATTACK, ${EPOCHS}ep, lr=$LR, optim=$OPTIM, seed=$SEED)"
  $PYTHON src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default model=$MODEL_CONFIG \
    model.model_args.pretrained_model_name_or_path=$CKPT \
    model.model_args.attn_implementation=eager \
    data/datasets@data.train=$DS \
    trainer.args.num_train_epochs=$EPOCHS \
    trainer.args.learning_rate=$LR \
    trainer.args.eval_on_start=false \
    ++trainer.args.remove_unused_columns=false \
    ++trainer.args.optim=$OPTIM \
    trainer.args.seed=$SEED \
    paths.output_dir=$RELEARN_DIR task_name=${TAG} >> $LOG 2>&1
  RC=$?
  if [[ $RC -ne 0 ]]; then
    log "FAIL relearn (exit=$RC) — removing partial RELEARN_DIR"
    rm -rf $RELEARN_DIR
    exit $RC
  fi
  log "DONE relearn"
fi

if [[ -f $EVAL_POST/TOFU_EVAL.json ]]; then
  log "SKIP eval-post (exists)"
else
  log "START eval-post"
  $PYTHON src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default model=$MODEL_CONFIG \
    model.model_args.pretrained_model_name_or_path=$RELEARN_DIR \
    model.model_args.attn_implementation=eager \
    forget_split=forget10 \
    paths.output_dir=$EVAL_POST task_name=${TAG}_eval >> $LOG 2>&1
  log "DONE eval-post"
fi

SUMMARY=$EVAL_POST/TOFU_SUMMARY.json
if [[ -f $SUMMARY ]] && grep -q extraction_strength $SUMMARY 2>/dev/null; then
  python3 - <<PYEOF
import json
b = json.load(open("$SUMMARY"))
line = "$TAG postES=%.4f postU=%.4f" % (b["extraction_strength"], b["model_utility"])
print(line)
open("$OUTDIR/attack_results.txt", "a").write(line + "\n")
PYEOF
  if [[ -d $RELEARN_DIR ]] && [[ "${KEEP_RELEARN_DIR:-0}" != "1" ]]; then
    log "CLEANUP rm RELEARN_DIR=$RELEARN_DIR (eval JSON preserved)"
    rm -rf $RELEARN_DIR
  fi
else
  log "CLEANUP skipped (summary missing)"
fi
log "ALL DONE"
