#!/bin/bash
# One relearning attack, plus the post-attack evaluation, against an existing unlearned
# checkpoint. Use it to add the forget and forget+retain attacks to a checkpoint that
# scripts/unlearn_baselines.sh has already produced and attacked on the retain split.
#
#   ATTACK=retain         fine-tune on the retain split only (Hu et al. protocol)
#   ATTACK=forget         fine-tune on the forget split (direct leak)
#   ATTACK=forget_retain  fine-tune on the union (strongest attacker)
#
# Table 1 uses EPOCHS=1, LR=1e-5, AdamW, seed 42. The other knobs reproduce the attack
# variants of Appendix C.3 (learning rate, optimizer, horizon).
#
#   CKPT=<path> NAME=<tag> SCALE=1B FS=forget10 ATTACK=forget_retain GPU=0 bash scripts/attack.sh
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

SCALE="${SCALE:-1B}"
FS="${FS:-forget10}"
case "$FS" in
  forget01) RS=retain99 ;;
  forget05) RS=retain95 ;;
  forget10) RS=retain90 ;;
  *) echo "bad FS=$FS (expected forget01|forget05|forget10)"; exit 1 ;;
esac
case "$SCALE" in
  1B|3B) MODEL_CONFIG="Llama-3.2-${SCALE}-Instruct" ;;
  *) echo "bad SCALE=$SCALE (expected 1B|3B)"; exit 1 ;;
esac

# The attack set is selected per split via an hf_args.name override, so one config
# serves all three forget sizes. TOFU_QA_full (the union) is split-independent.
case "$ATTACK" in
  retain)        DS=TOFU_QA_retain; NAME_OVR="data.train.TOFU_QA_retain.args.hf_args.name=$RS" ;;
  forget)        DS=TOFU_QA_forget; NAME_OVR="data.train.TOFU_QA_forget.args.hf_args.name=$FS" ;;
  forget_retain) DS=TOFU_QA_full;   NAME_OVR="" ;;
  *) echo "bad ATTACK=$ATTACK"; exit 1 ;;
esac

TAG="atk_${SCALE}_${NAME}_${FS}_${ATTACK}${EPOCHS}ep_lr${LR}"
[[ "$OPTIM" != "adamw_torch" ]] && TAG="${TAG}_${OPTIM}"
[[ "$SEED" != "42" ]] && TAG="${TAG}_s${SEED}"
RELEARN_DIR=saves/finetune/${TAG}
EVAL_POST=saves/eval/${TAG}
LOGDIR=$OUTDIR/logs
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
    data/datasets@data.train=$DS ${NAME_OVR:+++$NAME_OVR} \
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
    forget_split=$FS \
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
