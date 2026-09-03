#!/bin/bash
# FRP (and its retain-veto variant FRP-RV) end to end on TOFU: prune -> evaluate ->
# run the three relearning attacks -> evaluate again. This produces one row of Table 1.
#
# Pruning uses the canonical setting from Appendix B.1: rank-space scoring with beta=0.05
# at 3% MLP sparsity, 400 calibration sequences, seed 42, weights zeroed. P is the
# retain-veto fraction: P=0 is plain FRP, P>0 protects the top-P fraction of
# retain-exclusive weights per row from pruning, trading robustness for utility.
# Table 1 reports P=0 and P=0.15 at 1B, P=0 and P=0.05 at 3B.
#
#   SCALE=1B P=0.15 FS=forget10 GPU=0 bash scripts/frp_tofu.sh
#
# Environment:
#   OPEN_UNLEARNING  OpenUnlearning checkout used for training and evaluation
#   FRAG_REPO        this repository (defaults to the parent of this script)
#   OUTDIR           where per-run summaries are written (default: $FRAG_REPO/runs)
#   PYTHON           interpreter (default: $OPEN_UNLEARNING/.venv/bin/python)
set -u

FRAG_REPO="${FRAG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OU="${OPEN_UNLEARNING:-$HOME/open-unlearning}"
OUTDIR="${OUTDIR:-$FRAG_REPO/runs}"
PY="${PYTHON:-$OU/.venv/bin/python}"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

SCALE="${SCALE:-1B}"
P="${P:?need P, the retain-veto fraction (0 for plain FRP)}"
FS="${FS:-forget10}"
GPU="${GPU:-0}"
SPARSITY="${SPARSITY:-0.03}"
BETA="${BETA:-0.05}"

case "$FS" in
  forget01) RS=retain99 ;;
  forget05) RS=retain95 ;;
  forget10) RS=retain90 ;;
  *) echo "bad FS=$FS (expected forget01|forget05|forget10)"; exit 1 ;;
esac
case "$SCALE" in
  1B|3B) MODEL_CFG="Llama-3.2-${SCALE}-Instruct" ;;
  *) echo "bad SCALE=$SCALE (expected 1B|3B)"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=$GPU
cd "$OU" || exit 1
mkdir -p "$OUTDIR"

FULL=$(ls -d "$HUB"/models--open-unlearning--tofu_Llama-3.2-${SCALE}-Instruct_full/snapshots/*/ \
       2>/dev/null | head -1 | sed 's:/$::')
[ -n "$FULL" ] || { echo "original model not found under $HUB"; exit 1; }
RETAIN_LOGS="saves/eval/tofu_${MODEL_CFG}_retain90/TOFU_EVAL.json"

TAG="frp_${SCALE}_${FS}_p$(printf '%03d' "$(python3 -c "print(round(${P}*100))")")"
UN="saves/unlearn/${TAG}"
LOG="${OUTDIR}/${TAG}.log"; : > "$LOG"
OUT="${OUTDIR}/${TAG}.txt"

say() { echo "[$(date +%H:%M:%S) $TAG] $*" | tee -a "$LOG"; }
es() { python3 -c "import json;print('%.4f'%json.load(open('$1'))['extraction_strength'])" 2>/dev/null; }
ut() { python3 -c "import json;print('%.4f'%json.load(open('$1'))['model_utility'])" 2>/dev/null; }

# ---------------------------------------------------------------- 1. prune
if [ -f "$UN/config.json" ]; then
  say "skip prune (checkpoint exists)"
else
  say "prune: rank_combo beta=$BETA sparsity=$SPARSITY, retain veto p=$P"
  $PY "$FRAG_REPO/frp/prune.py" --model "$FULL" --dataset tofu \
    --forget_split "$FS" --retain_split "$RS" \
    --scoring rank_combo --beta "$BETA" --sparsity "$SPARSITY" \
    --target mlp --prune_mode shrink --shrink_factor 0.0 \
    --dtype bf16 --num_samples 400 --seed 42 \
    --protect_by excl --protect_p "$P" --output_dir "$UN" >>"$LOG" 2>&1
  [ -f "$UN/config.json" ] || { say "prune FAILED"; exit 1; }
fi

# ---------------------------------------------------------------- 2. pre-attack eval
EP="saves/eval/${TAG}_pure"
if [ ! -f "$EP/TOFU_SUMMARY.json" ]; then
  say "eval (pre-attack)"
  $PY src/eval.py --config-name=eval.yaml experiment=eval/tofu/default model="$MODEL_CFG" \
    forget_split="$FS" model.model_args.pretrained_model_name_or_path="$UN" \
    model.model_args.attn_implementation=eager \
    paths.output_dir="$EP" task_name="${TAG}_pure" >>"$LOG" 2>&1
fi
preES=$(es "$EP/TOFU_SUMMARY.json"); preU=$(ut "$EP/TOFU_SUMMARY.json")
say "pre-attack: ES=$preES U=$preU"
{ echo "# $TAG  sparsity=$SPARSITY beta=$BETA protect_by=excl protect_p=$P"
  echo "pure $preES $preU"; } > "$OUT"

# ---------------------------------------------------------------- 3. three relearning attacks
# One epoch of AdamW at lr 1e-5 on the retain split, the forget split, or their union
# (Appendix B.2). The union is the strongest attacker.
for A in retain forget forget_retain; do
  case $A in
    retain)        DATA=TOFU_QA_retain; NAME_OVR="data.train.TOFU_QA_retain.args.hf_args.name=$RS" ;;
    forget)        DATA=TOFU_QA_forget; NAME_OVR="data.train.TOFU_QA_forget.args.hf_args.name=$FS" ;;
    forget_retain) DATA=TOFU_QA_full;   NAME_OVR="" ;;
  esac
  RL="saves/finetune/${TAG}_${A}1ep"
  EPOST="saves/eval/${TAG}_postrelearn${A}1ep"
  if [ ! -f "$EPOST/TOFU_SUMMARY.json" ]; then
    say "attack $A: relearn 1 epoch, lr 1e-5"
    $PY src/train.py --config-name=train.yaml experiment=finetune/tofu/default model="$MODEL_CFG" \
      model.model_args.pretrained_model_name_or_path="$UN" \
      model.model_args.attn_implementation=eager \
      data/datasets@data.train=$DATA ${NAME_OVR:+++$NAME_OVR} \
      trainer.args.num_train_epochs=1 trainer.args.learning_rate=1e-5 \
      trainer.args.eval_on_start=false ++trainer.args.remove_unused_columns=false \
      ++trainer.args.optim=adamw_torch trainer.args.seed=42 \
      paths.output_dir="$RL" task_name="${TAG}_${A}1ep" >>"$LOG" 2>&1
    [ -f "$RL/config.json" ] || { say "relearn FAILED ($A)"; continue; }
    say "attack $A: eval"
    $PY src/eval.py --config-name=eval.yaml experiment=eval/tofu/default model="$MODEL_CFG" \
      forget_split="$FS" model.model_args.pretrained_model_name_or_path="$RL" \
      model.model_args.attn_implementation=eager retain_logs_path="$RETAIN_LOGS" \
      paths.output_dir="$EPOST" task_name="${TAG}_post${A}" >>"$LOG" 2>&1
    rm -rf "$RL"   # the attacked checkpoint is large and not needed once evaluated
  fi
  pES=$(es "$EPOST/TOFU_SUMMARY.json"); pU=$(ut "$EPOST/TOFU_SUMMARY.json")
  dES=$(python3 -c "print('%+.4f'%(${pES}-(${preES})))" 2>/dev/null)
  echo "$A $pES $pU $dES" >> "$OUT"
  say "attack $A: ES=$pES dES=$dES U=$pU"
done

say "done -> $OUT"
cat "$OUT"
