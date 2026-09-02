#!/bin/bash
# Full 1B FLP sparsity sweep. Reuses canonical_one.sh per point so that the
# 3.0% row is bit-exact with the main Table-1 FLP/1B/forget10 cell.
#
# CANONICAL PROTOCOL: PRUNE_MODE=zero (matches main table).
# Per-point: prune -> eval-pure -> L2 -> retain1ep relearn -> eval-post -> L2 -> cleanup.
# Cleanup: delete RELEARN_DIR + UNLEARN_DIR after L2 (eval JSONs + L2 JSONs preserved).
#
# Usage:
#   GPU=0 SPARSITIES="0.005 0.010 0.015" bash scripts/sweep_sparsity.sh
# Launch on 4 GPUs in parallel with disjoint SPARSITIES; outputs are keyed by tag.

set -u
FRAG_REPO="${FRAG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OU="${OPEN_UNLEARNING:-$HOME/open-unlearning}"
PYTHON="${PYTHON:-$OU/.venv/bin/python}"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
OUTDIR="${OUTDIR:-$FRAG_REPO/runs}"
cd "$OU"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU
SPARSITIES="${SPARSITIES:-0.005 0.010 0.015 0.020 0.025 0.030 0.040 0.050 0.070 0.100}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
SEED="${SEED:-42}"
PRUNE_MODE="${PRUNE_MODE:-zero}"   # canonical: zero-out (matches main Table 1)

OUT_JSON="$OUTDIR/flp_sparsity_sweep_1B.json"
LOGDIR=logs/flp_sparsity_sweep_1B
mkdir -p "$LOGDIR" "$(dirname "$OUT_JSON")"
LOG=$LOGDIR/driver_gpu${GPU}.log
ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] DRIVER/GPU$GPU $1" | tee -a "$LOG"; }

# Init output JSON once (use python for proper locking)
if [[ ! -f "$OUT_JSON" ]]; then
  echo '{"sweep":"1B_FLP_sparsity_v2","protocol":"canonical_retain1ep_maskfree","prune_mode":"'"$PRUNE_MODE"'","seed":'"$SEED"',"forget_split":"'"$FORGET_SPLIT"'","rows":{}}' > "$OUT_JSON"
fi

for SP in $SPARSITIES; do
  SP_TAG=$($PYTHON -c "print(f'{${SP}:.3f}'.replace('.', 'p'))")  # 0.030 -> 0p030
  TASK="flp_sweep_1B_sp${SP_TAG}"
  UNLEARN_DIR="saves/unlearn/${TASK}"
  RELEARN_DIR="saves/finetune/${TASK}_retain1ep"
  EVAL_PURE="saves/eval/${TASK}_pure"
  EVAL_POST="saves/eval/${TASK}_postrelearnretain1ep"
  L2_PRE="$OUTDIR/l2_per_task/${TASK}_pre.json"
  L2_POST="$OUTDIR/l2_per_task/${TASK}_retain_post.json"

  log "BEGIN sparsity=$SP tag=$SP_TAG task=$TASK"
  T0=$(date +%s)

  SPARSITY="$SP" TASK="$TASK" METHOD=RC SCALE=1B GPU="$GPU" \
    FORGET_SPLIT="$FORGET_SPLIT" SEED="$SEED" PRUNE_MODE="$PRUNE_MODE" \
    bash "$FRAG_REPO/scripts/unlearn_baselines.sh"
  RC=$?
  T_PIPE=$(( $(date +%s) - T0 ))
  if [[ $RC -ne 0 ]]; then
    log "FAIL pipeline sparsity=$SP (exit=$RC, ${T_PIPE}s)"
    continue
  fi
  log "OK pipeline sparsity=$SP (${T_PIPE}s)"

  $PYTHON - "$OUT_JSON" "$SP_TAG" "$SP" "$EVAL_PURE" "$EVAL_POST" "$L2_PRE" "$L2_POST" "$T_PIPE" <<'PY'
import json, sys, pathlib, fcntl, time
out_json, sp_tag, sp, eval_pure, eval_post, l2_pre_p, l2_post_p, tpipe = sys.argv[1:]
def es_util(d):
    s = json.loads((pathlib.Path(d)/"TOFU_SUMMARY.json").read_text())
    return s.get("extraction_strength"), s.get("model_utility")
def l2_total(p):
    j = json.loads(pathlib.Path(p).read_text())
    if isinstance(j, list): j = j[0]
    return j.get("total_l2")
pre_es, pre_u  = es_util(eval_pure)
post_es, post_u = es_util(eval_post)
l2_pre = l2_total(l2_pre_p) if pathlib.Path(l2_pre_p).exists() else None
l2_post = l2_total(l2_post_p) if pathlib.Path(l2_post_p).exists() else None
# locked update
for _ in range(20):
    try:
        f = open(out_json, "r+"); fcntl.flock(f, fcntl.LOCK_EX); break
    except BlockingIOError: time.sleep(0.2)
data = json.loads(f.read() or "{}")
data.setdefault("rows", {})[sp_tag] = {
    "sparsity": float(sp),
    "pre_ES": pre_es, "post_ES": post_es,
    "pre_U":  pre_u,  "post_U":  post_u,
    "delta_ES": (post_es - pre_es) if (pre_es is not None and post_es is not None) else None,
    "L2_pre": l2_pre, "L2_post": l2_post,
    "pipeline_seconds": int(tpipe),
}
f.seek(0); f.truncate(); f.write(json.dumps(data, indent=2))
fcntl.flock(f, fcntl.LOCK_UN); f.close()
print("WROTE", sp_tag, data["rows"][sp_tag])
PY

  log "CLEANUP sparsity=$SP — rm $UNLEARN_DIR $RELEARN_DIR"
  rm -rf "$UNLEARN_DIR" "$RELEARN_DIR"
  log "END sparsity=$SP total=$(( $(date +%s) - T0 ))s"
done

log "DRIVER DONE for SPARSITIES=$SPARSITIES"
