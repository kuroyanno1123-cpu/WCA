#!/bin/bash
# 250エポック本番実験 (スクリーニング3本終了後に自動実行)

cd /home/kairisasaki/WCA
DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data

SEED=0
MAX_EPOCH=250
BATCH_SIZE=128
GPU=0
SOURCE=haar
TARGET=db8
LEVEL=1
SWAP_PROB=0.2

# ── AugMix+JSD ウェイター (PID 4180685) の終了待ち ───────────────────────────
AUGMIX_WAITER=4180685
if kill -0 $AUGMIX_WAITER 2>/dev/null; then
    echo "[$(date)] スクリーニング終了を待機中 (PID $AUGMIX_WAITER)..."
    while kill -0 $AUGMIX_WAITER 2>/dev/null; do sleep 60; done
    echo "[$(date)] スクリーニング完了"
fi

run_exp() {
    local AUG=$1
    local JSD=$2
    local MEMO=$3
    local EXTRA=$4
    local OUTFOLDER=./results/${MEMO}
    mkdir -p $OUTFOLDER
    echo "[$(date)] === Start 250ep: $MEMO ==="

    conda run -n apr python main.py \
      $EXTRA \
      --jsd-lambda $JSD \
      --data $DATA \
      --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
      --seed $SEED \
      --outfolder $OUTFOLDER --memo $MEMO \
      > $OUTFOLDER/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      $EXTRA \
      --jsd-lambda $JSD \
      --data $DATA --data-c $DATA_C \
      --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
      --seed $SEED \
      --outfolder $OUTFOLDER --memo $MEMO \
      >> $OUTFOLDER/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
    grep -E "clean accuracy|Mean Error|mean error" $OUTFOLDER/train.log | tail -3
}

# ── 実験1: WCA + CE-only ──────────────────────────────────────────────────────
run_exp wca 0 full_wca_ce \
  "--aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB"

# ── 実験2: WCA + JSD ─────────────────────────────────────────────────────────
run_exp wca 12 full_wca_jsd \
  "--aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB"

# ── 実験3: AugMix + JSD ──────────────────────────────────────────────────────
run_exp augmix 12 full_augmix_jsd \
  "--aug augmix"

echo "[$(date)] === 全250ep実験完了 ==="
