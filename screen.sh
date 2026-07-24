#!/bin/bash
# WCA スクリーニング: CE-only と JSD を同一シードで比較
# VIPAug (PID 3927134) 終了後に自動で順実行

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data

SEED=0
MAX_EPOCH=50
BATCH_SIZE=128
GPU=0
SOURCE=haar
TARGET=db8
LEVEL=1
SWAP_PROB=0.2

# ── VIPAug 終了待ち ────────────────────────────────────────────────────────────
VIPAUG_PID=3927134
if kill -0 $VIPAUG_PID 2>/dev/null; then
    echo "[$(date)] VIPAug (PID $VIPAUG_PID) 終了を待機中..."
    while kill -0 $VIPAUG_PID 2>/dev/null; do sleep 60; done
    echo "[$(date)] VIPAug 終了確認"
fi

# ── 実験 1: WCA + CE-only (jsd-lambda 0) ──────────────────────────────────────
MEMO=screen_wca_ce
OUTFOLDER=./results/${MEMO}
mkdir -p $OUTFOLDER
echo "[$(date)] === Start: $MEMO ==="

conda run -n apr python main.py \
  --aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
  --jsd-lambda 0 \
  --data $DATA \
  --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
  --seed $SEED \
  --outfolder $OUTFOLDER --memo $MEMO \
  > $OUTFOLDER/train.log 2>&1

echo "[$(date)] Training done: $MEMO"

conda run -n apr python main.py --eval eval \
  --aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
  --jsd-lambda 0 \
  --data $DATA --data-c $DATA_C \
  --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
  --seed $SEED \
  --outfolder $OUTFOLDER --memo $MEMO \
  >> $OUTFOLDER/train.log 2>&1

echo "[$(date)] Eval done: $MEMO"
grep -E "clean accuracy|Mean Error|mean error" $OUTFOLDER/train.log | tail -3

# ── 実験 2: WCA + JSD (jsd-lambda 12) ─────────────────────────────────────────
MEMO=screen_wca_jsd
OUTFOLDER=./results/${MEMO}
mkdir -p $OUTFOLDER
echo "[$(date)] === Start: $MEMO ==="

conda run -n apr python main.py \
  --aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
  --jsd-lambda 12 \
  --data $DATA \
  --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
  --seed $SEED \
  --outfolder $OUTFOLDER --memo $MEMO \
  > $OUTFOLDER/train.log 2>&1

echo "[$(date)] Training done: $MEMO"

conda run -n apr python main.py --eval eval \
  --aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
  --jsd-lambda 12 \
  --data $DATA --data-c $DATA_C \
  --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
  --seed $SEED \
  --outfolder $OUTFOLDER --memo $MEMO \
  >> $OUTFOLDER/train.log 2>&1

echo "[$(date)] Eval done: $MEMO"
grep -E "clean accuracy|Mean Error|mean error" $OUTFOLDER/train.log | tail -3

# ── 実験 3: AugMix + JSD (jsd-lambda 12) ──────────────────────────────────────
MEMO=screen_augmix_jsd
OUTFOLDER=./results/${MEMO}
mkdir -p $OUTFOLDER
echo "[$(date)] === Start: $MEMO ==="

conda run -n apr python main.py \
  --aug augmix \
  --jsd-lambda 12 \
  --data $DATA \
  --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
  --seed $SEED \
  --outfolder $OUTFOLDER --memo $MEMO \
  > $OUTFOLDER/train.log 2>&1

echo "[$(date)] Training done: $MEMO"

conda run -n apr python main.py --eval eval \
  --aug augmix \
  --jsd-lambda 12 \
  --data $DATA --data-c $DATA_C \
  --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
  --seed $SEED \
  --outfolder $OUTFOLDER --memo $MEMO \
  >> $OUTFOLDER/train.log 2>&1

echo "[$(date)] Eval done: $MEMO"
grep -E "clean accuracy|Mean Error|mean error" $OUTFOLDER/train.log | tail -3

echo "[$(date)] === All screening done ==="
