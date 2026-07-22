#!/bin/bash
# WCA 実験スクリプト
# 使い方:
#   bash run.sh              # WCA + JSD (デフォルト)
#   bash run.sh --jsd-lambda 0   # WCA + CE only
#   bash run.sh --aug none       # Baseline

cd /home/kairisasaki/WCA
BASE=/home/kairisasaki/WCA
DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data

# ── パラメータ (ここを変える) ──────────────────────────────
AUG=wca
SOURCE=haar
TARGET=db8
LEVEL=1
SWAP_PROB=0.2
JSD_LAMBDA=12.0
MAX_EPOCH=250
BATCH_SIZE=128
GPU=0
MEMO=wca_jsd
# ──────────────────────────────────────────────────────────

OUTFOLDER=$BASE/results/${MEMO}
mkdir -p $OUTFOLDER

echo "[$(date)] Start training: $MEMO"

conda run -n apr python main.py \
  --aug $AUG \
  --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
  --jsd-lambda $JSD_LAMBDA \
  --data $DATA \
  --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
  --outfolder $OUTFOLDER --memo $MEMO \
  "$@" \
  > $OUTFOLDER/train.log 2>&1

echo "[$(date)] Training done. Starting eval..."

conda run -n apr python main.py --eval eval \
  --aug $AUG \
  --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
  --jsd-lambda $JSD_LAMBDA \
  --data $DATA --data-c $DATA_C \
  --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
  --outfolder $OUTFOLDER --memo $MEMO \
  >> $OUTFOLDER/train.log 2>&1

echo "[$(date)] All done. Results: $OUTFOLDER/logs.txt"
