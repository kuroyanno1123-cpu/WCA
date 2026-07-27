#!/bin/bash
# APR-S vs APR-S-soft 50ep スクリーニング
# 同一シード・同一ハイパーパラメータで 2 本を順番に実行する

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=50
BATCH_SIZE=128

# ── APR-S (hard swap, apply_prob=0.5 = 原実装の流儀) ─────────────────────────
MEMO=screen_apr_s
OUT=./results/$MEMO
mkdir -p $OUT
echo "[$(date)] === Start: $MEMO ==="

conda run -n apr python main.py \
  --aug apr-s \
  --data $DATA \
  --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
  --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  > $OUT/train.log 2>&1

echo "[$(date)] Training done: $MEMO"

conda run -n apr python main.py --eval eval \
  --aug apr-s \
  --data $DATA --data-c $DATA_C \
  --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
  --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  >> $OUT/train.log 2>&1

echo "[$(date)] Eval done: $MEMO"

# ── APR-S-soft (t ~ Uniform(0,1), E[t]=0.5 = APR-S と期待強度一致) ──────────
MEMO=screen_apr_s_soft
OUT=./results/$MEMO
mkdir -p $OUT
echo "[$(date)] === Start: $MEMO ==="

conda run -n apr python main.py \
  --aug apr-s-soft --t-max 1.0 --label-k 1.0 --clean-prob 0.0 \
  --data $DATA \
  --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
  --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  > $OUT/train.log 2>&1

echo "[$(date)] Training done: $MEMO"

conda run -n apr python main.py --eval eval \
  --aug apr-s-soft --t-max 1.0 --label-k 1.0 --clean-prob 0.0 \
  --data $DATA --data-c $DATA_C \
  --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
  --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  >> $OUT/train.log 2>&1

echo "[$(date)] Eval done: $MEMO"

# ── 結果サマリー ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === APR スクリーニング 結果サマリー ==="
echo ""
printf "%-25s  %10s  %10s\n" "run" "clean_acc" "mCE"

for RUN in screen_apr_s screen_apr_s_soft; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-25s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE"
done

echo ""
echo "[$(date)] === 完了 ==="
