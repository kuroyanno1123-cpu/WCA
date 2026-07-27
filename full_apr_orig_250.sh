#!/bin/bash
# apr-s-orig 3 設定の 250ep フル実験 (seed 0)
# 1. apr-s-orig (素)
# 2. apr-s-orig-cls γ=0.05
# 3. apr-s-orig-cls γ=0.3

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=250
BATCH_SIZE=128

# ── 1. apr-s-orig ─────────────────────────────────────────────────────────────
MEMO=full_apr_s_orig
OUT=./results/$MEMO
mkdir -p $OUT
echo "[$(date)] === Start: $MEMO ==="
conda run -n apr python main.py \
  --aug apr-s-orig \
  --data $DATA --dataset cifar10 \
  --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  > $OUT/train.log 2>&1
echo "[$(date)] Training done: $MEMO"
conda run -n apr python main.py --eval eval \
  --aug apr-s-orig \
  --data $DATA --data-c $DATA_C --dataset cifar10 \
  --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  >> $OUT/train.log 2>&1
echo "[$(date)] Eval done: $MEMO"

# ── 2. apr-s-orig-cls γ=0.05 ─────────────────────────────────────────────────
MEMO=full_apr_s_orig_cls_gs0.05
OUT=./results/$MEMO
mkdir -p $OUT
echo "[$(date)] === Start: $MEMO ==="
conda run -n apr python main.py \
  --aug apr-s-orig-cls --gamma-swap 0.05 --uncond-smooth 0.0 \
  --data $DATA --dataset cifar10 \
  --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  > $OUT/train.log 2>&1
echo "[$(date)] Training done: $MEMO"
conda run -n apr python main.py --eval eval \
  --aug apr-s-orig-cls --gamma-swap 0.05 --uncond-smooth 0.0 \
  --data $DATA --data-c $DATA_C --dataset cifar10 \
  --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  >> $OUT/train.log 2>&1
echo "[$(date)] Eval done: $MEMO"

# ── 3. apr-s-orig-cls γ=0.3 ──────────────────────────────────────────────────
MEMO=full_apr_s_orig_cls_gs0.3
OUT=./results/$MEMO
mkdir -p $OUT
echo "[$(date)] === Start: $MEMO ==="
conda run -n apr python main.py \
  --aug apr-s-orig-cls --gamma-swap 0.3 --uncond-smooth 0.0 \
  --data $DATA --dataset cifar10 \
  --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  > $OUT/train.log 2>&1
echo "[$(date)] Training done: $MEMO"
conda run -n apr python main.py --eval eval \
  --aug apr-s-orig-cls --gamma-swap 0.3 --uncond-smooth 0.0 \
  --data $DATA --data-c $DATA_C --dataset cifar10 \
  --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
  --outfolder $OUT --memo $MEMO \
  >> $OUT/train.log 2>&1
echo "[$(date)] Eval done: $MEMO"

# ── 結果サマリー ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === 250ep 結果サマリー ==="
printf "%-40s  %10s  %8s\n" "run" "clean_acc" "mCE"
for MEMO in full_apr_s_orig full_apr_s_orig_cls_gs0.05 full_apr_s_orig_cls_gs0.3; do
    DIR=./results/$MEMO
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-40s  %10s  %8s\n" "$MEMO" "$CLEAN" "$MCE"
done
echo "[$(date)] === 完了 ==="
