#!/bin/bash
# APR-S-orig 50ep スクリーニング
# 比較対象: screen_apr_s (29.25%), screen_apr_s_cls_gs0.2_us0.1 (28.78%), augmix_jsd (11.55%)

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=50
BATCH_SIZE=128
MEMO=screen_apr_s_orig
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

echo ""
echo "[$(date)] === APR-S-orig スクリーニング 結果 ==="
printf "%-40s  %10s  %10s\n" "run" "clean_acc" "mCE"

# 既存比較対象
for RUN in screen_apr_s screen_apr_s_cls_gs0.2_us0.1; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-40s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE"
done

# 今回
DIR=./results/$MEMO
CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
printf "%-40s  %10s  %10s\n" "$MEMO" "$CLEAN" "$MCE"

echo "[$(date)] === 完了 ==="
