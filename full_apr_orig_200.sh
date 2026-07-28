#!/bin/bash
# apr-s-orig 3 設定の 200ep フル実験 (seed 3)

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=3
MAX_EPOCH=200
BATCH_SIZE=128

run() {
    local AUG=$1
    local GS=$2
    local US=$3
    local MEMO=$4
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO ==="

    conda run -n apr python main.py \
      --aug $AUG --gamma-swap $GS --uncond-smooth $US \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug $AUG --gamma-swap $GS --uncond-smooth $US \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

run apr-s-orig      0.0  0.0  full_apr_s_orig_200ep_s3
run apr-s-orig-cls  0.05 0.0  full_apr_s_orig_cls_gs0.05_200ep_s3
run apr-s-orig-cls  0.3  0.0  full_apr_s_orig_cls_gs0.3_200ep_s3

echo ""
echo "[$(date)] === 200ep (seed 3) 結果サマリー ==="
printf "%-50s  %10s  %8s\n" "run" "clean_acc" "mCE"
for MEMO in full_apr_s_orig_200ep_s3 full_apr_s_orig_cls_gs0.05_200ep_s3 full_apr_s_orig_cls_gs0.3_200ep_s3; do
    DIR=./results/$MEMO
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-50s  %10s  %8s\n" "$MEMO" "$CLEAN" "$MCE"
done
echo "[$(date)] === 完了 ==="
