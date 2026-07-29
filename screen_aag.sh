#!/bin/bash
# APR-AAG 50ep スクリーニング (seed 0)
#   Run 1: CE-only    (--aag-jsd-weight 0.0)  ← 本命
#   Run 2: DAT忠実    (--aag-jsd-weight 2.0)  ← 比較用

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=50
BATCH_SIZE=128

run_aag() {
    local JSD=$1
    local MEMO=screen_apr_aag_jsd${JSD}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO (jsd_weight=${JSD}) ==="

    conda run -n apr python main.py \
      --aug apr-aag \
      --aag-jsd-weight $JSD --aag-mix-beta 1.0 --aag-g-lr 0.1 --aag-z-dim 100 \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug apr-aag \
      --aag-jsd-weight $JSD --aag-mix-beta 1.0 --aag-g-lr 0.1 --aag-z-dim 100 \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

run_aag 0.0
run_aag 2.0

echo ""
echo "[$(date)] === APR-AAG スクリーニング 結果 ==="
printf "%-40s  %10s  %8s\n" "run" "clean_acc" "mCE"

for JSD in 0.0 2.0; do
    MEMO=screen_apr_aag_jsd${JSD}
    DIR=./results/$MEMO
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-40s  %10s  %8s\n" "$MEMO" "$CLEAN" "$MCE"
done

echo ""
echo "比較ベースライン: apr-s-orig 50ep mCE = 25.97 (±1.43, 3シード)"
echo "[$(date)] === 完了 ==="
