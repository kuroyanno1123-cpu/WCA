#!/bin/bash
# apr-s-orig-cls γ=0.35/0.40/0.45/0.50 スイープ (50ep, seed 0)

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=50
BATCH_SIZE=128

run_cls() {
    local GS=$1
    local MEMO=screen_apr_s_orig_cls_gs${GS}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO (γ=${GS}) ==="

    conda run -n apr python main.py \
      --aug apr-s-orig-cls --gamma-swap $GS --uncond-smooth 0.0 \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug apr-s-orig-cls --gamma-swap $GS --uncond-smooth 0.0 \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

run_cls 0.35
run_cls 0.40
run_cls 0.45
run_cls 0.50

echo ""
echo "[$(date)] === γ=0.35-0.50 スイープ 結果 ==="
printf "%-45s  %10s  %8s\n" "run" "clean_acc" "mCE"

# 既存の低γ結果と並べて表示
for GS in 0.05 0.1 0.2 0.3 0.35 0.40 0.45 0.50; do
    RUN=screen_apr_s_orig_cls_gs${GS}
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-45s  %10s  %8s\n" "$RUN" "$CLEAN" "$MCE"
done
echo "[$(date)] === 完了 ==="
