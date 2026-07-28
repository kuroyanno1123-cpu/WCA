#!/bin/bash
# WaveletFusion 50ep スクリーニング: wf-sign / wf-max (seed 0)

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=50
BATCH_SIZE=128

run_wf() {
    local AUG=$1
    local MEMO=screen_${AUG//-/_}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO ==="

    conda run -n apr python main.py \
      --aug $AUG --wf-wavelet haar --wf-level 1 --wf-apply-prob 0.5 \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug $AUG --wf-wavelet haar --wf-level 1 --wf-apply-prob 0.5 \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

run_wf wf-sign
run_wf wf-max

echo ""
echo "[$(date)] === WaveletFusion スクリーニング 結果 ==="
printf "%-35s  %10s  %8s  %10s\n" "run" "clean_acc" "mCE" "apply_rate"

for AUG in wf-sign wf-max; do
    MEMO=screen_${AUG//-/_}
    DIR=./results/$MEMO
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    AR=$(tail -1 $DIR/history.csv 2>/dev/null | awk -F',' '{print $7}')
    printf "%-35s  %10s  %8s  %10s\n" "$MEMO" "$CLEAN" "$MCE" "$AR"
done

echo "[$(date)] === 完了 ==="
