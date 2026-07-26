#!/bin/bash
# basis_random スクリーニング (50ep): wca_basisrand_jsd12
# grid_lambda.sh (PID 6823) 終了後に自動実行

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data

MEMO=wca_basisrand_jsd12
OUT=./results/$MEMO
GPU=0

# ── grid_lambda.sh 終了待ち ────────────────────────────────────────────────────
WAIT_PID=6823
if kill -0 $WAIT_PID 2>/dev/null; then
    echo "[$(date)] grid_lambda.sh 終了を待機中 (PID $WAIT_PID)..."
    while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
    echo "[$(date)] 前実験完了"
fi

mkdir -p $OUT
echo "[$(date)] === Start: $MEMO ==="

conda run -n apr python main.py \
  --aug wca --basis-random --level 1 --swap-prob 0.2 \
  --jsd-lambda 12 \
  --data $DATA \
  --dataset cifar10 --batch-size 128 --max-epoch 50 --gpu $GPU \
  --seed 0 \
  --outfolder $OUT --memo $MEMO \
  > $OUT/train.log 2>&1

echo "[$(date)] Training done: $MEMO"

conda run -n apr python main.py --eval eval \
  --aug wca --basis-random --level 1 --swap-prob 0.2 \
  --jsd-lambda 12 \
  --data $DATA --data-c $DATA_C \
  --dataset cifar10 --batch-size 128 --gpu $GPU \
  --seed 0 \
  --outfolder $OUT --memo $MEMO \
  >> $OUT/train.log 2>&1

echo "[$(date)] Eval done: $MEMO"

# ── 結果サマリー ───────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === basis_random スクリーニング 結果サマリー ==="
echo ""
printf "%-30s  %10s  %10s\n" "run" "clean_acc" "mCE"

for RUN in screen_wca_ce screen_wca_jsd wca_basisrand_jsd12; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-30s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE"
done

echo ""
echo "[$(date)] === 完了 ==="
