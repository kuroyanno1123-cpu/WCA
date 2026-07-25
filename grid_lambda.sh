#!/bin/bash
# λ グリッドサーチ (50ep): λ = 3, 6, 18, 21
# WBS noamp ウェイター (PID 5274) 終了後に自動実行

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

# ── WBS noamp ウェイター終了待ち ───────────────────────────────────────────────
WAIT_PID=5274
if kill -0 $WAIT_PID 2>/dev/null; then
    echo "[$(date)] WBS noamp 終了を待機中 (PID $WAIT_PID)..."
    while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
    echo "[$(date)] 前実験完了"
fi

run_lambda() {
    local LAM=$1
    local MEMO=screen_wca_jsd_l${LAM}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO (λ=${LAM}) ==="

    conda run -n apr python main.py \
      --aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
      --jsd-lambda $LAM \
      --data $DATA \
      --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
      --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB \
      --jsd-lambda $LAM \
      --data $DATA --data-c $DATA_C \
      --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
      --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

for LAM in 3 6 18 21; do
    run_lambda $LAM
done

# ── 結果サマリー ───────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === λ グリッドサーチ 結果サマリー ==="
echo ""
printf "%-30s  %10s  %10s  %10s\n" "run" "clean_acc" "mCE" "best_clean_acc(csv)"

# 比較基準: 既存スクリーニング
for RUN in screen_wca_ce screen_wca_jsd; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    BEST=$(awk -F',' 'NR>1 && $6!="" {if($6>max) max=$6} END{print max}' $DIR/history.csv 2>/dev/null)
    printf "%-30s  %10s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE" "$BEST"
done

# グリッドサーチ結果
for LAM in 3 6 18 21; do
    RUN=screen_wca_jsd_l${LAM}
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    BEST=$(awk -F',' 'NR>1 && $6!="" {if($6>max) max=$6} END{print max}' $DIR/history.csv 2>/dev/null)
    printf "%-30s  %10s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE" "$BEST"
done

echo ""
echo "[$(date)] === 完了 ==="
