#!/bin/bash
# APR-S-cls 50ep スクリーニング (3 runs)
# screen_apr_s (ベースライン) は既に完了済み。

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=50
BATCH_SIZE=128

run_cls() {
    local GS=$1
    local US=$2
    local MEMO=screen_apr_s_cls_gs${GS}_us${US}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO ==="

    conda run -n apr python main.py \
      --aug apr-s-cls --gamma-swap $GS --uncond-smooth $US \
      --data $DATA \
      --dataset cifar10 --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU \
      --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug apr-s-cls --gamma-swap $GS --uncond-smooth $US \
      --data $DATA --data-c $DATA_C \
      --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
      --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

# Run 2: gamma-swap=0.1
run_cls 0.1 0.0

# Run 3: gamma-swap=0.2
run_cls 0.2 0.0

# Run 4: uncond-smooth=0.1 (対照: gamma-swap は無視される)
run_cls 0.2 0.1

# ── 結果サマリー ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === APR-S-cls スクリーニング 結果サマリー ==="
echo ""
printf "%-40s  %10s  %10s\n" "run" "clean_acc" "mCE"

# ベースライン (既存)
for RUN in screen_apr_s; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-40s  %10s  %10s\n" "$RUN (baseline)" "$CLEAN" "$MCE"
done

# 新規 runs
for GS_US in "0.1_0.0" "0.2_0.0" "0.2_0.1"; do
    GS=${GS_US%_*}
    US=${GS_US#*_}
    RUN=screen_apr_s_cls_gs${GS}_us${US}
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-40s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE"
done

echo ""
echo "[$(date)] === 完了 ==="
