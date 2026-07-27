#!/bin/bash
# APR-S-orig-cls γ スイープ (50ep × 4 本, seed 0)
# apr-s-orig ベースラインの結果確認後に実行する

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

# γ スイープ: 0.05, 0.1, 0.2, 0.3
run_cls 0.05
run_cls 0.1
run_cls 0.2
run_cls 0.3

# ── 結果サマリー ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === APR-S-orig-cls γ スイープ 結果 ==="
printf "%-45s  %10s  %10s  %10s\n" "run" "clean_acc" "mCE" "swap_rate"

# ベースライン
DIR=./results/screen_apr_s_orig
CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
printf "%-45s  %10s  %10s  %10s\n" "screen_apr_s_orig (baseline)" "$CLEAN" "$MCE" "N/A"

# γ スイープ
for GS in 0.05 0.1 0.2 0.3; do
    RUN=screen_apr_s_orig_cls_gs${GS}
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    SR=$(grep "swap_rate" $DIR/history.csv 2>/dev/null | tail -1 | cut -d, -f8)
    printf "%-45s  %10s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE" "$SR"
done

echo ""
echo "[$(date)] === 完了 ==="
