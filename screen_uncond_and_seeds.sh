#!/bin/bash
# Step 1: 無条件スムージング対照 (2 本)
# Step 2: シード追試 seed 1, 2 × 3 設定 (6 本)
# 計 8 本 / seed 0 は既存結果を流用

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
MAX_EPOCH=50
BATCH_SIZE=128

# ── 共通実行関数 ──────────────────────────────────────────────────────────────

run_orig() {
    local SEED=$1
    local MEMO=screen_apr_s_orig_s${SEED}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO (seed=${SEED}) ==="
    conda run -n apr python main.py \
      --aug apr-s-orig \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1
    conda run -n apr python main.py --eval eval \
      --aug apr-s-orig \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1
    echo "[$(date)] Done: $MEMO"
}

run_cls() {
    local GS=$1
    local US=$2
    local SEED=$3
    # memo にシードを含めて file_name に反映させる
    local MEMO=screen_apr_s_orig_cls_gs${GS}_us${US}_s${SEED}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO (γ=${GS}, uncond=${US}, seed=${SEED}) ==="
    conda run -n apr python main.py \
      --aug apr-s-orig-cls --gamma-swap $GS --uncond-smooth $US \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1
    conda run -n apr python main.py --eval eval \
      --aug apr-s-orig-cls --gamma-swap $GS --uncond-smooth $US \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1
    echo "[$(date)] Done: $MEMO"
}

# ── Step 1: 無条件スムージング対照 (seed 0) ──────────────────────────────────
# γ=0.05 条件付きの期待スムージング量 = swap_rate × γ ≈ 0.5 × 0.05 = 0.025
run_cls 0.0 0.025 0
# γ=0.3 条件付きの期待スムージング量 = 0.5 × 0.3 = 0.15
run_cls 0.0 0.15  0

# ── Step 2: シード追試 (seed 1, 2) ───────────────────────────────────────────
# seed 1
run_orig 1
run_cls 0.05 0.0 1
run_cls 0.3  0.0 1
# seed 2
run_orig 2
run_cls 0.05 0.0 2
run_cls 0.3  0.0 2

# ── 結果サマリー ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === 全結果サマリー ==="
printf "%-55s  %5s  %10s  %8s\n" "run" "seed" "clean_acc" "mCE"

# seed 0 既存分
for RUN in \
    screen_apr_s_orig \
    screen_apr_s_orig_cls_gs0.05 \
    screen_apr_s_orig_cls_gs0.3; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-55s  %5s  %10s  %8s\n" "$RUN" "0" "$CLEAN" "$MCE"
done

# 無条件対照 (seed 0)
for US in 0.025 0.15; do
    RUN=screen_apr_s_orig_cls_gs0.0_us${US}_s0
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-55s  %5s  %10s  %8s\n" "$RUN" "0" "$CLEAN" "$MCE"
done

# シード追試 (seed 1, 2)
for SEED in 1 2; do
    for RUN_BASE in \
        screen_apr_s_orig \
        "screen_apr_s_orig_cls_gs0.05_us0.0" \
        "screen_apr_s_orig_cls_gs0.3_us0.0"; do
        RUN=${RUN_BASE}_s${SEED}
        DIR=./results/$RUN
        CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
        MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
        printf "%-55s  %5s  %10s  %8s\n" "$RUN" "$SEED" "$CLEAN" "$MCE"
    done
done

echo ""
echo "[$(date)] === 完了 ==="
