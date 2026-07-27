#!/bin/bash
# 全スクリーニング済み実験の 250ep フル実行
# 既完了: full_wca_ce / full_wca_jsd(λ=12) / full_augmix_jsd はスキップ

cd /home/kairisasaki/WCA

DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
GPU=0
SEED=0
MAX_EPOCH=250
BATCH_SIZE=128

# ── WCA + JSD (λ グリッド) ─────────────────────────────────────────────────────

run_wca_jsd() {
    local LAM=$1
    local MEMO=full_wca_jsd_l${LAM}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO (λ=${LAM}) ==="

    conda run -n apr python main.py \
      --aug wca --source haar --target db8 --level 1 --swap-prob 0.2 \
      --jsd-lambda $LAM \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug wca --source haar --target db8 --level 1 --swap-prob 0.2 \
      --jsd-lambda $LAM \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

# ── WCA + basis_random ────────────────────────────────────────────────────────

run_basisrand() {
    local PROB=$1
    local LAM=$2
    local MEMO=full_wca_basisrand_p${PROB/./}_jsd${LAM}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO (p=${PROB}, λ=${LAM}) ==="

    conda run -n apr python main.py \
      --aug wca --basis-random --level 1 --swap-prob $PROB \
      --jsd-lambda $LAM \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug wca --basis-random --level 1 --swap-prob $PROB \
      --jsd-lambda $LAM \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

# ── APR-S ────────────────────────────────────────────────────────────────────

run_apr_s() {
    local MEMO=full_apr_s
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO ==="

    conda run -n apr python main.py \
      --aug apr-s \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug apr-s \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

# ── APR-S-cls ────────────────────────────────────────────────────────────────

run_apr_cls() {
    local GS=$1
    local US=$2
    local MEMO=full_apr_s_cls_gs${GS}_us${US}
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO ==="

    conda run -n apr python main.py \
      --aug apr-s-cls --gamma-swap $GS --uncond-smooth $US \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug apr-s-cls --gamma-swap $GS --uncond-smooth $US \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

# ── APR-S-soft ───────────────────────────────────────────────────────────────

run_apr_soft() {
    local MEMO=full_apr_s_soft
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Start: $MEMO ==="

    conda run -n apr python main.py \
      --aug apr-s-soft --t-max 1.0 --label-k 1.0 --clean-prob 0.0 \
      --data $DATA --dataset cifar10 \
      --batch-size $BATCH_SIZE --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1

    echo "[$(date)] Training done: $MEMO"

    conda run -n apr python main.py --eval eval \
      --aug apr-s-soft --t-max 1.0 --label-k 1.0 --clean-prob 0.0 \
      --data $DATA --data-c $DATA_C --dataset cifar10 \
      --batch-size $BATCH_SIZE --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      >> $OUT/train.log 2>&1

    echo "[$(date)] Eval done: $MEMO"
}

# ── 実行順序 (50ep 結果が良い順) ─────────────────────────────────────────────

run_wca_jsd 18          # 50ep mCE 23.06% ← 最良
run_wca_jsd 6           # 50ep mCE 24.83%
run_wca_jsd 21          # 50ep mCE 23.63%
run_wca_jsd 3           # 50ep mCE 27.88%
run_basisrand 0.4 12    # 50ep basis_random p=0.4
run_basisrand 0.2 12    # 50ep basis_random p=0.2
run_apr_s               # 50ep mCE 29.25%
run_apr_cls 0.2 0.1     # 50ep mCE 28.78% ← APR-cls 最良
run_apr_cls 0.2 0.0     # 50ep mCE 29.05%
run_apr_cls 0.1 0.0     # 50ep mCE 30.19%
run_apr_soft            # 50ep mCE 43.62% (最後)

# ── 結果サマリー ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] === 250ep 全実験 結果サマリー ==="
echo ""
printf "%-40s  %10s  %10s\n" "run" "clean_acc" "mCE"

# 既完了分
for RUN in full_wca_ce full_wca_jsd full_augmix_jsd; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-40s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE"
done

# 今回実行分
for RUN in \
    full_wca_jsd_l18 full_wca_jsd_l6 full_wca_jsd_l21 full_wca_jsd_l3 \
    full_wca_basisrand_p04_jsd12 full_wca_basisrand_p02_jsd12 \
    full_apr_s \
    full_apr_s_cls_gs0.2_us0.1 full_apr_s_cls_gs0.2_us0.0 full_apr_s_cls_gs0.1_us0.0 \
    full_apr_s_soft; do
    DIR=./results/$RUN
    CLEAN=$(grep "clean accuracy" $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    MCE=$(grep "Mean Error"    $DIR/logs.txt 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-40s  %10s  %10s\n" "$RUN" "$CLEAN" "$MCE"
done

echo ""
echo "[$(date)] === 完了 ==="
