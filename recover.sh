#!/bin/bash
# 再起動後の回復実行スクリプト
# full_wca_ce の eval + 残り2実験の訓練+eval

cd /home/kairisasaki/WCA
DATA=/home/kairisasaki/data
DATA_C=/home/kairisasaki/APR_phase/data
SEED=0; MAX_EPOCH=250; BATCH_SIZE=128; GPU=0
SOURCE=haar; TARGET=db8; LEVEL=1; SWAP_PROB=0.2

run_eval() {
    local MEMO=$1; local EXTRA=$2; local JSD=$3
    echo "[$(date)] === Eval: $MEMO ==="
    conda run -n apr python main.py --eval eval \
      $EXTRA --jsd-lambda $JSD \
      --data $DATA --data-c $DATA_C \
      --dataset cifar10 --batch-size $BATCH_SIZE --gpu $GPU \
      --seed $SEED --outfolder ./results/$MEMO --memo $MEMO \
      >> ./results/$MEMO/train.log 2>&1
    grep -E "clean accuracy|Mean Error|mean error" ./results/$MEMO/train.log | tail -3
}

run_train_eval() {
    local MEMO=$1; local EXTRA=$2; local JSD=$3
    local OUT=./results/$MEMO
    mkdir -p $OUT
    echo "[$(date)] === Train 250ep: $MEMO ==="
    conda run -n apr python main.py \
      $EXTRA --jsd-lambda $JSD \
      --data $DATA --dataset cifar10 --batch-size $BATCH_SIZE \
      --max-epoch $MAX_EPOCH --gpu $GPU --seed $SEED \
      --outfolder $OUT --memo $MEMO \
      > $OUT/train.log 2>&1
    echo "[$(date)] Training done: $MEMO"
    run_eval $MEMO "$EXTRA" $JSD
}

WCA_ARGS="--aug wca --source $SOURCE --target $TARGET --level $LEVEL --swap-prob $SWAP_PROB"

# 1. full_wca_ce の eval のみ (訓練済み)
run_eval full_wca_ce "$WCA_ARGS" 0

# 2. WCA + JSD (250ep)
run_train_eval full_wca_jsd "$WCA_ARGS" 12

# 3. AugMix + JSD (250ep)
run_train_eval full_augmix_jsd "--aug augmix" 12

echo "[$(date)] === 全回復完了 ==="
