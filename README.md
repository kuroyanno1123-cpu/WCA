# WCA: Wavelet Cross-synthesis Augmentation + JSD Consistency Loss

WaveletBasisSwap (WBS) augmentation と Jensen-Shannon Divergence (JSD) 整合損失  
(AugMix, Hendrycks et al., ICLR 2020) を組み合わせた CIFAR-10 学習パイプライン。

## Setup

```bash
conda activate apr
pip install pywt
```

## Dataset structure

```
{data}/
  cifar10/
    cifar-10-batches-py/   ← torchvision が自動ダウンロード

{data_c}/
  CIFAR-10-C/
    gaussian_noise.npy
    labels.npy
    ...  (15 corruption types)
```

## Training

```bash
# WCA + JSD (新手法, デフォルト aug_order=crop_first)
python3 main.py \
  --aug wca --source haar --target db8 --level 1 --swap-prob 0.2 \
  --jsd-lambda 12 \
  --data {data} --outfolder ./results/wca_jsd --memo wca_jsd

# WCA + CE only (jsd_lambda=0 → Dataset は1枚返し, 単一順伝播, aug_order=wca_first)
python3 main.py \
  --aug wca --source haar --target db8 --level 1 --swap-prob 0.2 \
  --jsd-lambda 0 \
  --data {data} --outfolder ./results/wca_ce --memo wca_ce

# aug_order を明示上書き
python3 main.py --aug wca --jsd-lambda 12 --aug-order wca_first ...

# Baseline (拡張なし, CE only)
python3 main.py --aug none --jsd-lambda 0 --data {data} --outfolder ./results/baseline
```

## Evaluation

```bash
python3 main.py \
  --aug wca --source haar --target db8 --level 1 --swap-prob 0.2 \
  --jsd-lambda 12 \
  --data {data} --data-c {data_c} \
  --outfolder ./results/wca_jsd --memo wca_jsd --eval eval
```

## Augmentation order (--aug-order)

| フラグ | CE モード (jsd_lambda=0) | JSD モード (jsd_lambda>0) |
|---|---|---|
| `wca_first` | WCA→Crop/Flip→ToTensor | x_clean: Crop/Flip, x_aug: WCA→Crop/Flip (独立) |
| `crop_first` | Crop/Flip→WCA→ToTensor | shared base: Crop/Flip; x_aug1/2: WCA(base) |
| **(default)** | **wca_first** | **crop_first** |

## λ=0 のときの等価性

`--jsd-lambda 0` を指定すると:
- `CIFAR10TrainDataset` は `(x, y)` を返す (3枚タプルなし)
- 順伝播は1枚分のみ (`torch.cat` なし)
- 損失 = `CrossEntropy(logits, targets)` のみ
- BatchNorm 統計は通常の CE 学習と完全等価

検証コマンド (出力 loss が CE-only と一致することを確認):

```bash
python3 main.py --aug wca --jsd-lambda 0 --max-epoch 1 \
  --data {data} --outfolder ./results/verify_lambda0 --memo test
```

## Reference result: mCE 11.33%

VIPAug_phase コードベース (amplitude_mix 込み) での再現コマンド:

```bash
# /home/kairisasaki/VIPAug_phase/ で実行
python main.py \
  --aug wbs_db8_p02_l1 \
  --dataset cifar10 --batch-size 128 --max-epoch 250 \
  --data /home/kairisasaki/data \
  --data-c /home/kairisasaki/APR_phase/data \
  --outfolder ./results/wbs_db8_p02_l1 --memo wbs_db8_p02_l1
```

> **Note**: VIPAug_phase の学習ループは WBS 拡張に加えて `amplitude_mix`  
> (APR 方式の振幅スワップ) を適用しています。本 WCA コードはこれを JSD 損失に  
> 置き換えた実装です。`--jsd-lambda 0` (CE only) の結果は amplitude_mix なしの  
> WBS 単体性能に対応します。

## File structure

```
WCA/
├── main.py              # エントリーポイント (--eval eval で評価モード)
├── core/
│   ├── wca.py           # WaveletBasisSwap / WaveletBasisSwapMol / WaveletCrossAug(V2)
│   └── losses.py        # jsd_loss()
├── datasets/
│   └── cifar.py         # CIFAR10TrainDataset / TestDataset / C-Dataset / build_loaders()
├── model/
│   └── resnet.py        # ResNet-18 (VIPAug_phase と同一)
└── utils/
    └── metrics.py       # AverageMeter / Logger / save_networks / test / test_robustness
```
