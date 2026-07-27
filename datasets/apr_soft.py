"""APR-S and APR-S-soft augmentation datasets.

APR-S (hard swap, ported from gary23ai/APR):
  - 50% 確率で振幅スワップを適用（原実装の流儀に従う）
  - 振幅: 別サンプル(distractor)の振幅で完全置換
  - 位相: 元画像のものを保持 (Phase-invariant = APR-S)
  - ラベル: y_own のみ (通常 CE)

APR-S-soft (提案手法):
  - 毎サンプル t ~ Uniform(0,1) を引いて連続混合
  - A_mixed = (1-t*t_max)*A_own + t*t_max*A_dist
  - 位相: 元画像のものを保持
  - ラベル: gamma = (t*t_max)^k で soft label を計算
  - E[t] = 0.5 = APR-S の期待混合強度と一致

APR-S-orig (原実装忠実版, gary23ai/APR APRecombination):
  - image-space aug 2種を同一画像に適用して2つのビューを生成
  - 両ビューのFFT振幅・位相を再結合 (2方向ランダム選択)
  - distractor は別画像ではなく x.copy() に別 aug を適用したもの
  - clip なし, astype(uint8) のオーバーフロー挙動を完全再現
  - 原実装順序: APRecombination → RandomCrop → RandomHorizontalFlip → ToTensor

FFT の流儀: gary23ai/APR に合わせ np.fft.fftn (H×W×C の 3D FFT) + fftshift を使用。
t_eff=1 のとき APR-S hard swap と数値的に一致する。
"""

import random
import numpy as np
from PIL import Image
import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from datasets.cifar import (
    _RawCIFAR10, _BASIC_AUG, _TO_TENSOR_NORM,
    _worker_init_fn, CIFAR10TestDataset, CIFAR10CDataset, CORRUPTION_TYPES,
)
import datasets.augmentations_orig as _aug_orig_mod

# aug_list は原実装と同一の 9 演算
_ORIG_AUG_LIST = _aug_orig_mod.augmentations

# APROrigDataset 用 post-APR 変換 (crop → flip → tensor → normalize)
_APR_ORIG_POST = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])


# ── FFT 振幅混合 ──────────────────────────────────────────────────────────────

def _fft_amplitude_mix(img_own: np.ndarray, img_dist: np.ndarray, t_eff: float) -> np.ndarray:
    """振幅スペクトルを連続補間して再構成する (gary23ai/APR の fftn 流儀)。

    A_mixed = (1-t_eff)*A_own + t_eff*A_dist
    位相は img_own のものを保持。

    t_eff=0: img_own の round-trip 再構成 (恒等変換)
    t_eff=1: img_dist の振幅 + img_own の位相 (APR-S hard swap)

    入力: uint8 ndarray (H, W, C)
    出力: uint8 ndarray (H, W, C)
    """
    x = img_own.astype(np.float64)
    d = img_dist.astype(np.float64)

    fft_x = np.fft.fftshift(np.fft.fftn(x))
    fft_d = np.fft.fftshift(np.fft.fftn(d))

    amp_x   = np.abs(fft_x)
    amp_d   = np.abs(fft_d)
    phase_x = np.angle(fft_x)

    amp_mixed = (1.0 - t_eff) * amp_x + t_eff * amp_d
    fft_mixed = amp_mixed * np.exp(1j * phase_x)

    out = np.fft.ifftn(np.fft.ifftshift(fft_mixed)).real
    return np.clip(out, 0, 255).astype(np.uint8)


# ── APR-S-soft Dataset ────────────────────────────────────────────────────────

class APRSoftDataset(Dataset):
    """APR-S-soft: 連続振幅混合 + 劣化度連動ラベル混合。

    __getitem__ が返すもの:
      (x_aug, t_eff, y_own, y_dist)
      x_aug  : normalized tensor (C, H, W)
      t_eff  : scalar float32 tensor, t_eff = t * t_max
      y_own  : int - 元画像のクラスラベル
      y_dist : int - distractor のクラスラベル

    distractor はデータセット全体からランダムにサンプリングする。
    clean_prob=0 のとき毎サンプル t ~ Uniform(0,1) を適用する。
    """

    def __init__(self, root, t_max=1.0, label_k=1.0, clean_prob=0.0, download=True):
        self._raw      = _RawCIFAR10(root, train=True, download=download)
        self.t_max     = t_max
        self.label_k   = label_k
        self.clean_prob = clean_prob
        self._n        = len(self._raw)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        img, y_own = self._raw[idx]

        # t をサンプル (Python random は _worker_init_fn でシード済み)
        t = random.random()
        if random.random() < self.clean_prob:
            t = 0.0
        t_eff = t * self.t_max

        # 基本 aug (Crop/Flip) を元画像に適用
        img_aug = _BASIC_AUG(img)

        # distractor をデータセット全体からランダムサンプリング
        dist_idx = random.randrange(self._n - 1)
        if dist_idx >= idx:
            dist_idx += 1
        img_dist, y_dist = self._raw[dist_idx]
        img_dist_aug = _BASIC_AUG(img_dist)

        # FFT 振幅混合
        mixed = _fft_amplitude_mix(
            np.array(img_aug), np.array(img_dist_aug), t_eff
        )

        x_tensor = _TO_TENSOR_NORM(Image.fromarray(mixed))
        return x_tensor, torch.tensor(t_eff, dtype=torch.float32), y_own, y_dist


# ── APR-S Dataset (hard swap ベースライン) ────────────────────────────────────

class APRHardDataset(Dataset):
    """APR-S: 振幅ハードスワップ (ported from gary23ai/APR)。

    apply_prob=0.5: 原実装の流儀に従い 50% の確率で適用。
    適用なし時は元画像 (Crop/Flip のみ) を返す。
    APR-S の期待混合強度 E[effective_t] = apply_prob * 1.0 = 0.5。

    return_flag=False: (x_aug, y_own)          ← apr-s 用
    return_flag=True:  (x_aug, y_own, swapped)  ← apr-s-cls 用
      swapped: bool tensor, スワップが実際に適用されたか
    """

    # 原実装の適用確率 (gary23ai/APR: `if p > 0.5: return x`)
    DEFAULT_APPLY_PROB = 0.5

    def __init__(self, root, apply_prob=DEFAULT_APPLY_PROB,
                 return_flag=False, download=True):
        self._raw        = _RawCIFAR10(root, train=True, download=download)
        self.apply_prob  = apply_prob
        self.return_flag = return_flag
        self._n          = len(self._raw)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        img, y_own = self._raw[idx]
        img_aug = _BASIC_AUG(img)

        swapped = False
        if random.random() < self.apply_prob:
            dist_idx = random.randrange(self._n - 1)
            if dist_idx >= idx:
                dist_idx += 1
            img_dist, _ = self._raw[dist_idx]
            img_dist_aug = _BASIC_AUG(img_dist)

            mixed = _fft_amplitude_mix(
                np.array(img_aug), np.array(img_dist_aug), 1.0  # hard swap: t_eff=1
            )
            x_tensor = _TO_TENSOR_NORM(Image.fromarray(mixed))
            swapped = True
        else:
            x_tensor = _TO_TENSOR_NORM(img_aug)

        if self.return_flag:
            return x_tensor, y_own, torch.tensor(swapped, dtype=torch.bool)
        return x_tensor, y_own


# ── APR-S-orig: 原実装 APRecombination 忠実移植 ──────────────────────────────

def _apr_orig_core(x: Image.Image):
    """APRecombination の共通コア。PIL image と swapped フラグを返す。

    swapped=False: p>0.5 で早期リターン (FFT 再結合なし)
    swapped=True:  FFT 振幅・位相の再結合を適用

    既存の _apr_orig_call はこの関数を薄くラップするだけで、
    バイト一致テストへの影響はない。
    """
    op = np.random.choice(_ORIG_AUG_LIST)
    x = op(x, 3)

    p = random.uniform(0, 1)
    if p > 0.5:
        return x, False   # early return: no amplitude recombination

    x_aug = x.copy()
    op = np.random.choice(_ORIG_AUG_LIST)
    x_aug = op(x_aug, 3)

    x_np     = np.array(x).astype(np.uint8)
    x_aug_np = np.array(x_aug).astype(np.uint8)

    fft_1 = np.fft.fftshift(np.fft.fftn(x_np))
    fft_2 = np.fft.fftshift(np.fft.fftn(x_aug_np))

    abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
    abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

    fft_1 = abs_1 * np.exp(1j * angle_2)   # amp(x) + phase(x_aug)
    fft_2 = abs_2 * np.exp(1j * angle_1)   # amp(x_aug) + phase(x)

    p = random.uniform(0, 1)
    if p > 0.5:
        out = np.fft.ifftn(np.fft.ifftshift(fft_1))
    else:
        out = np.fft.ifftn(np.fft.ifftshift(fft_2))

    out = out.astype(np.uint8)   # no clip: overflow/wraparound as original
    return Image.fromarray(out), True


def _apr_orig_call(x: Image.Image) -> Image.Image:
    """gary23ai/APR の APRecombination.__call__ と同一ロジック。

    バイト一致保証のため原コードの挙動をそのまま再現:
    - np.random.choice で aug を選択 (np.random)
    - p = random.uniform(0, 1) (Python random)
    - x.copy() に別 aug を適用 (distractor は別画像でない)
    - 2 方向からランダム選択 (p>0.5: amp1+phase2, else: amp2+phase1)
    - clip なし、astype(uint8) のオーバーフロー挙動を再現
    """
    img, _ = _apr_orig_core(x)
    return img


class APROrigDataset(Dataset):
    """APR-S 原実装忠実版 (gary23ai/APR APRecombination を完全再現)。

    原実装順序: APRecombination → RandomCrop(32, pad=4) → HFlip → ToTensor → Normalize
    ラベル: y_own のみ (CE), 返り値は (x_tensor, y_own)
    """

    def __init__(self, root, download=True):
        self._raw = _RawCIFAR10(root, train=True, download=download)
        self._n   = len(self._raw)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        img, y_own = self._raw[idx]            # PIL 32×32
        img = _apr_orig_call(img)              # APRecombination (PIL → PIL)
        x_tensor = _APR_ORIG_POST(img)         # crop → flip → tensor → normalize
        return x_tensor, y_own


class APROrigClsDataset(Dataset):
    """APR-S-orig + 条件付きラベルスムージング (apr-s-orig-cls)。

    _apr_orig_core でスワップフラグを取得し:
      swapped=True:  y_soft = (1-γ)·one_hot(y_own) + γ/C
      swapped=False: 通常 CE (y_soft = one_hot(y_own))

    返り値: (x_tensor, y_own, swapped: BoolTensor)
    """

    def __init__(self, root, download=True):
        self._raw = _RawCIFAR10(root, train=True, download=download)
        self._n   = len(self._raw)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        img, y_own = self._raw[idx]
        img, swapped = _apr_orig_core(img)
        x_tensor = _APR_ORIG_POST(img)
        return x_tensor, y_own, torch.tensor(swapped, dtype=torch.bool)


# ── DataLoader ファクトリ ──────────────────────────────────────────────────────

def build_apr_loaders(args, eval_mode=False):
    """APR-S / APR-S-soft / APR-S-orig 用 DataLoader を構築する。"""
    data_root = os.path.join(args.data, 'cifar10')

    if args.aug == 'apr-s-soft':
        train_ds = APRSoftDataset(
            data_root,
            t_max=args.t_max,
            label_k=args.label_k,
            clean_prob=args.clean_prob,
        )
    elif args.aug == 'apr-s-cls':
        train_ds = APRHardDataset(data_root, return_flag=True)
    elif args.aug == 'apr-s-orig':
        train_ds = APROrigDataset(data_root)
    elif args.aug == 'apr-s-orig-cls':
        train_ds = APROrigClsDataset(data_root)
    else:  # apr-s
        train_ds = APRHardDataset(data_root)

    test_ds = CIFAR10TestDataset(data_root)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    corruption_loaders = {}
    if eval_mode:
        c_root = os.path.join(args.data_c, 'CIFAR-10-C')
        for ctype in CORRUPTION_TYPES:
            c_ds = CIFAR10CDataset(c_root, ctype)
            corruption_loaders[ctype] = DataLoader(
                c_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )

    return train_loader, test_loader, corruption_loaders
