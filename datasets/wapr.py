"""WaveletAPR (wapr-orig) — APR-S の DTCWT 版。

APR-S (FFT) との差分は周波数分解領域のみ。それ以外は _apr_orig_core と完全同一:
  - aug_list からのオペ選択 (np.random.choice)
  - severity=3 固定
  - 2つの p ゲートの位置・判定順 (random.uniform)
  - 同一画像の 2 ビュー生成 (x.copy())
  - 2方向再結合のランダム選択 (p > 0.5 で方向 A)
  - RNG 消費回数・順序が _apr_orig_core と完全一致

既知の逸脱 (DEVIATION コメントで明記):
  1. チャネルの扱い: APR-S は fftn (H,W,C の 3D FFT, チャネル間結合)。
     DTCWT は 2D → チャネルが独立になる。これは原理的な差分。
  2. クリッピング: APR-S は astype(uint8) のオーバーフロー挙動を再現。
     DTCWT 版は逆変換後に [0,255] へクリップ (飽和) して uint8 変換。
  3. 射影損失: DTCWT は 4 倍冗長。任意の (振幅, 位相) の組は解析作用素の
     像に入らない可能性がある。Gate 2 で定量化。
"""

import random
import numpy as np
from PIL import Image
import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from pytorch_wavelets import DTCWTForward, DTCWTInverse

from datasets.cifar import (
    _RawCIFAR10, _worker_init_fn, CIFAR10TestDataset, CIFAR10CDataset, CORRUPTION_TYPES,
)
import datasets.augmentations_orig as _aug_orig_mod

# APR-S-orig と同一の aug_list (再定義禁止)
_ORIG_AUG_LIST = _aug_orig_mod.augmentations

# APR-S-orig と同一の post-APR 変換
_APR_ORIG_POST = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])


# ── PIL ↔ Torch 変換ユーティリティ ──────────────────────────────────────────

def _pil_to_torch(img: Image.Image) -> torch.Tensor:
    """PIL (H,W,3) uint8 → torch (1,3,H,W) float32 in [0,1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _torch_to_uint8(t: torch.Tensor):
    """(1,3,H,W) float32 → uint8 ndarray (H,W,3) with saturation logging.

    Returns (uint8_array, sat_rate).
    # DEVIATION: clip to [0,255] (APR-S uses astype(uint8) overflow/wraparound)
    """
    arr = t.squeeze(0).permute(1, 2, 0).numpy() * 255.0  # (H,W,3) float
    sat_rate = float(np.mean((arr < 0.0) | (arr > 255.0)))
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr, sat_rate


# ── DTCWT 再結合 (テスト可能な低レベル関数) ──────────────────────────────────

def _dtcwt_recombine(yl1, yh1, yl2, yh2, lam: float, lowpass_mode: str, dir_a: bool, ifm):
    """DTCWT 係数の再結合 + 逆変換。

    dir_a=True  (p>0.5): amp(x1) + phase(x2)  ← APR-S fft_1 に対応
    dir_a=False (p≤0.5): amp(x2) + phase(x1)  ← APR-S fft_2 に対応

    lam=0: 位相源の振幅で z_new を作る → 同一ソースなら恒等 (恒等性 A)
    lam=1: 振幅源が 100% になる → 同一ソースなら恒等 (恒等性 B)
    """
    yh_new = []
    for j in range(len(yh1)):
        h1 = yh1[j]   # (B, C, 6, H_j, W_j, 2)
        h2 = yh2[j]
        r1, i1 = h1[..., 0], h1[..., 1]
        r2, i2 = h2[..., 0], h2[..., 1]

        mag1 = torch.sqrt(r1 * r1 + i1 * i1)
        mag2 = torch.sqrt(r2 * r2 + i2 * i2)

        if dir_a:
            # amp from x1 (own), phase from x2 (aug)
            mixed_mag = lam * mag1 + (1.0 - lam) * mag2
            phase_src_r, phase_src_i = r2, i2
            phase_src_mag = mag2
        else:
            # amp from x2 (aug), phase from x1 (own)
            mixed_mag = lam * mag2 + (1.0 - lam) * mag1
            phase_src_r, phase_src_i = r1, i1
            phase_src_mag = mag1

        # 位相源が near-zero → 位相が数値的に未定義 → x1 係数でフォールバック
        safe = (phase_src_mag >= 1e-8)
        phase = torch.atan2(phase_src_i, phase_src_r)

        z_new_r = torch.where(safe, mixed_mag * torch.cos(phase), r1)
        z_new_i = torch.where(safe, mixed_mag * torch.sin(phase), i1)

        yh_new.append(torch.stack([z_new_r, z_new_i], dim=-1))

    if lowpass_mode == 'amp':
        yl_new = yl2
    elif lowpass_mode == 'phase':
        yl_new = yl1
    else:  # 'mix'
        yl_new = lam * yl2 + (1.0 - lam) * yl1

    with torch.no_grad():
        out_t = ifm((yl_new, yh_new))   # (B,C,H,W) float

    return out_t


# ── WAPR コア ────────────────────────────────────────────────────────────────

def _wapr_orig_core(
    x: Image.Image,
    xfm: DTCWTForward,
    ifm: DTCWTInverse,
    lam,            # float または 'uniform'
    lowpass_mode: str,  # 'amp' | 'phase' | 'mix'
):
    """DTCWT 版 APRecombination コア。

    (PIL image, swapped: bool, sat_rate: float) を返す。

    RNG 消費順序 (_apr_orig_core と完全一致):
      1. np.random.choice(_ORIG_AUG_LIST)       ← op1 選択
      2. [op1 内部の np.random 消費]
      3. random.uniform(0, 1)                   ← ゲート1
      [早期リターン or 続行]
      4. np.random.choice(_ORIG_AUG_LIST)       ← op2 選択
      5. [op2 内部の np.random 消費]
      6. random.uniform(0, 1)                   ← ゲート2 / 方向選択
      [lam='uniform' の場合のみ: random.random() を追加消費]

    # DEVIATION (channel): APR-S は 3D FFT でチャネル間が結合するが、
    #   DTCWT は 2D (チャネル独立)。これは原理的に埋められない差分。
    """
    # ── ステップ 1-2: op1 選択・適用 (_apr_orig_core と同一) ─────────────────
    op = np.random.choice(_ORIG_AUG_LIST)
    x = op(x, 3)

    # ── ステップ 3: ゲート1 (_apr_orig_core と同一) ──────────────────────────
    p = random.uniform(0, 1)
    if p > 0.5:
        return x, False, 0.0

    # ── ステップ 4-5: op2 選択・適用 (_apr_orig_core と同一) ─────────────────
    x_aug = x.copy()
    op = np.random.choice(_ORIG_AUG_LIST)
    x_aug = op(x_aug, 3)

    # ── DTCWT 分解 (RNG 消費なし) ────────────────────────────────────────────
    x1_t = _pil_to_torch(x)       # (1,3,H,W)
    x2_t = _pil_to_torch(x_aug)

    with torch.no_grad():
        yl1, yh1 = xfm(x1_t)     # yl:(1,3,H_l,W_l), yh[j]:(1,3,6,H_j,W_j,2)
        yl2, yh2 = xfm(x2_t)

    # ── ステップ 6: ゲート2 / 方向選択 (_apr_orig_core と同一) ───────────────
    p = random.uniform(0, 1)
    # dir_a=True  (p>0.5): amp(x1) + phase(x2)  ← APR-S の fft_1 に対応
    # dir_a=False (p≤0.5): amp(x2) + phase(x1)  ← APR-S の fft_2 に対応
    dir_a = (p > 0.5)

    # lam サンプリング (lam='uniform' 時のみ追加 RNG 消費。方向選択より後なので
    # APR-S との RNG 一致テストに影響しない。デフォルト lam=1.0 では消費なし)
    actual_lam = random.random() if lam == 'uniform' else float(lam)

    # ── 再結合と逆変換 ────────────────────────────────────────────────────────
    out_t = _dtcwt_recombine(
        yl1, yh1, yl2, yh2,
        actual_lam, lowpass_mode, dir_a, ifm,
    )

    # ── PIL 変換 (clip → uint8) ───────────────────────────────────────────────
    # DEVIATION: clip to [0,255] (APR-S uses astype(uint8) overflow/wraparound)
    out_arr, sat_rate = _torch_to_uint8(out_t)
    return Image.fromarray(out_arr), True, sat_rate


# ── WAPROrigDataset ──────────────────────────────────────────────────────────

class WAPROrigDataset(Dataset):
    """wapr-orig: APR-S の DTCWT 版。APROrigDataset と同一インタフェース。

    返り値: (x_tensor, y_own)  ← apr-s-orig と同一 (CE 学習)
    """

    def __init__(
        self,
        root,
        J: int = 3,
        lam=1.0,
        lowpass_mode: str = 'amp',
        download: bool = True,
    ):
        self._raw  = _RawCIFAR10(root, train=True, download=download)
        self._n    = len(self._raw)
        self.lam   = lam
        self.lowpass_mode = lowpass_mode

        # DTCWT 変換 (DataLoader worker ごとに 1 インスタンス)
        self._xfm = DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b')
        self._ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        img, y_own = self._raw[idx]              # PIL 32×32
        img, _, _  = _wapr_orig_core(
            img, self._xfm, self._ifm,
            self.lam, self.lowpass_mode,
        )
        x_tensor = _APR_ORIG_POST(img)           # crop → flip → tensor → normalize
        return x_tensor, y_own


# ── DataLoader ファクトリ ──────────────────────────────────────────────────────

def build_wapr_loaders(args, eval_mode=False):
    """wapr-orig 用 DataLoader を構築する。"""
    data_root = os.path.join(args.data, 'cifar10')

    # lam: float か文字列 'uniform'
    lam = args.wapr_lam if args.wapr_lam != 'uniform' else 'uniform'
    if lam != 'uniform':
        lam = float(lam)

    train_ds = WAPROrigDataset(
        data_root,
        J=args.wapr_levels,
        lam=lam,
        lowpass_mode=args.wapr_lowpass,
    )

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
