"""ウェーブレット係数レベルA/B融合拡張 (WaveletFusion)。

モード:
  sign_align : 出力係数 = sign(A係数) × |B係数|
               符号はA、絶対値はBから取る
  max_coef   : 各位置で |A| と |B| を比較し、絶対値が大きい方の係数をそのまま採用

共通仕様:
- LL（近似）サブバンドは常にAを使う
- wavelet基底・分解レベルは引数指定（デフォルト haar / level 1）
- apply_prob=0.5 のゲートで適用判定（非適用時はAをそのまま返す）
- チャンネル（RGB）ごとに独立に変換
- 再構成後 [0,255] clip → PIL → _TO_TENSOR_NORM（WCAのclip方針）
- ラベルは常にAのもの
- 返り値: (x_tensor, y_own, applied: BoolTensor)
"""

import random
import numpy as np
from PIL import Image
import os

import pywt
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.cifar import (
    _RawCIFAR10, _BASIC_AUG, _TO_TENSOR_NORM,
    _worker_init_fn, CIFAR10TestDataset, CIFAR10CDataset, CORRUPTION_TYPES,
)


# ── コア変換 ──────────────────────────────────────────────────────────────────

def _wavelet_fusion(a: np.ndarray, b: np.ndarray,
                    mode: str, wavelet: str = 'haar', level: int = 1) -> np.ndarray:
    """ウェーブレット係数レベルでAとBを融合して再構成する。

    引数:
        a, b   : (H, W, C) float64
        mode   : 'sign_align' or 'max_coef'
        wavelet: pywt wavelet 名
        level  : 分解レベル

    返り値:
        (H, W, C) float64, clip なし
    """
    H, W, C = a.shape
    out = np.empty_like(a, dtype=np.float64)

    for c in range(C):
        coeffs_a = pywt.wavedec2(a[:, :, c], wavelet, level=level)
        coeffs_b = pywt.wavedec2(b[:, :, c], wavelet, level=level)

        # LL は A 固定
        fused = [coeffs_a[0]]

        for lvl in range(1, len(coeffs_a)):
            detail_a = coeffs_a[lvl]   # (LH, HL, HH) tuple
            detail_b = coeffs_b[lvl]
            fused_detail = []

            for da, db in zip(detail_a, detail_b):
                if mode == 'sign_align':
                    # 符号=A, 絶対値=B
                    coef = np.sign(da) * np.abs(db)
                else:  # max_coef
                    # 絶対値が大きい方を符号ごと採用（同値はA優先）
                    coef = np.where(np.abs(da) >= np.abs(db), da, db)
                fused_detail.append(coef)

            fused.append(tuple(fused_detail))

        rec = pywt.waverec2(fused, wavelet)
        # waverec2 は入力より大きい shape を返す場合があるので crop
        out[:, :, c] = rec[:H, :W]

    return out  # float64, no clip


# ── Dataset ──────────────────────────────────────────────────────────────────

class WaveletFusionDataset(Dataset):
    """ウェーブレット係数融合拡張 Dataset。

    mode='sign_align' / 'max_coef' で切替。
    apply_prob=0.5 ゲート: 非適用時はAの basic aug 結果をそのまま返す。

    返り値: (x_tensor, y_own, applied: BoolTensor)
    """

    def __init__(self, root, mode, wavelet='haar', level=1,
                 apply_prob=0.5, download=True):
        assert mode in ('sign_align', 'max_coef'), f'Unknown mode: {mode}'
        self._raw       = _RawCIFAR10(root, train=True, download=download)
        self.mode       = mode
        self.wavelet    = wavelet
        self.level      = level
        self.apply_prob = apply_prob
        self._n         = len(self._raw)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        img_a, y_a = self._raw[idx]
        img_a_aug  = _BASIC_AUG(img_a)

        applied = False
        if random.random() < self.apply_prob:
            b_idx = random.randrange(self._n - 1)
            if b_idx >= idx:
                b_idx += 1
            img_b, _  = self._raw[b_idx]
            img_b_aug = _BASIC_AUG(img_b)

            a_np = np.array(img_a_aug).astype(np.float64)
            b_np = np.array(img_b_aug).astype(np.float64)

            fused = _wavelet_fusion(a_np, b_np, self.mode, self.wavelet, self.level)

            # WCA の clip 方針に合わせて [0,255] clip → PIL 変換
            fused_pil = Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8))
            x_tensor  = _TO_TENSOR_NORM(fused_pil)
            applied   = True
        else:
            x_tensor = _TO_TENSOR_NORM(img_a_aug)

        return x_tensor, y_a, torch.tensor(applied, dtype=torch.bool)


# ── DataLoader ファクトリ ─────────────────────────────────────────────────────

def build_wf_loaders(args, eval_mode=False):
    """WaveletFusion 用 DataLoader を構築する。"""
    data_root = os.path.join(args.data, 'cifar10')
    mode      = 'sign_align' if args.aug == 'wf-sign' else 'max_coef'

    train_ds = WaveletFusionDataset(
        data_root, mode=mode,
        wavelet=args.wf_wavelet, level=args.wf_level,
        apply_prob=args.wf_apply_prob,
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
