"""
WCA: Wavelet Cross-synthesis Augmentation

WaveletCrossAug (旧実装):
  1. db4 で DWT(1レベル) → LL, LH, HL, HH
  2. LL: そのまま保持（低周波・輪郭を保護）
  3. 高周波(LH, HL, HH): プールからランダムに選んだ別ウェーブレットBでIDWT
     X_low  = IDWT(LL, 0, 0, 0;  db4)
     X_high = IDWT(0, LH, HL, HH; B)
  4. 出力 = X_low + X_high

WaveletCrossAugV2 (wavedec2/waverec2版):
  1. source_wavelet で wavedec2(level=L) → 全係数
  2. 同じ係数を target_wavelet で waverec2 → 再構成
  入出力: PIL Image
"""

import random
import numpy as np
import pywt
from PIL import Image

# --basis-random モードで使用する基底プール（pywt + level=1 + 32x32 で動作確認済み）
BASIS_POOL = ["haar", "db4", "db8", "sym4", "sym8", "coif2"]


class WaveletBasisSwap:
    """WBS: Wavelet Basis Swap Augmentation（単一画像）
    Haar で分解後、各HF係数（LH, HL, HH）を独立にswap_probの確率でdb4基底で再構成。
    LLは常にsource_waveletで再構成（低周波保護）。
    レベルLなら3L個のHF係数がそれぞれ独立にswapされる。

    basis_random=True のとき、__call__ ごとに BASIS_POOL からランダムに
    (source, target) を選ぶ（source != target を保証）。
    _last_pair に最後に使ったペアを記録する。
    """

    def __init__(self, source_wavelet='haar', target_wavelet='db4',
                 level=1, swap_prob=0.5, mode='periodization',
                 basis_random=False, basis_pool=None):
        self.source_wavelet = source_wavelet
        self.target_wavelet = target_wavelet
        self.level = level
        self.swap_prob = swap_prob
        self.mode = mode
        self.basis_random = basis_random
        self.basis_pool = basis_pool if basis_pool is not None else BASIS_POOL
        self._last_pair = None  # (src, tgt) — basis_random=True のときのみ更新

    def _pick_pair(self, avoid_pair=None):
        """basis_pool からランダムに (src, tgt) を選ぶ。
        src != tgt、かつ avoid_pair と一致しない組み合わせを引き直す。"""
        pool = self.basis_pool
        while True:
            src = random.choice(pool)
            tgt = random.choice(pool)
            if src != tgt and (src, tgt) != avoid_pair:
                return src, tgt

    def _process_channel(self, channel, src_w, tgt_w):
        h, w = channel.shape
        coeffs = pywt.wavedec2(channel, wavelet=src_w,
                               level=self.level, mode=self.mode)

        cA = coeffs[0]
        detail_levels = coeffs[1:]
        zeros_cA = np.zeros_like(cA)

        # LL: 常にsrc_wで再構成（低周波保護）
        zeros_details = [(np.zeros_like(d[0]), np.zeros_like(d[1]), np.zeros_like(d[2]))
                         for d in detail_levels]
        result = pywt.waverec2([cA] + zeros_details, wavelet=src_w, mode=self.mode)

        # 各レベルの各HF係数（LH, HL, HH）を独立にswap
        for lvl_i, (LH, HL, HH) in enumerate(detail_levels):
            z = [(np.zeros_like(d[0]), np.zeros_like(d[1]), np.zeros_like(d[2]))
                 for d in detail_levels]

            for coef_i, coef in enumerate([LH, HL, HH]):
                w_use = tgt_w if random.random() < self.swap_prob else src_w

                details = [list(z[i]) for i in range(len(detail_levels))]
                details[lvl_i][coef_i] = coef
                details = [tuple(d) for d in details]

                recon = pywt.waverec2([zeros_cA] + details, wavelet=w_use, mode=self.mode)
                result = result + recon

        return result[:h, :w]

    def __call__(self, img_pil, avoid_pair=None):
        if self.basis_random:
            src_w, tgt_w = self._pick_pair(avoid_pair)
            self._last_pair = (src_w, tgt_w)
        else:
            src_w = self.source_wavelet
            tgt_w = self.target_wavelet

        img = np.array(img_pil).astype(np.float64)
        h, w = img.shape[:2]

        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = self._process_channel(img[:, :, c], src_w, tgt_w)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


class WaveletBasisSwapMol:
    """WBS + Mollification用: スワップ数に基づいてtを返す。
    t = 実際にスワップされた係数数 / 全HF係数数 (0〜1)
    """

    def __init__(self, source_wavelet='haar', target_wavelet='db8',
                 level=1, swap_prob=0.2, mode='periodization'):
        self.source_wavelet = source_wavelet
        self.target_wavelet = target_wavelet
        self.level = level
        self.swap_prob = swap_prob
        self.mode = mode

    def _process_channel(self, channel, swap_decisions):
        h, w = channel.shape
        coeffs = pywt.wavedec2(channel, wavelet=self.source_wavelet,
                               level=self.level, mode=self.mode)
        cA = coeffs[0]
        detail_levels = coeffs[1:]
        zeros_cA = np.zeros_like(cA)

        zeros_details = [(np.zeros_like(d[0]), np.zeros_like(d[1]), np.zeros_like(d[2]))
                         for d in detail_levels]
        result = pywt.waverec2([cA] + zeros_details, wavelet=self.source_wavelet, mode=self.mode)

        coef_idx = 0
        for lvl_i, (LH, HL, HH) in enumerate(detail_levels):
            z = [(np.zeros_like(d[0]), np.zeros_like(d[1]), np.zeros_like(d[2]))
                 for d in detail_levels]
            for coef_i, coef in enumerate([LH, HL, HH]):
                w_use = self.target_wavelet if swap_decisions[coef_idx] else self.source_wavelet
                coef_idx += 1
                details = [list(z[i]) for i in range(len(detail_levels))]
                details[lvl_i][coef_i] = coef
                details = [tuple(d) for d in details]
                recon = pywt.waverec2([zeros_cA] + details, wavelet=w_use, mode=self.mode)
                result = result + recon

        return result[:h, :w]

    def __call__(self, img_pil, t_mode='count'):
        img = np.array(img_pil).astype(np.float64)
        h, w = img.shape[:2]

        total_coefs = 3 * self.level
        swap_decisions = [random.random() < self.swap_prob for _ in range(total_coefs)]

        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = self._process_channel(img[:, :, c], swap_decisions)

        if t_mode == 'energy':
            diff = result - img
            denom = np.sum(img ** 2) + 1e-8
            t = float(np.sum(diff ** 2) / denom)
            t = min(t, 1.0)
        else:
            t = sum(swap_decisions) / total_coefs

        aug_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        return aug_img, t


class WaveletCrossAug:

    TARGET_POOL = ['db3', 'db6', 'sym4', 'coif2', 'bior2.2']

    def __init__(
        self,
        source_wavelet='db4',
        target_pool=None,
        mode='periodization',
    ):
        self.source_wavelet = source_wavelet
        self.target_pool = target_pool if target_pool is not None else self.TARGET_POOL
        self.mode = mode

    def _process_channel(self, channel, target_wavelet):
        LL, (LH, HL, HH) = pywt.dwt2(channel, self.source_wavelet, mode=self.mode)

        zeros_hf = (np.zeros_like(LH), np.zeros_like(HL), np.zeros_like(HH))
        X_low = pywt.idwt2((LL, zeros_hf), self.source_wavelet, mode=self.mode)

        zeros_ll = np.zeros_like(LL)
        X_high = pywt.idwt2((zeros_ll, (LH, HL, HH)), target_wavelet, mode=self.mode)

        return X_low + X_high

    def __call__(self, img_pil):
        img = np.array(img_pil).astype(np.float64)
        h, w = img.shape[:2]

        target_wavelet = random.choice(self.target_pool)

        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = self._process_channel(img[:, :, c], target_wavelet)

        result = result[:h, :w]
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


class WaveletCrossAugV2:
    """wavedec2/waverec2ベースのWCA。
    protect_ll=True: LLをsource_waveletで保護し、HFのみtarget_waveletで再構成。
    protect_ll=False: 全係数をtarget_waveletで再構成。
    """

    def __init__(
        self,
        source_wavelet='haar',
        target_wavelet='sym4',
        level=1,
        mode='periodization',
        protect_ll=True,
    ):
        self.source_wavelet = source_wavelet
        self.target_wavelet = target_wavelet
        self.level = level
        self.mode = mode
        self.protect_ll = protect_ll

    def _process_channel(self, channel):
        h, w = channel.shape
        coeffs = pywt.wavedec2(channel, wavelet=self.source_wavelet,
                               level=self.level, mode=self.mode)

        if not self.protect_ll:
            recon = pywt.waverec2(coeffs, wavelet=self.target_wavelet, mode=self.mode)
            return recon[:h, :w]

        # LLはsource_waveletで再構成（低周波保護）
        cA = coeffs[0]
        detail_levels = coeffs[1:]
        zeros_details = [(np.zeros_like(d[0]), np.zeros_like(d[1]), np.zeros_like(d[2]))
                         for d in detail_levels]
        coeffs_low = [cA] + zeros_details
        X_low = pywt.waverec2(coeffs_low, wavelet=self.source_wavelet, mode=self.mode)

        # HFはtarget_waveletで再構成
        zeros_cA = np.zeros_like(cA)
        coeffs_high = [zeros_cA] + list(detail_levels)
        X_high = pywt.waverec2(coeffs_high, wavelet=self.target_wavelet, mode=self.mode)

        return (X_low + X_high)[:h, :w]

    def __call__(self, img_pil):
        img = np.array(img_pil).astype(np.float64)
        h, w = img.shape[:2]

        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = self._process_channel(img[:, :, c])

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
