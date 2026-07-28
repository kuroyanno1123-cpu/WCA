"""Unit tests for datasets/wavelet_fusion.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import traceback
import random
import numpy as np
import pywt
import torch
import torch.nn.functional as F

from datasets.wavelet_fusion import _wavelet_fusion, WaveletFusionDataset, build_wf_loaders

DATA_ROOT = '/home/kairisasaki/data'


def _rand_img(seed, H=32, W=32, C=3):
    return np.random.RandomState(seed).randint(0, 256, (H, W, C)).astype(np.float64)


# ── Test 1: sign_align — 出力係数の符号=A, 絶対値=B ──────────────────────────

def test_sign_align_coef():
    a = _rand_img(0)
    b = _rand_img(1)
    wavelet, level = 'haar', 1

    fused = _wavelet_fusion(a, b, 'sign_align', wavelet, level)

    for c in range(3):
        ca = pywt.wavedec2(a[:, :, c], wavelet, level=level)
        cb = pywt.wavedec2(b[:, :, c], wavelet, level=level)
        cf = pywt.wavedec2(fused[:, :, c], wavelet, level=level)

        # LL は A と一致
        assert np.allclose(cf[0], ca[0], atol=1e-9), f'ch{c}: LL should match A'

        for lvl in range(1, len(ca)):
            for i, (da, db, df) in enumerate(zip(ca[lvl], cb[lvl], cf[lvl])):
                # 期待値: sign(A) * |B|  (da=0→0, db=0→0 の場合も一括で検証)
                expected = np.sign(da) * np.abs(db)
                assert np.allclose(df, expected, atol=1e-9), \
                    f'ch{c} lvl{lvl} band{i}: coef should be sign(A)*|B| ' \
                    f'(max_diff={np.max(np.abs(df - expected)):.2e})'


# ── Test 2: max_coef — 採用係数の絶対値 = max(|A|, |B|) ──────────────────────

def test_max_coef_coef():
    a = _rand_img(2)
    b = _rand_img(3)
    wavelet, level = 'haar', 1

    fused = _wavelet_fusion(a, b, 'max_coef', wavelet, level)

    for c in range(3):
        ca = pywt.wavedec2(a[:, :, c], wavelet, level=level)
        cb = pywt.wavedec2(b[:, :, c], wavelet, level=level)
        cf = pywt.wavedec2(fused[:, :, c], wavelet, level=level)

        # LL は A と一致
        assert np.allclose(cf[0], ca[0], atol=1e-9), f'ch{c}: LL should match A'

        for lvl in range(1, len(ca)):
            for i, (da, db, df) in enumerate(zip(ca[lvl], cb[lvl], cf[lvl])):
                expected_abs = np.maximum(np.abs(da), np.abs(db))
                assert np.allclose(np.abs(df), expected_abs, atol=1e-9), \
                    f'ch{c} lvl{lvl} band{i}: |coef| should be max(|A|,|B|)'


# ── Test 3: A==B のとき恒等変換 ───────────────────────────────────────────────

def test_identity_when_equal():
    a = _rand_img(4)

    for mode in ('sign_align', 'max_coef'):
        fused = _wavelet_fusion(a, a.copy(), mode, 'haar', 1)
        max_diff = np.max(np.abs(fused - a))
        assert max_diff < 1e-6, \
            f'{mode}: A==B should give identity (max_diff={max_diff:.2e})'


# ── Test 4: apply_prob=0 で applied=False になる ─────────────────────────────

def test_apply_prob_zero():
    ds = WaveletFusionDataset(
        os.path.join(DATA_ROOT, 'cifar10'),
        mode='sign_align', apply_prob=0.0,
    )
    random.seed(42)
    for i in range(8):
        _, _, applied = ds[i]
        assert not applied.item(), f'apply_prob=0: applied=True at idx={i}'


# ── Test 5: LL サブバンドが A のまま（level=2 でも確認）──────────────────────

def test_ll_from_a():
    a = _rand_img(5)
    b = _rand_img(6)

    for mode in ('sign_align', 'max_coef'):
        for level in (1, 2):
            fused = _wavelet_fusion(a, b, mode, 'haar', level)
            for c in range(3):
                ca = pywt.wavedec2(a[:, :, c], 'haar', level=level)
                cf = pywt.wavedec2(fused[:, :, c], 'haar', level=level)
                assert np.allclose(cf[0], ca[0], atol=1e-9), \
                    f'{mode} level={level} ch{c}: LL should match A'


# ── Test 6: seed 固定 回帰テスト（forward pass が NaN にならない） ────────────

def _make_wf_args(aug, **kw):
    import argparse
    defaults = dict(
        data=DATA_ROOT, data_c='/home/kairisasaki/APR_phase/data',
        dataset='cifar10', batch_size=8, workers=0, gpu='0',
        aug=aug, wf_wavelet='haar', wf_level=1, wf_apply_prob=0.5,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_regression_wf_sign():
    from model.resnet import ResNet18
    args = _make_wf_args('wf-sign')
    loader, _, _ = build_wf_loaders(args, eval_mode=False)
    net = ResNet18(num_classes=10)
    x_aug, y_own, applied = next(iter(loader))
    loss = F.cross_entropy(net(x_aug), y_own)
    assert not torch.isnan(loss), 'NaN loss for wf-sign'
    assert loss.item() > 0


def test_regression_wf_max():
    from model.resnet import ResNet18
    args = _make_wf_args('wf-max')
    loader, _, _ = build_wf_loaders(args, eval_mode=False)
    net = ResNet18(num_classes=10)
    x_aug, y_own, applied = next(iter(loader))
    loss = F.cross_entropy(net(x_aug), y_own)
    assert not torch.isnan(loss), 'NaN loss for wf-max'
    assert loss.item() > 0


# ── テストランナー ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('test_sign_align_coef',    test_sign_align_coef),
        ('test_max_coef_coef',      test_max_coef_coef),
        ('test_identity_when_equal',test_identity_when_equal),
        ('test_apply_prob_zero',    test_apply_prob_zero),
        ('test_ll_from_a',          test_ll_from_a),
        ('test_regression_wf_sign', test_regression_wf_sign),
        ('test_regression_wf_max',  test_regression_wf_max),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            traceback.print_exc()
            failed += 1

    print(f'\n{passed}/{passed+failed} tests passed')
    if failed:
        sys.exit(1)
