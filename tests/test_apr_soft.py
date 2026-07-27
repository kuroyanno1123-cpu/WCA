"""Unit tests for APR-S-soft implementation (pytest-free, standalone)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import traceback
import random
import numpy as np
import torch
import torch.nn.functional as F

from datasets.apr_soft import _fft_amplitude_mix, APRSoftDataset, APRHardDataset


DATA_ROOT = '/home/kairisasaki/data/cifar10'

# ── テスト用ランダム画像 ──────────────────────────────────────────────────────

def _rand_img():
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)


# ── Test 1: t_eff=0 のとき出力が元画像と一致する ─────────────────────────────

def test_identity_at_t0():
    rng = np.random.RandomState(42)
    x = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    d = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    out = _fft_amplitude_mix(x, d, t_eff=0.0)
    # FFT round-trip: IFFT(FFTSHIFT(FFTN(x))) ≈ x in uint8 精度
    assert np.allclose(out.astype(np.float64), x.astype(np.float64), atol=1.0), \
        f'max diff = {np.max(np.abs(out.astype(int) - x.astype(int)))}'


# ── Test 2: t_eff=1 のとき APRHard と同じ出力になる ──────────────────────────

def test_t1_matches_hard_swap():
    rng = np.random.RandomState(7)
    x = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    d = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    out_soft = _fft_amplitude_mix(x, d, t_eff=1.0)

    # 手動で hard swap を再現
    fft_x = np.fft.fftshift(np.fft.fftn(x.astype(np.float64)))
    fft_d = np.fft.fftshift(np.fft.fftn(d.astype(np.float64)))
    fft_hard = np.abs(fft_d) * np.exp(1j * np.angle(fft_x))
    out_hard = np.clip(np.fft.ifftn(np.fft.ifftshift(fft_hard)).real, 0, 255).astype(np.uint8)

    assert np.array_equal(out_soft, out_hard), \
        f'max diff = {np.max(np.abs(out_soft.astype(int) - out_hard.astype(int)))}'


# ── Test 3: gamma 計算 (t_eff=0.5, k=2 → gamma=0.25) ───────────────────────

def test_gamma_formula():
    t_eff = torch.tensor([0.5])
    k = 2.0
    gamma = t_eff ** k
    assert abs(gamma.item() - 0.25) < 1e-6, f'gamma={gamma.item()}'


# ── Test 4: y_soft の各行の和が 1 になる ──────────────────────────────────────

def test_y_soft_sums_to_one():
    C = 10
    bs = 16
    y_own  = torch.randint(0, C, (bs,))
    y_dist = torch.randint(0, C, (bs,))
    t_eff  = torch.rand(bs)
    k      = 1.0

    gamma  = t_eff ** k
    y_soft = (
        (1 - gamma).unsqueeze(1) * F.one_hot(y_own,  C).float()
        + gamma.unsqueeze(1)     * F.one_hot(y_dist, C).float()
    )
    row_sums = y_soft.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(bs), atol=1e-6), \
        f'row_sums not all 1: {row_sums}'


# ── Test 5: 既存パスの回帰テスト (1 バッチ forward が通る) ──────────────────

def _make_args(**kwargs):
    import argparse
    defaults = dict(
        data=DATA_ROOT.replace('/cifar10', ''),
        data_c='/home/kairisasaki/APR_phase/data',
        dataset='cifar10', batch_size=8, workers=0, gpu='0',
        aug='wca', source='haar', target='db8', level=1, swap_prob=0.2,
        jsd_lambda=0.0, aug_order='wca_first', basis_random=False,
        t_max=1.0, label_k=1.0, clean_prob=0.0,
        gamma_swap=0.1, uncond_smooth=0.0,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _forward_one_batch(aug_name, **extra):
    from datasets.cifar import build_loaders
    from datasets.apr_soft import build_apr_loaders
    from model.resnet import ResNet18

    args = _make_args(aug=aug_name, **extra)

    if aug_name in ('apr-s', 'apr-s-soft', 'apr-s-cls'):
        loader, _, _ = build_apr_loaders(args, eval_mode=False)
    else:
        loader, _, _ = build_loaders(args, eval_mode=False)

    net = ResNet18(num_classes=10)
    batch = next(iter(loader))

    if aug_name == 'apr-s-soft':
        x_aug, t_eff, y_own, y_dist = batch
        logits = net(x_aug)
        C = 10
        gamma  = t_eff ** args.label_k
        y_soft = (
            (1 - gamma).unsqueeze(1) * F.one_hot(y_own,  C).float()
            + gamma.unsqueeze(1)     * F.one_hot(y_dist, C).float()
        )
        loss = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    elif aug_name == 'apr-s-cls':
        x_aug, y_own, swapped = batch
        C = 10
        y_one_hot = F.one_hot(y_own, C).float()
        y_smooth  = (1 - args.gamma_swap) * y_one_hot + args.gamma_swap / C
        mask      = swapped.float().unsqueeze(1)
        y_soft    = mask * y_smooth + (1 - mask) * y_one_hot
        logits = net(x_aug)
        loss = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    elif aug_name == 'apr-s':
        x_aug, y_own = batch
        logits = net(x_aug)
        loss = F.cross_entropy(logits, y_own)
    else:
        if isinstance(batch[0], (list, tuple)):
            (x_clean, x_aug1, x_aug2), y = batch
            logits = net(x_clean)
        else:
            x, y = batch
            logits = net(x)
        loss = F.cross_entropy(logits, y if not isinstance(batch[0], (list, tuple)) else y)

    assert not torch.isnan(loss), f'NaN loss for aug={aug_name}'
    assert loss.item() > 0


def test_regression_wca_ce():
    _forward_one_batch('wca', jsd_lambda=0.0)


def test_regression_wca_jsd():
    _forward_one_batch('wca', jsd_lambda=12.0, aug_order='crop_first')


def test_regression_augmix():
    _forward_one_batch('augmix', jsd_lambda=12.0, aug_order='crop_first')


def test_regression_apr_s():
    _forward_one_batch('apr-s')


def test_regression_apr_s_soft():
    _forward_one_batch('apr-s-soft', t_max=1.0, label_k=1.0, clean_prob=0.0)


# ── Test 6: apr-s-cls swapped=False → y_soft == one_hot ──────────────────────

def test_cls_not_swapped_is_onehot():
    C, bs = 10, 8
    y_own   = torch.randint(0, C, (bs,))
    swapped = torch.zeros(bs, dtype=torch.bool)
    gamma_swap = 0.1

    y_one_hot = F.one_hot(y_own, C).float()
    y_smooth  = (1 - gamma_swap) * y_one_hot + gamma_swap / C
    mask      = swapped.float().unsqueeze(1)
    y_soft    = mask * y_smooth + (1 - mask) * y_one_hot

    assert torch.allclose(y_soft, y_one_hot), 'not-swapped samples must equal one_hot'


# ── Test 7: apr-s-cls swapped=True → 行和=1, 正解確率=1-γ+γ/C ───────────────

def test_cls_swapped_label_values():
    C, bs = 10, 8
    gamma_swap = 0.1
    y_own   = torch.zeros(bs, dtype=torch.long)   # 全サンプルクラス0
    swapped = torch.ones(bs, dtype=torch.bool)

    y_one_hot = F.one_hot(y_own, C).float()
    y_smooth  = (1 - gamma_swap) * y_one_hot + gamma_swap / C
    mask      = swapped.float().unsqueeze(1)
    y_soft    = mask * y_smooth + (1 - mask) * y_one_hot

    expected_correct = 1 - gamma_swap + gamma_swap / C
    assert torch.allclose(y_soft.sum(dim=1), torch.ones(bs), atol=1e-6), '行和が1でない'
    assert torch.allclose(y_soft[:, 0], torch.full((bs,), expected_correct), atol=1e-6), \
        f'正解クラス確率: expected {expected_correct:.4f}, got {y_soft[0, 0]:.4f}'


def test_regression_apr_s_cls():
    _forward_one_batch('apr-s-cls', gamma_swap=0.1, uncond_smooth=0.0)


# ── テストランナー ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('test_identity_at_t0',             test_identity_at_t0),
        ('test_t1_matches_hard_swap',        test_t1_matches_hard_swap),
        ('test_gamma_formula',               test_gamma_formula),
        ('test_y_soft_sums_to_one',          test_y_soft_sums_to_one),
        ('test_regression_wca_ce',           test_regression_wca_ce),
        ('test_regression_wca_jsd',          test_regression_wca_jsd),
        ('test_regression_augmix',           test_regression_augmix),
        ('test_regression_apr_s',            test_regression_apr_s),
        ('test_regression_apr_s_soft',       test_regression_apr_s_soft),
        ('test_cls_not_swapped_is_onehot',   test_cls_not_swapped_is_onehot),
        ('test_cls_swapped_label_values',    test_cls_swapped_label_values),
        ('test_regression_apr_s_cls',        test_regression_apr_s_cls),
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
