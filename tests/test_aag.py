"""Tests for core/aag.py (GeneratorMLP + FFT utilities)."""

import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/kairisasaki/DAT')

import torch
import torch.nn.functional as F
import numpy as np

from core.aag import GeneratorMLP, get_fft, inverse_fft, denorm, renorm


# ── Test 1: GeneratorMLP の出力が DAT Generator_MLP と一致 ───────────────────

def test_generator_match_dat():
    """同一シード・同一入力で WCA GeneratorMLP == DAT Generator_MLP。"""
    from Generator import Generator_MLP as DATGen

    torch.manual_seed(42)
    G_wca = GeneratorMLP(z_dim=100, out_channels=3, img_h=32, img_w=32, num_classes=10)

    torch.manual_seed(42)
    G_dat = DATGen(in_channel=100, out_channel=3, img_h=32, img_w=32, num_class=10)

    G_wca.eval()
    G_dat.eval()

    torch.manual_seed(0)
    z    = torch.randn(8, 100)
    feat = torch.randn(8, 10)

    with torch.no_grad():
        out_wca = G_wca(z, feat)
        out_dat = G_dat(z, feat)

    max_diff = torch.abs(out_wca - out_dat).max().item()
    assert max_diff < 1e-6, \
        f'Output mismatch: max_diff={max_diff:.2e}'


# ── Test 2: FFT round-trip (get_fft → inverse_fft ≈ 恒等変換) ──────────────

def test_fft_roundtrip():
    """get_fft → inverse_fft が [0,1] 画像に対して恒等変換になる。"""
    torch.manual_seed(0)
    x_raw = torch.rand(4, 3, 32, 32)
    amp, pha = get_fft(x_raw)
    x_recon  = inverse_fft(amp, pha)
    max_diff = torch.abs(x_recon - x_raw.clamp(0, 1)).max().item()
    assert max_diff < 1e-5, f'Round-trip error: max_diff={max_diff:.2e}'


# ── Test 3: 交互更新 — net と G の両パラメータが 1 バッチで更新される ─────────

def test_alternating_update():
    """1バッチで net と GeneratorMLP の両パラメータが変化する。"""
    from model.resnet import ResNet18

    torch.manual_seed(0)
    net = ResNet18(num_classes=10)
    G   = GeneratorMLP()
    opt_net = torch.optim.SGD(net.parameters(), lr=0.01)
    opt_G   = torch.optim.SGD(G.parameters(),   lr=0.01)

    net_params_before = [p.clone().detach() for p in net.parameters()]
    g_params_before   = [p.clone().detach() for p in G.parameters()]

    bs    = 4
    x_raw = torch.rand(bs, 3, 32, 32)
    x_norm = renorm(x_raw)
    y     = torch.randint(0, 10, (bs,))

    # conditioning
    net.eval()
    with torch.no_grad():
        feat = net(x_norm)
    net.train()

    z       = torch.randn(bs, 100)
    amp_G   = G(z, feat.detach())

    amp_orig, pha_orig = get_fft(x_raw.detach())
    # beta=0.5: 非ゼロなので G への勾配が確実に流れる
    beta1 = 0.5 * torch.ones(bs, 1, 1, 1)
    amp_mixed    = beta1 * amp_G + (1 - beta1) * amp_orig
    x_auto1_raw  = inverse_fft(amp_mixed, pha_orig)
    x_auto1      = renorm(x_auto1_raw)

    # Model update
    logits_b = net(x_norm)
    logits_a = net(x_auto1)
    loss = (F.cross_entropy(logits_b, y) + F.cross_entropy(logits_a, y)) / 2
    opt_net.zero_grad()
    loss.backward(retain_graph=True)
    opt_net.step()

    # Generator update
    logits_b2 = net(x_norm)
    logits_a2 = net(x_auto1)
    g_loss = -((F.cross_entropy(logits_b2, y) + F.cross_entropy(logits_a2, y)) / 2)
    opt_G.zero_grad()
    g_loss.backward()
    opt_G.step()

    net_changed = any(not torch.equal(p, p0)
                      for p, p0 in zip(net.parameters(), net_params_before))
    g_changed   = any(not torch.equal(p, p0)
                      for p, p0 in zip(G.parameters(), g_params_before))
    assert net_changed, 'net parameters did not change after model update'
    assert g_changed,   'G parameters did not change after generator update'


# ── Test 4: 既存テスト群の PASS 確認 ─────────────────────────────────────────

def test_existing_wavelet_fusion():
    """test_wavelet_fusion の全テストが PASS。"""
    import subprocess
    r = subprocess.run([sys.executable, 'tests/test_wavelet_fusion.py'],
                       capture_output=True, text=True)
    assert r.returncode == 0, \
        f'test_wavelet_fusion failed:\n{r.stdout}\n{r.stderr}'


def test_existing_apr_soft():
    """test_apr_soft の全テストが PASS。"""
    import subprocess
    r = subprocess.run([sys.executable, 'tests/test_apr_soft.py'],
                       capture_output=True, text=True)
    assert r.returncode == 0, \
        f'test_apr_soft failed:\n{r.stdout}\n{r.stderr}'


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('test_generator_match_dat',    test_generator_match_dat),
        ('test_fft_roundtrip',          test_fft_roundtrip),
        ('test_alternating_update',     test_alternating_update),
        ('test_existing_wavelet_fusion',test_existing_wavelet_fusion),
        ('test_existing_apr_soft',      test_existing_apr_soft),
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
