"""wapr_gates.py — 学習前の 4 つのゲート確認スクリプト。

使い方:
  cd /home/kairisasaki/WCA
  conda run -n apr python3 wapr_gates.py --data /home/kairisasaki/data

Gate 1: 摂動強度キャリブレーション (PSNR 比較)
Gate 2: 冗長性による射影損失
Gate 3: 目視比較画像の出力
Gate 4: スループット計測
"""

import sys, os, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from pytorch_wavelets import DTCWTForward, DTCWTInverse

from datasets.cifar import _RawCIFAR10, _worker_init_fn
from datasets.apr_soft import _apr_orig_core
from datasets.wapr import _wapr_orig_core, _pil_to_torch, WAPROrigDataset

parser = argparse.ArgumentParser(description='wapr-orig pre-training gates')
parser.add_argument('--data',    type=str, default='/home/kairisasaki/data',
                    help='dataset root (contains cifar10/)')
parser.add_argument('--n-imgs',  type=int, default=1000, help='Gate1/2 使用枚数')
parser.add_argument('--seed',    type=int, default=0)
parser.add_argument('--gate',    type=int, default=0,
                    help='0=全Gate, 1-4=個別 Gate 番号')
parser.add_argument('--out-dir', type=str, default='./wapr_gate_output',
                    help='Gate3 の目視画像出力先')
parser.add_argument('--workers', type=int, default=4, help='Gate4 DataLoader workers')
parser.add_argument('--batch-size', type=int, default=128, help='Gate4 batch size')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

DATA_ROOT = os.path.join(args.data, 'cifar10')
os.makedirs(args.out_dir, exist_ok=True)


def _psnr(ref: np.ndarray, out: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - out.astype(np.float64)) ** 2)
    return float('inf') if mse < 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)


def _set_seed(s):
    random.seed(s)
    np.random.seed(s)


# ── Gate 1: 摂動強度キャリブレーション ──────────────────────────────────────────

def gate1():
    print('\n' + '=' * 65)
    print('Gate 1: 摂動強度キャリブレーション (PSNR)')
    print('=' * 65)

    raw = _RawCIFAR10(DATA_ROOT, train=True, download=False)
    imgs = [raw[i][0] for i in range(args.n_imgs)]  # PIL list

    # APR-S-orig PSNR
    apr_psnrs = []
    for i, img in enumerate(imgs):
        _set_seed(i)
        ref = np.array(img)
        out_pil, swapped = _apr_orig_core(img.copy())
        if swapped:
            apr_psnrs.append(_psnr(ref, np.array(out_pil)))

    # wapr-orig の PSNR を J × lowpass_mode グリッドで計算
    J_vals       = [1, 2, 3]
    lowpass_vals = ['phase', 'amp', 'mix']

    print(f'\napr-s-orig: PSNR = {np.mean(apr_psnrs):.2f} ± {np.std(apr_psnrs):.2f} dB'
          f'  (n={len(apr_psnrs)} swapped)')

    print(f'\n{"J":>3}  {"lowpass":>6}  {"mean_PSNR":>10}  {"std":>6}  {"n_swapped":>9}  {"pass(±1dB)":>10}')
    print('-' * 60)

    target_psnr = np.mean(apr_psnrs)
    for J in J_vals:
        xfm = DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b')
        ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        for lp in lowpass_vals:
            w_psnrs = []
            for i, img in enumerate(imgs):
                _set_seed(i)
                ref = np.array(img)
                out_pil, swapped, sat = _wapr_orig_core(
                    img.copy(), xfm, ifm, lam=1.0, lowpass_mode=lp
                )
                if swapped:
                    w_psnrs.append(_psnr(ref, np.array(out_pil)))

            mean_p = np.mean(w_psnrs)
            std_p  = np.std(w_psnrs)
            ok = '✓ PASS' if abs(mean_p - target_psnr) <= 1.0 else '✗ FAIL'
            print(f'{J:>3}  {lp:>6}  {mean_p:>10.2f}  {std_p:>6.2f}  '
                  f'{len(w_psnrs):>9}  {ok:>10}')

    print(f'\n判定基準: |wapr_PSNR - apr_PSNR| ≤ 1.0 dB')
    print(f'apr-s-orig の基準値: {target_psnr:.2f} dB')


# ── Gate 2: 冗長性による射影損失 ─────────────────────────────────────────────────

def gate2():
    print('\n' + '=' * 65)
    print('Gate 2: 冗長性による射影損失 (rel_err = ‖W(W⁻¹(c′))−c′‖/‖c′‖)')
    print('  Fourier では 0 になるのに対し、DTCWT は 4× 冗長なので非ゼロ。')
    print('=' * 65)

    J = 3
    xfm = DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b')
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
    raw = _RawCIFAR10(DATA_ROOT, train=True, download=False)

    rel_errs = []
    for i in range(min(args.n_imgs, 200)):
        img = raw[i][0]
        x1_t = _pil_to_torch(img)
        x2_t = _pil_to_torch(img.copy())  # 同一画像を aug として (効果を中立化)

        with torch.no_grad():
            yl1, yh1 = xfm(x1_t)
            yl2, yh2 = xfm(x2_t)

        # 再結合係数 c' を構成 (lam=1, dir_b: amp=yh2, phase=yh1)
        yh_prime = []
        for j in range(len(yh1)):
            h1, h2 = yh1[j], yh2[j]
            r1, i1 = h1[..., 0], h1[..., 1]
            r2, i2 = h2[..., 0], h2[..., 1]
            mag2 = torch.sqrt(r2 * r2 + i2 * i2)
            phase1 = torch.atan2(i1, r1)
            safe = (torch.sqrt(r1 * r1 + i1 * i1) >= 1e-8)
            z_r = torch.where(safe, mag2 * torch.cos(phase1), r1)
            z_i = torch.where(safe, mag2 * torch.sin(phase1), i1)
            yh_prime.append(torch.stack([z_r, z_i], dim=-1))
        yl_prime = yl2

        # 逆変換 → 再分解
        with torch.no_grad():
            out = ifm((yl_prime, yh_prime))
            yl_pp, yh_pp = xfm(out)

        # rel_err = ‖W(W⁻¹(c′)) − c′‖ / ‖c′‖ (全係数を flatten して計算)
        c_prime_norms = []
        c_diff_norms  = []
        for j in range(len(yh_prime)):
            cp = yh_prime[j]
            cpp = yh_pp[j]
            c_prime_norms.append(cp.pow(2).sum().item())
            c_diff_norms.append((cpp - cp).pow(2).sum().item())

        # lowpass
        c_prime_norms.append(yl_prime.pow(2).sum().item())
        c_diff_norms.append((yl_pp - yl_prime).pow(2).sum().item())

        norm_c  = np.sqrt(sum(c_prime_norms))
        norm_diff = np.sqrt(sum(c_diff_norms))
        rel_err = norm_diff / (norm_c + 1e-12)
        rel_errs.append(rel_err)

    print(f'\nrel_err (J={J}, lam=1, lowpass=amp):')
    print(f'  mean = {np.mean(rel_errs):.4f}')
    print(f'  std  = {np.std(rel_errs):.4f}')
    print(f'  max  = {np.max(rel_errs):.4f}')
    print(f'\n  Fourier での対応値: 0.0000 (射影は常に完全)')
    print(f'  上記の非ゼロ値が「意図した摂動のうち逆変換で捨てられる割合」を示す。')

    # 飽和率も報告
    sat_rates = []
    for i in range(min(args.n_imgs, 200)):
        img = raw[i][0]
        x1_t = _pil_to_torch(img)
        with torch.no_grad():
            yl_, yh_ = xfm(x1_t)
            out = ifm((yl_, yh_))
        arr = out.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        sat = float(np.mean((arr < 0.0) | (arr > 255.0)))
        sat_rates.append(sat)

    print(f'\n飽和率 (clip[0,255] が必要なピクセルの割合):')
    print(f'  mean = {np.mean(sat_rates)*100:.3f}%')
    print(f'  max  = {np.max(sat_rates)*100:.3f}%')


# ── Gate 3: 目視比較 ──────────────────────────────────────────────────────────

def gate3():
    """CIFAR-10 の 6 クラスについて apr-s-orig と wapr-orig を並べた比較画像を出力。"""
    print('\n' + '=' * 65)
    print('Gate 3: 目視比較画像の出力')
    print('=' * 65)

    try:
        import torchvision
    except ImportError:
        print('torchvision 未インストール。Gate3 をスキップ。')
        return

    xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
    raw = _RawCIFAR10(DATA_ROOT, train=True, download=False)

    target_classes = list(range(6))   # クラス 0-5
    lowpass_modes  = ['phase', 'amp', 'mix']
    n_per_class    = 3

    # クラスごとにサンプルを収集
    class_imgs = {c: [] for c in target_classes}
    for idx in range(len(raw)):
        img, label = raw[idx]
        if label in class_imgs and len(class_imgs[label]) < n_per_class:
            class_imgs[label].append(img)
        if all(len(v) >= n_per_class for v in class_imgs.values()):
            break

    # 列構成: [原画像, APR-S, WAPR-phase, WAPR-amp, WAPR-mix]
    cols = ['original', 'apr-s-orig', 'wapr-phase', 'wapr-amp', 'wapr-mix']
    scale = 4
    cell  = 32 * scale
    cols_n = len(cols)
    rows_n = sum(len(v) for v in class_imgs.values())
    canvas = Image.new('RGB', (cell * cols_n, cell * rows_n), color=(200, 200, 200))

    row = 0
    for cls in target_classes:
        for img in class_imgs[cls]:
            imgs_row = [img.resize((cell, cell), Image.NEAREST)]

            # APR-S-orig
            _set_seed(row)
            out_apr, _ = _apr_orig_core(img.copy())
            imgs_row.append(out_apr.resize((cell, cell), Image.NEAREST))

            # wapr-orig × 3 lowpass modes
            for lp in lowpass_modes:
                _set_seed(row)
                out_w, _, _ = _wapr_orig_core(img.copy(), xfm, ifm, lam=1.0, lowpass_mode=lp)
                imgs_row.append(out_w.resize((cell, cell), Image.NEAREST))

            for col_idx, cell_img in enumerate(imgs_row):
                canvas.paste(cell_img, (col_idx * cell, row * cell))
            row += 1

    out_path = os.path.join(args.out_dir, 'gate3_visual_comparison.png')
    canvas.save(out_path)
    print(f'\n  cols: {cols}')
    print(f'  rows: 6 classes × {n_per_class} samples = {rows_n} rows')
    print(f'  saved → {out_path}')


# ── Gate 4: スループット計測 ──────────────────────────────────────────────────

def gate4():
    print('\n' + '=' * 65)
    print('Gate 4: DataLoader スループット計測')
    print('=' * 65)

    import argparse as _ap
    fake_args = _ap.Namespace(
        data=args.data,
        data_c='/dev/null',
        batch_size=args.batch_size,
        workers=args.workers,
        wapr_levels=3,
        wapr_lam='1.0',
        wapr_lowpass='amp',
    )

    from datasets.wapr import build_wapr_loaders
    loader, _, _ = build_wapr_loaders(fake_args, eval_mode=False)
    n_batches = len(loader)

    # apr-s-orig ローダー (比較用)
    import argparse as _ap2
    from datasets.apr_soft import build_apr_loaders
    fake_args2 = _ap2.Namespace(
        data=args.data,
        data_c='/dev/null',
        batch_size=args.batch_size,
        workers=args.workers,
        aug='apr-s-orig',
    )
    loader_apr, _, _ = build_apr_loaders(fake_args2, eval_mode=False)

    def _time_loader(ldr, n_batch=50):
        t0 = time.time()
        cnt = 0
        for x, y in ldr:
            cnt += 1
            if cnt >= n_batch:
                break
        return time.time() - t0, cnt

    print(f'\napr-s-orig:')
    t_apr, n = _time_loader(loader_apr)
    imgs_per_sec_apr = n * args.batch_size / t_apr
    print(f'  {n} batches in {t_apr:.1f}s → {imgs_per_sec_apr:.0f} imgs/s')

    print(f'\nwapr-orig (J=3):')
    t_wapr, n = _time_loader(loader)
    imgs_per_sec_wapr = n * args.batch_size / t_wapr
    print(f'  {n} batches in {t_wapr:.1f}s → {imgs_per_sec_wapr:.0f} imgs/s')

    ratio = imgs_per_sec_wapr / imgs_per_sec_apr
    print(f'\nスループット比: wapr/apr = {ratio:.2f}×')
    if ratio < 0.5:
        print('  WARNING: GPU が待たされる可能性あり。バッチ化 (学習ループ内移動) を検討してください。')
    else:
        print('  OK: スループットは学習上問題ない範囲です。')


# ── メイン ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_all = (args.gate == 0)

    if run_all or args.gate == 1:
        gate1()
    if run_all or args.gate == 2:
        gate2()
    if run_all or args.gate == 3:
        gate3()
    if run_all or args.gate == 4:
        gate4()

    print('\n=== Gates 完了 ===')
