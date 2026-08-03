import os
import sys
import csv
import random
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from datasets.cifar import build_loaders, CORRUPTION_TYPES
from core.losses import jsd_loss
from utils.metrics import AverageMeter, Logger, save_networks, load_networks, test, test_robustness

# ── Argument parser ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()

# dataset paths
parser.add_argument('--data',    type=str, default='/ws/data',   help='CIFAR-10 root')
parser.add_argument('--data-c',  type=str, default='/ws/data_c', help='CIFAR-10-C root')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--outfolder', type=str, default='./results')
parser.add_argument('--memo',    type=str, default='none')

# augmentation
parser.add_argument('--aug',       type=str,   default='wca',
                    choices=['wca', 'augmix', 'none', 'apr-s', 'apr-s-soft', 'apr-s-cls',
                             'apr-s-orig', 'apr-s-orig-cls', 'wf-sign', 'wf-max', 'apr-aag',
                             'wapr-orig'],
                    help='augmentation type')
parser.add_argument('--source',    type=str,   default='haar', help='source wavelet')
parser.add_argument('--target',    type=str,   default='db8',  help='target wavelet')
parser.add_argument('--level',     type=int,   default=1,      help='DWT decomposition level')
parser.add_argument('--swap-prob', type=float, default=0.2,    help='HF coefficient swap probability')
parser.add_argument('--aug-order', type=str,   default=None,
                    choices=['wca_first', 'crop_first'],
                    help='Preprocessing order. '
                         'Default: wca_first when --jsd-lambda 0, crop_first otherwise.')
parser.add_argument('--basis-random', action='store_true', default=False,
                    help='Randomize (source, target) wavelet pair per sample from BASIS_POOL. '
                         'Requires --aug wca. Ignores --source/--target when enabled.')

# APR-S-soft specific
parser.add_argument('--t-max',      type=float, default=1.0,
                    help='APR-S-soft: upper bound of amplitude mixing ratio (default 1.0)')
parser.add_argument('--label-k',    type=float, default=1.0,
                    help='APR-S-soft: label schedule exponent k. gamma=(t_eff)^k (default 1.0)')
parser.add_argument('--clean-prob', type=float, default=0.0,
                    help='APR-S-soft: probability of forcing t=0 (clean pass) per sample')
parser.add_argument('--gamma-swap',    type=float, default=0.1,
                    help='APR-S-cls: label smoothing gamma for swapped samples')
parser.add_argument('--uncond-smooth', type=float, default=0.0,
                    help='APR-S-cls: >0 applies uniform smoothing to ALL samples (ablation)')

# WaveletAPR (wapr-orig) specific
parser.add_argument('--wapr-levels',              type=int,   default=3,
                    help='wapr-orig: DTCWT decomposition levels J (default 3)')
parser.add_argument('--wapr-lam',                 type=str,   default='1.0',
                    help='wapr-orig: amplitude mix ratio. float or "uniform" (default 1.0)')
parser.add_argument('--wapr-lowpass',             type=str,   default='amp',
                    choices=['amp', 'phase', 'mix'],
                    help='wapr-orig: lowpass subband treatment (default amp)')
parser.add_argument('--wapr-phase-sigma-vital',   type=float, default=0.0,
                    help='wapr-orig: phase perturbation sigma for vital coeffs (default 0.0=off)')
parser.add_argument('--wapr-phase-sigma-nonvital',type=float, default=0.0,
                    help='wapr-orig: phase perturbation sigma for non-vital coeffs (default 0.0=off)')
parser.add_argument('--wapr-q-vital',             type=float, default=0.75,
                    help='wapr-orig: quantile threshold for vital judgment (default 0.75)')

# AAG (Adversarial Amplitude Generator) specific
parser.add_argument('--aag-mix-beta',   type=float, default=1.0,
                    help='AAG: upper bound of per-sample beta ~ U(0, mix_beta) (default 1.0)')
parser.add_argument('--aag-jsd-weight', type=float, default=0.0,
                    help='AAG: JSD consistency loss weight (0=CE-only, DAT=2.0)')
parser.add_argument('--aag-g-lr',       type=float, default=0.1,
                    help='AAG: generator SGD lr (DAT default 0.1)')
parser.add_argument('--aag-z-dim',      type=int,   default=100,
                    help='AAG: noise vector dimension (default 100)')

# WaveletFusion specific
parser.add_argument('--wf-wavelet',    type=str,   default='haar',
                    help='WaveletFusion: pywt wavelet basis (default haar)')
parser.add_argument('--wf-level',      type=int,   default=1,
                    help='WaveletFusion: decomposition level (default 1)')
parser.add_argument('--wf-apply-prob', type=float, default=0.5,
                    help='WaveletFusion: probability of applying fusion per sample (default 0.5)')

# loss
parser.add_argument('--jsd-lambda', type=float, default=12.0,
                    help='JSD consistency loss weight (λ). '
                         '0 = CE only: dataset returns 1 image, single forward pass, '
                         'loss = CrossEntropy only. BatchNorm stats identical to standard CE.')

# optimization
parser.add_argument('--batch-size', type=int,   default=128,
                    help='Per-GPU batch size. Effective batch is 3× in JSD mode.')
parser.add_argument('--lr',         type=float, default=0.1)
parser.add_argument('--max-epoch',  type=int,   default=250)

# model
parser.add_argument('--model', type=str, default='resnet18')

# eval
parser.add_argument('--eval', type=str, default='none', choices=['none', 'eval'])

# misc
parser.add_argument('--workers',    type=int,   default=16)
parser.add_argument('--gpu',        type=str,   default='0')
parser.add_argument('--print-freq', type=int,   default=100)
parser.add_argument('--seed',       type=int,   default=0)
parser.add_argument('--grad-clip',  type=float, default=None,
                    help='gradient clipping max norm. Default: 1.0 in JSD mode, None in CE mode.')

args = parser.parse_args()

# APR-AAG は独自の JSD を管理するため、グローバル jsd_lambda は無効化
if args.aug == 'apr-aag':
    args.jsd_lambda = 0.0

# ── Aug-order default: wca_first for CE, crop_first for JSD ──────────────────
if args.aug_order is None:
    args.aug_order = 'crop_first' if args.jsd_lambda > 0 else 'wca_first'

# ── Grad-clip default: 1.0 in JSD mode ───────────────────────────────────────
if args.grad_clip is None and args.jsd_lambda > 0:
    args.grad_clip = 1.0

os.makedirs(args.outfolder, exist_ok=True)
sys.stdout = Logger(osp.join(args.outfolder, 'logs.txt'))


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _csv_init(path, extra_headers=()):
    if not osp.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'lr', 'loss_total', 'loss_ce', 'loss_jsd', 'test_acc']
                + list(extra_headers)
            )


def _csv_append(path, epoch, lr, losses, acc, extra_vals=()):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch,
            f'{lr:.8f}',
            f'{losses["total"]:.6f}',
            f'{losses["ce"]:.6f}',
            f'{losses["jsd"]:.6f}' if losses['jsd'] is not None else '',
            f'{acc:.5f}'           if acc is not None             else '',
        ] + [f'{v:.6f}' if v is not None else '' for v in extra_vals])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), 'CUDA not available'
    cudnn.benchmark = True

    print(f'GPU: {args.gpu}  seed={args.seed}')
    if args.aug == 'wca':
        if args.basis_random:
            from core.wca import BASIS_POOL
            print(f'aug=wca  basis_random=True  pool={BASIS_POOL}  '
                  f'level={args.level}  swap_prob={args.swap_prob}')
        else:
            print(f'aug={args.aug}  source={args.source}  target={args.target}  '
                  f'level={args.level}  swap_prob={args.swap_prob}')
    elif args.aug in ('apr-s', 'apr-s-soft', 'apr-s-cls', 'apr-s-orig', 'apr-s-orig-cls'):
        print(f'aug={args.aug}')
        if args.aug == 'apr-s-soft':
            print(f't_max={args.t_max}  label_k={args.label_k}  clean_prob={args.clean_prob}')
        elif args.aug in ('apr-s-cls', 'apr-s-orig-cls'):
            print(f'gamma_swap={args.gamma_swap}  uncond_smooth={args.uncond_smooth}')
        elif args.aug == 'apr-s':
            print(f'[APR-S] apply_prob=0.5 (matching gary23ai/APR original)')
        elif args.aug == 'apr-s-orig':
            print('[APR-S-orig] faithful port: same-image 2-view FFT recombination, no clip, uint8 overflow')
        elif args.aug == 'apr-s-orig-cls':
            print('[APR-S-orig-cls] faithful port + conditional label smoothing on swapped samples')
    elif args.aug == 'wapr-orig':
        print(f'aug=wapr-orig  J={args.wapr_levels}  lam={args.wapr_lam}  '
              f'lowpass={args.wapr_lowpass}  phase_sigma_vital={args.wapr_phase_sigma_vital}')
    elif args.aug in ('wf-sign', 'wf-max'):
        print(f'aug={args.aug}  wf_wavelet={args.wf_wavelet}  '
              f'wf_level={args.wf_level}  wf_apply_prob={args.wf_apply_prob}')
    elif args.aug == 'apr-aag':
        print(f'aug=apr-aag  jsd_weight={args.aag_jsd_weight}  '
              f'mix_beta={args.aag_mix_beta}  g_lr={args.aag_g_lr}  z_dim={args.aag_z_dim}')
    else:
        print(f'aug={args.aug}')
    print(f'jsd_lambda={args.jsd_lambda}  aug_order={args.aug_order}  grad_clip={args.grad_clip}')

    eval_mode = (args.eval == 'eval')
    if args.aug in ('apr-s', 'apr-s-soft', 'apr-s-cls', 'apr-s-orig', 'apr-s-orig-cls'):
        from datasets.apr_soft import build_apr_loaders
        train_loader, test_loader, corruption_loaders = build_apr_loaders(args, eval_mode=eval_mode)
    elif args.aug == 'wapr-orig':
        from datasets.wapr import build_wapr_loaders
        train_loader, test_loader, corruption_loaders = build_wapr_loaders(args, eval_mode=eval_mode)
    elif args.aug in ('wf-sign', 'wf-max'):
        from datasets.wavelet_fusion import build_wf_loaders
        train_loader, test_loader, corruption_loaders = build_wf_loaders(args, eval_mode=eval_mode)
    else:
        # apr-aag も標準 DataLoader を使う (振幅操作はバッチループ内で実施)
        train_loader, test_loader, corruption_loaders = build_loaders(args, eval_mode=eval_mode)

    from model.resnet import ResNet18
    net = ResNet18(num_classes=10)
    net = torch.nn.DataParallel(net).cuda()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'Parameters: {n_params:,}')

    if args.aug == 'apr-s-soft':
        file_name = (
            f'{args.model}_{args.dataset}_{args.aug}'
            f'_tmax{args.t_max}_k{args.label_k}_cp{args.clean_prob}'
            f'_{args.memo}'
        )
    elif args.aug in ('apr-s', 'apr-s-cls'):
        file_name = (
            f'{args.model}_{args.dataset}_{args.aug}'
            f'_gs{args.gamma_swap}_us{args.uncond_smooth}'
            f'_{args.memo}'
        )
    elif args.aug == 'apr-s-orig':
        file_name = f'{args.model}_{args.dataset}_{args.aug}_{args.memo}'
    elif args.aug == 'wapr-orig':
        file_name = (
            f'{args.model}_{args.dataset}_wapr-orig'
            f'_J{args.wapr_levels}_lam{args.wapr_lam}_lp{args.wapr_lowpass}'
            f'_{args.memo}'
        )
    elif args.aug == 'apr-s-orig-cls':
        file_name = (
            f'{args.model}_{args.dataset}_{args.aug}'
            f'_gs{args.gamma_swap}_us{args.uncond_smooth}'
            f'_{args.memo}'
        )
    elif args.aug in ('wf-sign', 'wf-max'):
        file_name = (
            f'{args.model}_{args.dataset}_{args.aug}'
            f'_wv{args.wf_wavelet}_l{args.wf_level}_p{args.wf_apply_prob}'
            f'_{args.memo}'
        )
    elif args.aug == 'apr-aag':
        file_name = (
            f'{args.model}_{args.dataset}_apr-aag'
            f'_jsd{args.aag_jsd_weight}_mb{args.aag_mix_beta}'
            f'_{args.memo}'
        )
    else:
        file_name = (
            f'{args.model}_{args.dataset}_{args.aug}'
            f'_src{args.source}_tgt{args.target}_l{args.level}_p{args.swap_prob}'
            f'_jsd{args.jsd_lambda}_{args.memo}'
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if eval_mode:
        net = load_networks(net, args.outfolder, file_name)
        clean_acc = test(net, test_loader)
        print(f'clean accuracy: {clean_acc}')
        acc_list = []
        for key in CORRUPTION_TYPES:
            res = test_robustness(net, corruption_loaders[key])
            print(f'{key} (%): {res["ACC"]:.3f}\t')
            print(f'corruption error: {100 - res["ACC"]}')
            acc_list.append(res['ACC'])
        mean_acc   = float(np.mean(acc_list))
        mean_error = 100.0 - mean_acc
        print(f'Mean ACC: {mean_acc}')
        print(f'Mean Error: {mean_error}')
        print(f'mean acc: {mean_acc} mean error: {mean_error}')
        return

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 190], gamma=0.2)

    # AAG generator (apr-aag のみ)
    aag_G, aag_optim_G, aag_scheduler_G = None, None, None
    if args.aug == 'apr-aag':
        from core.aag import GeneratorMLP
        aag_G = GeneratorMLP(z_dim=args.aag_z_dim).cuda()
        aag_optim_G = torch.optim.SGD(
            aag_G.parameters(), lr=args.aag_g_lr, momentum=0.9, weight_decay=5e-4)
        aag_scheduler_G = lr_scheduler.MultiStepLR(
            aag_optim_G, milestones=[60, 120, 160, 190], gamma=0.2)
        n_g = sum(p.numel() for p in aag_G.parameters())
        print(f'AAG Generator parameters: {n_g:,}')

    csv_path   = osp.join(args.outfolder, 'history.csv')
    if args.aug == 'apr-s-soft':
        _apr_extra_headers = ('t_mean', 't_std', 'gamma_mean', 'gamma_std')
    elif args.aug in ('apr-s-cls', 'apr-s-orig-cls'):
        _apr_extra_headers = ('swap_rate', 'gamma_mean')
    elif args.aug in ('wf-sign', 'wf-max'):
        _apr_extra_headers = ('apply_rate',)
    elif args.aug == 'apr-aag':
        _apr_extra_headers = ('g_loss', 'ce_base', 'ce_auto', 'jsd_raw')
    else:
        _apr_extra_headers = ()
    _csv_init(csv_path, extra_headers=_apr_extra_headers)
    best_acc   = 0.0
    start_time = time.time()

    for epoch in range(args.max_epoch):
        t0 = time.time()
        print(f'==> Epoch {epoch + 1}/{args.max_epoch}')

        if args.aug == 'apr-aag':
            losses = _train_epoch_aag(
                net, optimizer, train_loader, aag_G, aag_optim_G, epoch)
        else:
            losses = _train_epoch(net, optimizer, train_loader)
        if args.aug == 'apr-aag':
            ce_diff = losses['ce_auto'] - losses['ce_base']
            jsd_str = (f'  JSD={losses["jsd_raw"]:.4f}' if losses['jsd_raw'] is not None else '')
            print(f'epoch_loss: {losses["total"]:.6f}  '
                  f'CE_base={losses["ce_base"]:.6f}  CE_auto={losses["ce_auto"]:.6f}  '
                  f'CE_diff={ce_diff:+.6f}  g_loss={losses["g_loss"]:.6f}{jsd_str}')
        elif losses['jsd'] is not None:
            print(f'epoch_loss: {losses["total"]:.6f}  '
                  f'CE: {losses["ce"]:.6f}  JSD(pre-λ): {losses["jsd"]:.6f}')
        elif losses.get('t_mean') is not None:
            print(f'epoch_loss: {losses["total"]:.6f}  '
                  f't={losses["t_mean"]:.3f}±{losses["t_std"]:.3f}  '
                  f'γ={losses["gamma_mean"]:.3f}±{losses["gamma_std"]:.3f}')
        elif losses.get('swap_rate') is not None:
            print(f'epoch_loss: {losses["total"]:.6f}  '
                  f'swap_rate={losses["swap_rate"]:.3f}  '
                  f'γ_mean={losses["gamma_mean"]:.4f}')
        else:
            print(f'epoch_loss: {losses["total"]:.6f}')

        acc        = None
        eval_start = int(args.max_epoch * 0.6)
        if epoch >= eval_start or epoch == args.max_epoch - 1:
            print('==> Test')
            acc = test(net, test_loader)
            print(f'accuracy: {acc}')
            if acc > best_acc:
                best_acc = acc
                print(f'Best Acc (%): {best_acc:.3f}')
                save_networks(net, args.outfolder, file_name)

        scheduler.step()
        if aag_scheduler_G is not None:
            aag_scheduler_G.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f'lr: {lr_now:.6f}  epoch_time(min): {(time.time() - t0) // 60}')
        if args.aug == 'apr-s-soft':
            _apr_extra_vals = (
                losses.get('t_mean'), losses.get('t_std'),
                losses.get('gamma_mean'), losses.get('gamma_std'),
            )
        elif args.aug in ('apr-s-cls', 'apr-s-orig-cls'):
            _apr_extra_vals = (losses.get('swap_rate'), losses.get('gamma_mean'))
        elif args.aug in ('wf-sign', 'wf-max'):
            _apr_extra_vals = (losses.get('apply_rate'),)
        elif args.aug == 'apr-aag':
            _apr_extra_vals = (
                losses.get('g_loss'), losses.get('ce_base'),
                losses.get('ce_auto'), losses.get('jsd_raw'),
            )
        else:
            _apr_extra_vals = ()
        _csv_append(csv_path, epoch + 1, lr_now, losses, acc, extra_vals=_apr_extra_vals)

    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print(f'Finished. Total elapsed time (h:m:s): {elapsed}')
    print(f'best accuracy: {best_acc}')


def _train_epoch(net, optimizer, loader):
    net.train()
    meter     = AverageMeter()
    meter_ce  = AverageMeter()
    meter_jsd = AverageMeter()

    if args.aug == 'apr-s-soft':
        # ── APR-S-soft モード ─────────────────────────────────────────────────
        # dataset が (x_aug, t_eff, y_own, y_dist) を返す
        C = 10  # CIFAR-10
        t_vals, gamma_vals = [], []

        for batch_idx, (x_aug, t_eff, y_own, y_dist) in enumerate(loader):
            x_aug  = x_aug.cuda()
            t_eff  = t_eff.cuda()    # (bs,)
            y_own  = y_own.cuda()
            y_dist = y_dist.cuda()
            bs     = x_aug.size(0)

            gamma  = t_eff ** args.label_k   # (bs,)
            y_soft = (
                (1 - gamma).unsqueeze(1) * F.one_hot(y_own,  C).float()
                + gamma.unsqueeze(1)     * F.one_hot(y_dist, C).float()
            )  # (bs, C)

            logits = net(x_aug)
            loss   = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            meter.update(loss.item(), bs)
            t_vals.extend(t_eff.detach().cpu().tolist())
            gamma_vals.extend(gamma.detach().cpu().tolist())

            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})  '
                      f't={t_eff.mean():.3f}  γ={gamma.mean():.3f}')

        t_arr     = np.array(t_vals)
        gamma_arr = np.array(gamma_vals)
        return {
            'total': meter.avg, 'ce': meter.avg, 'jsd': None,
            't_mean': float(t_arr.mean()),     't_std': float(t_arr.std()),
            'gamma_mean': float(gamma_arr.mean()), 'gamma_std': float(gamma_arr.std()),
        }

    elif args.aug in ('apr-s-cls', 'apr-s-orig-cls'):
        # ── APR-S-cls / APR-S-orig-cls モード (条件付きラベルスムージング) ─────
        # dataset が (x_aug, y_own, swapped) を返す
        C = 10
        total_swapped = 0
        total_samples = 0

        for batch_idx, (x_aug, y_own, swapped) in enumerate(loader):
            x_aug   = x_aug.cuda()
            y_own   = y_own.cuda()
            swapped = swapped.cuda()   # (bs,) bool
            bs      = x_aug.size(0)

            y_one_hot = F.one_hot(y_own, C).float()

            if args.uncond_smooth > 0.0:
                # 全サンプルに一律スムージング (対照実験用)
                y_soft = (1 - args.uncond_smooth) * y_one_hot + args.uncond_smooth / C
            else:
                # スワップしたサンプルのみ γ=gamma_swap、それ以外は one_hot
                y_smooth = (1 - args.gamma_swap) * y_one_hot + args.gamma_swap / C
                mask     = swapped.float().unsqueeze(1)  # (bs, 1)
                y_soft   = mask * y_smooth + (1 - mask) * y_one_hot

            logits = net(x_aug)
            loss   = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            meter.update(loss.item(), bs)
            total_swapped += int(swapped.sum().item())
            total_samples += bs

            if (batch_idx + 1) % args.print_freq == 0:
                sr = swapped.float().mean().item()
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})  swap={sr:.2f}')

        swap_rate = total_swapped / total_samples if total_samples > 0 else 0.0
        if args.uncond_smooth > 0.0:
            gamma_mean = args.uncond_smooth
        else:
            gamma_mean = swap_rate * args.gamma_swap
        return {'total': meter.avg, 'ce': meter.avg, 'jsd': None,
                'swap_rate': swap_rate, 'gamma_mean': gamma_mean}

    elif args.aug in ('wf-sign', 'wf-max'):
        # ── WaveletFusion モード (CE, apply_rate ログ) ────────────────────────
        # dataset が (x_aug, y_own, applied: bool) を返す
        total_applied = 0
        total_samples = 0

        for batch_idx, (x_aug, y_own, applied) in enumerate(loader):
            x_aug   = x_aug.cuda()
            y_own   = y_own.cuda()
            bs      = x_aug.size(0)

            logits = net(x_aug)
            loss   = F.cross_entropy(logits, y_own)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            meter.update(loss.item(), bs)
            total_applied += int(applied.sum().item())
            total_samples += bs

            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})')

        apply_rate = total_applied / total_samples if total_samples > 0 else 0.0
        return {'total': meter.avg, 'ce': meter.avg, 'jsd': None, 'apply_rate': apply_rate}

    elif args.aug in ('apr-s', 'apr-s-orig', 'wapr-orig'):
        # ── APR-S / APR-S-orig / wapr-orig モード (通常 CE) ──────────────────
        # dataset が (x_aug, y_own) を返す
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs  = inputs.cuda()
            targets = targets.cuda()

            logits = net(inputs)
            loss   = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            meter.update(loss.item(), targets.size(0))
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})')

        return {'total': meter.avg, 'ce': meter.avg, 'jsd': None}

    elif args.jsd_lambda > 0:
        # ── JSD モード ───────────────────────────────────────────────────────
        # dataset が ((x_clean, x_aug1, x_aug2), y) を返す
        for batch_idx, ((x_clean, x_aug1, x_aug2), targets) in enumerate(loader):
            x_clean  = x_clean.cuda()
            x_aug1   = x_aug1.cuda()
            x_aug2   = x_aug2.cuda()
            targets  = targets.cuda()
            bs       = x_clean.size(0)

            # 3枚を cat して1回の順伝播
            logits = net(torch.cat([x_clean, x_aug1, x_aug2], dim=0))
            l_clean, l_aug1, l_aug2 = torch.split(logits, bs, dim=0)

            loss_ce  = F.cross_entropy(l_clean, targets)
            loss_jsd = jsd_loss(l_clean, l_aug1, l_aug2)
            loss     = loss_ce + args.jsd_lambda * loss_jsd

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            meter.update(loss.item(),         bs)
            meter_ce.update(loss_ce.item(),   bs)
            meter_jsd.update(loss_jsd.item(), bs)
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})  '
                      f'CE {meter_ce.val:.6f} ({meter_ce.avg:.6f})  '
                      f'JSD {meter_jsd.val:.6f} ({meter_jsd.avg:.6f})')

        return {'total': meter.avg, 'ce': meter_ce.avg, 'jsd': meter_jsd.avg}

    else:
        # ── CE-only モード ───────────────────────────────────────────────────
        # dataset が (x, y) を返す（3枚 cat なし、単一順伝播）
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs  = inputs.cuda()
            targets = targets.cuda()

            logits = net(inputs)
            loss   = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            meter.update(loss.item(), targets.size(0))
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})')

        return {'total': meter.avg, 'ce': meter.avg, 'jsd': None}


def _aag_compute_loss(logits_b, logits_a, y, jsd_weight):
    """APR-AAG 用損失: (CE_base + CE_auto)/2 + jsd_weight * JSD。"""
    ce_b  = F.cross_entropy(logits_b, y)
    ce_a  = F.cross_entropy(logits_a, y)
    total = (ce_b + ce_a) / 2
    jsd_val = None
    if jsd_weight > 0:
        p_b = F.softmax(logits_b, dim=1)
        p_a = F.softmax(logits_a, dim=1)
        m   = torch.clamp((p_b + p_a) / 2., 1e-7, 1).log()
        jsd_val = (F.kl_div(m, p_b, reduction='batchmean') +
                   F.kl_div(m, p_a, reduction='batchmean')) / 2
        total = total + jsd_weight * jsd_val
    return total, ce_b, ce_a, jsd_val


def _train_epoch_aag(net, optimizer, loader, aag_G, aag_optim_G, epoch):
    """APR-AAG 学習エポック: DAT 公式の交互更新ループ (AT 機構なし)。

    バッチごとに:
      1. model.eval() で feat = model(x_base) を取得 (generator 条件付け)
      2. G(z, feat) で振幅生成 → beta * amp_G + (1-beta) * amp_orig で混合
      3. inverse_fft → clip → renorm で x_auto1 を構成
      4. model 損失 = (CE_base + CE_auto)/2 + jsd_weight*JSD を backward(retain_graph)
      5. g_loss = -(同損失の再計算) を G に backward
    """
    from core.aag import get_fft, inverse_fft, denorm, renorm
    from torchvision.utils import save_image

    net.train()
    aag_G.train()

    meter_total  = AverageMeter()
    meter_ce_b   = AverageMeter()
    meter_ce_a   = AverageMeter()
    meter_jsd    = AverageMeter()
    meter_g_loss = AverageMeter()

    last_x_auto1_raw = None

    for batch_idx, (x_base, y) in enumerate(loader):
        x_base = x_base.cuda()
        y      = y.cuda()
        bs     = x_base.size(0)

        # ── 1. 条件付け (DAT 公式通り eval mode, detach) ──────────────────────
        net.eval()
        with torch.no_grad():
            feat = net(x_base)           # raw logits (B, 10)
        net.train()

        # ── 2. Generator forward ──────────────────────────────────────────────
        z     = torch.randn(bs, args.aag_z_dim, device=x_base.device)
        amp_G = aag_G(z, feat.detach()) # (B,3,32,32), [0,1]; G.params → amp_G

        # ── 3. 振幅混合 (DAT 公式通り: [0,1] スケールの amp_G と生 FFT 振幅を混合) ──
        x_raw            = denorm(x_base).detach()   # [0,1], no grad
        amp_orig, pha_orig = get_fft(x_raw)          # no grad

        beta1 = torch.from_numpy(
            np.random.uniform(0, args.aag_mix_beta, (bs,)).astype(np.float32)
        ).to(x_base.device).view(bs, 1, 1, 1)

        amp_mixed    = beta1 * amp_G + (1 - beta1) * amp_orig  # grad via amp_G
        x_auto1_raw  = inverse_fft(amp_mixed, pha_orig)        # [0,1], grad via amp_G
        x_auto1      = renorm(x_auto1_raw)                     # normalized, grad via amp_G

        last_x_auto1_raw = x_auto1_raw.detach().cpu()

        # ── 4. Model update ───────────────────────────────────────────────────
        logits_b = net(x_base)
        logits_a = net(x_auto1)
        loss, ce_b, ce_a, jsd_val = _aag_compute_loss(
            logits_b, logits_a, y, args.aag_jsd_weight)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)    # グラフ保持: g_loss のために必要
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()

        # ── 5. Generator update (updated model で再 forward) ──────────────────
        logits_b2 = net(x_base)
        logits_a2 = net(x_auto1)           # x_auto1 → amp_G → G.params (retained)
        g_loss_pos, _, _, _ = _aag_compute_loss(
            logits_b2, logits_a2, y, args.aag_jsd_weight)
        g_loss = -g_loss_pos               # G は model 損失を最大化

        aag_optim_G.zero_grad()
        g_loss.backward()
        aag_optim_G.step()

        # ── メータ ─────────────────────────────────────────────────────────────
        meter_total.update(loss.item(), bs)
        meter_ce_b.update(ce_b.item(),  bs)
        meter_ce_a.update(ce_a.item(),  bs)
        if jsd_val is not None:
            meter_jsd.update(jsd_val.item(), bs)
        meter_g_loss.update(g_loss_pos.item(), bs)   # 正値でログ

        if (batch_idx + 1) % args.print_freq == 0:
            ce_diff = ce_a.item() - ce_b.item()
            print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                  f'Loss {meter_total.val:.4f} ({meter_total.avg:.4f})  '
                  f'CE_base={ce_b.item():.4f}  CE_auto={ce_a.item():.4f}  '
                  f'CE_diff={ce_diff:+.4f}  g_loss={g_loss_pos.item():.4f}')

    # ── 10 エポックごとにサンプル画像保存 ────────────────────────────────────
    if (epoch + 1) % 10 == 0 and last_x_auto1_raw is not None:
        save_dir = osp.join(args.outfolder, 'aug_samples')
        os.makedirs(save_dir, exist_ok=True)
        save_image(last_x_auto1_raw[:8],
                   osp.join(save_dir, f'ep{epoch + 1:03d}.png'), nrow=8)

    return {
        'total':   meter_total.avg,
        'ce':      meter_ce_b.avg,
        'jsd':     None,                  # legacy field
        'g_loss':  meter_g_loss.avg,
        'ce_base': meter_ce_b.avg,
        'ce_auto': meter_ce_a.avg,
        'jsd_raw': meter_jsd.avg if args.aag_jsd_weight > 0 else None,
    }


if __name__ == '__main__':
    main()
