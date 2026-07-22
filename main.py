import os
import sys
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
                    choices=['wca', 'none'], help='augmentation type')
parser.add_argument('--source',    type=str,   default='haar', help='source wavelet')
parser.add_argument('--target',    type=str,   default='db8',  help='target wavelet')
parser.add_argument('--level',     type=int,   default=1,      help='DWT decomposition level')
parser.add_argument('--swap-prob', type=float, default=0.2,    help='HF coefficient swap probability')
parser.add_argument('--aug-order', type=str,   default=None,
                    choices=['wca_first', 'crop_first'],
                    help='Preprocessing order. '
                         'Default: wca_first when --jsd-lambda 0, crop_first otherwise.')

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
parser.add_argument('--workers',    type=int, default=16)
parser.add_argument('--gpu',        type=str, default='0')
parser.add_argument('--print-freq', type=int, default=100)

args = parser.parse_args()

# ── Aug-order default: wca_first for CE, crop_first for JSD ──────────────────
if args.aug_order is None:
    args.aug_order = 'crop_first' if args.jsd_lambda > 0 else 'wca_first'

os.makedirs(args.outfolder, exist_ok=True)
sys.stdout = Logger(osp.join(args.outfolder, 'logs.txt'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), 'CUDA not available'
    cudnn.benchmark = True

    print(f'GPU: {args.gpu}')
    print(f'aug={args.aug}  source={args.source}  target={args.target}  '
          f'level={args.level}  swap_prob={args.swap_prob}')
    print(f'jsd_lambda={args.jsd_lambda}  aug_order={args.aug_order}')

    eval_mode = (args.eval == 'eval')
    train_loader, test_loader, corruption_loaders = build_loaders(args, eval_mode=eval_mode)

    from model.resnet import ResNet18
    net = ResNet18(num_classes=10)
    net = torch.nn.DataParallel(net).cuda()

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

    best_acc   = 0.0
    start_time = time.time()

    for epoch in range(args.max_epoch):
        t0 = time.time()
        print(f'==> Epoch {epoch + 1}/{args.max_epoch}')

        loss_all = _train_epoch(net, optimizer, train_loader)
        print(f'epoch_loss: {loss_all}')

        if epoch > 150:
            print('==> Test')
            acc = test(net, test_loader)
            print(f'accuracy: {acc}')
            if acc > best_acc:
                best_acc = acc
                print(f'Best Acc (%): {best_acc:.3f}')
                save_networks(net, args.outfolder, file_name)

        scheduler.step()
        print(f'epoch_time(min): {(time.time() - t0) // 60}')

    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print(f'Finished. Total elapsed time (h:m:s): {elapsed}')
    print(f'best accuracy: {best_acc}')


def _train_epoch(net, optimizer, loader):
    net.train()
    meter    = AverageMeter()
    loss_all = 0.0

    if args.jsd_lambda > 0:
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

            loss = F.cross_entropy(l_clean, targets) + \
                   args.jsd_lambda * jsd_loss(l_clean, l_aug1, l_aug2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), bs)
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})')
            loss_all += meter.avg

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
            optimizer.step()

            meter.update(loss.item(), targets.size(0))
            if (batch_idx + 1) % args.print_freq == 0:
                print(f'Batch {batch_idx + 1}/{len(loader)}\t'
                      f'Loss {meter.val:.6f} ({meter.avg:.6f})')
            loss_all += meter.avg

    return loss_all


if __name__ == '__main__':
    main()
