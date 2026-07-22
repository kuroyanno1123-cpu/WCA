import os
import sys
import os.path as osp

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class Logger:
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file    = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'w')

    def __del__(self):           self.close()
    def __enter__(self):         pass
    def __exit__(self, *args):   self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file:
            self.file.close()


def save_networks(net, result_dir, name):
    ckpt_dir = osp.join(result_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(net.state_dict(), osp.join(ckpt_dir, f'{name}.pth'))


def load_networks(net, result_dir, name):
    path = osp.join(result_dir, 'checkpoints', f'{name}.pth')
    net.load_state_dict(torch.load(path))
    return net


@torch.no_grad()
def test(net, loader):
    net.eval()
    correct = total = 0
    torch.cuda.empty_cache()
    for data, labels in loader:
        data, labels = data.cuda(), labels.cuda()
        preds = net(data, _eval=True).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    acc = 100.0 * correct / total
    print(f'Acc: {acc:.5f}')
    return acc


@torch.no_grad()
def test_robustness(net, loader):
    net.eval()
    correct = total = 0
    torch.cuda.empty_cache()
    for data, labels in loader:
        data, labels = data.cuda(), labels.cuda()
        preds = net(data, _eval=True).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    acc = 100.0 * correct / total
    return {'ACC': acc}
