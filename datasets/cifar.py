import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2023, 0.1994, 0.2010]

CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

_TO_TENSOR_NORM = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

_BASIC_AUG = T.Compose([
    T.RandomCrop(32, padding=4, fill=128),
    T.RandomHorizontalFlip(),
])


class _RawCIFAR10(Dataset):
    """torchvision CIFAR10 を transform なしで返すラッパー。"""
    def __init__(self, root, train=True, download=True):
        self._ds = CIFAR10(root=root, train=train, download=download, transform=None)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]  # (PIL Image, int)


class CIFAR10TrainDataset(Dataset):
    """
    CE モード (jsd_mode=False):
        (x, y) を返す。x は拡張済みテンソル。
        aug_order='wca_first' : WCA → Crop/Flip → ToTensor+Norm  ← VIPAug_phase と同一順序
        aug_order='crop_first': Crop/Flip → WCA → ToTensor+Norm

    JSD モード (jsd_mode=True):
        ((x_clean, x_aug1, x_aug2), y) を返す。
        aug_order='crop_first': base = Crop/Flip(img)
            x_clean = Norm(base), x_aug1 = Norm(WCA(base)), x_aug2 = Norm(WCA(base))
        aug_order='wca_first' : 各ブランチが独立に WCA → Crop/Flip を受ける
            x_clean = Norm(Crop/Flip(img))
            x_aug1  = Norm(Crop/Flip(WCA(img)))
            x_aug2  = Norm(Crop/Flip(WCA(img)))
    """

    def __init__(self, root, wca_aug, jsd_mode=False, aug_order='wca_first', download=True):
        if jsd_mode and wca_aug is None:
            raise ValueError('--aug none と --jsd-lambda > 0 の組み合わせは未対応です。')
        self._raw     = _RawCIFAR10(root, train=True, download=download)
        self.wca_aug  = wca_aug
        self.jsd_mode = jsd_mode
        self.aug_order = aug_order

    def __len__(self):
        return len(self._raw)

    def __getitem__(self, idx):
        img, label = self._raw[idx]

        if not self.jsd_mode:
            # ── CE モード ────────────────────────────────────────────────
            if self.aug_order == 'wca_first' and self.wca_aug is not None:
                img = self.wca_aug(img)   # WCA 先
                img = _BASIC_AUG(img)     # Crop/Flip 後
            else:                          # crop_first
                img = _BASIC_AUG(img)
                if self.wca_aug is not None:
                    img = self.wca_aug(img)
            return _TO_TENSOR_NORM(img), label

        else:
            # ── JSD モード ───────────────────────────────────────────────
            if self.aug_order == 'crop_first':
                base    = _BASIC_AUG(img)                      # 共有ベース
                x_clean = _TO_TENSOR_NORM(base)
                x_aug1  = _TO_TENSOR_NORM(self.wca_aug(base))  # 独立 WCA ×2
                x_aug2  = _TO_TENSOR_NORM(self.wca_aug(base))
            else:  # wca_first
                x_clean = _TO_TENSOR_NORM(_BASIC_AUG(img))
                x_aug1  = _TO_TENSOR_NORM(_BASIC_AUG(self.wca_aug(img)))
                x_aug2  = _TO_TENSOR_NORM(_BASIC_AUG(self.wca_aug(img)))
            return (x_clean, x_aug1, x_aug2), label


class CIFAR10TestDataset(Dataset):
    def __init__(self, root, download=True):
        self._raw = _RawCIFAR10(root, train=False, download=download)

    def __len__(self):
        return len(self._raw)

    def __getitem__(self, idx):
        img, label = self._raw[idx]
        return _TO_TENSOR_NORM(img), label


class CIFAR10CDataset(Dataset):
    """CIFAR-10-C (5 severities 合計 50,000 サンプル)。"""
    def __init__(self, root, corruption_type):
        self.data    = np.load(os.path.join(root, corruption_type + '.npy'))
        self.targets = np.load(os.path.join(root, 'labels.npy'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        return _TO_TENSOR_NORM(img), int(self.targets[idx])


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_loaders(args, eval_mode=False):
    data_root = os.path.join(args.data, 'cifar10')

    wca_aug = None
    if args.aug == 'wca':
        from core.wca import WaveletBasisSwap
        wca_aug = WaveletBasisSwap(
            source_wavelet=args.source,
            target_wavelet=args.target,
            level=args.level,
            swap_prob=args.swap_prob,
        )
    elif args.aug == 'augmix':
        from torchvision.transforms import AugMix
        wca_aug = AugMix()  # severity=3, mixture_width=3, chain_depth=-1

    jsd_mode  = (args.jsd_lambda > 0)
    aug_order = args.aug_order

    train_ds = CIFAR10TrainDataset(data_root, wca_aug, jsd_mode=jsd_mode, aug_order=aug_order)
    test_ds  = CIFAR10TestDataset(data_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              worker_init_fn=_worker_init_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

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
