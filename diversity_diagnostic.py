"""
Diversity diagnostic: measures how "different" aug1 vs aug2 actually are,
for WCA and AugMix, in both pixel space and feature space.

This directly tests the hypothesis: "WCA's low JSD(pre-lambda) reflects a
lack of augmentation diversity, not learned invariance."

USAGE
-----
Run from the WCA repo root:
    python diversity_diagnostic.py

OUTPUT
------
Per-method: mean/std of
  - pixel-space L2 distance between aug1 and aug2
  - feature-space L2 distance (penultimate layer embedding)
  - feature-space cosine distance
Printed as a table, plus a plain verdict line.

If WCA distances are meaningfully smaller than AugMix's across all three
metrics, that's direct (training-free) evidence for the diversity-deficit
hypothesis. If they're comparable, the problem lies elsewhere (e.g.
perturbation strength calibration in a specific frequency band, not
overall diversity).
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# ── WCA augmentation ──────────────────────────────────────────────────────────
# WaveletBasisSwap: haar→db8, level=1, swap_prob=0.2 (スクリーニングと同一設定)
from core.wca import WaveletBasisSwap
_wbs = WaveletBasisSwap(source_wavelet='haar', target_wavelet='db8',
                        level=1, swap_prob=0.2, basis_random=True)
WCA_AUG_FN = _wbs  # PIL Image → PIL Image

# ── AugMix augmentation ───────────────────────────────────────────────────────
_augmix = T.AugMix()  # severity=3, mixture_width=3, chain_depth=-1
AUGMIX_AUG_FN = _augmix  # PIL Image → PIL Image

# ── Checkpoint ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = (
    './results/screen_wca_jsd/checkpoints/'
    'resnet18_cifar10_wca_srchaar_tgtdb8_l1_p0.2_jsd12.0_screen_wca_jsd.pth'
)
DATA_ROOT = '/home/kairisasaki/data/cifar10'
N_SAMPLES = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_feature_extractor(checkpoint_path):
    """Load our custom ResNet-18 and use penultimate features (rf=True)."""
    from model.resnet import ResNet18
    model = ResNet18(num_classes=10)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    # DataParallel checkpoint has 'module.' prefix — strip it
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model


_to_tensor = T.ToTensor()
_normalize  = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])


def get_two_views(img_pil, aug_fn):
    """Apply aug_fn twice independently (mirrors aug1/aug2 in JSD training)."""
    v1 = aug_fn(img_pil)
    v2 = aug_fn(img_pil)
    if not torch.is_tensor(v1):
        v1 = _normalize(_to_tensor(v1))
    if not torch.is_tensor(v2):
        v2 = _normalize(_to_tensor(v2))
    return v1, v2


@torch.no_grad()
def measure(aug_fn, dataset, model, n_samples):
    pixel_dists, feat_l2, feat_cos = [], [], []
    for i in range(n_samples):
        img_pil, _ = dataset[i]
        v1, v2 = get_two_views(img_pil, aug_fn)

        # pixel L2 (on normalized tensors)
        pixel_dists.append(torch.norm(v1 - v2).item())

        batch = torch.stack([v1, v2]).to(DEVICE)
        # rf=True → (features, logits); features is 512-dim penultimate embedding
        emb, _ = model(batch, rf=True)
        feat_l2.append(torch.norm(emb[0] - emb[1]).item())
        cos = F.cosine_similarity(emb[0:1], emb[1:2]).item()
        feat_cos.append(1.0 - cos)

    return {
        'pixel_L2':     (np.mean(pixel_dists), np.std(pixel_dists)),
        'feat_L2':      (np.mean(feat_l2),     np.std(feat_l2)),
        'feat_cos_dist': (np.mean(feat_cos),   np.std(feat_cos)),
    }


def main():
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=None
    )

    model = load_feature_extractor(CHECKPOINT_PATH)
    print(f'Running diagnostic on {N_SAMPLES} images, device={DEVICE}\n')

    results = {}
    for name, fn in [('WCA', WCA_AUG_FN), ('AugMix', AUGMIX_AUG_FN)]:
        print(f'-- measuring {name} --')
        results[name] = measure(fn, dataset, model, N_SAMPLES)

    print('\n=== RESULTS (mean ± std) ===')
    print(f"{'method':<10}{'pixel_L2':<24}{'feat_L2':<24}{'feat_cos_dist':<24}")
    for name, r in results.items():
        print(
            f"{name:<10}"
            f"{r['pixel_L2'][0]:.4f}±{r['pixel_L2'][1]:.4f}        "
            f"{r['feat_L2'][0]:.4f}±{r['feat_L2'][1]:.4f}        "
            f"{r['feat_cos_dist'][0]:.4f}±{r['feat_cos_dist'][1]:.4f}"
        )

    wca_feat = results['WCA']['feat_L2'][0]
    aug_feat = results['AugMix']['feat_L2'][0]
    ratio = wca_feat / aug_feat if aug_feat > 0 else float('inf')

    print('\n=== VERDICT ===')
    if ratio < 0.5:
        print(
            f'WCA feature-space distance is {ratio:.2f}x AugMix\'s -- '
            'strong support for the diversity-deficit hypothesis. '
            'Prioritize basis-pair/subband randomization.'
        )
    elif ratio < 0.8:
        print(
            f'WCA feature-space distance is {ratio:.2f}x AugMix\'s -- '
            'mild support for diversity deficit, but not conclusive alone. '
            'Worth running the randomization experiment, but also check '
            'perturbation strength calibration.'
        )
    else:
        print(
            f'WCA feature-space distance is {ratio:.2f}x AugMix\'s -- '
            'comparable diversity. The low JSD is likely learned invariance, '
            'not a diversity deficit. Look elsewhere (e.g. which frequency '
            'bands the perturbation targets, or train/test distribution '
            'mismatch in the wavelet basis swap).'
        )


if __name__ == '__main__':
    main()
