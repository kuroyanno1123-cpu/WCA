"""Adversarial Amplitude Generator (AAG) — AT-free port from DAT (NeurIPS 2024).

DAT公式: github.com/Feng-peng-Li/DAT  Generator_MLP / get_fft / inverse_fft を
AT機構なしで移植。スケール処理は公式通り(amp_G:[0,1]、na_amp:生FFT振幅)。
"""

import torch
import torch.nn as nn

# ── CIFAR-10 正規化定数 ───────────────────────────────────────────────────────
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """CIFAR-10正規化済みテンソル → [0,1]。"""
    mean = _MEAN.to(x.device)
    std  = _STD.to(x.device)
    return x * std + mean


def renorm(x: torch.Tensor) -> torch.Tensor:
    """[0,1]テンソル → CIFAR-10正規化。"""
    mean = _MEAN.to(x.device)
    std  = _STD.to(x.device)
    return (x - mean) / std


# ── FFT utilities (DAT公式と同一) ─────────────────────────────────────────────

def get_fft(x: torch.Tensor):
    """[0,1] 画像 → (fftshift済み振幅, 非shift位相)。DAT get_fft と同一。"""
    fft_im  = torch.fft.fftn(x, dim=(-2, -1))
    fft_amp = torch.fft.fftshift(torch.abs(fft_im), dim=(-2, -1))
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha


def inverse_fft(fft_amp: torch.Tensor, fft_pha: torch.Tensor) -> torch.Tensor:
    """(fftshift済み振幅, 非shift位相) → [0,1] clip。DAT inverse_fft と同一。"""
    amp = torch.fft.ifftshift(fft_amp, dim=(-2, -1))
    img = torch.fft.ifftn(amp * torch.exp(1j * fft_pha), dim=(-2, -1))
    return torch.clip(torch.real(img).float(), 0., 1.)


# ── Generator ─────────────────────────────────────────────────────────────────

def _init_weights(m):
    """DAT init_weights と同一。"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class GeneratorMLP(nn.Module):
    """DAT Generator_MLP の忠実な移植。

    入力: concat(z[B,z_dim], feat[B,num_classes]) → MLP → 振幅 [B,C,H,W], Sigmoid。
    dead params (feature_emb, linear) も含め、init順・重みが公式と一致する。
    """

    def __init__(self, z_dim=100, out_channels=3, img_h=32, img_w=32, num_classes=10):
        super().__init__()
        self.out_channels = out_channels
        self.h = img_h
        self.w = img_w

        # DAT公式の dead params — init_weights の RNG 消費順を合わせるために必要
        self.feature_emb = nn.Embedding(num_classes, num_classes)
        self.linear       = nn.Linear(num_classes, 10)

        def block(in_f, out_f, normalize=True):
            layers = [nn.Linear(in_f, out_f)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_f, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, out_channels * img_h * img_w),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        x   = torch.cat([z, feat], dim=1)
        out = self.model(x)
        return out.view(out.shape[0], self.out_channels, self.h, self.w)
