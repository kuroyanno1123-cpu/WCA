import torch
import torch.nn.functional as F


def jsd_loss(logits_clean, logits_aug1, logits_aug2):
    """Jensen-Shannon Divergence consistency loss.
    AugMix (Hendrycks et al., ICLR 2020) と同方式。
    M のクランプは数値安定化のみが目的（値域の変更ではない）。
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1  = F.softmax(logits_aug1,  dim=1)
    p_aug2  = F.softmax(logits_aug2,  dim=1)

    M = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, min=1e-7, max=1.0)
    log_M = torch.log(M)

    kl_clean = F.kl_div(log_M, p_clean, reduction='batchmean')
    kl_aug1  = F.kl_div(log_M, p_aug1,  reduction='batchmean')
    kl_aug2  = F.kl_div(log_M, p_aug2,  reduction='batchmean')

    return (kl_clean + kl_aug1 + kl_aug2) / 3.0
