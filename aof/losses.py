"""
losses.py — CFAL 损失（v4.1 §5.7）: L = L_event + λ1·L_sep + λ2·L_aud + λ3·L_reg
  L_event: 多标签 BCE（unknown 类掩码置 0）
  L_sep:   分离层增强图 vs 干净 log-mel 的 MSE（反事实监督, CF-Sampler 提供对）
  L_aud:   掩码 Huber(δ=2dB) 逐类 SNR̂ vs 外生 r_dB
  L_reg:   SNR̂ 输出 L2 正则（1e-4）
"""
import torch
import torch.nn.functional as F

from . import config as C


def cfal_loss(event_logits, event_targets, masks,
              enhanced, clean_log_mel,
              snr_db, snr_true_db, snr_masks,
              event_weight=None,
              lam_sep=C.LAM_SEP, lam_aud=C.LAM_AUD, lam_reg=C.LAM_REG):
    # L_event: BCE with logits, unknown 类掩码（masks: [B,10] 逐类标签掩码）
    loss_event = F.binary_cross_entropy_with_logits(event_logits, event_targets, reduction="none")
    m = masks * torch.ones_like(loss_event)
    m[:, C.UNKNOWN_CLASS] = 0.0
    if event_weight is not None:                  # 按腐蚀 SNR 加权（低 SNR 事件损失降权）
        m = m * event_weight.unsqueeze(1)
    loss_event = (loss_event * m).sum() / max(m.sum(), 1.0)

    # L_sep: 增强图(24ch) vs 干净 log-mel 广播到 24 通道
    loss_sep = F.mse_loss(enhanced, clean_log_mel.repeat(1, enhanced.shape[1], 1, 1))

    # L_aud: 掩码 Huber δ=2dB
    delta = torch.tensor(C.HUBER_DELTA_DB, device=snr_db.device)
    err = (snr_db - snr_true_db).abs()
    huber = torch.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    loss_aud = (huber * snr_masks).sum() / max(snr_masks.sum(), 1.0)

    # L_reg
    loss_reg = snr_db.pow(2).mean()

    total = loss_event + lam_sep * loss_sep + lam_aud * loss_aud + lam_reg * loss_reg
    return total, {"event": loss_event, "sep": loss_sep,
                   "aud": loss_aud, "reg": loss_reg}
