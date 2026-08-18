"""train_variants.py — 基线训练变体（v4.1 §6.4 B7/B8/B9）
  B7 SelectiveNet (1901.09192): 选择性头 + 辅助任务 + 覆盖率惩罚 λ·max(0, cov_target−R̂)²
  B8 Deep Gamblers  (1907.00208): 拒绝类 + 熵正则（拒绝代价 α）
  B9 shift-aware    (2405.05160): 域偏移加权 BCE（腐蚀类型条件权重, 低 SNR 降权同 CFAL）
与 train.py 共用 PairDataset/DataLoader/cosine_lr; 训练循环返回同一 (model, res) 契约。
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from .model import SONTRA_A
from .train import PairDataset, cosine_lr


class SelectiveNetLoss(torch.nn.Module):
    """B7: L = α·L_task(选择样本) + L_aux(全部样本) + λ·max(0, C−R̂)²
    C = 目标覆盖率; R̂ = 选择比例; 选择 = sel_head 分数 top-k。"""

    def __init__(self, cov_target: float = 0.7, lam_cov: float = 1.0):
        super().__init__()
        self.cov_target = cov_target
        self.lam_cov = lam_cov

    def forward(self, event_logits, y, sel_logits, mask):
        probs = torch.sigmoid(event_logits)
        sel = torch.sigmoid(sel_logits)                       # [B]
        k = max(int(self.cov_target * len(sel)), 1)
        keep = torch.topk(sel, k, dim=0).indices              # 选 top-k 样本
        # L_task: 被选样本的事件 BCE（unknown 掩码）
        loss_task = F.binary_cross_entropy_with_logits(
            event_logits[keep], y[keep], reduction="none") * mask[keep]
        loss_task = loss_task.sum() / max(mask[keep].sum(), 1.0)
        # L_aux: 全部样本（辅助任务, 权重 α=0.5）
        loss_aux = F.binary_cross_entropy_with_logits(event_logits, y, reduction="none") * mask
        loss_aux = loss_aux.sum() / max(mask.sum(), 1.0)
        # 覆盖率惩罚
        r_hat = k / len(sel)
        loss_cov = self.lam_cov * torch.clamp(self.cov_target - r_hat, min=0.0) ** 2
        return 0.5 * loss_task + 0.5 * loss_aux + loss_cov


class DeepGamblersLoss(torch.nn.Module):
    """B8: 拒绝类 logit 参与 softmax; L = CE(类/拒绝) − β·H(概率)。"""

    def __init__(self, n_classes: int = C.N_CLASSES, beta: float = 0.1):
        super().__init__()
        self.beta = beta
        self.refuse_idx = n_classes                    # 附加拒绝类

    def forward(self, event_logits, y, refuse_logits, mask):
        logits = torch.cat([event_logits, refuse_logits.unsqueeze(1)], dim=1)
        # 目标: 多标签 → 取第一个正类（简化, 与 sample_clips 单主类口径一致）
        tgt = y[:, :C.N_CLASSES - 1].argmax(dim=1)
        tgt = torch.where(mask[:, :C.N_CLASSES - 1].any(1), tgt,
                          torch.full_like(tgt, self.refuse_idx))
        loss_ce = F.cross_entropy(logits, tgt)
        p = F.softmax(logits, dim=1)
        ent = -(p * p.clamp_min(1e-7).log()).sum(1).mean()
        return loss_ce - self.beta * ent


def train_variant(model: SONTRA_A, pairs_train, pairs_val, variant: str,
                  epochs: int = 30, batch: int = C.BATCH_SIZE, device="cpu",
                  out_dir=None, seed: int = C.SEED):
    """variant ∈ {B7, B8, B9}。B9 = CFAL 事件损失 + 域权重（重用 train.py 主体）。
    返回 (model, {"best_val": ...})。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    loader = DataLoader(PairDataset(pairs_train), batch_size=batch, shuffle=True)
    vloader = DataLoader(PairDataset(pairs_val), batch_size=batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
    warmup = int(C.WARMUP_FRAC * epochs * len(loader))
    total_steps = epochs * len(loader)

    if variant == "B7":
        sel_head = torch.nn.Linear(64, 1).to(device)     # 选择头（接 belief 特征）
        crit = SelectiveNetLoss()
        params = list(model.parameters()) + list(sel_head.parameters())
        opt = torch.optim.AdamW(params, lr=C.LR, weight_decay=1e-4)
    elif variant == "B8":
        refuse_head = torch.nn.Linear(64, 1).to(device)
        crit = DeepGamblersLoss()
        params = list(model.parameters()) + list(refuse_head.parameters())
        opt = torch.optim.AdamW(params, lr=C.LR, weight_decay=1e-4)
    elif variant == "B9":
        crit = None                                     # 用 CFAL 主体
    else:
        raise ValueError(variant)

    best = float("inf")
    for ep in range(epochs):
        model.train()
        for step, (xc, xcl, y, r, m, snr) in enumerate(loader):
            xc, xcl, y, r, m, snr = (t.to(device) for t in (xc, xcl, y, r, m, snr))
            gstep = ep * len(loader) + step
            for g in opt.param_groups:
                g["lr"] = cosine_lr(gstep, total_steps, warmup)
            opt.zero_grad()
            out = model(xc)
            if variant == "B7":
                sel = sel_head(out["features"])
                loss = crit(out["event_logits"], y, sel.squeeze(-1), m)
            elif variant == "B8":
                ref = refuse_head(out["features"])
                loss = crit(out["event_logits"], y, ref.squeeze(-1), m)
            else:  # B9: CFAL + 域权重（w_ev 与 train.py 同口径）
                from .losses import cfal_loss
                w_ev = 1.0 / (1.0 + torch.exp(-(snr + 10.0) / 6.0))
                loss, _ = cfal_loss(out["event_logits"], y, torch.ones_like(y),
                                    model.separation(xc)[0], xcl, out["snr_db"], r, m,
                                    event_weight=w_ev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # 验证: 事件 BCE（多标签, unknown 掩码）
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xc, xcl, y, r, m, snr in vloader:
                xc = xc.to(device)
                out = model(xc)
                bce = F.binary_cross_entropy_with_logits(out["event_logits"].cpu(), y,
                                                         reduction="none") * m
                vloss += float(bce.sum() / max(m.sum(), 1.0))
        vloss /= max(len(vloader), 1)
        print(f"[{variant}] ep{ep} val_BCE={vloss:.4f}")
        if vloss < best - 0.001:
            best = vloss
            if out_dir:
                torch.save(model.state_dict(), f"{out_dir}/{variant}_ep{ep}.pt")
    return model, {"best_val": best}
