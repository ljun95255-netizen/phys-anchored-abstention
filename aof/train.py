"""
train.py — CFAL 训练循环（v4.1 §6.2）: AdamW lr=3e-4, warmup 5%, cosine, ≤50 epochs 早停
数据: CF-Sampler 产生的 (corrupted log-mel, clean log-mel, labels, r_dB) 对
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .losses import cfal_loss
from .model import SONTRA_A


class PairDataset(Dataset):
    """CF-Sampler 产物: (x_corr[B,1,24,128], x_clean[B,1,24,128], y[B,10], r_dB[B,10], mask[B,10])"""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        xc, xcl, y, r, m, snr = self.pairs[i]
        return (torch.from_numpy(xc), torch.from_numpy(xcl),
                torch.from_numpy(y).float(), torch.from_numpy(r).float(),
                torch.from_numpy(m).float(), torch.tensor(snr, dtype=torch.float32))


def cosine_lr(step, total_steps, warmup_steps, base_lr=C.LR):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def train_epoch(model, loader, opt, device, epoch, total_epochs, warmup_steps, log_every=50):
    """R3 原始损失配方（实证: 分离层特征质量决定双头学习）。
    事件损失按腐蚀 SNR 加权: w = 1/(1+exp(−(snr+10)/6)) —— 低 SNR 对不可分类,
    降权防"从噪声里预测标签"（R8）。"""
    model.train()
    total_steps = len(loader) * total_epochs
    losses, aud_mae = [], []
    t0 = time.time()
    for step, (xc, xcl, y, r, m, snr) in enumerate(loader):
        xc, xcl, y, r, m, snr = (t.to(device) for t in (xc, xcl, y, r, m, snr))
        global_step = epoch * len(loader) + step
        for g in opt.param_groups:
            g["lr"] = cosine_lr(global_step, total_steps, warmup_steps)
        opt.zero_grad()
        enhanced, _ = model.separation(xc)
        out = model(xc)
        w_ev = 1.0 / (1.0 + torch.exp(-(snr + 10.0) / 6.0))    # [B]
        # R10: aud 梯度 10% 穿透骨干（混合: 骨干保留 SNR 轴, 但事件头竞争压力大减）
        feats, enc = out["features"], out["enc_pooled"]
        snr_full = model.ahead(feats, enc)
        snr_det = model.ahead(feats.detach(), enc.detach())
        snr_hyb = snr_det + 0.1 * (snr_full - snr_det)
        total, comp = cfal_loss(out["event_logits"], y, torch.ones_like(y),
                                enhanced, xcl, snr_hyb, r, m, event_weight=w_ev)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(total.detach()))
        with torch.no_grad():
            err = (out["snr_db"] - r).abs() * m
            aud_mae.append(float(err.sum() / max(m.sum(), 1)))
        if step % log_every == 0:
            print(f"  ep{epoch} step{step}/{len(loader)} loss={total:.3f} "
                  f"snrMAE={aud_mae[-1]:.2f}dB lr={g['lr']:.2e} {time.time()-t0:.0f}s "
                  f"[ev={float(comp['event'].detach()):.2f} sep={float(comp['sep'].detach()):.0f} "
                  f"aud={float(comp['aud'].detach()):.1f}]")
    return float(np.mean(losses)), float(np.mean(aud_mae))


def train(model, pairs_train, pairs_val, epochs=C.MAX_EPOCHS, batch=C.BATCH_SIZE,
          device="mps", out_dir=None, seed=C.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    ds = PairDataset(pairs_train)
    dv = PairDataset(pairs_val)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    vloader = DataLoader(dv, batch_size=batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
    warmup_steps = int(WARMUP := C.WARMUP_FRAC * epochs * len(loader))
    best = float("inf")
    patience = 0
    for ep in range(epochs):
        tr_loss, tr_mae = train_epoch(model, loader, opt, device, ep, epochs, warmup_steps)
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        nv = 0
        with torch.no_grad():
            for xc, xcl, y, r, m, snr in vloader:
                xc, xcl, y, r, m, snr = (t.to(device) for t in (xc, xcl, y, r, m, snr))
                out = model(xc)
                _, comp = cfal_loss(out["event_logits"], y, torch.ones_like(y),
                                    model.separation(xc)[0], xcl, out["snr_db"], r, m)
                val_loss += float(comp["event"].detach()) + C.LAM_AUD * float(comp["aud"].detach())
                err = (out["snr_db"] - r).abs() * m
                val_mae += float(err.sum() / max(m.sum(), 1))
                nv += 1
        val_loss /= max(nv, 1)
        val_mae /= max(nv, 1)
        print(f"[val] ep{ep} loss={val_loss:.4f} snrMAE={val_mae:.2f}dB")
        if val_mae < best - 0.01:
            best = val_mae
            patience = 0
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(out_dir, f"sontra_a_ep{ep}.pt"))
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                print(f"early stop @ ep{ep}")
                break
    return model, {"best_snr_mae": best}
