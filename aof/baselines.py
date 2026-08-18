"""
baselines.py — 基线矩阵 v5（§6.4）: B0/B1/B2+B4/B3/B7/B8/B9/B11/B11a/Rand/Oracle
统一接口: System.predict(win) → (event_probs[B,10] or None, snr_db[B,10] or None, meta)
  None = 免训练检测器（只有 SNR 无类别概率）→ 走 detector-tier 评估。
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .af_rule import pe_theory_rn, r_min_theory, n_eff
from .model import SONTRA_A
from .wsosim import _band


# ---------- 免训练检测器层（detector-tier）----------

def estimate_noise_floor(x, w_ref, fs, event_band, low_band=(300.0, 800.0)):
    """经典 SP 锚噪声底估计（B11a 设计）: 测试窗低带[300,800]Hz 功率（风噪主导）
    × 参考窗谱形比 shape = P_event_band/P_low_band（wind-only 谱形）→ 事件带噪声功率。
    优点: 噪声级取自测试窗自身（尺度无关）; 参考窗只提供谱形。"""
    lo, hi = event_band
    llo, lhi = low_band
    p_low = float(np.sum(_band(x, fs, llo, lhi) ** 2)) + 1e-15
    p_ref_e = float(np.sum(_band(w_ref, fs, lo, hi) ** 2)) + 1e-15
    p_ref_l = float(np.sum(_band(w_ref, fs, llo, lhi) ** 2)) + 1e-15
    shape = p_ref_e / p_ref_l
    return max(p_low * shape, 1e-15)


class EnergyDetectorB0:
    """B0: 纯能量检测器（低带噪声级 + 谱形比 → r̂, 与 AF-Rule 同口径）。"""

    def __init__(self, band=C.EVENT_BAND, fs=C.SAMPLE_RATE):
        self.band, self.fs = band, fs

    def snr_db(self, x, w_ref):
        """r_dB = 10log10((Σx² − P̂_n)/P̂_n)（事件带能量比）。"""
        f_lo, f_hi = self.band
        pn = estimate_noise_floor(x, w_ref, self.fs, self.band)
        xb = _band(x, self.fs, f_lo, f_hi)
        r = max(float(np.sum(xb ** 2)) / pn - 1.0, 1e-9)
        return 10 * math.log10(r)


class SPAnchorB11:
    """B11: 经典 SP 锚（Boll 谱系）: 同 B0 噪声底估计 + Welch PSD 谱形。"""

    def __init__(self, band=C.EVENT_BAND, fs=C.SAMPLE_RATE, nperseg=512):
        self.band, self.fs, self.nperseg = band, fs, nperseg

    def _noise_energy(self, x, w_ref):
        from scipy.signal import welch
        f_lo, f_hi = self.band
        # 测试窗低带 Welch 功率（风噪主导）
        llo, lhi = 300.0, 800.0
        f_l, P_l = welch(x, fs=self.fs, nperseg=self.nperseg,
                         noverlap=self.nperseg // 2, window="hann")
        m_l = (f_l >= llo) & (f_l <= lhi)
        p_low = float(np.trapezoid(P_l[m_l], f_l[m_l])) + 1e-15
        # 参考窗谱形比（Welch）
        f_r, P_r = welch(w_ref, fs=self.fs, nperseg=self.nperseg,
                         noverlap=self.nperseg // 2, window="hann")
        m_e = (f_r >= f_lo) & (f_r <= f_hi)
        m_lr = (f_r >= llo) & (f_r <= lhi)
        shape = (float(np.trapezoid(P_r[m_e], f_r[m_e])) + 1e-15) / \
                (float(np.trapezoid(P_r[m_lr], f_r[m_lr])) + 1e-15)
        return max(p_low * shape * x.shape[0], 1e-15)

    def snr_db(self, x, w_ref):
        """r_dB = 10log10((Σx² − P̂_n)/P̂_n)。"""
        f_lo, f_hi = self.band
        pn = self._noise_energy(x, w_ref)
        xb = _band(x, self.fs, f_lo, f_hi)
        r = max(float(np.sum(xb ** 2)) / pn - 1.0, 1e-9)
        return 10 * math.log10(r)


# ---------- 学习式系统（system-tier）----------

class ThresholdSystem:
    """B1/B2+B4/B3 共用: 骨干 + 分数函数 + 阈值（按覆盖率或理论 r_min）。"""

    def __init__(self, model, score_fn, name):
        self.model = model
        self.score_fn = score_fn
        self.name = name

    @torch.no_grad()
    def predict(self, log_mel):
        out = self.model(log_mel)
        return out, self.score_fn(out)


def score_ts(out):        # B1: 温度缩放置信度（T=1.0 恒等; 校准在评估时拟合）
    return out["event_probs"].max(dim=1).values

def score_entropy(out):   # B2: 负熵
    p = out["event_probs"].clamp(1e-7, 1)
    return -(p * p.log()).sum(dim=1)

def score_energy(out):    # B4: Energy score (2010.03759)
    return torch.logsumexp(out["event_logits"], dim=1)

def score_dropout(model, log_mel, T=10):   # B3: MC-dropout
    model.train()
    probs = []
    with torch.no_grad():
        for _ in range(T):
            out = model(log_mel)
            probs.append(out["event_probs"])
    p = torch.stack(probs).mean(0)
    return p, p.max(dim=1).values


# ---------- 随机弃权 / Oracle ----------

def random_abstain(n, coverage_frac, rng=np.random.default_rng(C.SEED)):
    """Rand: 随机弃权@匹配覆盖率（gap 量纲校准）。"""
    keep = rng.random(n) < coverage_frac
    return keep


def oracle_snr(r_db_true, r_min):
    """Oracle: 真实 SNR 阈值（=ED 前沿）。"""
    return (10 ** (r_db_true / 10)) >= r_min


def make_system(name, model: SONTRA_A, device="cpu"):
    model = model.to(device).eval()
    if name in ("B1",):
        return ThresholdSystem(model, score_ts, name)
    if name in ("B2", "B4"):
        return ThresholdSystem(model, score_entropy if name == "B2" else score_energy, name)
    raise ValueError(name)


# ---------- 训练变体（B7/B8/B9 在 train.py 中以 loss 变体实现）----------
# B7 SelectiveNet / B8 Deep Gamblers / B9 shift-aware: 训练目标不同, 评估接口同上
