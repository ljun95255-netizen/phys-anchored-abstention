"""
wsosim.py — WSO-Sim 腐蚀模拟器（v4.1 §5.5 的 PyTorch/NumPy 实现）
三类退化 + SNR×T 网格混合，外生标签 r = E_s/P_n（与 E0 同口径）:
  wind: 低频整形噪声 + 三正弦 AM（5/9/14 Hz, m=0.5）
  occlusion: 一阶低通 f_c∈{0.5,1,2,4}kHz + 衰减 {0,-6,-12}dB
  self-motion: 周期增益调制 1/2 Hz, 深度 0.3
确定性种子派生（同一 clip 同一种子 → 可复现）。事件带 [1k,4k] 内精确控制 SNR。
"""
import math
import numpy as np
import scipy.signal as sig

from . import config as C


def _wind(n: int, fs: int, seed: int):
    """风噪: 白噪声 → 一阶低通 fc=500Hz → 三正弦 AM。返回 (w, env)。"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    b, a = sig.butter(1, C.WIND_FC / (fs / 2), btype="low")
    w = sig.lfilter(b, a, w)
    t = np.arange(n) / fs
    env = 1.0 + C.WIND_AM_DEPTH * sum(
        np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi)) for f in C.WIND_AM_FREQS
    )
    return w * env, env


def _occlusion_filter(fc: float, fs: int):
    b, a = sig.butter(1, fc / (fs / 2), btype="low")
    return b, a


def _band(x, fs: int, f_lo: float, f_hi: float):
    b, a = sig.butter(4, [f_lo / (fs / 2), f_hi / (fs / 2)], btype="band")
    return sig.lfilter(b, a, x)


def corrupt(clean: np.ndarray, kind: str, snr_db: float, seed: int,
            fs: int = C.SAMPLE_RATE, band: tuple = C.EVENT_BAND):
    """对 clean 施加 kind 类腐蚀，腐蚀后事件带内 SNR = snr_db。
    返回 (x_corr, r_db, meta): r_db = 10log10(E_s_corr/P_n)（可听性口径, 构造即等于 snr_db）。
    事件完全不可听（带内能量≈0）时 r_db 钳到 −60dB（保留样本, A-Head 学"不可听"）。"""
    n = clean.shape[0]
    rng = np.random.default_rng(seed)
    f_lo, f_hi = band
    # 1) 腐蚀后的事件分量（不含噪声）
    if kind == "wind":
        event_corr = clean
    elif kind == "occlusion":
        fc = C.OCCL_FREQS[int(seed % len(C.OCCL_FREQS))]
        attn_db = C.OCCL_ATTN_DB[(seed // 7) % len(C.OCCL_ATTN_DB)]
        b, a = sig.butter(1, fc / (fs / 2), btype="low")
        event_corr = sig.lfilter(b, a, clean) * 10 ** (attn_db / 20)
    elif kind == "self_motion":
        t = np.arange(n) / fs
        env = 1.0 + C.SELFMOTION_DEPTH * sum(
            np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
            for f in C.SELFMOTION_FREQS
        )
        event_corr = clean * env
    else:
        raise ValueError(kind)

    eb = _band(event_corr, fs, f_lo, f_hi)
    e_corr = float(np.sum(eb ** 2)) + 1e-15
    if e_corr < 1e-12 * n:                       # 事件在带内完全不可听
        return event_corr.astype(np.float32), -60.0, {"kind": kind, "snr_db": snr_db, "inaudible": True}

    # 2) 噪声注入, 增益相对腐蚀后事件能量标定
    w, _ = _wind(n, fs, seed)
    wb = _band(w, fs, f_lo, f_hi)
    pn = float(np.sum(wb ** 2)) + 1e-15
    g = math.sqrt(e_corr / (pn * 10 ** (snr_db / 10)))
    x = event_corr + g * w
    r_db = 10 * math.log10(e_corr / (g * g * pn + 1e-15))   # 构造即 ≈ snr_db
    return x.astype(np.float32), r_db, {"kind": kind, "snr_db": snr_db}


def corrupt_grid(clean: np.ndarray, kinds, snr_db, seed_base: int = C.SEED,
                 fs: int = C.SAMPLE_RATE):
    """同一 clip 在全部 (kind × snr) 网格上的腐蚀版本（CF-Sampler 用）。"""
    out = {}
    for k in kinds:
        for s in snr_db:
            x, r_db, meta = corrupt(clean, k, s, seed_base + hash((k, s)) % 100000)
            out[(k, s)] = (x, r_db)
    return out
