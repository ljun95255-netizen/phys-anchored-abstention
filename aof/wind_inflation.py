"""wind_inflation.py — v3 §11.6 风噪膨胀闭合
AM 风噪（WSO-Sim wind）使检测统计量方差膨胀 σ0 ×3.7-4.2（方差 ×14-18, E0 实证）。
闭合实验: 把膨胀注入 σ̂² 模型 → 公式预测 Pe vs 实测 Pe 逐单元对照。
公式: 有效样本数 n_eff = n / (σ̃²), σ̃² = 膨胀因子（AM 包络方差 1 + m²/2 的贡献 + 谱着色）。
"""
import math

import numpy as np

from . import config as C
from .af_rule import pe_theory_rn, n_eff


def am_variance_inflation(m: float = C.WIND_AM_DEPTH, f_am_list=C.WIND_AM_FREQS) -> float:
    """幅值调制 A(t) = 1 + m·Σ sin(2πf_i t + φ_i) 的方差膨胀因子。
    Var(A) = m²·(n_tones)/2（相位均匀独立）; 统计量方差 ∝ E[A²] = 1 + Var(A)。"""
    n_tones = len(f_am_list)
    var_a = m * m * n_tones / 2.0
    return 1.0 + var_a


def spectrum_inflation(x_wind: np.ndarray, fs: int, band: tuple) -> float:
    """谱着色膨胀: 事件带内噪声非白 → 有效样本数折减 = 带内 PSD 峰均比。"""
    from scipy.signal import welch
    from .wsosim import _band
    f_lo, f_hi = band
    f, P = welch(x_wind, fs=fs, nperseg=512, noverlap=256)
    m_b = (f >= f_lo) & (f <= f_hi)
    if m_b.sum() < 2:
        return 1.0
    psd = P[m_b]
    return float(psd.max() / (psd.mean() + 1e-15))


def inflation_factor(wind: np.ndarray, fs: int = C.SAMPLE_RATE,
                     band: tuple = C.EVENT_BAND) -> dict:
    """总膨胀 = AM 包络 × 谱着色（E0 实测 σ0 ×3.7-4.2 即此组合）。"""
    am = am_variance_inflation()
    sp = spectrum_inflation(wind, fs, band)
    return {"am": am, "spectral": sp, "total": am * sp}


def pe_predicted_with_inflation(snr_db: float, band_hz: float, t_sec: float,
                                infl: float) -> float:
    """注入膨胀后的公式预测: n_eff = n/infl, r 不变（带内能量比口径）。"""
    n = n_eff(band_hz, t_sec)
    n_eff_v = max(n / infl, 2.0)
    r = 10.0 ** (snr_db / 10.0)
    return pe_theory_rn(r, n_eff_v)


def pe_empirical_on_wind(clean: np.ndarray, snr_db: float, seed: int,
                         band: tuple = C.EVENT_BAND, fs: int = C.SAMPLE_RATE,
                         trials: int = 200) -> dict:
    """实测: 在 AM 风噪上按目标 SNR 混合, 能量检测统计量 → 经验 Pe（等先验）。
    返回 {pe_emp, pe_theory_no_infl, pe_theory_infl, infl}。"""
    from .wsosim import _wind, _band
    from e0_reference.energy_detector import threshold_radiometer
    rng = np.random.default_rng(seed)
    f_lo, f_hi = band
    w, _ = _wind(clean.shape[0], fs, seed)
    eb = _band(clean, fs, f_lo, f_hi)
    wb = _band(w, fs, f_lo, f_hi)
    pe = float(np.sum(eb ** 2)) + 1e-15
    pn = float(np.sum(wb ** 2)) + 1e-15
    g = math.sqrt(pe / (pn * 10.0 ** (snr_db / 10.0)))
    n = clean.shape[0]
    bt = (f_hi - f_lo) * (n / fs)
    thr = threshold_radiometer(0.5, n)          # z 以 σ² 为单位（等先验 Pe）
    n_fp = n_fn = 0
    for _ in range(trials):
        w2, _ = _wind(n, fs, seed + _)
        x = eb + g * _band(w2, fs, f_lo, f_hi)
        z = float(np.sum(x ** 2)) / (g * g * pn / n + 1e-15)
        if z > thr:
            n_fp += 1
        x0 = g * _band(w2, fs, f_lo, f_hi)
        z0 = float(np.sum(x0 ** 2)) / (g * g * pn / n + 1e-15)
        if z0 <= thr:
            n_fn += 1
    pe_emp = 0.5 * (n_fp + n_fn) / trials
    infl = inflation_factor(w, fs, band)
    return {
        "pe_emp": pe_emp,
        "pe_theory_no_infl": pe_predicted_with_inflation(snr_db, f_hi - f_lo, n / fs, 1.0),
        "pe_theory_infl": pe_predicted_with_inflation(snr_db, f_hi - f_lo, n / fs,
                                                      infl["total"]),
        "infl": infl,
        "snr_db": snr_db,
    }
