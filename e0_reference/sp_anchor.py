"""
sp_anchor.py — B11a 经典 SP 锚（谱减 SNR 估计，Boll 谱系）
链路: 参考窗 Welch PSD -> 带内噪声功率 P̂_n -> 谱减(半波整流+地板) -> r̂ = P̂_s/P̂_n
单位与能量检测器一致: r = E_s/P_n（窗内离散能量比），阈值可直接用理论 r_min(α,n)。

detect(x_test, w_ref, fs, f_lo, f_hi) -> r_hat (线性比), 或 dB
"""
import numpy as np
import scipy.signal as sig

def band_noise_energy(w_ref, fs, f_lo, f_hi, n_eff, nperseg=512):
    """参考窗 Welch PSD -> 带内噪声能量估计 P̂_n（离散能量单位: n_eff·带内功率）。"""
    f, P = sig.welch(w_ref, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    m = (f >= f_lo) & (f <= f_hi)
    p_band = float(np.trapezoid(P[m], f[m]))      # 带内功率 (PSD 积分)
    return max(p_band * n_eff, 1e-15)             # 离散能量: sum(x²) ≈ n·power

def spectral_subtract_energy(x_band, pn_energy):
    """谱减（能量域）: P̂_s = max(sum(x²) - P̂_n, 0)。"""
    return max(float(np.sum(x_band ** 2)) - pn_energy, 0.0)

def detect(x_test, w_ref, fs, f_lo, f_hi, n_eff):
    """返回 r̂ = P̂_s/P̂_n（谱减 SNR 的能量比形式）。"""
    pn = band_noise_energy(w_ref, fs, f_lo, f_hi, n_eff)
    # 测试窗带通（与 to_band 同路径，保证功率口径一致）
    b, a = sig.butter(4, [f_lo / (fs / 2), f_hi / (fs / 2)], btype="band")
    xb = sig.resample_poly(sig.lfilter(b, a, x_test), 3, 8)
    ps = spectral_subtract_energy(xb, pn)
    return ps / pn if pn > 0 else 0.0
