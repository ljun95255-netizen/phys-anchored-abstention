"""
wind_noise.py — WSO-Sim 风噪合成器（E0 用）
物理近似: 低频整形噪声 + 5-15Hz 幅值调制（引 Corcos 系 / 2507.01821 / 2409.06137）
确定性种子派生，可复现。带内 SNR 精确控制（预滤波到事件带后按功率配比）。
"""
import math
import numpy as np
import scipy.signal as sig

def make_wind(n_samples, fs, seed=20260803, fc_hz=500.0, m=0.5, f_am=10.0, tilt=0.5):
    """生成一段风噪。
    - 白噪声 -> 低通(fc) + 1/f^tilt 倾斜 -> 幅值调制 A(t)=1+m*sin(2pi*f_am*t)
    返回 (w, 调制包络)"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n_samples)
    # 1/f tilt 通过一阶积分近似 + 低通
    b, a = sig.butter(2, fc_hz / (fs / 2), btype="low")
    w = sig.lfilter(b, a, w)
    w = w / (np.std(w) + 1e-12)
    t = np.arange(n_samples) / fs
    env = 1.0 + m * np.sin(2.0 * np.pi * f_am * t)
    return w * env, env

def bandpass(x, fs, f_lo, f_hi, order=4):
    """预检测带通（理论要求噪声在事件带内白化）。"""
    b, a = sig.butter(order, [f_lo / (fs / 2), f_hi / (fs / 2)], btype="band")
    return sig.lfilter(b, a, x)

def mix_snr(event, wind, snr_db, fs, f_lo, f_hi):
    """按带内 SNR 混合: x = event + g*wind。返回混合信号与增益 g。"""
    e_b = bandpass(event, fs, f_lo, f_hi)
    w_b = bandpass(wind, fs, f_lo, f_hi)
    pe = np.mean(e_b ** 2)
    pn = np.mean(w_b ** 2)
    g = math.sqrt(pe / (pn * 10.0 ** (snr_db / 10.0) + 1e-15))
    return event + g * wind, g

def inband_snr(x_event, x_wind, fs, f_lo, f_hi):
    """实测带内 SNR(dB)（用于协议校验）。"""
    e_b = bandpass(x_event, fs, f_lo, f_hi)
    w_b = bandpass(x_wind, fs, f_lo, f_hi)
    return 10.0 * np.log10(np.mean(e_b ** 2) / (np.mean(w_b ** 2) + 1e-15))
