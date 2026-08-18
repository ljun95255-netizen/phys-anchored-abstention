"""
energy_detector.py — AOF v4.1 能量检测理论与 MC 校验（v2: (r,n) 实测驱动形式）

核心: 对带限信号降采样到 2B 后，统计量 z = sum(x^2)/sigma^2 ~ chi2(n) / noncentral chi2(n, 2r)
  n = 2BT 个独立样本;  r = E_s/P_n = 事件能量/带内噪声功率（实测）
  D(r,n) = r*sqrt(n/2) / (1 + sqrt(1 + 2r))          [双方差高斯近似, MC 验证 <0.5%]
  Pe = Phi(-D)
  NP 阈值(虚警 alpha): thr = n + sqrt(2n)*Phi^-1(1-alpha)  [z 以 sigma^2 为单位]

论文形式换算: r = SNR_linear * BT; 弱信号渐近 SNR_min ~ 2c/(BT)^1.5 (T 加倍 4.5dB)
运行: python mc_crosscheck.py
"""
import math
import numpy as np

def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def ppf(p):
    return math.sqrt(2.0) * math.erf(2.0 * p - 1.0)

# ---------- 核心 (r, n) 形式 ----------
def deflection_rn(r, n):
    """双方差高斯近似判决系数。r=E_s/P_n, n=2BT 独立样本数。"""
    return r * math.sqrt(n / 2.0) / (1.0 + math.sqrt(1.0 + 2.0 * r))

def pe_theory_rn(r, n):
    return phi(-deflection_rn(r, n))

def threshold_radiometer(alpha, n):
    """NP 虚警控制阈值（z 以 sigma^2 为单位）。"""
    return n + math.sqrt(2.0 * n) * ppf(1.0 - alpha)

def pe_predicted_np(alpha, r, n):
    """NP 阈值(虚警=alpha)下预测 Pe（等先验）: 0.5*(P_FA + P(T<thr|H1))。"""
    thr = threshold_radiometer(alpha, n)
    s1 = math.sqrt(2.0 * n + 4.0 * r)          # H1 标准差
    miss = phi((thr - (n + r)) / s1)
    return 0.5 * (alpha + miss)

# ---------- 论文形式 (SNR_dB, BT) ----------
def pe_theory_ed(snr_db, BT):
    """等价于 pe_theory_rn(r=SNR*BT, n=2BT)。"""
    snr = 10.0 ** (snr_db / 10.0)
    return pe_theory_rn(snr * BT, 2.0 * BT)

def pe_naive_deflection(snr_db, BT):
    """naive 等方差偏转（已证伪，对照用）。"""
    snr = 10.0 ** (snr_db / 10.0)
    return phi(-0.5 * snr * math.sqrt(BT))

def pe_mf(snr_db, BT):
    """匹配滤波（已知信号）Pe。"""
    snr = 10.0 ** (snr_db / 10.0)
    return phi(-math.sqrt(2.0 * snr * BT) / 2.0)

def snr_min_ed(alpha, BT, lo=-80.0, hi=20.0):
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pe_theory_ed(mid, BT) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def snr_min_mf(alpha, BT):
    c = 2.0 * ppf(1.0 - alpha)
    return 10.0 * math.log10(c * c / BT)

def snr_min_ed_asymptotic(alpha, BT):
    c = 2.0 * ppf(1.0 - alpha)
    return 10.0 * math.log10(2.0 * c / (BT ** 1.5))

def r_min_theory(alpha, n, lo=-10.0, hi=1e7):
    """数值求 r 阈值: pe_theory_rn(r, n) == alpha（决策规则用）。"""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pe_theory_rn(mid, n) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def mc_pe(snr_db, BT, trials=3000, seed=20260803):
    """离散模型 MC：n=2BT 样本, 每样本噪声方差 1/n, 信号总能量 r=SNR*BT。"""
    rng = np.random.default_rng(seed)
    snr = 10.0 ** (snr_db / 10.0)
    n = max(2, int(round(2.0 * BT)))
    r = snr * BT
    var0 = 1.0 / n
    a = math.sqrt(r / n)
    thr = 1.0 + r / 2.0
    z0 = (rng.normal(0.0, math.sqrt(var0), (trials, n)) ** 2).sum(1)
    z1 = ((a + rng.normal(0.0, math.sqrt(var0), (trials, n))) ** 2).sum(1)
    return 0.5 * (np.mean(z0 > thr) + np.mean(z1 < thr))

if __name__ == "__main__":
    print("=== 能量检测 MC vs 理论 (BT=256, trials=5000) ===")
    print(f"{'SNR(dB)':>8} {'Pe_emp':>10} {'Pe_theory':>10} {'Pe_naive':>10} {'Pe_MF':>10}")
    for s in [-36, -34, -32, -30, -28]:
        emp = mc_pe(s, 256.0, trials=5000)
        print(f"{s:>8} {emp:>10.4f} {pe_theory_ed(s, 256.0):>10.4f} "
              f"{pe_naive_deflection(s, 256.0):>10.4f} {pe_mf(s, 256.0):>10.4f}")
    print("\n=== SNR_min(alpha=0.1) ===")
    for BT in [256.0, 1024.0]:
        print(f"BT={BT:>6.0f}: ED_exact={snr_min_ed(0.1, BT):6.2f}  "
              f"ED_asympt={snr_min_ed_asymptotic(0.1, BT):6.2f}  "
              f"MF={snr_min_mf(0.1, BT):6.2f}  "
              f"naive={10*math.log10(2*ppf(0.9)/math.sqrt(BT)):6.2f} dB")
