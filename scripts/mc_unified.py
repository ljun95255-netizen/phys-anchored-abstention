"""mc_unified.py — MC 校验, 统一约定重跑（2026-08-07, 回应审稿人 SNR 单位约定批评）
约定: 论文 r = 带内功率比（每样本, dB 直接入式）; P_e(r, BT) = Φ(−r√(BT)/(1+√(1+2r))).
MC 模型与 e0_reference/mc_pe 相同（n=2BT 样本, 每样本噪声方差 1/n, 等误阈值 1+r/2）,
唯一改动: r = 10^(dB/10) 直接注入（不再 ×BT）。
"""
import math

import numpy as np


def pe_theory(r, BT):
    from scipy.special import erf
    x = -r * math.sqrt(BT) / (1.0 + math.sqrt(1.0 + 2.0 * r))
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def mc_pe(snr_db, BT, trials=10000, seed=20260803):
    rng = np.random.default_rng(seed)
    r = 10.0 ** (snr_db / 10.0)          # 统一约定: 直接注入
    n = max(2, int(round(2.0 * BT)))
    var0 = 1.0 / n
    a = math.sqrt(r / n)
    thr = 1.0 + r / 2.0
    z0 = (rng.normal(0.0, math.sqrt(var0), (trials, n)) ** 2).sum(1)
    z1 = ((a + rng.normal(0.0, math.sqrt(var0), (trials, n))) ** 2).sum(1)
    return 0.5 * (np.mean(z0 > thr) + np.mean(z1 < thr))


def r_min(alpha, BT):
    lo, hi = 1e-9, 1e6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pe_theory(mid, BT) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    BT = 256.0
    grid = [-10.0, -8.0, -7.6, -6.0, -4.0]   # r_min(0.1)≈−7.6dB @BT=256
    print(f"{'SNR(dB)':>8} {'Pe_theory':>10} {'Pe_mc':>10} {'|err|':>8}")
    rows = []
    for s in grid:
        th = pe_theory(10 ** (s / 10), BT)
        emp = mc_pe(s, BT, trials=10000)
        rows.append((s, th, emp, abs(th - emp)))
        print(f"{s:>8.1f} {th:>10.4f} {emp:>10.4f} {abs(th-emp):>8.4f}")
    rmin = r_min(0.1, BT)
    rm_emp = mc_pe(10 * math.log10(rmin), BT, trials=10000)
    print(f"r_min(α=0.1): theory {10*math.log10(rmin):.2f} dB | MC P_e @ r_min: {rm_emp:.4f}")
    import json
    out = {"bt": BT, "grid": [{"snr_db": s, "theory": round(th, 5), "mc": round(e, 5),
                               "err": round(a, 5)} for s, th, e, a in rows],
           "r_min_db": round(10 * math.log10(rmin), 2), "mc_pe_at_rmin": round(rm_emp, 5)}
    with open("outputs/mc_unified.json", "w") as f:
        json.dump(out, f, indent=1)
    print("DONE → outputs/mc_unified.json")


if __name__ == "__main__":
    main()
