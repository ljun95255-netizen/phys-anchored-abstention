"""
mc_crosscheck.py — AOF 前沿公式的 Monte Carlo 交叉校验（论文附录证据生成器）
复现 2026-08-04 的验证：双方差高斯近似 vs 实测（BT=256/1024, SNR 扫描）
输出: 控制台表 + outputs/e0_mc_crosscheck.csv
运行: python mc_crosscheck.py
"""
import csv, os
from energy_detector import (mc_pe, pe_theory_ed, pe_naive_deflection,
                             snr_min_ed, snr_min_ed_asymptotic, snr_min_mf)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

def main():
    rows = []
    print("=== MC vs 双方差高斯近似 vs naive 偏转 (trials=10000) ===")
    print(f"{'BT':>6} {'SNR(dB)':>8} {'Pe_emp':>9} {'Pe_theory':>10} {'Pe_naive':>9}")
    for BT in [256.0, 1024.0]:
        grid = [-36, -34, -32, -30, -28] if BT == 256 else [-46, -44, -42, -40, -38]
        for s in grid:
            emp = mc_pe(s, BT, trials=10000)
            th = pe_theory_ed(s, BT)
            nv = pe_naive_deflection(s, BT)
            print(f"{BT:>6.0f} {s:>8} {emp:>9.4f} {th:>10.4f} {nv:>9.4f}")
            rows.append({"BT": BT, "snr_db": s, "pe_empirical": round(emp, 5),
                         "pe_theory_gauss2var": round(th, 5),
                         "pe_naive_deflection": round(nv, 5)})

    print("\n=== SNR_min(alpha=0.1): ED exact / ED asympt / MF ===")
    for BT in [256.0, 1024.0]:
        rows.append({"BT": BT, "snr_db": "SNR_min@0.1",
                     "pe_empirical": round(snr_min_ed(0.1, BT), 2),
                     "pe_theory_gauss2var": round(snr_min_ed_asymptotic(0.1, BT), 2),
                     "pe_naive_deflection": round(snr_min_mf(0.1, BT), 2)})
        print(f"BT={BT:>6.0f}: ED_exact={snr_min_ed(0.1, BT):6.2f}  "
              f"ED_asympt={snr_min_ed_asymptotic(0.1, BT):6.2f}  MF={snr_min_mf(0.1, BT):6.2f}")

    with open(os.path.join(OUT, "e0_mc_crosscheck.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[written] {os.path.join(OUT, 'e0_mc_crosscheck.csv')}")

if __name__ == "__main__":
    main()
