"""paper_stats.py — 期刊版统计检验（2026-08-07, 全部从现有聚合 JSON 计算, 不重跑实验）
1. Wilson 95% CI for risk（每系统行; FSD50K 冻结行 n_decide = cov × 6765）
2. 单侧二项检验 risk ≤ α（H0: p ≥ α）
3. 双比例 z 检验: B12 vs B12a（阈值代价形式化; 同语义可比）
4. per-kind Wilson CI（SC-10/US8K）
输出: outputs/paper_stats.json + LaTeX 表格片段
"""
import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
ALPHA = 0.1


def wilson_ci(k, n, z=1.959963985):
    """Wilson score interval for binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def one_sided_binomial(k, n, p0):
    """H0: p >= p0; H1: p < p0. 返回 P(X <= k | Bin(n, p0))."""
    if n == 0:
        return 1.0
    # 精确累积（n 最大 ~33k, 用正则化不完全 beta）
    from math import lgamma, exp
    # P(X<=k) = I_{1-p0}(n-k, k+1)
    def log_beta(a, b):
        return lgamma(a) + lgamma(b) - lgamma(a + b)
    # 用 scipy 若可用
    try:
        from scipy.stats import binom
        return float(binom.cdf(k, n, p0))
    except ImportError:
        # 近似: 连续校正正态近似
        mu = n * p0
        sd = math.sqrt(n * p0 * (1 - p0))
        from scipy.special import erfc
        return 0.5 * erfc((k + 0.5 - mu) / (sd * math.sqrt(2)))


def ztest_two_prop(k1, n1, k2, n2):
    """双比例 z 检验（同语义风险比较）: H0 p1=p2."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    if p * (1 - p) == 0:
        return (float("nan"), 1.0)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    from scipy.special import erfc
    pv = erfc(abs(z) / math.sqrt(2))  # 双尾
    return (z, pv)


def main():
    sc = json.load(open(os.path.join(OUT, "exp_sc10_eval.json")))
    us = json.load(open(os.path.join(OUT, "exp_us8k_eval.json")))

    # FSD50K 冻结表（Table II 源值, n = cov × 6765）
    fsd = {
        "B12_tau0.5": {"risk": 0.446, "cov": 0.495},
        "B12_tau0.95": {"risk": 0.284, "cov": 0.183},
        "B12a_true_snr": {"risk": 0.449, "cov": 0.487},
        "B13_phys": {"risk": 0.494, "cov": 0.612},
        "B11_sp": {"risk": 0.228, "cov": 0.468},
        "B0_energy": {"risk": 0.175, "cov": 0.427},
        "AGRC": {"risk": 0.083, "cov": 0.011},
    }
    N_FSD = 6765

    out = {"wilson_ci": {}, "risk_le_alpha": {}, "z_b12_vs_b12a": {}, "per_kind_ci": {}}

    # 1. Wilson CI
    def add_ci(ds_name, row_name, risk, cov, n_windows, n_exact=None):
        n = n_exact if n_exact else int(round(cov * n_windows))
        k = int(round(risk * n))
        lo, hi = wilson_ci(k, n)
        out["wilson_ci"][f"{ds_name}/{row_name}"] = {
            "n_decide": n, "k_err": k, "risk": round(risk, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        return n, k

    for row_name, r in fsd.items():
        add_ci("FSD50K", row_name, r["risk"], r["cov"], N_FSD)
    for ds, js in (("SC10", sc), ("US8K", us)):
        nw = js["n_windows"]
        for row_name, r in js["systems"].items():
            if row_name in ("B12_tau0.05", "B12_tau0.10"):
                continue
            add_ci(ds, row_name, r["risk"], r["coverage"], nw, n_exact=r.get("n_decide"))
        for row_name, r in js["detectors"].items():
            if "risk" not in r:
                continue
            add_ci(ds, row_name, r["risk"], r["coverage"], nw, n_exact=r.get("n_decide"))

    # 2. 单侧二项检验 risk <= alpha
    def risk_le_alpha(ds, row, risk, n):
        k = int(round(risk * n))
        pv = one_sided_binomial(k, n, ALPHA)
        out["risk_le_alpha"][f"{ds}/{row}"] = {"n": n, "k": k, "p_value": round(pv, 6),
                                               "reject_H0_risk_ge_alpha": pv < 0.05}

    risk_le_alpha("FSD50K", "B12_tau0.5", 0.446, int(round(0.495 * N_FSD)))
    risk_le_alpha("FSD50K", "B12_tau0.95", 0.284, int(round(0.183 * N_FSD)))
    risk_le_alpha("SC10", "B12_tau0.5", sc["systems"]["B12_tau0.50"]["risk"],
                  sc["systems"]["B12_tau0.50"]["n_decide"])
    risk_le_alpha("SC10", "B12_tau0.95", sc["systems"]["B12_tau0.95"]["risk"],
                  sc["systems"]["B12_tau0.95"]["n_decide"])
    risk_le_alpha("US8K", "B12_tau0.5", us["systems"]["B12_tau0.50"]["risk"],
                  us["systems"]["B12_tau0.50"]["n_decide"])

    # 3. B12 vs B12a z 检验（同语义: 分类错误率）
    def z_b12_b12a(ds, js, nw):
        b12 = js["systems"]["B12_tau0.50"]
        b12a = js["systems"]["B12_tau0.5"] if "B12_tau0.5" in js["systems"] else None
        # B12a 在 detectors 里?
        b12a = js["detectors"].get("B12a_true_snr") or js["detectors"].get("B12a")
        if b12a is None:
            return
        z, pv = ztest_two_prop(int(round(b12["risk"] * b12["n_decide"])), b12["n_decide"],
                               int(round(b12a["risk"] * b12a["n_decide"])), b12a["n_decide"])
        out["z_b12_vs_b12a"][ds] = {"z": round(z, 3), "p": round(pv, 5)}

    # FSD50K: B12 0.446(n=3349) vs B12a 0.449(n=3294)
    z, pv = ztest_two_prop(int(round(0.446 * 3349)), 3349, int(round(0.449 * 3294)), 3294)
    out["z_b12_vs_b12a"]["FSD50K"] = {"z": round(z, 3), "p": round(pv, 5)}
    for ds, js in (("SC10", sc), ("US8K", us)):
        b12 = js["systems"]["B12_tau0.50"]
        b12a = js["systems"].get("B12a_true_snr")
        if b12a and "risk" in b12a and b12a.get("n_decide"):
            z, pv = ztest_two_prop(int(round(b12["risk"] * b12["n_decide"])), b12["n_decide"],
                                   int(round(b12a["risk"] * b12a["n_decide"])), b12a["n_decide"])
            out["z_b12_vs_b12a"][ds] = {"z": round(z, 3), "p": round(pv, 5)}

    # 4. per-kind Wilson CI（SC-10/US8K）
    for ds, js in (("SC10", sc), ("US8K", us)):
        out["per_kind_ci"][ds] = {}
        for kind, r in js["per_kind"].items():
            n = r["n_decide"]
            k = int(round(r["risk"] * n))
            lo, hi = wilson_ci(k, n)
            out["per_kind_ci"][ds][kind] = {"risk": round(r["risk"], 4), "n": n,
                                            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    with open(os.path.join(OUT, "paper_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:2600])


if __name__ == "__main__":
    main()
