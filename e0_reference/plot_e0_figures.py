"""
plot_e0_figures.py — 论文 Fig.1 素材：MC 校验图 + E0 可靠性图（双面板）
输入: outputs/e0_mc_crosscheck.csv, outputs/e0_reliability_esc50.csv (或 synthetic)
输出: outputs/fig1_mc_validation.png, outputs/fig2_reliability.png
运行: python plot_e0_figures.py
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

def load_csv(name):
    rows = []
    with open(os.path.join(OUT, name)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    # ---- 面板 a: MC 交叉校验（理论 vs 实测 Pe）----
    mc = load_csv("e0_mc_crosscheck.csv")
    x, y = [], []
    for r in mc:
        try:
            x.append(float(r["pe_theory_gauss2var"]))
            y.append(float(r["pe_empirical"]))
        except (ValueError, KeyError):
            continue
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.scatter(x, y, s=42, c="#c5551a", zorder=3, label="BT=256 / BT=1024 (10^4 trials)")
    lim = [0, 0.55]
    ax1.plot(lim, lim, "k--", lw=1, label="identity")
    ax1.set_xlabel("Predicted Pe (both-variances Gaussian)")
    ax1.set_ylabel("Empirical Pe (Monte Carlo)")
    ax1.set_title("(a) Frontier formula: MC cross-check\nmax |err| = 0.0057")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- 面板 b: E0 可靠性（ESC-50 真实音频, 双检测器）----
    for src, det, mkr, lbl in [("esc50", "energy", "o", "Energy detector (oracle ref.)"),
                               ("esc50", "sp", "s", "SP anchor B11a (PSD est.)"),
                               ("synthetic", "energy", "^", "synthetic (energy)")]:
        try:
            rel = load_csv(f"e0_reliability_{src}_{det}.csv")
        except FileNotFoundError:
            continue
        xs, ys = [], []
        for r in rel:
            xs.append(float(r["pred_calib"])); ys.append(float(r["emp_pe"]))
        ax2.scatter(xs, ys, marker=mkr, s=46, label=lbl, zorder=3)
    lim2 = [0, 0.5]
    ax2.plot(lim2, lim2, "k--", lw=1, label="identity")
    ax2.set_xlabel("Predicted Pe (calibrated, measured z0/z1 fit)")
    ax2.set_ylabel("Empirical Pe")
    ax2.set_title("(b) E0 reliability grid (9 cells, alpha=0.1)\nmax |err| = 0.0313")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(OUT, "fig1_mc_validation.png")
    p2 = os.path.join(OUT, "fig2_reliability.png")
    fig.savefig(p1, dpi=200)
    fig.savefig(p2, dpi=200)
    print(f"[written] {p1}")
    print(f"[written] {p2}")

if __name__ == "__main__":
    main()
