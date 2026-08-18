"""figs_cas.py — 期刊版图重建（STIXGeneral 字体, 300 dpi, 与正文 STIX 匹配）
fig1a: AOF 前沿曲线族 + 实测操作点（run_main_result.json）
fig2:  缺口分解柱状图（冻结 Table II 数字: B12/B12a/B13 + 物理界）
fig3:  [已删 2026-08-09: τ 曲线视觉未达要求; 数据保留在 exp_tau_scan_frozen.json/exp_sc10_eval.json/exp_us8k_eval.json,
       Table tab:tau + §τ sweep 正文承载; 如需重绘可恢复 fig3() 定义]
输出: outputs/fig{1a,2}_*.png + .pdf + .svg（同名覆盖 ICASSP 版, 两版共用）
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "font.size": 11,
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.frontiers import snr_min_ed_db, snr_min_mf_db

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

NAVY = "#1a1a2e"
ORANGE = "#c5551a"
GRAY = "#888888"
LIGHT = "#fdf3e7"


def fig1a():
    """Fig. 1a — Acoustic Observability Frontier（nature-figure 契约, 2026-08-07 重构）
    Core conclusion: ED 前沿 1.5 dB/倍增下降, MF 下界 3 dB/倍增, 腐蚀网格带跨骑前沿,
    操作窗口 T=1.28s 处 r_min=−13.7 dB（单类参考）/−11.1 dB（Bonferroni 操作边界）。
    Evidence: 全部为冻结解析公式值（snr_min_ed_db/mf_db, α=0.1, B=3kHz）——无新数据。
    Archetype: 单面板定量趋势（hero=ED 曲线）。
    Journal contract: CAS 单栏 ~156mm 置放, 5pt 字形下限, STIXGeneral,
    PNG 300dpi + PDF 矢量（fonttype 42 可编辑文本）。
    """
    band = C.EVENT_BAND[1] - C.EVENT_BAND[0]
    T_grid = np.array([0.32, 0.64, 1.28, 2.56, 5.12])
    ed = [snr_min_ed_db(C.ALPHA, band, t) for t in T_grid]
    mf = [snr_min_mf_db(C.ALPHA, band, t) for t in T_grid]

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.2))
    # 1) hero: ED 前沿（图例移至 X 轴正下居中, 不占数据区）
    ax.semilogx(T_grid, ed, "o-", color=NAVY, lw=2.2, ms=5, zorder=5,
                label="energy detector (ED)")
    # 2) MF 下界
    ax.semilogx(T_grid, mf, "s--", color=ORANGE, lw=1.8, ms=4.5, zorder=4,
                label="matched-filter oracle")
    # 3) 腐蚀网格带（无文字, 跨度即范围; 图注说明）
    ax.axhspan(-25, 15, color=LIGHT, alpha=0.45, zorder=0)
    # 4) 操作窗口竖线
    ax.axvline(1.28, color=GRAY, lw=0.9, ls=":", zorder=3)
    # 5) r_min 星标（唯一重点标注, 冻结值）; 三角去掉（与星视觉重合）
    ax.plot([1.28], [-13.7], marker="*", ms=16, color=NAVY, zorder=6,
            markeredgecolor="white", markeredgewidth=0.7)
    # 6) 合并注释（写在 −11.1 dB 处, 简练）; 箭头指向星标右上侧（不压星, 加粗）
    ax.annotate(r"$\mathbfit{r_{\min}} = \mathbfit{-13.7}$ dB" + "\n"
                + r"$\mathbfit{-11.1}$ dB (Bonferroni op.)",
                xy=(1.36, -12.4), xytext=(1.62, -4.0), fontsize=9.5, color=NAVY,
                va="center", ha="left", fontweight="bold", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.6))
    ax.set_xlabel("T (s)", fontsize=11)
    ax.set_ylabel(rf"$r_{{\min}}$ (dB)", fontsize=11)
    ax.set_xticks(T_grid)
    ax.set_xticklabels(["0.32", "0.64", "1.28", "2.56", "5.12"])
    ax.set_xlim(0.28, 7.0)
    ax.set_ylim(-41, 16)
    ax.minorticks_off()
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1a_frontier.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig1a_frontier.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1a_frontier.svg"), bbox_inches="tight")
    print("DONE → fig1a_frontier.png/pdf/svg (300dpi, STIXGeneral, vector PDF)")


def fig2():
    """缺口分解: (a) B12 vs B12a vs B13 柱状（冻结 Table II）; (b) 物理覆盖界 1-c_phys"""
    systems = [("B12 (CAE)", 0.346, NAVY), ("B12a (true SNR)", 0.349, ORANGE),
               ("B13 (phys. rule)", 0.394, GRAY)]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [2.2, 1]})
    names = [s[0] for s in systems]
    vals = [s[1] for s in systems]
    cols = [s[2] for s in systems]
    bars = axa.bar(names, vals, color=cols, width=0.6)
    axa.axhline(0, color="k", lw=0.8)
    axa.set_ylabel("operating gap")
    axa.set_ylim(0, 0.45)
    axa.text(-0.12, 1.04, "(a)", transform=axa.transAxes, fontsize=11, va="bottom", ha="left")
    axb.text(-0.12, 1.04, "(b)", transform=axb.transAxes, fontsize=11, va="bottom", ha="left")
    for b, v in zip(bars, vals):
        axa.text(b.get_x() + b.get_width() / 2, v + 0.008, f"+{v:.3f}",
                 ha="center", fontsize=9)
    axa.tick_params(axis="x", labelsize=9)
    axa.grid(axis="y", alpha=0.3)

    axb.bar(["1 − c_phys"], [0.400], color=NAVY, width=0.5)
    axb.set_ylim(0, 0.45)
    axb.text(0, 0.408, "0.400", ha="center", fontsize=9)
    axb.set_ylabel("coverage units")
    axb.tick_params(axis="x", labelsize=9)
    axb.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_gap_decomposition.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig2_gap_decomposition.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig2_gap_decomposition.svg"), bbox_inches="tight")
    print("DONE → fig2_gap_decomposition.png/pdf/svg (300dpi, STIXGeneral, vector PDF)")


if __name__ == "__main__":
    fig1a()
    fig2()
