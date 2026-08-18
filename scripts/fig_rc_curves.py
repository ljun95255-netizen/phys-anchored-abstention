"""fig_rc_curves.py — 补充材料 RC 曲线图（2026-08-18, optimization #7）
Chow 式 risk-coverage 曲线（选择性分类文献惯例）: 三数据集各一面板,
  B12 (SONTRA-A) τ 扫实线 + CLAP-FT 探针 τ 扫虚线 + oracle 点 (cov 0.600, risk 0)
  + α=0.1 水平线; 头条操作点加星标。全部来自冻结 JSON, 无新数据。
输出: outputs/fig_supp_rc_curves.{png,pdf,svg}（STIX, 300dpi, 图例 X 轴正下居中）
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
    "font.size": 9,
    "mathtext.fontset": "stix",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
NAVY = "#1a1a2e"
ORANGE = "#c5551a"
GRAY = "#888888"
ALPHA = 0.1


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def probe_curve(ds):
    d = load(f"exp_clap_finetune_{ds}_20260816.json")
    rows = d["evaluation"]["rows"]
    pts = []
    for k, r in rows.items():
        if r["coverage"] == 0.0 and r["risk"] == 0.0:
            continue  # 退化行（US8K τ≥0.7 零决策）不进曲线
        pts.append((r["coverage"], r["risk"]))
    return sorted(pts)


def b12_curve(source, bucket=None, key_fmt="{:.2f}"):
    d = load(source)
    if bucket:
        d = d[bucket]
    pts = []
    for k, r in d.items():
        if not k.startswith("B12_tau"):
            continue
        t = float(k[len("B12_tau"):])
        pts.append((r["coverage"], r["risk"]))
    return sorted(pts)


def main():
    panels = [
        ("FSD50K-10", "B12 $\\tau$ sweep", "CLAP-FT probe", (0.495, 0.446), (0.163, 0.074)),
        ("SC-10", "B12 $\\tau$ sweep", "CLAP-FT probe (control)", (0.542, 0.081), (0.123, 0.105)),
        ("UrbanSound8K", "B12 $\\tau$ sweep", "CLAP-FT probe", (0.424, 0.420), (0.303, 0.136)),
    ]
    curves = [
        b12_curve("exp_tau_scan_frozen.json"),
        b12_curve("exp_sc10_eval.json", bucket="systems"),
        b12_curve("exp_us8k_eval.json", bucket="systems"),
    ]
    probes = [probe_curve("fsd50k10"), probe_curve("sc10"), probe_curve("us8k")]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=True)
    for ax, (title, blab, plab, b_head, p_head), bc, pc in zip(axes, panels, curves, probes):
        cb = np.array(bc)
        cp = np.array(pc)
        ax.plot(cb[:, 0], cb[:, 1], "o-", color=NAVY, lw=1.6, ms=3.2, zorder=4, label=blab)
        ax.plot(cp[:, 0], cp[:, 1], "s--", color=ORANGE, lw=1.4, ms=3.0, zorder=4, label=plab)
        ax.plot([0.600], [0.0], marker="*", ms=11, color="black", zorder=6,
                markeredgecolor="white", markeredgewidth=0.5, label="oracle (frontier)")
        ax.axhline(ALPHA, color=GRAY, lw=0.9, ls=":")
        ax.text(0.97, ALPHA + 0.012, r"$\alpha$", color=GRAY, fontsize=8, ha="right")
        ax.plot([b_head[0]], [b_head[1]], "o", ms=7, mfc="none", mec=NAVY, mew=1.4, zorder=5)
        ax.plot([p_head[0]], [p_head[1]], "s", ms=6.5, mfc="none", mec=ORANGE, mew=1.3, zorder=5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("coverage")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 0.62)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("risk (error among decisions)")
    axes[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_supp_rc_curves.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    print("DONE → outputs/fig_supp_rc_curves.{png,pdf,svg}")


if __name__ == "__main__":
    main()
