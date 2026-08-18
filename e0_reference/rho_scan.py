"""rho_scan.py — 噪声功率不确定 ρ 扫描（2026-08-18, optimization #5）

实证正文 "The SNR wall and in-window noise estimation" 的 CFAR 论证:
  名义边界 r_min(−13.7 dB, BT=3840) 低于 SNR wall 7.4 dB, 合法只因系统不假设
  噪声功率而改为窗内估计。本实验在 r_min 处把噪声方差按 ρ 缩放, 对比:
    wall 场景  参考窗取名义噪声（固定不确定, wall 的适用条件）→ 门限失真,
               P_FA 随 ρ 爆炸（决策泄漏到前沿以下 = 非法）
    cfar 场景  参考窗取自同一 ρ 缩放噪声（估计跟随真实功率）→ P_FA 钉在
               α, 检测性能按 r/ρ 的真实 SNR 退化（边界保持诚实）
  预测 = 双方差高斯（与 e0_reliability 同公式）; 实测 = 蒙特卡洛。
  附实噪声 mismatch 面板: 冻结 e0_reliability CSV 的 pred vs emp_pe（零成本）。

输出: outputs/exp_rho_scan_20260818.json + outputs/fig_supp_rho_mismatch.{png,pdf,svg}
运行: cd e0_reference && python rho_scan.py
"""
import argparse
import csv
import json
import math
import os

import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_detector import ppf, phi
from wind_noise import make_wind

FS = 16000
F_LO, F_HI = 1000.0, 4000.0
RATE2 = 2 * (F_HI - F_LO)
ALPHA = 0.1
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

R_MIN_DB = -13.7                 # 参考边界（单类, BT=3840）
R_MIN = 10.0 ** (R_MIN_DB / 10.0)
RHO_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def to_band(x):
    b, a = sig.butter(4, [F_LO / (FS / 2), F_HI / (FS / 2)], btype="band")
    return sig.resample_poly(sig.lfilter(b, a, x), 3, 8)


def pe_pred_two_moment(r_eff, n, rho, alpha):
    """双方差高斯预测: z0~(rho, 2rho²/n), z1~(r_eff+rho, 2(r_eff+rho)²/n)。
    r_eff = 事件能量 / 名义噪声（真实 SNR = r_eff/rho）。"""
    thr = 1.0 + 2.0 / math.sqrt(n) * ppf(1.0 - alpha)
    s0 = math.sqrt(2.0) * rho / math.sqrt(n)
    s1 = math.sqrt(2.0) * (r_eff + rho) / math.sqrt(n)
    pfa = 1.0 - phi((thr - rho) / s0)
    miss = phi((thr - (r_eff + rho)) / s1)
    return pfa, miss, 0.5 * (pfa + miss)


def run_rho_cell(events, r_target, n_eff, rho, scenario, seed, alpha=ALPHA):
    """scenario: 'wall'（参考窗名义噪声）| 'cfar'（参考窗 ρ 缩放噪声）。
    测试窗噪声一律 ρ 缩放。返回经验 P_FA / P_miss / P_e 与 z 统计。"""
    rng = np.random.default_rng(seed)
    thr = 1.0 + 2.0 / math.sqrt(n_eff) * ppf(1.0 - alpha)
    n_t = 0
    fa, miss = [], []
    z0s, z1s = [], []
    for ev in events:
        n_src = int(round(1.28 * FS))
        for start in range(0, ev.shape[0] - n_src + 1, n_src):
            n_t += 1
            seg = ev[start:start + n_src]
            w_ref0, _ = make_wind(n_src, FS, seed=seed + 2 * n_t)
            w_tst, _ = make_wind(n_src, FS, seed=seed + 2 * n_t + 1)
            w_tst = w_tst * math.sqrt(rho)          # 测试窗噪声 ρ 缩放（真实 SNR = r_target/ρ）
            eb = to_band(seg)
            e_energy = float(np.sum(eb ** 2))
            denom_true = float(np.sum(to_band(w_ref0) ** 2)) + 1e-15
            if e_energy < 1e-9 * n_eff:
                continue
            g = math.sqrt(e_energy / (denom_true * r_target))   # 事件固定, 按名义 σ₀ 配比
            if scenario == "cfar":
                w_ref = w_ref0 * math.sqrt(rho)     # 参考窗跟随噪声功率（估计口径）
            else:
                w_ref = w_ref0                      # 名义参考（wall: 假设已知 σ₀）
            denom = float(np.sum(to_band(w_ref) ** 2)) + 1e-15
            z0 = float(np.sum(to_band(w_tst) ** 2)) / denom
            z1 = float(np.sum(to_band(seg + g * w_tst) ** 2)) / (g * g * denom)
            z0s.append(z0); z1s.append(z1)
            fa.append(z0 > thr)
            miss.append(z1 < thr)
    if n_t == 0:
        return None
    z0, z1 = np.array(z0s), np.array(z1s)
    return {"rho": rho, "scenario": scenario, "n": int(n_t),
            "emp_pfa": round(float(np.mean(fa)), 4), "emp_pmiss": round(float(np.mean(miss)), 4),
            "emp_pe": round(0.5 * (np.mean(fa) + np.mean(miss)), 4),
            "z0_mean": round(float(z0.mean()), 4), "z0_std": round(float(z0.std()), 4),
            "z1_mean": round(float(z1.mean()), 4), "z1_std": round(float(z1.std()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    n_eff = int(round(RATE2 * 1.28))
    rng = np.random.default_rng(args.seed)
    events = []
    b, a = sig.butter(4, [F_LO / (FS / 2), F_HI / (FS / 2)], btype="band")
    for _ in range(400):
        n = int(1.28 * FS)
        ev = sig.lfilter(b, a, rng.standard_normal(n))
        events.append(ev / (np.std(ev) + 1e-12) * 0.5)

    rows = []
    print(f"r_min = {R_MIN_DB} dB (r={R_MIN:.4f}), n={n_eff}, ρ grid {RHO_GRID}")
    for rho in RHO_GRID:
        for scen in ("wall", "cfar"):
            res = run_rho_cell(events, R_MIN, n_eff, rho, scen, args.seed)
            res["pred_pfa"], res["pred_pmiss"], res["pred_pe"] = pe_pred_two_moment(
                R_MIN, n_eff, rho, ALPHA)
            # CFAR 预测: FA 钉 α, miss 按有效 SNR r/ρ（双方差, 参考窗估计无系统偏差）
            rr = R_MIN / rho
            thr = 1.0 + 2.0 / math.sqrt(n_eff) * ppf(1.0 - ALPHA)
            s1 = math.sqrt(2.0) * (1.0 + rr) / math.sqrt(n_eff)
            res["pred_cfar_pfa"] = ALPHA
            res["pred_cfar_pmiss"] = phi((thr - (1.0 + rr)) / s1)
            res["pred_cfar_pe"] = 0.5 * (ALPHA + res["pred_cfar_pmiss"])
            rows.append(res)
            print(f"  ρ={rho:4.1f} {scen:5s}: emp P_FA={res['emp_pfa']:.4f} "
                  f"P_miss={res['emp_pmiss']:.4f} P_e={res['emp_pe']:.4f} "
                  f"| pred wall P_e={res['pred_pe']:.4f}, cfar P_e={res['pred_cfar_pe']:.4f}", flush=True)

    with open(os.path.join(OUT, "exp_rho_scan_20260818.json"), "w") as f:
        json.dump({"tag": "20260818", "r_min_db": R_MIN_DB, "n_eff": n_eff,
                   "alpha": ALPHA, "rho_grid": RHO_GRID, "cells": rows}, f, indent=2)

    # ---- 图: (a) 实噪声 mismatch（冻结 e0 CSV）; (b) ρ 扫描 ----
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 9,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    NAVY, ORANGE, GRAY = "#1a1a2e", "#c5551a", "#888888"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))

    # (a) mismatch: pred_oracle vs emp_pe（三数据集/两源 CSV 取 synthetic + esc50 energy）
    for src, mkr, lbl in (("synthetic", "o", "synthetic events"),
                          ("esc50", "s", "real (ESC-50) events")):
        xs, ys = [], []
        try:
            with open(os.path.join(OUT, f"e0_reliability_{src}_energy.csv")) as f:
                for r in csv.DictReader(f):
                    xs.append(float(r["pred_oracle"])); ys.append(float(r["emp_pe"]))
        except FileNotFoundError:
            continue
        ax1.scatter(xs, ys, s=34, c=NAVY if src == "synthetic" else ORANGE,
                    marker=mkr, zorder=3, label=lbl)
    lim = [0, 0.5]
    ax1.plot(lim, lim, "k--", lw=0.9, label="identity")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("predicted $P_e$ (Gaussian, nominal $\\sigma$)")
    ax1.set_ylabel("empirical $P_e$ (wind channel)")
    ax1.set_title("(a) real-noise mismatch\n($\\sigma_0\\!\\times\\!\\approx$3.9 measured)", fontsize=8.5)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)

    # (b) ρ 扫描
    for scen, col, mk, lbl in (("wall", NAVY, "o", "wall: reference at nominal $\\sigma$"),
                               ("cfar", ORANGE, "s", "CFAR: reference tracks noise")):
        rs = [r for r in rows if r["scenario"] == scen]
        ax2.plot([r["rho"] for r in rs], [r["pred_pe"] for r in rs], ls=":",
                 color=col, lw=1.2, label=f"{lbl} (predicted)")
        ax2.plot([r["rho"] for r in rs], [r["emp_pe"] for r in rs], mk + "-",
                 color=col, lw=1.5, ms=4, label=f"{lbl} (measured)")
    ax2.axhline(ALPHA, color=GRAY, lw=0.9, ls="--")
    ax2.text(7.6, ALPHA + 0.015, "$\\alpha$", color=GRAY, fontsize=8, ha="right")
    ax2.axvline(3.9, color=GRAY, lw=0.8, ls=":")
    ax2.text(3.95, 0.46, "measured $\\sigma_0\\!\\times\\!\\approx$3.9", fontsize=7, color=GRAY)
    ax2.set_xscale("log")
    ax2.set_xticks(RHO_GRID)
    ax2.set_xticklabels([f"{r:g}" for r in RHO_GRID])
    ax2.set_xlim(0.9, 9)
    ax2.set_ylim(0, 0.55)
    ax2.set_xlabel("noise-power inflation $\\rho$ (linear)")
    ax2.set_ylabel("$P_e$ at $r_{\\min}$")
    ax2.set_title("(b) $\\rho$-scan at the frontier (BT=3840)", fontsize=8.5)
    ax2.legend(fontsize=6.8, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_supp_rho_mismatch.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    print("\nDONE → outputs/exp_rho_scan_20260818.json + fig_supp_rho_mismatch.{png,pdf,svg}")


if __name__ == "__main__":
    main()
