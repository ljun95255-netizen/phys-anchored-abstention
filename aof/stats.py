"""stats.py — v3 §11.5 统计纪律工具（预注册）
  1. source-cluster bootstrap: 按源录音聚类重采样（同源片段共享录音 → 方差不低估）
  2. one-sided risk certificate: H0: risk > α 的单侧 95% 上界（非 ECE）
  3. worst-cell ECE: 沿前沿最差单元的校准（不报全体均值）
  4. MMD / centroid distance: 腐蚀族特征空间正交性诊断（高风噪≈低截止遮挡风险）
输入契约: recs = list[dict] 或 (decide, correct, cluster_id) 三元组。
"""
import numpy as np
from scipy.stats import norm


def paired_bootstrap(gap_a: np.ndarray, gap_b: np.ndarray, n_boot: int = 1000,
                     seed: int = 20260804) -> dict:
    """配对 bootstrap（同种子同 clip 的 B12 vs B11）: 差值分布 + 单侧 P 值。
    gap_a/gap_b: 逐 clip 的 gap（或任意逐单元指标）, 配对排列。"""
    if len(gap_a) != len(gap_b):
        raise ValueError("配对比较要求等长")
    d = np.asarray(gap_a, dtype=float) - np.asarray(gap_b, dtype=float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    n = len(d)
    d_centered = d - d.mean()                       # 按 H0（差=0）中心化
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(d_centered[idx].mean())
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals + d.mean(), [2.5, 97.5])   # CI 用未中心化分布
    # H0: 差值均值 ≥ 0（A 不优于 B）→ 单侧 p = P(d̄*_centered ≤ d̄_obs)
    p_value = float((vals <= d.mean()).mean())
    return {"mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": p_value, "n": n, "n_boot": n_boot}


def cluster_bootstrap(decide: np.ndarray, correct: np.ndarray, cluster: np.ndarray,
                      metric_fn, n_boot: int = 1000, seed: int = 20260804) -> dict:
    """按 cluster（源录音 ID）聚类 bootstrap。返回 metric 的均值 ± 95% CI。
    metric_fn(decide_sub, correct_sub) → float。每个 bootstrap 抽取整簇（放回）。"""
    clusters = np.unique(cluster)
    if len(clusters) < 8:
        raise ValueError(f"聚类数 {len(clusters)} < 8 → CI 退化（预注册: ≥8 源簇/类）")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([np.where(cluster == c)[0] for c in picked])
        try:
            vals.append(metric_fn(decide[idx], correct[idx]))
        except (ValueError, ZeroDivisionError):
            continue
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": int(len(vals))}


def risk_upper_bound(decide: np.ndarray, correct: np.ndarray, alpha: float,
                     z: float = 1.645) -> dict:
    """单侧 95% 上界: H0: risk > α 拒绝当 n 足够大且 R̂ 的上界 < α。
    Wilson score 区间上界（保守, 聚类相关性由调用方用 cluster_bootstrap 复核）。"""
    n = int(decide.sum())
    if n == 0:
        return {"n": 0, "risk": float("nan"), "upper": float("inf"),
                "reject_H0": False}
    risk = float(1.0 - correct[decide].mean())
    # Wilson score 单侧上界
    z2 = z * z
    p_hat = risk
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom
    upper = float(center + half)
    return {"n": n, "risk": risk, "upper": upper,
            "reject_H0": upper < alpha}


def worst_cell_ece(event_probs: np.ndarray, correct: np.ndarray, decide: np.ndarray,
                   cells: np.ndarray, n_bins: int = 10) -> dict:
    """沿前沿单元（cells 为单元 ID, 如 SNR 仓）的逐单元 ECE, 报告最差单元。
    防高 SNR 易单元把均值洗白（v3 §11.5）。"""
    cell_ids = np.unique(cells)
    ece_by_cell = {}
    for c in cell_ids:
        m = (decide & (cells == c))
        if m.sum() < 20:
            continue
        conf = event_probs[m].max(axis=1)
        acc = correct[m]
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            b = (conf >= edges[i]) & (conf < edges[i + 1])
            if b.sum() == 0:
                continue
            ece += b.sum() / m.sum() * abs(acc[b].mean() - conf[b].mean())
        ece_by_cell[int(c)] = float(ece)
    if not ece_by_cell:
        return {"worst": float("nan"), "mean": float("nan"), "cells": {}}
    worst = max(ece_by_cell, key=ece_by_cell.get)
    return {"worst": ece_by_cell[worst], "worst_cell": worst,
            "mean": float(np.mean(list(ece_by_cell.values()))), "cells": ece_by_cell}


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """RBF-MMD（无偏估计）: 特征空间正交性诊断。x,y: [n, d]。"""
    n = x.shape[0]
    m = y.shape[0]
    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma * sigma))
    return float(k(x, x).sum() / (n * (n - 1)) + k(y, y).sum() / (m * (m - 1))
                  - 2 * k(x, y).mean())


def centroid_distance(x: np.ndarray, y: np.ndarray) -> float:
    """质心欧氏距离（MMD 的补充, 对分布平移敏感）。"""
    return float(np.linalg.norm(x.mean(0) - y.mean(0)))


def orthogonality_report(feat_by_kind: dict, sigma: float = 1.0) -> dict:
    """feat_by_kind: {kind: [n,d] 特征矩阵} → 两两 MMD + 质心距离。"""
    kinds = list(feat_by_kind.keys())
    out = {}
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            a, b = kinds[i], kinds[j]
            out[f"{a}↔{b}"] = {
                "mmd": mmd_rbf(feat_by_kind[a], feat_by_kind[b], sigma),
                "centroid": centroid_distance(feat_by_kind[a], feat_by_kind[b]),
            }
    return out
