"""
metrics.py — v4.1 指标集（§6.5）
  Operating Gap (带符号): gap(α) = R̂(α) − α
  覆盖率 vs 物理上限 c_phys(α)
  event-level AURC（风险-覆盖率曲线下面积）
  rank-AUC（SNR̂ 排序代理质量）
  selective-ECE@coverage（选择性概率校准）
  MR: Guard Mode 漏报率（不可听且未告警）
"""
import numpy as np
import torch

from . import config as C
from .af_rule import pe_theory_rn


def operating_gap(decide: np.ndarray, correct: np.ndarray, alpha: float = C.ALPHA):
    """R̂(α) = 决策样本中的错误率; gap = R̂ − α。带符号: 负 gap = 低于 ED 前沿。"""
    n_dec = int(decide.sum())
    if n_dec == 0:
        return float("nan"), 0.0
    risk = 1.0 - correct[decide].mean()          # 决策样本错误率
    return risk - alpha, float(risk)


def coverage(decide: np.ndarray) -> float:
    return float(decide.mean())


def physical_coverage(r_db: np.ndarray, r_min: float) -> float:
    """c_phys(α) = P(r ≥ r_min)。"""
    return float((10 ** (r_db / 10) >= r_min).mean())


def auroc(decide: np.ndarray, correct: np.ndarray):
    """风险-覆盖率曲线下面积（覆盖率 0→1 排序）。"""
    order = np.argsort(-decide.astype(float))     # 先决策的在前（按置信度排序需外部传入分数）
    risk_curve, cov_curve = [], []
    for i in range(1, len(order) + 1):
        cov = i / len(order)
        dec = order[:i]
        risk = 1.0 - correct[dec].mean()
        risk_curve.append(risk)
        cov_curve.append(cov)
    return float(np.trapezoid(risk_curve, cov_curve))


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray):
    """按分数降序决策的风险-覆盖率曲线（AURC 用）。"""
    order = np.argsort(-scores)
    s = correct[order]
    cum_err = np.cumsum(1 - s)
    cov = np.arange(1, len(s) + 1) / len(s)
    risk = cum_err / np.arange(1, len(s) + 1)
    return risk, cov


def auroc_from_scores(scores: np.ndarray, correct: np.ndarray) -> float:
    risk, cov = risk_coverage_curve(scores, correct)
    return float(np.trapezoid(risk, cov))


def rank_auc(snr_db: np.ndarray, correct: np.ndarray) -> float:
    """SNR̂ 排序对"决策是否正确"的 AUC（排序代理质量）。"""
    order = np.argsort(-snr_db)
    y = correct[order]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.where(y == 1)[0] + 1
    return float((ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selective_ece(event_probs: np.ndarray, correct: np.ndarray, decide: np.ndarray,
                  n_bins: int = 10) -> float:
    """选择性概率校准: 对决策样本按 max prob 分箱比较准确率。"""
    idx = np.where(decide)[0]
    if len(idx) == 0:
        return float("nan")
    conf = event_probs[idx].max(axis=1)
    acc = correct[idx]
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() / len(idx) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def miss_rate(events_inaudible: np.ndarray, warned: np.ndarray) -> float:
    """MR = P(未告警 | 事件不可听)。"""
    if events_inaudible.sum() == 0:
        return float("nan")
    return float(((~warned) & events_inaudible).sum() / events_inaudible.sum())


def snr_mae(snr_pred: np.ndarray, snr_true: np.ndarray, mask: np.ndarray) -> float:
    """逐类 SNR MAE（dB），掩码类不计。"""
    m = mask.astype(bool)
    if m.sum() == 0:
        return float("nan")
    return float(np.abs(snr_pred[m] - snr_true[m]).mean())
