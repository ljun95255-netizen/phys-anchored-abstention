"""conformal_rc.py — v3 §11.4 conformal 基线族（自实现, 诚实简化标注）
实现（Split/加权 split conformal risk control 血统）:
  SplitRiskControl      (AGRC 简化, Bates/Angelopoulos 2021): 校准集上选使经验风险
                        ≤ α−β_n 的最大阈值 λ（β_n = sqrt(log(1/δ)/(2n)) Hoeffding 余量）
  SelectiveCRC          (SCRC 2512.12844 简化): 先按分数选 top-k 子集, 再在该子集上风险控制
  DriftAwareCRC         (2606.15964 简化): 非交换权重 w_i（相似度）的加权分位数阈值
  AnytimeValidCRC       (2602.04364 简化): 校准集增长时 e-process 式保守合并
  NonExchangeableCRC    (2310.01262 简化): 加权 + 敏感度诊断

输入契约: 分数 s（大=更可信）, 校准集 (s_cal, err_cal), 测试分数 s_test。
输出: 决策掩码（1=决策）。所有实现均为免训练（无梯度）。
NOTE: 完整复现各论文超出脚本范围; 此处为"可辩护的 split-conformal 简化版",
      论文中须写明与各原论文的差异（非加权 → 加权/两阶段/时序扩展）。
"""
import math

import numpy as np


def _hoeffding_margin(n: int, delta: float = 0.05) -> float:
    """β_n = sqrt(log(1/δ)/(2n)): 经验风险上界余量（Hoeffding）。"""
    return math.sqrt(math.log(1.0 / delta) / (2.0 * max(n, 1)))


class SplitRiskControl:
    """AGRC 简化: 阈值 λ = 最小分数使校准集经验风险 ≤ α−β_n。"""

    def __init__(self, alpha: float = 0.1, delta: float = 0.05):
        self.alpha, self.delta = alpha, delta
        self.lam = None

    def fit(self, s_cal: np.ndarray, err_cal: np.ndarray) -> float:
        n = len(s_cal)
        margin = _hoeffding_margin(n, self.delta)
        budget = self.alpha - margin
        order = np.argsort(-s_cal)                       # 分数降序 = 决策优先序
        cum_err = np.cumsum(err_cal[order])
        counts = np.arange(1, n + 1)
        risk = cum_err / counts
        # 最大 k 使 risk ≤ budget; λ = 第 k 个样本的分数（低于 λ 拒绝）
        ok = np.where(risk <= budget)[0]
        k = int(ok[-1]) + 1 if len(ok) else 0
        self.lam = s_cal[order[k - 1]] if k > 0 else np.inf
        return float(self.lam)

    def decide(self, s_test: np.ndarray) -> np.ndarray:
        if self.lam is None:
            raise RuntimeError("fit first")
        return (s_test >= self.lam).astype(bool)


class SelectiveCRC:
    """SCRC 简化: 两阶段——先按覆盖率 cov 选 top-k 子集, 再在子集上做 split risk control。"""

    def __init__(self, alpha: float = 0.1, cov: float = 0.5, delta: float = 0.05):
        self.alpha, self.cov, self.delta = alpha, cov, delta
        self.lam = None

    def fit(self, s_cal: np.ndarray, err_cal: np.ndarray) -> float:
        n = len(s_cal)
        k = int(round(self.cov * n))
        order = np.argsort(-s_cal)[:k]
        sub_s, sub_e = s_cal[order], err_cal[order]
        rc = SplitRiskControl(self.alpha, self.delta)
        self.lam = rc.fit(sub_s, sub_e)
        return float(self.lam)

    def decide(self, s_test: np.ndarray) -> np.ndarray:
        if self.lam is None:
            raise RuntimeError("fit first")
        return (s_test >= self.lam).astype(bool)


class DriftAwareCRC:
    """2606.15964 简化: 非交换权重 w_i（与测试分布的相似度, 由调用方给出）的加权风险控制。
    λ 选择: 加权经验风险 ≤ α − β_n（β_n 用有效样本量 n_eff = (Σw)²/Σw²）。"""

    def __init__(self, alpha: float = 0.1, delta: float = 0.05):
        self.alpha, self.delta = alpha, delta
        self.lam = None

    def fit(self, s_cal: np.ndarray, err_cal: np.ndarray, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        w = w / w.sum()
        n_eff = 1.0 / float(np.sum(w ** 2))
        margin = _hoeffding_margin(n_eff, self.delta)
        budget = self.alpha - margin
        order = np.argsort(-s_cal)
        s_ord, e_ord, w_ord = s_cal[order], err_cal[order], w[order]
        cum_w = np.cumsum(w_ord)
        risk = np.cumsum(w_ord * e_ord) / cum_w
        ok = np.where(risk <= budget)[0]
        k = int(ok[-1]) + 1 if len(ok) else 0
        self.lam = s_ord[k - 1] if k > 0 else np.inf
        return float(self.lam)

    def decide(self, s_test: np.ndarray) -> np.ndarray:
        if self.lam is None:
            raise RuntimeError("fit first")
        return (s_test >= self.lam).astype(bool)


class AnytimeValidCRC:
    """2602.04364 简化: 校准集按块增长时的保守合并——每块做 split risk control,
    阈值取跨块最保守（最大）值。真正的 anytime-valid 需要 e-process; 此处为块级 Bonferroni 近似。"""

    def __init__(self, alpha: float = 0.1, delta: float = 0.05, block: int = 100):
        self.alpha, self.delta, self.block = alpha, delta, block
        self.lam = None

    def fit_stream(self, s_cal: np.ndarray, err_cal: np.ndarray) -> float:
        n = len(s_cal)
        lams = []
        for start in range(0, n, self.block):
            end = min(start + self.block, n)
            rc = SplitRiskControl(self.alpha, self.delta)
            rc.fit(s_cal[start:end], err_cal[start:end])
            lams.append(rc.lam)
        self.lam = max(lams) if lams else np.inf
        return float(self.lam)

    def decide(self, s_test: np.ndarray) -> np.ndarray:
        if self.lam is None:
            raise RuntimeError("fit first")
        return (s_test >= self.lam).astype(bool)


class NonExchangeableCRC:
    """2310.01262 简化: 加权 + 敏感度诊断（对权重扰动报告 λ 变化范围）。"""

    def __init__(self, alpha: float = 0.1, delta: float = 0.05):
        self.alpha, self.delta = alpha, delta
        self.lam = None
        self.sensitivity = None

    def fit(self, s_cal: np.ndarray, err_cal: np.ndarray, w: np.ndarray,
            w_perturb: float = 0.5) -> float:
        base = DriftAwareCRC(self.alpha, self.delta)
        self.lam = base.fit(s_cal, err_cal, w)
        # 敏感度: 权重扰动 ±perturb（重新归一化）后 λ 的极值
        w1 = w + w_perturb * (w - w.mean())
        w1 = np.clip(w1, 1e-6, None)
        lo = DriftAwareCRC(self.alpha, self.delta)
        lo_lam = lo.fit(s_cal, err_cal, w1)
        self.sensitivity = (min(self.lam, lo_lam), max(self.lam, lo_lam))
        return float(self.lam)

    def decide(self, s_test: np.ndarray) -> np.ndarray:
        if self.lam is None:
            raise RuntimeError("fit first")
        return (s_test >= self.lam).astype(bool)
