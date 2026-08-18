"""
af_rule.py — AF-Rule（NP 双参决策）+ T2O + SNR wall（v4.1 §5.2-5.4）
理论来自 e0_reference/energy_detector.py（MC 验证 <0.6% 误差）。

口径: r = E_s/P_n（窗内能量比, A-Head 输出 dB）; n = 2·B·T 有效样本数。
  Pe(r,n) = Φ(−r·√(n/2)/(1+√(1+2r)))
  r_min(α,n): 数值求 Pe(r,n)=α 的反函数
  T2O: t*(α) = [2Φ⁻¹(1−α)]^{2/3} / (B · SNR_per^{2/3}),  SNR_per = r/n（每样本功率比）
  SNR wall: SNR_wall ≈ (ρ−1/ρ)/2（线性功率不确定 ρ）→ t* = ∞
"""
import math

import torch

from . import config as C


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ppf(p):
    from scipy.stats import norm
    return float(norm.ppf(p))


def pe_theory_rn(r: float, n: int) -> float:
    """双方差高斯近似: Pe = Φ(−r√(n/2)/(1+√(1+2r)))。"""
    return phi(-r * math.sqrt(n / 2) / (1.0 + math.sqrt(1.0 + 2.0 * r)))


def r_min_theory(alpha: float, n: int) -> float:
    lo, hi = -10.0, 1e7
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pe_theory_rn(mid, n) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def n_eff(band_hz: float, t_sec: float) -> int:
    return max(int(round(2 * band_hz * t_sec)), 2)


def snr_wall(rho_db: float = C.SNR_WALL_RHO_DB) -> float:
    """线性功率不确定 ρ 下的能量检测墙（Tandra–Sahai 型）→ 每样本 SNR 下限（dB）。"""
    rho = 10 ** (rho_db / 10)
    return 10 * math.log10(max((rho - 1.0 / rho) / 2.0, 1e-9))


def t2o(alpha: float, snr_per_db: float, band_hz: float, rho_db: float = C.SNR_WALL_RHO_DB):
    """Time-to-Observability: 还需听多久（秒）。SNR 低于 wall → ∞。
    snr_per_db = 每样本功率比 SNR（dB），= r_dB − 10log10(n)。"""
    wall = snr_wall(rho_db)
    if snr_per_db < wall:
        return math.inf
    snr = 10 ** (snr_per_db / 10)
    c = 2.0 * ppf(1.0 - alpha)
    return (c / snr) ** (2.0 / 3.0) / band_hz


class AFRule:
    """NP 双参决策: 事件集 S = {k: p̂_k > γ}; k∈S 需 P_D(r̂_k) ≥ 1−α_k 且 P_FA ≤ β 才决策。
    否则弃权 → Guard Mode（保守警告语义，本模块返回 abstain=True）。"""

    def __init__(self, alpha: float = C.ALPHA, beta: float = C.BETA,
                 gamma: float = C.GAMMA, n_classes: int = C.N_CLASSES,
                 band_hz: float = C.EVENT_BAND[1] - C.EVENT_BAND[0], t_sec: float = C.WINDOW_SEC):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = n_eff(band_hz, t_sec)
        # Bonferroni: 逐类 α_k = α/10
        self.alpha_k = alpha / n_classes
        self.r_min = r_min_theory(self.alpha_k, self.n)

    def p_d(self, r_db: torch.Tensor) -> torch.Tensor:
        """P_D(SNR̂) = Φ(d) = 1 − Pe(r̂, n)，r̂ 由 dB 转线性。"""
        r = torch.pow(10.0, r_db / 10.0)
        d = r * math.sqrt(self.n / 2) / (1.0 + torch.sqrt(1.0 + 2.0 * r))
        return 0.5 * (1.0 + torch.erf(d / math.sqrt(2.0)))

    def decide(self, event_probs: torch.Tensor, snr_db: torch.Tensor, tau=None):
        """→ (decide[B], pred_class[B], r_ratio[B]): r_ratio = r̂/r_min（>1 才可决策）。
        tau: 事件概率阈值（τ-Cal 校准后替代 gamma）。event_probs 已是概率（模型输出）。
        pred = 可闻类中概率最高者（非掩码首个——R16 修复）。"""
        probs = event_probs
        gamma = self.gamma if tau is None else tau
        cand = probs > gamma
        cand[:, -1] = False                    # unknown 不参与决策（最后列, 兼容 10/5 类）
        pd = self.p_d(snr_db)
        ok = pd >= (1.0 - self.alpha_k)                     # P_D ≥ 1−α_k
        eligible = cand & ok
        n_elig = eligible.sum(dim=1)
        pred = (eligible.float() * probs).argmax(dim=1)     # 可闻类按概率 argmax
        decide = (n_elig > 0) & (pred < probs.shape[1] - 1)
        r = torch.pow(10.0, snr_db / 10.0)
        r_ratio = r / self.r_min
        return decide, pred, r_ratio

    def coverage_ceiling(self, r_db: torch.Tensor) -> torch.Tensor:
        """c_phys(α) = P(r ≥ r_min)（物理覆盖率上限, 用 oracle r 评估）。"""
        return (torch.pow(10.0, r_db / 10.0) >= self.r_min).float()
