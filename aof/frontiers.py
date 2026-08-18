"""frontiers.py — AOF 双前沿曲线族（v4.1 §5.2 评估侧集成, 论文图 1a 用）
ED 主前沿 + MF oracle 上界。**口径统一为带内功率比 r = P_s/P_n**（与实验 SNR 网格
−25..15dB 同量纲）:
  ED: r_min(α,T) = 数值反解 pe_theory_rn(r, 2BT)=α（弱信号渐近 2Φ⁻¹(1−α)/√(BT)）
      @T=1.28s, B=3kHz → r_min ≈ 0.0422 = −13.7dB（带内比）
  MF: r_min,MF = 2[Φ⁻¹(1−α)]²/(BT)（E_s/N0 = r·BT = 2c² → r = 2c²/BT）
      @T=1.28s → −30.7dB —— **MF 在 ED 下方（已知信号更灵敏）**, 教科书顺序正确。
T 标度: ED 4.5dB/加倍, MF 3dB/加倍（v4.1 MC 实证, MF 3dB 勘误）。
"""
import math

import numpy as np

from . import config as C
from .af_rule import ppf, pe_theory_rn, r_min_theory


def snr_min_mf_db(alpha: float, band_hz: float, t_sec: float) -> float:
    """匹配滤波 oracle 上界（带内功率比 dB）: r_min,MF = 2[Φ⁻¹(1−α)]²/(BT)。"""
    bt = max(band_hz * t_sec, 1e-9)
    c = ppf(1.0 - alpha)
    return 10.0 * math.log10(2.0 * c * c / bt)


def snr_min_ed_db(alpha: float, band_hz: float, t_sec: float) -> float:
    """ED 主前沿（带内功率比 dB）: 数值反解 pe_theory_rn(r, 2BT)=α。"""
    bt = max(band_hz * t_sec, 1e-9)
    n = max(int(round(2 * bt)), 2)
    return 10.0 * math.log10(r_min_theory(alpha, n))


def frontier_table(alpha: float = C.ALPHA, band_hz: float = C.EVENT_BAND[1] - C.EVENT_BAND[0],
                   t_grid: tuple = (0.32, 0.64, 1.28, 2.56)) -> list:
    """双前沿曲线族: [(T, ED_dB, MF_dB)]。T 标度律检查: ED≈4.5dB/加倍, MF=6dB/加倍。"""
    rows = []
    for t in t_grid:
        rows.append((t, snr_min_ed_db(alpha, band_hz, t),
                     snr_min_mf_db(alpha, band_hz, t)))
    return rows


def t_scale_slope_db(rows: list) -> dict:
    """相邻 T 加倍斜率（dB/加倍）。"""
    slopes = {"ed": [], "mf": []}
    for (t1, e1, m1), (t2, e2, m2) in zip(rows, rows[1:]):
        if abs(math.log2(t2 / t1) - 1.0) < 1e-6:      # 恰好加倍
            slopes["ed"].append(e2 - e1)
            slopes["mf"].append(m2 - m1)
    return {"ed_per_doubling": float(np.mean(slopes["ed"])) if slopes["ed"] else float("nan"),
            "mf_per_doubling": float(np.mean(slopes["mf"])) if slopes["mf"] else float("nan")}
