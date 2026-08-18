"""test_af_rule.py + test_metrics.py + test_wsosim.py — 决策规则/指标/模拟器验证"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from aof.af_rule import AFRule, pe_theory_rn, r_min_theory, t2o, snr_wall, n_eff
from aof.metrics import operating_gap, auroc_from_scores, rank_auc, selective_ece, miss_rate, snr_mae
from aof.wsosim import corrupt, _wind
from aof import config as C


def test_rmin_matches_mc():
    """E0 MC 实证: SNR_min(α=0.1, BT=256) = −31.7dB（公式 SNR = r/BT 口径）。"""
    n = 2 * 256
    r = r_min_theory(0.1, n)
    snr_db = 10 * math.log10(r / 256)          # MC 口径 SNR = r/BT
    assert abs(snr_db - (-31.7)) < 0.5, f"{snr_db:.2f}"
    n2 = 2 * 1024
    r2 = r_min_theory(0.1, n2)
    snr2 = 10 * math.log10(r2 / 1024)
    assert abs(snr2 - (-40.9)) < 0.5, f"{snr2:.2f}"
    # T 加倍 4.5dB: 10log10((2c/(BT)^1.5) 对 4x BT → 9dB
    print(f"SNR_min(0.1): BT=256 → {snr_db:.2f} dB (MC −31.7) ; BT=1024 → {snr2:.2f} dB (MC −40.9)")
    print(f"T2O doubling check: 4×BT → {10*math.log10(4**1.5):.2f} dB (理论 9.02)")


def test_t2o_and_wall():
    # SNR 高于 wall: t* 有限且随 SNR 单调降
    t_hi = t2o(0.1, 0.0, 3000.0)
    t_lo = t2o(0.1, -6.0, 3000.0)
    assert t_hi < t_lo
    # 低于 wall → ∞
    assert t2o(0.1, snr_wall() - 3.0, 3000.0) == math.inf
    print(f"T2O@0dB={t_hi:.2f}s, T2O@-6dB={t_lo:.2f}s, wall={snr_wall():.2f}dB, below-wall=∞")


def test_afrule_decision():
    rule = AFRule()
    probs = torch.tensor([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]])
    snr_ok = torch.tensor([[20.0, -60, -60, -60, -60, -60, -60, -60, -60, -60]])
    snr_bad = torch.tensor([[-60.0, -60, -60, -60, -60, -60, -60, -60, -60, -60]])
    d1, p1, _ = rule.decide(probs, snr_ok)
    d2, p2, _ = rule.decide(probs, snr_bad)
    assert d1.item() is True and d2.item() is False
    print(f"AF-Rule: SNR=20dB → decide={d1.item()} (P_D={rule.p_d(snr_ok)[0,0].item():.4f} ≥ {1-rule.alpha_k:.4f}); "
          f"SNR=-60dB → abstain={not d2.item()}")


def test_metrics():
    rng = np.random.default_rng(0)
    n = 400
    scores = rng.normal(size=n)
    correct = (scores + rng.normal(0, 1, n) > 0)
    decide = scores > 0.3
    gap, risk = operating_gap(decide, correct, alpha=0.1)
    assert -1 < gap < 1
    a = auroc_from_scores(scores, correct)
    assert 0 <= a <= 1
    ra = rank_auc(scores, correct)
    assert abs(ra - 0.5) < 0.35
    ece = selective_ece(np.clip(np.column_stack([correct, 1-correct]), 1e-3, 1-1e-3).astype(float),
                        correct, decide)
    assert 0 <= ece <= 1
    mr = miss_rate(np.array([True, False, True]), np.array([True, True, False]))
    assert mr == 0.5
    mae = snr_mae(np.array([1.0, 5.0]), np.array([2.0, 3.0]), np.array([1, 0]))
    assert abs(mae - 1.0) < 1e-6
    print(f"metrics OK: gap={gap:.3f} risk={risk:.3f} AURC={a:.3f} rankAUC={ra:.3f} selECE={ece:.3f}")


def test_wsosim_determinism_and_r():
    n = C.WINDOW_SAMPLES
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(n).astype(np.float32) * 0.3
    x1, r1, m1 = corrupt(clean, "wind", 5.0, seed=42)
    x2, r2, m2 = corrupt(clean, "wind", 5.0, seed=42)
    assert np.array_equal(x1, x2) and abs(r1 - r2) < 1e-6
    # r 标签 = 带内功率比 P_s/P_n（构造即等于目标 SNR）; n_eff 只进理论阈值
    expect = 5.0
    assert abs(r1 - expect) < 1.5, f"r_db={r1:.2f} expect≈{expect:.2f}"
    n_eff_v = n_eff(C.EVENT_BAND[1] - C.EVENT_BAND[0], C.WINDOW_SEC)
    print(f"WSO-Sim deterministic ✓; r_db={r1:.2f} ≈ 目标 SNR {expect}dB; "
          f"n_eff={n_eff_v}（r_min(0.1, n)={r_min_theory(0.1, n_eff_v):.4f} ≈ {10*math.log10(r_min_theory(0.1, n_eff_v)):.1f}dB）; kinds={m1['kind']}")


if __name__ == "__main__":
    test_rmin_matches_mc()
    test_t2o_and_wall()
    test_afrule_decision()
    test_metrics()
    test_wsosim_determinism_and_r()
    print("ALL RULE/METRIC/SIM TESTS PASSED")
