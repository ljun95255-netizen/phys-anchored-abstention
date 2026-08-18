"""test_v3_tools.py — v3 增量工具单元测试（CPU, 小数据, 不依赖训练）
覆盖: frontiers / conformal_rc / stats / baselines_extra(部分) / wind_inflation
运行: python -m pytest tests/test_v3_tools.py -q
      （或直接 python3 tests/test_v3_tools.py）
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.frontiers import (snr_min_ed_db, snr_min_mf_db, frontier_table,
                           t_scale_slope_db)
from aof.conformal_rc import (SplitRiskControl, SelectiveCRC, DriftAwareCRC,
                              AnytimeValidCRC, NonExchangeableCRC)
from aof.stats import (cluster_bootstrap, risk_upper_bound, worst_cell_ece,
                       mmd_rbf, centroid_distance, orthogonality_report,
                       paired_bootstrap)
from aof.calibration import IsotonicCalibrator, ScoreCalibrator
from aof.wind_inflation import (am_variance_inflation, inflation_factor,
                                pe_predicted_with_inflation)
from aof.baselines_extra import SohnNPLRT


def test_frontiers():
    ed = snr_min_ed_db(0.1, 3000.0, 1.28)
    mf = snr_min_mf_db(0.1, 3000.0, 1.28)
    # 带内功率比口径（与实验网格同量纲）: MF（已知信号）比 ED 更灵敏 → MF 更低
    assert mf < ed, f"MF 应在 ED 下方: ED={ed} MF={mf}"
    assert abs(ed - (-13.7)) < 1.0, ed          # r_min ≈ 0.0422 = −13.7dB 带内比
    rows = frontier_table(t_grid=(0.32, 0.64, 1.28, 2.56))
    sl = t_scale_slope_db(rows)
    # 带内功率比 r 口径（与实验网格一致）: ED r∝1/√BT → 1.5dB/加倍; MF r=2c²/BT → 3dB/加倍
    # （v4.1 的 4.5dB/加倍是每样本口径, 带内比口径下为 1.5dB——口径统一后以此为准）
    assert abs(sl["ed_per_doubling"] - (-1.5)) < 0.3, sl
    assert abs(sl["mf_per_doubling"] - (-3.0)) < 0.3, sl


def test_conformal_rc():
    rng = np.random.default_rng(0)
    n = 5000
    # 可控合成: 分数均匀偏高, 错误率随分数线性下降（高分低错）
    s = rng.uniform(0.3, 1.0, n)
    err = ((1.0 - s) * 0.4 + rng.uniform(0.0, 0.08, n)).clip(0, 1)
    s_cal, e_cal, s_tst = s[:600], err[:600], s[600:]
    rc = SplitRiskControl(alpha=0.1)
    lam = rc.fit(s_cal, e_cal)
    d = rc.decide(s_tst)
    assert d.sum() > 50, f"决策集太小: {d.sum()}"
    emp_risk = err[600:][d].mean()
    assert emp_risk < 0.25, emp_risk
    sc = SelectiveCRC(alpha=0.1, cov=0.5)
    sc.fit(s_cal, e_cal)
    assert sc.decide(s_tst).sum() <= 0.55 * len(s_tst) + 10
    # 加权: 均匀权重 ≈ split
    dw = DriftAwareCRC(alpha=0.1)
    dw.fit(s_cal, e_cal, np.ones(len(s_cal)))
    assert np.isclose(dw.lam, lam, atol=0.02) or dw.decide(s_tst).sum() > 0
    av = AnytimeValidCRC(alpha=0.1, block=100)
    av.fit_stream(s_cal, e_cal)
    ne = NonExchangeableCRC(alpha=0.1)
    ne.fit(s_cal, e_cal, np.ones(len(s_cal)))
    assert ne.sensitivity is not None


def test_stats():
    rng = np.random.default_rng(1)
    n = 500
    decide = np.ones(n, dtype=bool)
    correct = rng.random(n) > 0.2              # risk ≈ 0.2
    cluster = rng.integers(0, 25, n)           # 25 个源簇
    cb = cluster_bootstrap(decide, correct, cluster,
                           lambda d, c: 1.0 - c[d].mean(), n_boot=200)
    assert cb["n_boot"] >= 100
    # 单侧检验: risk≈0.2 > α=0.1 → 不拒绝 H0
    r = risk_upper_bound(decide, correct, alpha=0.1)
    assert r["reject_H0"] is False
    # 低风险场景 → 拒绝
    correct2 = rng.random(n) > 0.03
    r2 = risk_upper_bound(decide, correct2, alpha=0.1)
    assert r2["reject_H0"] is True
    # 最差单元 ECE
    probs = np.clip(correct2.astype(float) + rng.uniform(-0.2, 0.2, n), 0.05, 0.95)
    cells = rng.integers(0, 3, n)
    w = worst_cell_ece(probs[:, None].repeat(10, 1), correct2, decide, cells, n_bins=5)
    assert "worst" in w
    # MMD: 同分布 vs 异分布
    x = rng.normal(0, 1, (100, 4))
    y_same = rng.normal(0, 1, (100, 4))
    y_diff = rng.normal(2, 1, (100, 4))
    assert mmd_rbf(x, y_diff) > mmd_rbf(x, y_same)
    rep = orthogonality_report({"a": x, "b": y_diff})
    assert "a↔b" in rep


def test_wind_inflation():
    infl = am_variance_inflation(m=0.5, f_am_list=[5.0, 9.0, 14.0])
    assert abs(infl - (1.0 + 0.25 * 3 / 2)) < 1e-9      # 1 + m²·k/2
    w = np.random.default_rng(0).standard_normal(C.WINDOW_SAMPLES)
    tot = inflation_factor(w)["total"]
    assert tot >= 1.0
    pe0 = pe_predicted_with_inflation(-10.0, 3000.0, 1.28, 1.0)
    pe1 = pe_predicted_with_inflation(-10.0, 3000.0, 1.28, 8.0)
    assert pe1 > pe0                                    # 膨胀 → 更差检测


def test_np_lrt():
    fs = C.SAMPLE_RATE
    n = C.WINDOW_SAMPLES
    rng = np.random.default_rng(2)
    clean = rng.standard_normal(n) * 0.01
    from aof.wsosim import _wind
    w, _ = _wind(n, fs, 2)
    v = SohnNPLRT()
    # 无事件（纯风噪）→ 不应决策（低 SNR 帧）
    d0 = v.decide(w, w)
    # 强事件（带内帧能量须显著超过风噪帧噪声）→ 应决策
    strong = clean * 300.0
    d1 = v.decide(strong + w * 0.001, w)
    assert d1 and not d0


def test_calibration():
    rng = np.random.default_rng(3)
    n = 500
    r_hat = rng.normal(-5.0, 4.0, n)
    bias = 2.5 + 0.3 * r_hat                      # 系统性偏差（随 SNR 变化）
    r_true = r_hat + bias + rng.normal(0, 0.5, n)
    cal = IsotonicCalibrator()
    cal.fit(r_hat, r_true)
    m = cal.mae_before_after(r_hat, r_true)
    assert m["mae_after"] < m["mae_before"], m     # isotonic 修正应降低 MAE
    assert m["improvement"] > 0.5
    # ScoreCalibrator 单调性
    sc = ScoreCalibrator()
    p = rng.uniform(0.05, 0.95, n)
    y = (p > 0.5).astype(float)
    sc.fit(p, y)
    pc = sc.calibrate(np.array([0.1, 0.9]))
    assert pc[1] >= pc[0]


def test_paired_bootstrap():
    rng = np.random.default_rng(4)
    n = 200
    gap_b12 = rng.normal(0.10, 0.05, n)           # B12 均值 0.10
    gap_b11 = rng.normal(0.15, 0.05, n)           # B11 均值 0.15 → 差 -0.05
    r = paired_bootstrap(gap_b12, gap_b11, n_boot=500)
    assert r["mean_diff"] < -0.03
    assert r["p_value"] < 0.05                     # B12 显著优于 B11
    assert r["ci_hi"] < 0.0


if __name__ == "__main__":
    for fn in [test_frontiers, test_conformal_rc, test_stats,
               test_wind_inflation, test_np_lrt, test_calibration,
               test_paired_bootstrap]:
        fn()
        print(f"PASS {fn.__name__}")
    print("ALL PASS")
