"""exp_conformal_hybrid.py — conformal-within-decidable hybrid（2026-08-16, decisive check #3）

问题: 论文 §4.7 显示全域校准的 split-conformal（AGRC）在混合 SNR 校准下塌缩
      （λ=1.0, coverage 0.011）。论文 Discussion 列出的 immediate extension:
      只在物理可决策子集内校准 → 得到带分布自由保证的 AF-Rule 变体。
      本脚本实现并测量该混合法, 回答: 校准域受限后 conformal 是否恢复有效覆盖?

协议（对齐 exp_conformal_baselines.py 的分割与分数契约）:
  数据: FSD50K-10 val 500 clips（fname 分层, seed C.SEED+1）→ 300 校准 + 200 测试;
        窗口 = SONTRA-A(B12, sontra_a_ep22.pt) 的 evaluate_system 记录。
  分数: max event prob（排除 unknown, 与基线一致）。
  物理门: 校准/测试都先按 r_true ≥ r_min 过滤出"可决策域"（oracle SNR 界定域）;
          SplitRiskControl 只在校准域上拟合 λ。
  测试变体:
    (a) ideal-gate: decide = (r_true ≥ r_min) AND score ≥ λ   ← 校准侧增益上界
    (b) deployable: decide = (B11 SNR̂ 物理门通过) AND score ≥ λ ← 可部署版
  基线: AGRC 全域（塌缩点 0.011）; AF-Rule B12 τ=0.5（0.346）; oracle（−0.100）。
判定（输出打印）: hybrid coverage 明显 > AGRC 全域且 risk ≤ α（门内校准的意义成立）
  → 可写入论文（limitation (v) 部分解决, 补"校准域受限"小节）; 否则不写。

用法: python scripts/exp_conformal_hybrid.py
      [--ckpt outputs/checkpoints/sontra_a_ep22.pt --n-val 500 --device mps --tag 20260816]
输出: outputs/exp_conformal_hybrid_{tag}.json
注意: 新增实验记录（非 R19 冻结 pass）; 复用 R19 冻结 ckpt 做推理不违反冻结纪律
      （不重跑 R19 已报告数字; 本实验输出独立 JSON）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.baselines import SPAnchorB11
from aof.conformal_rc import SplitRiskControl
from aof.evaluate import evaluate_system
from aof.metrics import operating_gap, coverage
from aof.model import SONTRA_A
from run_main import load_fsd50k_clips, sample_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "checkpoints", "sontra_a_ep22.pt"))
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="20260816")
    args = ap.parse_args()
    snrs = [-25.0, -15.0, -5.0, 5.0, 15.0]
    kinds = ["wind", "occlusion", "self_motion"]

    from aof.mapping import build_dev_index
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], args.n_val, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}
    va = [(f, idx_map[f]) for f, _ in val_sel]
    va_clips = [(x, f) for x, f in load_fsd50k_clips([f for f, _ in va])]
    from aof.cf_sampler import CFSampler
    sampler = CFSampler([])
    va_clips = [(sampler._best_window(x, C.WINDOW_SAMPLES), idx_map[f], f)
                for x, f in va_clips
                if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]

    model = SONTRA_A()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    rule = AFRule()
    recs = evaluate_system(model, rule, va_clips, kinds, snrs, device=args.device)
    print(f"evaluated {len(recs)} windows", flush=True)

    # 分割: 300 校准 + 200 测试（同 clip 不跨集）
    fnames = sorted({r["fname"] for r in recs})
    rng = np.random.default_rng(C.SEED)
    rng.shuffle(fnames)
    cal_fn, tst_fn = set(fnames[:args.n_cal]), set(fnames[args.n_cal:])
    cal, tst = [r for r in recs if r["fname"] in cal_fn], [r for r in recs if r["fname"] in tst_fn]

    def score(r):
        return float(r["event_probs"][: C.N_CLASSES - 1].max())

    rmin_db = 10 * np.log10(rule.r_min)

    # 全域 AGRC（复现塌缩, 基线）
    rc_full = SplitRiskControl(alpha=C.ALPHA)
    rc_full.fit(np.array([score(r) for r in cal]),
                np.array([0.0 if r["correct"] else 1.0 for r in cal]))
    lam_full = rc_full.lam
    decide_full = rc_full.decide(np.array([score(r) for r in tst]))
    correct_full = np.array([r["correct"] for r in tst])
    gap_f, risk_f = operating_gap(decide_full, correct_full, C.ALPHA)

    # 混合: 只在校准域（r_true ≥ r_min）上拟合
    cal_dec = [r for r in cal if r["r_true_db"] >= rmin_db]
    rc_hy = SplitRiskControl(alpha=C.ALPHA)
    lam_hy = rc_hy.fit(np.array([score(r) for r in cal_dec]),
                       np.array([0.0 if r["correct"] else 1.0 for r in cal_dec]))
    print(f"AGRC 全域 λ={lam_full:.3f} (cal {len(cal)} 窗) | "
          f"hybrid λ={lam_hy:.3f} (cal 域 {len(cal_dec)} 窗)", flush=True)

    # 测试变体 (a) ideal-gate
    s_tst = np.array([score(r) for r in tst])
    r_true = np.array([r["r_true_db"] for r in tst])
    phys_true = r_true >= rmin_db
    decide_a = phys_true & (s_tst >= lam_hy)
    correct_all = np.array([r["correct"] for r in tst])
    gap_a, risk_a = operating_gap(decide_a, correct_all, C.ALPHA)

    # 测试变体 (b) deployable: B11 SNR̂ 物理门
    b11 = SPAnchorB11()
    snr_hat = np.array([r["snr_hat_db"] for r in tst])
    r_lin = 10.0 ** (snr_hat / 10.0)
    d = r_lin * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r_lin))
    pd = 0.5 * (1.0 + erf(d / np.sqrt(2.0)))
    pd_ok = pd >= (1.0 - rule.alpha_k)
    decide_b = pd_ok & (s_tst >= lam_hy)
    gap_b, risk_b = operating_gap(decide_b, correct_all, C.ALPHA)

    # 基线: AF-Rule B12 τ=0.5（同表）
    best = np.array([score(r) for r in tst])
    decide_rule = pd_ok & (best > 0.5)
    gap_r, risk_r = operating_gap(decide_rule, correct_all, C.ALPHA)

    out = {"tag": args.tag, "ckpt": os.path.basename(args.ckpt), "seed": C.SEED,
           "n_cal": len(cal), "n_cal_decidable": len(cal_dec), "n_test": len(tst),
           "lambda_full_agrc": round(float(lam_full), 4) if np.isfinite(lam_full) else None,
           "lambda_hybrid": round(float(lam_hy), 4) if np.isfinite(lam_hy) else None,
           "rows": {
               "agrc_full_domain": {"gap": round(gap_f, 3), "risk": round(risk_f, 3),
                                    "coverage": round(coverage(decide_full), 3)},
               "hybrid_ideal_gate": {"gap": round(gap_a, 3), "risk": round(risk_a, 3),
                                     "coverage": round(coverage(decide_a), 3)},
               "hybrid_deployable": {"gap": round(gap_b, 3), "risk": round(risk_b, 3),
                                     "coverage": round(coverage(decide_b), 3)},
               "af_rule_b12_tau05": {"gap": round(gap_r, 3), "risk": round(risk_r, 3),
                                     "coverage": round(coverage(decide_rule), 3)}},
           "note": ("校准域 = r_true ≥ r_min（oracle SNR 界定物理可决策域）; "
                    "ideal-gate 用 r_true（校准侧增益上界）, deployable 用 B11 SNR̂"
                    "（可部署版, 门误差进入风险）")}
    with open(os.path.join(OUT, f"exp_conformal_hybrid_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["rows"], indent=2), flush=True)
    print(f"DONE → outputs/exp_conformal_hybrid_{args.tag}.json")


if __name__ == "__main__":
    main()
