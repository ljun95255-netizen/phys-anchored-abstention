"""exp_b13_snr_only.py — B13: 物理阈值 + A-Head SNR̂（弃权完全交给 SNR̂, 分类器只报类别）
缺口分解指出的修复路径: B11a（真实阈值+SP 估计器）= oracle; B13 用学习估计器（A-Head,
SNR MAE 0.1dB）替代 SP → 若逼近 oracle 则"分解→修复→物理极限"闭环。
实现: AFRule(tau=0.0) → probs>0 恒真 → eligible = P_D(SNR̂)≥1−α_k（纯 SNR 决策）;
      类别 = 可决策类概率 argmax。对照: oracle / B11a / B12(τ=0.5)。
用法: python scripts/exp_b13_snr_only.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.evaluate import evaluate_system
from aof.metrics import operating_gap, coverage, rank_auc
from aof.model import SONTRA_A
from run_main import load_fsd50k_clips, sample_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def main():
    ckpt = os.path.join(OUT, "checkpoints", "sontra_a_ep22.pt")
    snrs = [-25.0, -15.0, -5.0, 5.0, 15.0]
    kinds = ["wind", "occlusion", "self_motion"]

    from aof.mapping import build_dev_index
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], 500, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}
    va_clips = [(x, f) for x, f in load_fsd50k_clips([f for f, _ in val_sel])]
    from aof.cf_sampler import CFSampler
    sampler = CFSampler([])
    va_clips = [(sampler._best_window(x, C.WINDOW_SAMPLES), idx_map[f], f)
                for x, f in va_clips
                if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]

    model = SONTRA_A()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    print(f"ckpt: {os.path.basename(ckpt)}  va_clips: {len(va_clips)}", flush=True)

    rule_b13 = AFRule()                       # τ=0.0 → 纯 SNR 决策
    recs = evaluate_system(model, rule_b13, va_clips, kinds, snrs,
                           device="mps", tau=0.0)
    dec = np.array([r["decide"] for r in recs])
    corr = np.array([r["correct"] for r in recs])
    gap, risk = operating_gap(dec, corr, C.ALPHA)
    acc = float(corr[dec].mean()) if dec.sum() else float("nan")
    out = {"B13_snr_only": {"gap": round(gap, 3), "risk": round(risk, 3),
                            "coverage": round(coverage(dec), 3),
                            "acc_at_dec": round(acc, 3), "n_decide": int(dec.sum()),
                            "n_total": len(dec)},
           "compare": {"oracle": -0.100, "B11a": -0.100, "B12_tau05": 0.351,
                       "B12_tau095": 0.184},
           "note": "B13 = AFRule(tau=0.0): 弃权仅由 P_D(A-Head SNR̂)≥1−α_k 决定, 无概率阈值"}
    with open(os.path.join(OUT, "exp_b13.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"DONE → outputs/exp_b13.json")


if __name__ == "__main__":
    main()
