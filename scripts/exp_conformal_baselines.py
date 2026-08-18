"""exp_conformal_baselines.py — v3 §11.4 conformal 基线接入 gap 矩阵（R19 冻结）
协议: 500 val clips 按 fname 分 300 校准 + 200 测试（同 clip 不跨集）;
分数 = max event prob（模型输出, 不经 AF-Rule）; 校准集拟合 λ → 测试集决策 → gap/risk/cov。
基线: SplitRiskControl（AGRC 简化）/ SelectiveCRC（SCRC 简化）/ MC-Dropout 对照。
消融: 逐腐蚀族 gap（wind/occlusion/self_motion）——同一次评估输出。
用法: python scripts/exp_conformal_baselines.py
      --ckpt outputs/checkpoints/sontra_a_ep22.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.conformal_rc import SplitRiskControl, SelectiveCRC
from aof.evaluate import evaluate_system
from aof.metrics import operating_gap, coverage
from aof.model import SONTRA_A
from run_main import load_fsd50k_clips, sample_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "checkpoints", "sontra_a_ep22.pt"))
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--snrs", default="-25,-15,-5,5,15")
    args = ap.parse_args()
    snrs = [float(s) for s in args.snrs.split(",")]
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
    print(f"ckpt: {os.path.basename(args.ckpt)}  va_clips: {len(va_clips)}", flush=True)

    recs = evaluate_system(model, rule, va_clips, kinds, snrs, device=args.device)
    print(f"evaluated {len(recs)} windows", flush=True)

    # 按 fname 分组: 300 校准 + 200 测试
    fnames = sorted({r["fname"] for r in recs})
    rng = np.random.default_rng(C.SEED)
    rng.shuffle(fnames)
    cal_fn, tst_fn = set(fnames[:300]), set(fnames[300:])
    cal = [r for r in recs if r["fname"] in cal_fn]
    tst = [r for r in recs if r["fname"] in tst_fn]
    print(f"cal clips {len(cal_fn)} / tst clips {len(tst_fn)}; recs {len(cal)}/{len(tst)}", flush=True)

    def score(r):
        p = r["event_probs"]
        return float(p[: C.N_CLASSES - 1].max())          # 排除 unknown 类

    s_cal = np.array([score(r) for r in cal])
    e_cal = np.array([0.0 if r["correct"] else 1.0 for r in cal])
    s_tst = np.array([score(r) for r in tst])
    e_tst = np.array([0.0 if r["correct"] else 1.0 for r in tst])

    res = {}
    for name, rc in [("AGRC_split", SplitRiskControl(alpha=C.ALPHA)),
                     ("SCRC_selective", SelectiveCRC(alpha=C.ALPHA, cov=0.5))]:
        lam = rc.fit(s_cal, e_cal)
        d = rc.decide(s_tst)
        gap, risk = operating_gap(d, ~e_tst.astype(bool), C.ALPHA)
        res[name] = {"lambda": float(lam), "gap": gap, "risk": risk,
                     "coverage": coverage(d), "n_decide": int(d.sum())}
        print(f"  [{name}] lam={lam:.3f} gap={gap:.3f} risk={risk:.3f} cov={coverage(d):.3f}", flush=True)

    # 逐腐蚀族消融（B0/B11/B12 在各自 kind 上的 gap; B12 用 recs 的 AF-Rule 决策）
    per_kind = {}
    for k in kinds:
        rr = [r for r in recs if r["kind"] == k]
        dec = np.array([r["decide"] for r in rr])
        corr = np.array([r["correct"] for r in rr])
        gap12, risk12 = operating_gap(dec, corr, C.ALPHA)
        per_kind[k] = {"B12_gap": round(gap12, 3), "B12_risk": round(risk12, 3),
                       "n": len(rr)}
    res["per_kind_ablation"] = per_kind
    print(f"  per-kind: {per_kind}", flush=True)

    with open(os.path.join(OUT, "exp_conformal.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"DONE → outputs/exp_conformal.json")


if __name__ == "__main__":
    main()
