"""exp_p1_class_snr_heatmap.py — R20 前置探针 P1: per-class × SNR-bin 精度热图
定位分类器瓶颈: "所有类在低 SNR 端一起差"（→ SNR 分层加权, 方案 A）vs
"特定类在所有 SNR 都差"（→ 标签噪声/类内多样性天花板, 方案 A 无效）。
口径: B12 模型 raw 预测（argmax over [0..8], 坑 15）per (class, SNR bin) recall;
B13 物理决策域（r_true ≥ r_min 操作边界 −11.1dB）内 acc@dec 同表对比。
用法: python scripts/exp_p1_class_snr_heatmap.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule, r_min_theory
from aof.evaluate import evaluate_system
from aof.model import SONTRA_A
from run_main import load_fsd50k_clips, sample_clips

from aof.mapping import CLASS_NAMES as MAPPING_CLASS_NAMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
CLASS_NAMES = MAPPING_CLASS_NAMES[:9]   # 真实类顺序（mapping.py 冻结）, 勿手写


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
    va3 = [(sampler._best_window(x, C.WINDOW_SAMPLES), idx_map[f], f)
           for x, f in va_clips if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]

    model = SONTRA_A()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    recs = evaluate_system(model, AFRule(), va3, kinds, snrs, device="mps", tau=0.0)
    print(f"recs: {len(recs)}", flush=True)

    r_min_db = 10 * np.log10(r_min_theory(AFRule().alpha_k, 7680))   # 操作边界 −11.1dB

    # per (class, SNR bin) recall: 该类在 labels 中的窗里, 预测正确（argmax ∈ labels）的比例
    # raw_correct = 模型 raw argmax 正确（与决策规则无关）
    heat = np.zeros((9, len(snrs)))
    cnt = np.zeros((9, len(snrs)))
    bin_acc = []
    for bi, s in enumerate(snrs):
        rr = [r for r in recs if abs(r["snr_db"] - s) < 0.01]
        dec = np.array([r["decide"] for r in rr])
        corr = np.array([r["correct"] for r in rr])
        bin_acc.append({"snr": s, "n": len(rr),
                        "B13_cov": float(dec.mean()),
                        "B13_acc_at_dec": float(corr[dec].mean()) if dec.sum() else float("nan")})
        for r in rr:
            probs = r["event_probs"]
            pred = int(probs[:9].argmax())
            labels = set(r.get("labels") or [])
            for k in range(9):
                if k in labels:
                    cnt[k, bi] += 1
                    if pred in labels:
                        heat[k, bi] += 1
    recall = np.divide(heat, np.where(cnt == 0, 1, cnt))
    recall[cnt == 0] = float("nan")

    # 决策域低端 vs 高端的全局对比
    low_bins = [i for i, s in enumerate(snrs) if s >= -13.7 and s < 5]
    high_bins = [i for i, s in enumerate(snrs) if s >= 5]
    print("\nper-class recall by SNR bin (rows=class, cols=-25/-15/-5/5/15dB):", flush=True)
    hdr = "        " + "".join(f"{s:>7.0f}" for s in snrs)
    print(hdr, flush=True)
    for k in range(9):
        row = "".join(f"{recall[k, i] * 100:>7.1f}" if not np.isnan(recall[k, i]) else "     --"
                      for i in range(len(snrs)))
        lo = np.nanmean(recall[k, low_bins]) * 100 if cnt[k, low_bins].sum() else float("nan")
        hi = np.nanmean(recall[k, high_bins]) * 100 if cnt[k, high_bins].sum() else float("nan")
        print(f"{CLASS_NAMES[k]:>8s}{row}   low={lo:.1f}  high={hi:.1f}", flush=True)

    print("\nB13 decision-domain per-bin:", flush=True)
    for b in bin_acc:
        print(f"  {b['snr']:>5.0f}dB  n={b['n']}  cov={b['B13_cov']:.3f}  acc@dec={b['B13_acc_at_dec']:.3f}", flush=True)

    out = {"per_class_recall": recall.tolist(), "class_names": CLASS_NAMES,
           "snr_bins": snrs, "bin_stats": bin_acc,
           "r_min_op_db": round(float(r_min_db), 2)}
    with open(os.path.join(OUT, "exp_p1_heatmap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE → outputs/exp_p1_heatmap.json")


if __name__ == "__main__":
    main()
