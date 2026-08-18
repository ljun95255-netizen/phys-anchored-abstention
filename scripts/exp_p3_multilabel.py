"""exp_p3_multilabel.py — P3: 多标签语义 vs 分类器真实上限（物理决策域）
1. 多标签窗占比（argmax 单报在多标签窗天然漏报 → recall 上界被压）
2. 单标签窗 acc@dec（排除多标签混淆后分类器真实上限）
3. top-2/top-3 命中率（多标签输出的潜在上限）
4. per-class 单标签精度（坏类是数据问题还是全局问题）
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule, r_min_theory
from aof.evaluate import evaluate_system
from aof.model import SONTRA_A
from run_main import load_fsd50k_clips, sample_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def main():
    snrs = [-25.0, -15.0, -5.0, 5.0, 15.0]
    kinds = ["wind", "occlusion", "self_motion"]
    from aof.mapping import build_dev_index, CLASS_NAMES
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], 500, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}
    va_clips = [(x, f) for x, f in load_fsd50k_clips([f for f, _ in val_sel])]
    from aof.cf_sampler import CFSampler
    sampler = CFSampler([])
    va3 = [(sampler._best_window(x, C.WINDOW_SAMPLES), idx_map[f], f)
           for x, f in va_clips if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]

    model = SONTRA_A()
    model.load_state_dict(torch.load(os.path.join(OUT, "checkpoints", "sontra_a_ep22.pt"),
                                     map_location="cpu"), strict=False)
    recs = evaluate_system(model, AFRule(), va3, kinds, snrs, device="mps", tau=0.0)
    print(f"recs: {len(recs)}", flush=True)

    rmin = r_min_theory(AFRule().alpha_k, 7680)
    phys = [r for r in recs if r["r_true_db"] >= 10 * np.log10(rmin)]
    print(f"物理决策域窗: {len(phys)}", flush=True)

    nlab = np.array([len(r["labels"]) for r in phys])
    multi_frac = float((nlab >= 2).mean())
    print(f"\n标签数分布: 0标签={(nlab == 0).mean():.3f}  1标签={(nlab == 1).mean():.3f}  "
          f"2+标签={multi_frac:.3f}", flush=True)

    single = [r for r in phys if len(r["labels"]) == 1]
    dec = np.array([r["decide"] for r in single])
    corr = np.array([r["correct"] for r in single])
    print(f"\n单标签窗: n={len(single)}  cov={dec.mean():.3f}  "
          f"acc@dec={corr[dec].mean():.3f}  (n_dec={int(dec.sum())})", flush=True)

    def topk_hit(r, k):
        preds = np.argsort(r["event_probs"][:9])[::-1][:k]
        return bool(set(preds) & set(r["labels"]))

    dphys = [r for r in phys if r["decide"]]
    tk2 = np.mean([topk_hit(r, 2) for r in dphys])
    tk3 = np.mean([topk_hit(r, 3) for r in dphys])
    tk1 = np.mean([r["correct"] for r in dphys])
    print(f"\n物理决策域 top-1/top-2/top-3 命中: {tk1:.3f} / {tk2:.3f} / {tk3:.3f}", flush=True)

    cls_acc, cls_n = defaultdict(float), defaultdict(int)
    for r in single:
        if not r["decide"]:
            continue
        k = r["labels"][0]
        cls_n[k] += 1
        cls_acc[k] += 1 if r["correct"] else 0
    print("\n单标签物理域 per-class acc@dec:")
    for k in sorted(cls_n):
        print(f"  {k} {CLASS_NAMES[k]}: {cls_acc[k] / cls_n[k]:.3f}  (n={cls_n[k]})", flush=True)

    out = {"multi_label_frac_phys": round(multi_frac, 3),
           "single_label_acc_at_dec": round(float(corr[dec].mean()), 3) if dec.sum() else None,
           "top1_hit": round(float(tk1), 3), "top2_hit": round(float(tk2), 3),
           "top3_hit": round(float(tk3), 3)}
    with open(os.path.join(OUT, "exp_p3_multilabel.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nDONE → outputs/exp_p3_multilabel.json")


if __name__ == "__main__":
    main()
