"""exp_p4_clean_ceiling.py — P4: 干净域（无腐蚀）任务天花板探针
R20 关键分叉: 若干净域 top-1 命中 ~60% → FSD50K-10 任务本身天花板低（标签近似/类内异构/
1.28s 窗语义）, 82.5% 目标不可达 → 需任务定义修正或目标重设; 若干净域 85%+ → 腐蚀域是问题。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.evaluate import _to_mel
from aof.model import SONTRA_A
from run_main import load_fsd50k_clips, sample_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def main():
    from aof.mapping import build_dev_index
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], 500, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}
    va_clips = [(x, f) for x, f in load_fsd50k_clips([f for f, _ in val_sel])]
    from aof.cf_sampler import CFSampler
    sampler = CFSampler([])
    va3 = [(sampler._best_window(x, C.WINDOW_SAMPLES), idx_map[f], f)
           for x, f in va_clips if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]
    print(f"clips: {len(va3)}", flush=True)

    model = SONTRA_A()
    model.load_state_dict(torch.load(os.path.join(OUT, "checkpoints", "sontra_a_ep22.pt"),
                                     map_location="cpu"), strict=False)
    model = model.to("mps").eval()

    hit1 = hit2 = hit3 = n = 0
    per_class = {}
    with torch.no_grad():
        for x, y, f in va3:
            labels = set()
            for i, v in enumerate(y):
                if v and i < C.N_CLASSES - 1:
                    labels.add(i)
            if not labels:
                continue
            mel = torch.from_numpy(x).unsqueeze(0).float().to("mps")
            mel = _to_mel(mel)
            p = model(mel)["event_probs"][0].cpu().numpy()[:9]
            order = np.argsort(p)[::-1]
            n += 1
            hit1 += 1 if order[0] in labels else 0
            hit2 += 1 if len(set(order[:2]) & labels) else 0
            hit3 += 1 if len(set(order[:3]) & labels) else 0
            for k in labels:
                per_class.setdefault(k, [0, 0])[1] += 1
                per_class[k][0] += 1 if order[0] in labels else 0

    print(f"\n干净域（无腐蚀）: n={n}", flush=True)
    print(f"  top-1 命中: {hit1/n:.3f}   top-2: {hit2/n:.3f}   top-3: {hit3/n:.3f}", flush=True)
    from aof.mapping import CLASS_NAMES
    print("  per-class top-1 recall:", flush=True)
    for k in sorted(per_class):
        a, b = per_class[k]
        print(f"    {k} {CLASS_NAMES[k]}: {a/b:.3f} (n={b})", flush=True)

    res = {"n": n, "top1": round(hit1 / n, 3), "top2": round(hit2 / n, 3),
           "top3": round(hit3 / n, 3),
           "per_class_top1": {str(k): round(a / b, 3) for k, (a, b) in per_class.items()}}
    import json
    with open(os.path.join(OUT, "exp_p4_clean_ceiling.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("DONE → outputs/exp_p4_clean_ceiling.json")


if __name__ == "__main__":
    main()
