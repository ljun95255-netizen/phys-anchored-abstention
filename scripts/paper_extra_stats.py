"""paper_extra_stats.py — 期刊版加厚所需补充统计（2026-08-07, 全部真实计算）
1. FSD50K-10 每类 clip 计数（build_dev_index: train/val）
2. SC-10 每词计数（官方划分 train/val/test）
3. US8K 每类计数（官方 CSV, fold 1-10 总量）
4. T2O 数值实例（af_rule.t2o, 有代表性 per-sample SNR）
5. SC-10/US8K 逐类 acc@dec 摘要（heatmap JSON 聚合）
输出: outputs/paper_extra_stats.json
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def main():
    out = {}

    # 1. FSD50K-10 每类计数（train/val 划分）
    from aof.mapping import build_dev_index
    index = build_dev_index()  # [(fname, y, split)]
    from collections import Counter
    tr = Counter(); va = Counter()
    for f, y, s in index:
        names = [i for i, v in enumerate(y) if v]
        for i in names:
            (tr if s == "train" else va)[i] += 1
    out["fsd50k10_per_class"] = {
        "train": {k: tr[k] for k in sorted(tr)}, "val": {k: va[k] for k in sorted(va)}}
    out["fsd50k10_total"] = {"train": sum(tr.values()), "val": sum(va.values()),
                             "clips": len(index)}

    # 2. SC-10 每词计数
    sys.path.insert(0, os.path.join(OUT, "..", "scripts"))
    from exp_sc10 import load_sc_index
    sc_idx = load_sc_index()
    sc_counts = Counter(); sc_split = Counter()
    for f, y, s in sc_idx:
        w = f.split("/")[0] if "/" in f else str(int(max(range(len(y)), key=lambda i: y[i])))
        sc_counts[w] += 1
        sc_split[s] += 1
    out["sc10_per_word"] = dict(sc_counts)
    out["sc10_split"] = dict(sc_split)

    # 3. US8K 每类计数（官方 CSV）
    import csv
    us8k_csv = os.environ.get("US8K_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "urbansound8k", "UrbanSound8K", "metadata", "UrbanSound8K.csv"))
    us_counts = Counter()
    with open(us8k_csv) as f:
        for r in csv.DictReader(f):
            us_counts[r["class"]] += 1
    out["us8k_per_class"] = dict(us_counts)
    out["us8k_total"] = sum(us_counts.values())

    # 4. T2O 实例（af_rule.t2o: alpha=0.1, B=3kHz, rho=1dB）
    from aof.af_rule import t2o, snr_wall
    from aof import config as C
    band = C.EVENT_BAND[1] - C.EVENT_BAND[0]
    wall = snr_wall(C.SNR_WALL_RHO_DB)
    t2o_examples = {}
    for snr_per_db in [-35, -30, -27, -25, -20, -15, -10]:
        t = t2o(C.ALPHA, snr_per_db, band, C.SNR_WALL_RHO_DB)
        t2o_examples[str(snr_per_db)] = None if math.isinf(t) else round(t, 2)
    out["t2o"] = {"wall_per_sample_db": round(wall, 2), "examples_s": t2o_examples,
                  "band_hz": band, "alpha": C.ALPHA, "rho_db": C.SNR_WALL_RHO_DB}

    # 5. 逐类 acc@dec（heatmap 聚合: 决策域 -5/+5/+15dB 合并）
    for ds, fname in (("SC10", "exp_sc10_eval.json"), ("US8K", "exp_us8k_eval.json")):
        js = json.load(open(os.path.join(OUT, fname)))
        per_class = {}
        for cls, cells in js["heatmap"].items():
            tot_n = tot_ok = 0
            for snr in ["-5", "5", "15"]:
                c = cells[snr]
                if c["acc_at_dec"] is not None:
                    tot_n += c["n_decide"]
                    tot_ok += c["acc_at_dec"] * c["n_decide"]
            per_class[cls] = {"n_dec": tot_n,
                              "acc": round(tot_ok / tot_n, 3) if tot_n else None}
        out[f"{ds.lower()}_per_class_acc"] = per_class

    with open(os.path.join(OUT, "paper_extra_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:2200])


if __name__ == "__main__":
    main()
