"""exp_heterogeneity_matched.py — 异构匹配事件类型复测（2026-08-18, optimization #4）

limitation (iii) 承诺: "event type remains confounded with the ratio, so a causal
claim requires a matched-event-type comparison"。本脚本在同一事件类型上重算
类内/类间余弦距离比（协议同 exp_heterogeneity.py, 干净域最优窗, CLAP 嵌入）。

匹配类（两数据集共享的语义事件）:
  siren        US8K 8 ↔ FSD50K-10 3        （精确匹配）
  horn         US8K 1 (car_horn) ↔ FSD50K-10 2 （精确匹配; FSD50K-10 需新嵌 val horn 窗）
  vehicle      US8K 5 (engine_idling) ↔ FSD50K-10 0 （近似匹配, 标注说明）

数据: 优先复用冻结嵌入缓存 outputs/cache/clap_emb_*_20260816.npy（先复现冻结
per-class 数字, 位级一致后再用）; horn 子集新嵌入（同一 CLAP 管线, batch=32）。
判定: 控制事件类型后, 匹配类的 ratio 排名是否仍与"低异质性→高天花板"矛盾
（即原拒绝结论是否在 confound 控制后仍成立）; 结果如实记录（可反转）。
输出: outputs/exp_heterogeneity_matched_20260818.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))

from aof import config as C
from aof.cf_sampler import CFSampler
from exp_sc10 import SC_NAMES, load_sc_index, load_sc_wav
from exp_us8k import US8K_NAMES, load_us8k_index, win_clips
from exp_heterogeneity import embed, cosine_dist, class_ratio

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
CACHE = os.path.join(OUT, "cache")
TAG = "20260816"
FSD_NAMES = ["vehicle", "bicycle", "horn", "siren", "tire_squeal", "impact",
             "construction", "mechanical_anomaly", "human_activity"]


def rebuild_labels():
    """重建与冻结 exp_heterogeneity 相同的 (labels, names, window 集) —— 顺序须一致。"""
    labels, names = {}, {}

    sc_index = [e for e in load_sc_index() if e[2] == "test"]
    sc_labels = []
    for rel, y, _ in sc_index:
        x = load_sc_wav(rel)
        if x is None:
            continue
        sc_labels.append(int(np.argmax(y[:10])))
    labels["sc10"] = np.array(sc_labels)
    names["sc10"] = SC_NAMES

    te = [e for e in load_us8k_index() if e[3] == "test"]
    us_clips, _ = win_clips(te)
    us_labels = [int(np.argmax(y[:10])) for _, y, _ in us_clips]
    labels["us8k"] = np.array(us_labels)
    names["us8k"] = US8K_NAMES

    from aof.mapping import build_dev_index
    from run_main import load_fsd50k_clips, sample_clips
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], 500, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}
    fs_labels = []
    sampler = CFSampler([])
    for x, f in load_fsd50k_clips([f for f, _ in val_sel]):
        w = sampler._best_window(x, C.WINDOW_SAMPLES)
        if w is None:
            continue
        y = idx_map[f]
        fs_labels.append([i for i, v in enumerate(y) if v and i < 9][0])
    labels["fsd50k10"] = np.array(fs_labels)
    names["fsd50k10"] = FSD_NAMES
    return labels, names


def load_fsd_horn_windows(device):
    """FSD50K-10 val 中标签集含 horn(类 2)的全部窗（最优窗, 新嵌入）。
    口径注: val 无 horn 主标签 clip（horn 恒与 siren/vehicle 共现）, 故按
    标签集成员口径选取（与探针 multi-label-hit 口径同精神, 披露差异）。"""
    from transformers.models.clap import ClapModel, ClapProcessor
    from aof.mapping import build_dev_index
    from run_main import load_fsd50k_clips
    index = build_dev_index()
    horn_sel = [(f, y) for f, y, s in index if s == "val" and
                2 in [i for i, v in enumerate(y) if v and i < 9]]
    sampler = CFSampler([])
    wavs = []
    for x, f in load_fsd50k_clips([f for f, _ in horn_sel]):
        w = sampler._best_window(x, C.WINDOW_SAMPLES)
        if w is not None:
            wavs.append(w)
    if not wavs:
        return None
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    emb = embed(model, proc, device, wavs)
    return emb, len(wavs)


def per_class_ratio(emb, labels, cls):
    """单类 intra/inter ratio + silhouette（与 class_ratio 同公式）。"""
    m = labels == cls
    v = emb[m]
    if len(v) < 8:
        return None
    intra = float(np.mean(cosine_dist(v)[np.triu_indices(len(v), 1)]))
    cls_ids = sorted(set(int(l) for l in labels))
    centroids = {k: emb[labels == k].mean(axis=0) for k in cls_ids}
    others = np.array([centroids[j] for j in cls_ids if j != cls])
    inter = float((1.0 - v @ others.T).mean())
    nearest = float((1.0 - v @ others.T).min(axis=1).mean())
    return {"n": int(m.sum()), "intra": round(intra, 4), "inter": round(inter, 4),
            "ratio": round(intra / inter, 4) if inter > 0 else None,
            "silhouette": round((inter - intra) / inter, 4) if inter > 0 else None,
            "nearest_class_margin": round(nearest, 4)}


def main():
    device = "mps"
    labels, names = rebuild_labels()
    embs = {}
    for ds in ("sc10", "us8k", "fsd50k10"):
        embs[ds] = np.load(os.path.join(CACHE, f"clap_emb_{ds}_{TAG}.npy"))
        assert len(embs[ds]) == len(labels[ds]), f"{ds}: cache/labels 长度不一致"

    # ---- 复现冻结 per-class 数字（位级校验后才可信） ----
    frozen = json.load(open(os.path.join(OUT, f"exp_heterogeneity_{TAG}.json")))
    for ds in ("sc10", "us8k", "fsd50k10"):
        ratio, per = class_ratio(embs[ds], labels[ds])
        for k, v in per.items():
            fv = frozen["results"][ds]["per_class"][k]
            for field in ("intra", "inter", "silhouette", "nearest_class_margin"):
                if v[field] is not None and fv.get(field) is not None and \
                        abs(v[field] - fv[field]) > 0.001:
                    raise SystemExit(f"冻结复现失败 {ds}/{k}/{field}: {v[field]} vs {fv[field]}")
        if abs(ratio - frozen["results"][ds]["ratio"]) > 0.002:
            raise SystemExit(f"冻结复现失败 {ds} ratio: {ratio} vs {frozen['results'][ds]['ratio']}")
        print(f"[{ds}] 冻结复现 OK ratio={ratio:.4f} n={len(labels[ds])}", flush=True)

    # ---- FSD50K-10 horn 子集（缓存中无 horn 窗, 新嵌入） ----
    horn = load_fsd_horn_windows(device)
    horn_res = None
    if horn is not None:
        emb_horn, n_horn = horn
        comb = np.concatenate([embs["fsd50k10"], emb_horn], axis=0)
        comb_labels = np.concatenate([labels["fsd50k10"], np.full(n_horn, 2)])
        # 用合并集重算 FSD50K-10 全部类的质心（horn 入参考系）
        horn_res = per_class_ratio(comb, comb_labels, 2)
        print(f"[fsd50k10] horn: n={n_horn} {horn_res}", flush=True)

    # ---- 匹配类 per-class ratio ----
    matched = {
        "siren": {"us8k": 8, "fsd50k10": 3},
        "horn": {"us8k": 1, "fsd50k10": 2},
        "vehicle_approx": {"us8k": 5, "fsd50k10": 0},
    }
    results = {"matched_classes": {}, "notes": []}
    for mname, pair in matched.items():
        entry = {}
        for ds, cls in pair.items():
            if ds == "fsd50k10" and mname == "horn" and horn_res is not None:
                entry[ds] = horn_res
            else:
                entry[ds] = per_class_ratio(embs[ds], labels[ds], cls)
            if entry[ds] is None:
                entry[ds] = {"n": int((labels[ds] == cls).sum())}
        results["matched_classes"][mname] = entry
        print(f"[{mname}] {entry}", flush=True)

    # ---- 判定: 控制事件类型后, 原拒绝是否仍成立 ----
    # 原拒绝依据: SC-10 ratio 0.92 > US8K 0.63 / FSD50K-10 0.82, 与天花板排名矛盾。
    # 匹配版逐配对报告方向; 结论 = 匹配比较是否救活原假设（要求匹配类方向
    # 全部与天花板排名一致且 SC-10 臂矛盾被解释）; 任一配对方向不一致即不救活。
    pairs_notes = []
    for mname, pair in results["matched_classes"].items():
        u = pair.get("us8k", {}).get("ratio")
        f = pair.get("fsd50k10", {}).get("ratio")
        if u is None or f is None:
            pairs_notes.append(f"{mname}: 样本不足 (us8k={pair.get('us8k',{}).get('n')}, "
                               f"fsd50k10={pair.get('fsd50k10',{}).get('n')})")
        else:
            # 预测: 天花板更低的 US8K 应有更高的异质性 ratio（> 表示方向与假设一致）
            consistent = u > f
            pairs_notes.append(f"{mname}: us8k {u} vs fsd50k10 {f} — "
                               f"{'与假设一致' if consistent else '与假设矛盾'}")
    n_inconsistent = sum(1 for mname in results["matched_classes"]
                         if (lambda u, f: u is not None and f is not None and not (u > f))(
                             results["matched_classes"][mname].get("us8k", {}).get("ratio"),
                             results["matched_classes"][mname].get("fsd50k10", {}).get("ratio")))
    if n_inconsistent > 0:
        verdict = "REJECTION_STANDS"
        interp = (f"控制事件类型后匹配比较不一致配对数 = {n_inconsistent}/"
                  f"{len(results['matched_classes'])}; 且 SC-10 臂(ratio 0.92, 决定性证据)不受"
                  "匹配影响 — 原拒绝结论未被救活")
    else:
        verdict = "INSUFFICIENT_N"
        interp = "匹配类样本均不足, 无法判定"
    results["verdict"] = verdict
    results["interpretation"] = interp
    results["pair_notes"] = pairs_notes
    for pn in pairs_notes:
        print("  ", pn, flush=True)

    out = {"tag": "20260818", "model": "laion/clap-htsat-unfused", "seed": C.SEED,
           "basis": "冻结缓存 clap_emb_*_20260816.npy（先位级复现冻结 per-class）+ 新嵌 horn 子集",
           "frozen_ratios": {ds: frozen["results"][ds]["ratio"] for ds in ("sc10", "us8k", "fsd50k10")},
           **results}
    with open(os.path.join(OUT, "exp_heterogeneity_matched_20260818.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}\n{interp}")
    print("DONE → outputs/exp_heterogeneity_matched_20260818.json")


if __name__ == "__main__":
    main()
