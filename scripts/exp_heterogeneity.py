"""exp_heterogeneity.py — 嵌入空间类内/类间异质性量化（2026-08-16, decisive check #2）

问题: 论文把 FSD50K-10/US8K 的可决策域天花板归因于"类内声学异质性"（task-definition
      ceiling）, 但该归因目前是三角互证（SC-10 低异质性成功 / US8K 高异质性复现天花板）,
      异质性本身未被直接量化。本文档实现直接测量: 在共享 CLAP 嵌入空间上计算
      类内/类间余弦距离比（intra/inter ratio）。

协议（对齐论文冻结 pass 的窗口契约）:
  数据集窗口: SC-10 官方 test（4,074 clips, 1s→1.28s 零填充, 单窗）
              US8K fold-10 test（win_clips, ≥1.28s 最优窗）
              FSD50K-10 val（run_main 同款 sample_clips 500, 最优窗, 首标签指派）
  嵌入: CLAP(htsat-unfused) audio 特征, L2 归一化, 48kHz（与 exp_us8k_clap.py 同管线）
  距离: 余弦距离 = 1 − cos
  指标: intra_k = 类 k 样本到本类质心的平均距离; inter_k = 类 k 样本到其他类质心的
        平均距离（min 也可用, 取 mean 稳定）; ratio = Σ_k w_k·intra_k / Σ_k w_k·inter_k
        （w_k = 类样本占比）; 另报 per-class (inter−intra)/inter（silhouette 式）与
        nearest-class margin。
  干净域: 异质性是任务/类定义属性, 非信道属性 → 只用 CLEAN 窗（论文 clean-ceiling
        探针同口径）。

判定（go/no-go, 输出中直接打印）:
  CONFIRM: ratio_sc10 < ratio_us8k 且 ratio_sc10 < ratio_fsd50k（低异质性 → 高天花板）
  → 可写入论文（triangulation 升级为直接测量）
  否则: 不写入论文, 维持"异质性未直接量化"的诚实叙述。

用法: python scripts/exp_heterogeneity.py [--device mps] [--tag 20260816] [--max-clips 0]
输出: outputs/exp_heterogeneity_{tag}.json + 嵌入缓存 outputs/cache/clap_emb_{dataset}_{tag}.npy
注意: 新增实验记录（非 R19 冻结 pass）; 种子 C.SEED 派生; 冻结纪律遵守（不重跑 R19 数字）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.cf_sampler import CFSampler
from exp_sc10 import SC_NAMES, load_sc_index, load_sc_wav
from exp_us8k import US8K_NAMES, load_us8k_index, win_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
CACHE = os.path.join(OUT, "cache")
MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
SR_TARGET = 48000


def to_48k(x, fs=16000):
    from scipy.signal import resample_poly
    return resample_poly(x, SR_TARGET, fs).astype(np.float32)


def embed(model, proc, device, wavs, batch=32):
    """wavs: list[np.float32 @16k]; 返回 [B,512] L2 归一化嵌入。分批避免 MPS OOM。"""
    outs = []
    for i in range(0, len(wavs), batch):
        chunk = wavs[i:i + batch]
        audio = proc(audio=[to_48k(w) for w in chunk], sampling_rate=SR_TARGET,
                     return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_audio_features(**audio)
            outs.append(F.normalize(out.pooler_output, dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def cosine_dist(emb):
    """emb [N,D] 归一化 → [N,N] 余弦距离矩阵（1−cos）。"""
    return 1.0 - emb @ emb.T


def class_ratio(emb, labels):
    """labels: [N] 类索引。返回 (ratio, per_class dict)。"""
    n = len(emb)
    cls_ids = sorted(set(int(l) for l in labels))
    centroids = {}
    for k in cls_ids:
        centroids[k] = emb[labels == k].mean(axis=0)
    per = {}
    wsum, num = 0.0, 0.0
    for k in cls_ids:
        m = labels == k
        v = emb[m]
        intra = float(np.mean(cosine_dist(v)[np.triu_indices(len(v), 1)])) if len(v) > 1 else 0.0
        others = np.array([centroids[j] for j in cls_ids if j != k])
        dist_other = 1.0 - v @ others.T
        inter = float(dist_other.mean())
        nearest = float(dist_other.min(axis=1).mean())
        per[str(k)] = {"n": int(m.sum()), "intra": round(intra, 4),
                       "inter": round(inter, 4), "silhouette": round((inter - intra) / inter, 4)
                       if inter > 0 else None, "nearest_class_margin": round(nearest, 4)}
        wsum += len(v) * intra
        num += len(v) * inter
    ratio = wsum / num if num > 0 else None
    return ratio, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="20260816")
    ap.add_argument("--max-clips", type=int, default=0, help="0=全量; 调试用限量")
    args = ap.parse_args()

    from transformers.models.clap import ClapModel, ClapProcessor
    dev = args.device if torch.backends.mps.is_available() else "cpu"
    if args.device != "cpu" and dev != args.device:
        print(f"WARN: {args.device} 不可用, 回退 {dev}", flush=True)
    model = ClapModel.from_pretrained(MODEL_ID).to(dev).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    print(f"CLAP loaded on {dev}", flush=True)

    # ---- 三数据集窗口（干净域）----
    sets = {}

    # SC-10: 官方 test 全量
    sc_index = [e for e in load_sc_index() if e[2] == "test"]
    if args.max_clips:
        sc_index = sc_index[: args.max_clips]
    sc_wavs, sc_labels = [], []
    for rel, y, _ in sc_index:
        x = load_sc_wav(rel)
        if x is None:
            continue
        sc_wavs.append(x)
        sc_labels.append(int(np.argmax(y[:10])))
    sets["sc10"] = {"wavs": sc_wavs, "labels": sc_labels, "names": SC_NAMES}

    # US8K: fold-10 test, 最优窗
    te = [e for e in load_us8k_index() if e[3] == "test"]
    if args.max_clips:
        te = te[: args.max_clips]
    us_clips, dropped = win_clips(te)
    us_wavs, us_labels = [], []
    for x, y, fname in us_clips:
        us_wavs.append(x)
        us_labels.append(int(np.argmax(y[:10])))
    sets["us8k"] = {"wavs": us_wavs, "labels": us_labels, "names": US8K_NAMES, "dropped": dropped}

    # FSD50K-10: val 500（run_main 同款分层抽样, seed C.SEED+1）, 最优窗, 首标签指派
    from aof.mapping import build_dev_index
    from run_main import load_fsd50k_clips, sample_clips
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], 500, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}
    fs_wavs, fs_labels = [], []
    sampler = CFSampler([])
    for x, f in load_fsd50k_clips([f for f, _ in val_sel]):
        w = sampler._best_window(x, C.WINDOW_SAMPLES)
        if w is None:
            continue
        y = idx_map[f]
        primary = [i for i, v in enumerate(y) if v and i < 9][0]
        fs_wavs.append(w)
        fs_labels.append(primary)
    sets["fsd50k10"] = {"wavs": fs_wavs, "labels": fs_labels,
                        "names": ["vehicle", "bicycle", "horn", "siren", "tire_squeal",
                                  "impact", "construction", "mechanical_anomaly",
                                  "human_activity"]}

    # ---- 嵌入 + 指标 ----
    results = {}
    for name, s in sets.items():
        print(f"[{name}] embedding {len(s['wavs'])} windows ...", flush=True)
        if not s["wavs"]:
            results[name] = {"error": "no windows"}
            continue
        emb = embed(model, proc, dev, s["wavs"])
        os.makedirs(CACHE, exist_ok=True)
        np.save(os.path.join(CACHE, f"clap_emb_{name}_{args.tag}.npy"), emb)
        labels = np.array(s["labels"])
        ratio, per = class_ratio(emb, labels)
        results[name] = {"n": len(s["wavs"]), "ratio": round(ratio, 4) if ratio else None,
                         "per_class": per, "names": s["names"]}
        print(f"[{name}] ratio={ratio:.4f}", flush=True)

    # ---- go/no-go 判定 ----
    r_sc = results.get("sc10", {}).get("ratio")
    r_us = results.get("us8k", {}).get("ratio")
    r_fs = results.get("fsd50k10", {}).get("ratio")
    verdict = "CONFIRM" if (r_sc is not None and r_us is not None and r_fs is not None
                            and r_sc < r_us and r_sc < r_fs) else "NO_CONFIRM"
    out = {"tag": args.tag, "model": MODEL_ID, "device": dev, "seed": C.SEED,
           "verdict": verdict,
           "interpretation": ("SC-10 ratio < US8K 且 < FSD50K-10 → 低异质性→高天花板一致,"
                              " 可写入论文; 否则维持原诚实叙述"),
           "results": results}
    with open(os.path.join(OUT, f"exp_heterogeneity_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}  (sc10={r_sc}, us8k={r_us}, fsd50k10={r_fs})", flush=True)
    print(f"DONE → outputs/exp_heterogeneity_{args.tag}.json")


if __name__ == "__main__":
    main()
