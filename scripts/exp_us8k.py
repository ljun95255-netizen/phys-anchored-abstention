"""exp_us8k.py — UrbanSound8K 跨数据集验证（2026-08-06）

协议对齐 FSD50K-10 冻结 pass（run_main R19 配方）:
  训练网格 {-20,-5,10}dB × {wind,occlusion,self_motion}; 评估网格 {-25,-15,-5,5,15}dB
  1.28s best-window; 24-mel/128-frame; alpha=0.1; Bonferroni alpha_k=alpha/10
  （AFRule(n_classes=10) → 与论文相同的工作边界; 模型输出 11 维, unknown@10 掩码）
US8K 特有: 原生 10 类（无标签近似）; split = fold 1-9 train / fold 10 test;
  balanced quota（稀有类全量）。采样器自带（run_main 版硬编码 C.N_CLASSES-1=9, 漏 class 9）。
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.cf_sampler import CFSampler
from aof.data import log_mel
from aof.model import SONTRA_A
from aof.train import train
from aof.wsosim import corrupt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
US8K_ROOT = os.environ.get("US8K_ROOT", os.path.join(_REPO, "data", "urbansound8k", "UrbanSound8K"))
US8K_NAMES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
              "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
N_REAL = 10
N_OUT = 11                      # + unknown@10（掩码）
CKPT_DIR = os.path.join(OUT, "checkpoints_us8k")
TRAIN_SNRS = [-20.0, -5.0, 10.0]
KINDS = ["wind", "occlusion", "self_motion"]


def load_us8k_index(limit=None):
    """[(fname, fold, one_hot_y(11), split)]; split = fold<=9 ? train : test"""
    meta = os.path.join(US8K_ROOT, "metadata", "UrbanSound8K.csv")
    index = []
    with open(meta) as f:
        for r in csv.DictReader(f):
            fold = int(r["fold"])
            cid = int(r["classID"])
            y = [0.0] * N_OUT
            y[cid] = 1.0
            split = "train" if fold <= 9 else "test"
            index.append((r["fname"], fold, y, split))
            if limit and len(index) >= limit:
                break
    return index


def load_us8k_wav(fname, fold):
    import scipy.io.wavfile as wavf
    import scipy.signal as sig
    p = os.path.join(US8K_ROOT, "audio", f"fold{fold}", fname)
    sr, arr = wavf.read(p)
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        x = sig.resample_poly(x, C.SAMPLE_RATE, sr)
    return x.astype(np.float32)


def sample_balanced_us8k(entries, quota, seed):
    """entries: [(fname, fold, y11, split)]; 每类最多 quota（稀有类全量）。"""
    by_class = defaultdict(list)
    for fname, fold, y, split in entries:
        cls = [i for i, v in enumerate(y) if v and i < N_REAL]
        if cls:
            by_class[cls[0]].append((fname, fold, y))
    rng = random.Random(seed)
    picked = []
    for k in sorted(by_class):
        pool = by_class[k]
        rng.shuffle(pool)
        picked.extend(pool[:quota])
    return picked


def sample_clips_us8k(entries, n, seed):
    """分层抽样: 少样本类多抽, 保证 car_horn/gun_shot 不丢。池耗尽即停止。"""
    rng = random.Random(seed)
    picked, seen = [], set()
    keys = sorted({i for _, _, y, _ in entries for i in range(N_REAL) if y[i]})
    stall = 0
    while len(picked) < n:
        k = keys[rng.randrange(len(keys))]
        pool = [e for e in entries if e[2][k] and e[0] not in seen]
        if not pool:
            stall += 1
            if stall > 200:          # 池耗尽（冒烟 --limit 时必然）→ 停止
                break
            continue
        stall = 0
        e = rng.choice(pool)
        seen.add(e[0])
        picked.append(e)
    return picked


def build_pairs_us8k(clips_meta, kinds, snrs, sampler_seed=C.SEED):
    """clips_meta: [(x, y11, fname)] → 全网格 CF 对（n_classes=11 版 run_main.build_pairs）。"""
    rng = np.random.default_rng(sampler_seed)
    pairs = []
    for x, y, fname in clips_meta:
        for kind in kinds:
            for snr in snrs:
                seed = int(rng.integers(1 << 31))
                xc, r_db, meta = corrupt(x, kind, snr, seed)
                if meta.get("inaudible"):
                    continue
                yv = np.array(y, dtype=np.float32)
                r_vec = np.full(N_OUT, -60.0, dtype=np.float32)
                mask = np.zeros(N_OUT, dtype=np.float32)
                for i in range(N_REAL):
                    if yv[i]:
                        mask[i] = 1.0
                        r_vec[i] = r_db
                pairs.append((log_mel(torch.from_numpy(xc).unsqueeze(0).float()).squeeze(0).numpy(),
                              log_mel(torch.from_numpy(x).unsqueeze(0).float()).squeeze(0).numpy(),
                              yv, r_vec, mask, snr))
    return pairs


def win_clips(entries):
    """best-window 化; 返回 (clips[(x,y,fname)], n_dropped)。"""
    sampler = CFSampler([])
    out, dropped = [], 0
    for fname, fold, y, split in entries:
        x = load_us8k_wav(fname, fold)
        w = sampler._best_window(x, C.WINDOW_SAMPLES)
        if w is None:
            dropped += 1
            continue
        out.append((w, y, fname))
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: 只取前 N 条 index")
    ap.add_argument("--quota", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()

    index = load_us8k_index(args.limit or None)
    print(f"[{time.time()-t0:.0f}s] index: {len(index)}")

    tr_entries = [e for e in index if e[3] == "train"]
    te_entries = [e for e in index if e[3] == "test"]

    sel = sample_balanced_us8k(tr_entries, args.quota, C.SEED)
    sel_set = {f for f, _, _ in sel}
    tr_sel = [e for e in tr_entries if e[0] in sel_set]
    val_sel = sample_clips_us8k(tr_entries, 500, C.SEED + 1)
    val_set = {e[0] for e in val_sel}
    va_sel = [e for e in tr_entries if e[0] in val_set]
    print(f"[{time.time()-t0:.0f}s] sampled {len(tr_sel)} train / {len(va_sel)} val / {len(te_entries)} test")

    tr_clips, tr_drop = win_clips(tr_sel)
    va_clips, va_drop = win_clips(va_sel)
    te_clips, te_drop = win_clips(te_entries)
    print(f"[{time.time()-t0:.0f}s] windowed: {len(tr_clips)}/{len(va_clips)}/{len(te_clips)} "
          f"(dropped {tr_drop}/{va_drop}/{te_drop} <1.28s)")

    tr_pairs = build_pairs_us8k(tr_clips, KINDS, TRAIN_SNRS)
    va_pairs = build_pairs_us8k(va_clips, KINDS, TRAIN_SNRS)
    print(f"[{time.time()-t0:.0f}s] pairs: {len(tr_pairs)} train / {len(va_pairs)} val")

    cls_tr, cls_te = defaultdict(int), defaultdict(int)
    for _, y, _ in tr_clips:
        cls_tr[int(np.argmax(y))] += 1
    for _, y, _ in te_clips:
        cls_te[int(np.argmax(y))] += 1
    manifest = {"n_train_clips": len(tr_clips), "n_val_clips": len(va_clips),
                "n_test_clips": len(te_clips), "dropped_train": tr_drop,
                "dropped_test": te_drop, "class_counts_train": dict(cls_tr),
                "class_counts_test": dict(cls_te), "classes": US8K_NAMES}
    with open(os.path.join(OUT, "us8k_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{time.time()-t0:.0f}s] manifest: {json.dumps(manifest, indent=1)}")

    model = SONTRA_A(n_classes=N_OUT)
    torch.manual_seed(C.SEED)
    print(f"[{time.time()-t0:.0f}s] training (epochs={args.epochs}, batch={args.batch})...")
    model, res = train(model, tr_pairs, va_pairs, epochs=args.epochs, batch=args.batch,
                       device=args.device, out_dir=CKPT_DIR)
    out = {"manifest": manifest, "best_snr_mae": res["best_snr_mae"],
           "args": vars(args), "ckpt_dir": CKPT_DIR}
    with open(os.path.join(OUT, "exp_us8k_train.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{time.time()-t0:.0f}s] DONE best_snr_mae={res['best_snr_mae']:.3f} "
          f"→ outputs/exp_us8k_train.json")


if __name__ == "__main__":
    main()
