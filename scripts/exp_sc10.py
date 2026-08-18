"""exp_sc10.py — Speech Commands v0.02 10-core 跨数据集验证（2026-08-06）

协议对齐 FSD50K-10 冻结 pass（与 exp_us8k.py 同一配方）:
  训练网格 {-20,-5,10}dB × {wind,occlusion,self_motion}; 评估网格 {-25,-15,-5,5,15}dB
  alpha=0.1; Bonferroni alpha_k=alpha/10 (AFRule(n_classes=10)); 模型 11 输出 (unknown@10 掩码)
SC 特有: 10-core 单词子集（yes/no/up/down/left/right/on/off/stop/go, 原生标签无近似）;
  1s clip 零填充至 1.28s 窗（论文诚实标注）; 官方 validation/testing_list 划分;
  val = validation_list 抽 500（与 run_main 协议一致）; test = testing_list 全量。
"""
import argparse
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
from aof.data import log_mel
from aof.model import SONTRA_A
from aof.train import train
from aof.wsosim import corrupt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SC_ROOT = os.environ.get("SC_ROOT", os.path.join(_REPO, "data", "speech_commands"))
SC_NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
N_REAL = 10
N_OUT = 11                      # + unknown@10（掩码）
CKPT_DIR = os.path.join(OUT, "checkpoints_sc10")
TRAIN_SNRS = [-20.0, -5.0, 10.0]
KINDS = ["wind", "occlusion", "self_motion"]
PAD = C.WINDOW_SAMPLES          # 1s → 1.28s 零填充


def load_sc_index():
    """[(fname, y11, split)]; split ∈ train/val/test（官方列表）。"""
    val_set = set()
    with open(os.path.join(SC_ROOT, "validation_list.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                val_set.add(line)
    test_set = set()
    with open(os.path.join(SC_ROOT, "testing_list.txt")) as f:
        for line in f:
            line = line.strip()
            if line:
                test_set.add(line)
    index = []
    for i, word in enumerate(SC_NAMES):
        d = os.path.join(SC_ROOT, word)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".wav"):
                continue
            rel = f"{word}/{fn}"
            y = [0.0] * N_OUT
            y[i] = 1.0
            if rel in test_set:
                split = "test"
            elif rel in val_set:
                split = "val"
            else:
                split = "train"
            index.append((rel, y, split))
    return index


def load_sc_wav(rel):
    import scipy.io.wavfile as wavf
    p = os.path.join(SC_ROOT, rel)
    sr, arr = wavf.read(p)
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        from scipy.signal import resample_poly
        x = resample_poly(x, C.SAMPLE_RATE, sr)
    # 1s → 1.28s 零填充（超过则截断）
    if len(x) < PAD:
        x = np.pad(x, (0, PAD - len(x)))
    else:
        x = x[:PAD]
    return x.astype(np.float32)


def sample_balanced_sc(entries, quota, seed):
    """每词最多 quota（稀有词全量）。"""
    by_class = defaultdict(list)
    for rel, y, split in entries:
        cls = [i for i, v in enumerate(y) if v and i < N_REAL]
        if cls:
            by_class[cls[0]].append((rel, y))
    rng = random.Random(seed)
    picked = []
    for k in sorted(by_class):
        pool = by_class[k]
        rng.shuffle(pool)
        picked.extend(pool[:quota])
    return picked


def sample_clips_sc(entries, n, seed):
    rng = random.Random(seed)
    picked, seen = [], set()
    keys = sorted({i for _, y, _ in entries for i in range(N_REAL) if y[i]})
    stall = 0
    while len(picked) < n:
        k = keys[rng.randrange(len(keys))]
        pool = [e for e in entries if e[1][k] and e[0] not in seen]
        if not pool:
            stall += 1
            if stall > 200:
                break
            continue
        stall = 0
        e = rng.choice(pool)
        seen.add(e[0])
        picked.append(e)
    return picked


def build_pairs_sc(clips_meta, kinds, snrs, sampler_seed=C.SEED):
    """clips_meta: [(x, y11, rel)] → 全网格 CF 对。"""
    rng = np.random.default_rng(sampler_seed)
    pairs = []
    for x, y, rel in clips_meta:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: 每词只取前 N 条")
    ap.add_argument("--quota", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()

    index = load_sc_index()
    if args.limit:
        by_word = defaultdict(list)
        for rel, y, split in index:
            by_word[rel.split("/")[0]].append((rel, y, split))
        index = [e for w in SC_NAMES for e in by_word[w][:args.limit]]
    print(f"[{time.time()-t0:.0f}s] index: {len(index)} "
          f"({sum(1 for e in index if e[2]=='train')} train / "
          f"{sum(1 for e in index if e[2]=='val')} val / "
          f"{sum(1 for e in index if e[2]=='test')} test)")

    tr_entries = [e for e in index if e[2] == "train"]
    va_entries = [e for e in index if e[2] == "val"]
    te_entries = [e for e in index if e[2] == "test"]

    sel = sample_balanced_sc(tr_entries, args.quota, C.SEED)
    sel_set = {f for f, _ in sel}
    tr_sel = [e for e in tr_entries if e[0] in sel_set]
    val_sel = sample_clips_sc(va_entries, 500, C.SEED + 1)
    val_set = {e[0] for e in val_sel}
    va_sel = [e for e in va_entries if e[0] in val_set]
    print(f"[{time.time()-t0:.0f}s] sampled {len(tr_sel)} train / {len(va_sel)} val / {len(te_entries)} test")

    tr_clips = [(load_sc_wav(rel), y, rel) for rel, y, _ in tr_sel]
    va_clips = [(load_sc_wav(rel), y, rel) for rel, y, _ in va_sel]
    te_clips = [(load_sc_wav(rel), y, rel) for rel, y, _ in te_entries]
    print(f"[{time.time()-t0:.0f}s] loaded {len(tr_clips)}/{len(va_clips)}/{len(te_clips)}")

    tr_pairs = build_pairs_sc(tr_clips, KINDS, TRAIN_SNRS)
    va_pairs = build_pairs_sc(va_clips, KINDS, TRAIN_SNRS)
    print(f"[{time.time()-t0:.0f}s] pairs: {len(tr_pairs)} train / {len(va_pairs)} val")

    cls_tr, cls_te = defaultdict(int), defaultdict(int)
    for _, y, _ in tr_clips:
        cls_tr[int(np.argmax(y))] += 1
    for _, y, _ in te_clips:
        cls_te[int(np.argmax(y))] += 1
    manifest = {"n_train_clips": len(tr_clips), "n_val_clips": len(va_clips),
                "n_test_clips": len(te_clips), "class_counts_train": dict(cls_tr),
                "class_counts_test": dict(cls_te), "classes": SC_NAMES,
                "note": "v0.02, 10-core words, 1s clips zero-padded to 1.28s"}
    with open(os.path.join(OUT, "sc10_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{time.time()-t0:.0f}s] manifest: {json.dumps(manifest)}")

    model = SONTRA_A(n_classes=N_OUT)
    torch.manual_seed(C.SEED)
    print(f"[{time.time()-t0:.0f}s] training (epochs={args.epochs}, batch={args.batch})...")
    model, res = train(model, tr_pairs, va_pairs, epochs=args.epochs, batch=args.batch,
                       device=args.device, out_dir=CKPT_DIR)
    out = {"manifest": manifest, "best_snr_mae": res["best_snr_mae"],
           "args": vars(args), "ckpt_dir": CKPT_DIR}
    with open(os.path.join(OUT, "exp_sc10_train.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{time.time()-t0:.0f}s] DONE best_snr_mae={res['best_snr_mae']:.3f} "
          f"→ outputs/exp_sc10_train.json")


if __name__ == "__main__":
    main()
