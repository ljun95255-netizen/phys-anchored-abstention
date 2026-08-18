"""
run_main.py — D1 子集构建 + SONTRA-A 训练 + gap 主矩阵（首轮, 子集规模 [诚实标注]）
流程: dev 索引分层抽样 → CF 对(全 kind×SNR 网格) → CFAL 训练 → detector-tier(B0/B11/B11a/Oracle)
      + system-tier(B12) 评估 → outputs/run_main_result.json + 控制台表
用法: python run_main.py --n-train 2000 --n-val 500 --epochs 10
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from aof import config as C
from aof.af_rule import AFRule, r_min_theory
from aof.baselines import EnergyDetectorB0, SPAnchorB11
from aof.cf_sampler import CFSampler
from aof.data import log_mel, load_esc50
from aof.evaluate import evaluate_detector, evaluate_detector_oracle_threshold, evaluate_system
from aof.mapping import build_dev_index, CLASS_NAMES
from aof.metrics import (operating_gap, coverage, physical_coverage,
                         rank_auc, snr_mae)
from aof.model import SONTRA_A
from aof.train import train
from aof.wsosim import corrupt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
OUT = os.path.abspath(OUT)


def load_fsd50k_clips(fnames, split_dir="dev", fsd50k_dir=C.FSD50K_DIR):
    """按 fname 加载 wav（44.1k → 16k），返回 [(x f32, labels_set, fname)]。"""
    import scipy.io.wavfile as wavf
    import scipy.signal as sig
    clips = []
    d = os.path.join(fsd50k_dir, "clips", split_dir)
    for fn in fnames:
        p = os.path.join(d, fn + ".wav")
        try:
            sr, arr = wavf.read(p)
        except Exception:
            continue
        x = np.asarray(arr, dtype=np.float64)
        if sr != C.SAMPLE_RATE:
            x = sig.resample_poly(x, C.SAMPLE_RATE, sr)
        clips.append((x.astype(np.float32), fn))
    return clips


def sample_clips_balanced(index, quota, seed):
    """类别均衡采样: 每类最多 quota 个 clip（稀有类全量纳入）。
    R19: 修类别失衡（construction 14% / vehicle 83% 的逐类精度差）。"""
    by_class = defaultdict(list)
    for fname, y, split in index:
        cls = [i for i, v in enumerate(y) if v and i < C.N_CLASSES - 1]
        if cls:
            by_class[cls[0]].append((fname, y))
    rng = random.Random(seed)
    picked = []
    for k in sorted(by_class):
        pool = by_class[k]
        rng.shuffle(pool)
        picked.extend(pool[:quota])
    return picked


def sample_clips(index, n, seed):
    """按类别分层抽样（类别少的多抽, 保证 horn/bicycle 不丢）。"""
    from collections import defaultdict
    by_class = defaultdict(list)
    for fname, y, split in index:
        cls = [i for i, v in enumerate(y) if v and i < C.N_CLASSES - 1]
        if cls:
            by_class[cls[0]].append((fname, y))
    rng = random.Random(seed)
    picked, seen = [], set()
    keys = list(by_class.keys())
    while len(picked) < n:
        k = keys[rng.randrange(len(keys))]
        pool = [c for c in by_class[k] if c[0] not in seen]
        if not pool:
            continue
        fname, y = rng.choice(pool)
        seen.add(fname)
        picked.append((fname, y))
    return picked


def build_pairs(clips_meta, kinds, snrs, sampler_seed=C.SEED, cache=True):
    """clips_meta: [(x, fname, y)] → 每 clip × (kind × snr) 全网格 CF 对。"""
    rng = np.random.default_rng(sampler_seed)
    pairs = []
    for x, y, fname in clips_meta:      # (x, target, fname) 契约
        for kind in kinds:
            for snr in snrs:
                seed = int(rng.integers(1 << 31))
                xc, r_db, meta = corrupt(x, kind, snr, seed)
                if meta.get("inaudible"):
                    continue
                yv = np.array(y, dtype=np.float32)
                r_vec = np.full(C.N_CLASSES, -60.0, dtype=np.float32)
                mask = np.zeros(C.N_CLASSES, dtype=np.float32)
                for i in range(C.N_CLASSES - 1):
                    if yv[i]:
                        mask[i] = 1.0
                        r_vec[i] = r_db
                pairs.append((log_mel(torch.from_numpy(xc).unsqueeze(0).float()).squeeze(0).numpy(),
                              log_mel(torch.from_numpy(x).unsqueeze(0).float()).squeeze(0).numpy(),
                              yv, r_vec, mask, snr))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=int, default=0,
                    help="类别均衡采样: 每类配额（R19, 修稀有类欠训）; 非 0 时忽略 --n-train")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--kinds", default="wind,occlusion,self_motion")
    ap.add_argument("--train-snrs", default="-20,-5,10")     # 训练网格跨前沿(含弃权区)
    ap.add_argument("--eval-snrs", default="-25,-15,-5,5,15")  # 评估网格含前沿两侧
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    tr_snrs = [float(s) for s in args.train_snrs.split(",")]
    ev_snrs = [float(s) for s in args.eval_snrs.split(",")]

    t0 = time.time()
    index = build_dev_index()
    # ---- 采样（类别均衡或分层）----
    if args.balanced > 0:
        train_sel = sample_clips_balanced([r for r in index if r[2] == "train"], args.balanced, args.seed)
        print(f"[{time.time()-t0:.0f}s] balanced train clips: {len(train_sel)} (quota {args.balanced}/class)")
    else:
        train_sel = sample_clips([r for r in index if r[2] == "train"], args.n_train, args.seed)
    val_sel = sample_clips([r for r in index if r[2] == "val"], args.n_val, args.seed + 1)
    print(f"[{time.time()-t0:.0f}s] sampled {len(train_sel)} train / {len(val_sel)} val clips")
    idx_map = {f: y for f, y, s in index}
    tr_clips = [(x, f, idx_map[f]) for x, f in load_fsd50k_clips([f for f, _ in train_sel])]
    va_clips = [(x, f, idx_map[f]) for x, f in load_fsd50k_clips([f for f, _ in val_sel])]
    print(f"[{time.time()-t0:.0f}s] loaded wavs: {len(tr_clips)} train / {len(va_clips)} val")

    # 事件窗选择: 每 clip 取事件带能量最大 1.28s 窗（静音窗剔除）
    # 注意元组顺序 = (x, target, fname) —— evaluate_system 的契约（R12 修复 f/y 错位 bug）
    sampler = CFSampler([])
    tr_clips = [(sampler._best_window(x, C.WINDOW_SAMPLES), y, f)
                for x, f, y in tr_clips if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]
    va_clips = [(sampler._best_window(x, C.WINDOW_SAMPLES), y, f)
                for x, f, y in va_clips if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]
    print(f"[{time.time()-t0:.0f}s] windowed: {len(tr_clips)} train / {len(va_clips)} val (1.28s)")

    tr_pairs = build_pairs(tr_clips, kinds, tr_snrs)
    va_pairs = build_pairs(va_clips, kinds, tr_snrs)
    print(f"[{time.time()-t0:.0f}s] built {len(tr_pairs)} train pairs / {len(va_pairs)} val pairs "
          f"({len(kinds)*len(tr_snrs)}-way train grid)")

    model = SONTRA_A()
    torch.manual_seed(args.seed)
    print(f"[{time.time()-t0:.0f}s] training (epochs={args.epochs})...")
    model, res = train(model, tr_pairs, va_pairs, epochs=args.epochs,
                       batch=C.BATCH_SIZE, device=args.device,
                       out_dir=os.path.join(OUT, "checkpoints"))
    print(f"[{time.time()-t0:.0f}s] trained; best val SNR MAE = {res['best_snr_mae']:.2f} dB")

    # ---- detector-tier（评估网格 ev_snrs, 含前沿两侧）----
    rule = AFRule()
    rmin_db = 10 * np.log10(rule.r_min)
    results = {"config": vars(args), "snr_mae": res["best_snr_mae"], "detectors": {}, "systems": {}}
    for name, det in [("B0_energy", EnergyDetectorB0()), ("B11_sp", SPAnchorB11())]:
        recs = evaluate_detector(det, va_clips, kinds, ev_snrs)
        dec = np.array([r["decide"] for r in recs])
        r_true = np.array([r["r_true_db"] for r in recs])
        correct = ~dec | (r_true >= rmin_db)      # 决策且真可听=正确; 弃权不计入风险
        gap, risk = operating_gap(dec, correct, C.ALPHA)
        results["detectors"][name] = {
            "gap": gap, "risk": risk, "coverage": coverage(dec),
            "c_phys": physical_coverage(r_true, rule.r_min),
            "rank_auc": rank_auc(np.array([r["snr_hat_db"] for r in recs]), correct),
        }
        print(f"  [{name}] gap={gap:.3f} risk={risk:.3f} cov={coverage(dec):.3f} "
              f"c_phys={results['detectors'][name]['c_phys']:.3f}")

    # Oracle（真实 SNR 阈值, B0 估计器）
    recs_or = evaluate_detector(EnergyDetectorB0(), va_clips, kinds, ev_snrs)
    dec = np.array([r["r_true_db"] for r in recs_or]) >= rmin_db
    r_true = np.array([r["r_true_db"] for r in recs_or])
    correct = ~dec | (r_true >= rmin_db)
    gap, risk = operating_gap(dec, correct, C.ALPHA)
    results["detectors"]["oracle"] = {"gap": gap, "risk": risk, "coverage": coverage(dec)}
    print(f"  [oracle] gap={gap:.3f} risk={risk:.3f} cov={coverage(dec):.3f}")

    # B11a: SP 估计器 + 真实 SNR 阈值（经典可达上界——估计器噪声存在下的最优）
    recs_b11a = evaluate_detector_oracle_threshold(SPAnchorB11(), va_clips, kinds, ev_snrs)
    dec = np.array([r["decide"] for r in recs_b11a])
    r_true = np.array([r["r_true_db"] for r in recs_b11a])
    correct = ~dec | (r_true >= rmin_db)
    gap, risk = operating_gap(dec, correct, C.ALPHA)
    results["detectors"]["B11a_sp_oracle_thresh"] = {
        "gap": gap, "risk": risk, "coverage": coverage(dec),
        "rank_auc": rank_auc(np.array([r["snr_hat_db"] for r in recs_b11a]), correct),
    }
    print(f"  [B11a] gap={gap:.3f} risk={risk:.3f} cov={coverage(dec):.3f}")

    # ---- system-tier (B12: CAE 全栈) ----
    recs = evaluate_system(model, rule, va_clips, kinds, ev_snrs, device=args.device)
    dec = np.array([r["decide"] for r in recs])
    correct = np.array([r["correct"] for r in recs])
    gap, risk = operating_gap(dec, correct, C.ALPHA)
    raw_acc = float(np.mean([r["raw_correct"] for r in recs]))   # 诊断: 无弃权 argmax 准确率
    results["systems"]["B12_cae"] = {
        "gap": gap, "risk": risk, "coverage": coverage(dec),
        "rank_auc": rank_auc(np.array([r["snr_hat_db"] for r in recs]), correct),
        "raw_acc": raw_acc,
    }
    print(f"  [B12] gap={gap:.3f} risk={risk:.3f} cov={coverage(dec):.3f} "
          f"rank_auc={results['systems']['B12_cae']['rank_auc']:.3f} raw_acc={raw_acc:.3f}")

    with open(os.path.join(OUT, "run_main_result.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[{time.time()-t0:.0f}s] DONE → outputs/run_main_result.json")


if __name__ == "__main__":
    main()
