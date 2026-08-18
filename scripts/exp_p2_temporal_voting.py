"""exp_p2_temporal_voting.py — P2 时序多窗投票探针（2026-08-05）
流式场景红利: 事件持续 2-5s, 1.28s 窗滑动 → 同一事件被多个连续窗观测,
每窗独立决策, 事件级多数投票。零训练: 复用 R19 ep22 checkpoint。
对照: ① 同批 clip 前 K 窗单窗 acc@dec（无投票）② best-window 单窗（B 探针口径）
成功线: 投票后事件级 acc@dec ≥ 82.5%（反事实阈值, 10 类口径）。
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
CKPT = os.path.join(OUT, "checkpoints", "sontra_a_ep22.pt")
SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
KINDS = ["wind", "occlusion", "self_motion"]
K = 5  # 最多连续窗数（1.28s × 5 = 6.4s 观测）


def slide_windows(x, win=C.WINDOW_SAMPLES, hop=C.WINDOW_SAMPLES):
    n = x.shape[0]
    wins = []
    i = 0
    while i + win <= n and len(wins) < K:
        wins.append(x[i:i + win])
        i += hop
    return wins


def main():
    from aof.mapping import build_dev_index
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], 500, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}

    # 多窗事件集: 每 clip 取前 K 连续窗（不足 K 取全部可用）
    va_clips = load_fsd50k_clips([f for f, _ in val_sel])
    multi = []  # (win, y, fname)
    win_stats = defaultdict(int)
    for x, f in va_clips:
        wins = slide_windows(x)
        win_stats[len(wins)] += 1
        for w in wins:
            multi.append((w, idx_map[f], f))
    print(f"clip 窗数分布: {dict(sorted(win_stats.items()))}", flush=True)
    print(f"多窗评估集: {len(multi)} 窗 × {len(KINDS)} kinds × {len(SNRS)} SNR", flush=True)

    model = SONTRA_A()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    recs = evaluate_system(model, AFRule(), multi, KINDS, SNRS, device="mps", tau=0.0)
    print(f"recs: {len(recs)}", flush=True)

    rmin = r_min_theory(AFRule().alpha_k, 7680)
    # 事件格: (fname, kind, snr) → [窗 recs]
    events = defaultdict(list)
    for r in recs:
        events[(r["fname"], r["kind"], r["snr_db"])].append(r)

    # 物理决策域（外生 r_true ≥ r_min）内的事件
    ev_phys = {k: v for k, v in events.items()
               if v[0]["r_true_db"] >= 10 * np.log10(rmin)}
    print(f"物理决策域事件格: {len(ev_phys)}", flush=True)

    def event_pred(ev, m):
        """事件级预测: ≥m 窗 decide 才决策; 多数类, 平局取窗级 max 概率最大者。"""
        decided = [r for r in ev if r["decide"]]
        if len(decided) < m:
            return None
        votes = defaultdict(float)
        for r in decided:
            p = r["event_probs"][:9]
            c = int(np.argmax(p))
            votes[c] += float(p[c])
        return max(votes, key=votes.get)

    def ev_labels(ev):
        return set(ev[0]["labels"])

    rows = []
    for m in range(1, K + 1):
        n_ev, n_hit, n_cov = 0, 0, 0
        for ev in ev_phys.values():
            pred = event_pred(ev, m)
            if pred is None:
                continue
            n_cov += 1
            n_ev += 1
            if pred in ev_labels(ev):
                n_hit += 1
        acc = n_hit / n_cov if n_cov else float("nan")
        rows.append({"m": m, "acc_at_dec": round(acc, 3),
                     "coverage": round(n_cov / len(ev_phys), 3), "n_dec": n_cov})
        print(f"m={m}: acc@dec {acc:.3f}  cov {n_cov/len(ev_phys):.3f}  n_dec={n_cov}", flush=True)

    # 对照 1: 多窗集单窗 acc@dec（无投票; 窗级）
    dphys = [r for r in recs if r["decide"] and r["r_true_db"] >= 10 * np.log10(rmin)]
    single = np.mean([1.0 if r["correct"] else 0.0 for r in dphys]) if dphys else float("nan")
    print(f"对照 单窗 acc@dec（多窗集, 无投票）: {single:.3f}  n={len(dphys)}", flush=True)

    # 对照 2: best-window 单窗（B 探针口径, 同批 clip）
    from aof.cf_sampler import CFSampler
    sampler = CFSampler([])
    best = [(sampler._best_window(x, C.WINDOW_SAMPLES), idx_map[f], f)
            for x, f in va_clips if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]
    recs_b = evaluate_system(model, AFRule(), best, KINDS, SNRS, device="mps", tau=0.0)
    bphys = [r for r in recs_b if r["decide"] and r["r_true_db"] >= 10 * np.log10(rmin)]
    best_acc = np.mean([1.0 if r["correct"] else 0.0 for r in bphys]) if bphys else float("nan")
    print(f"对照 best-window 单窗 acc@dec: {best_acc:.3f}  n={len(bphys)}", flush=True)

    out = {"k_windows": K, "event_grid": len(ev_phys),
           "voting": rows, "single_window_acc_at_dec": round(float(single), 3),
           "best_window_acc_at_dec": round(float(best_acc), 3),
           "clip_win_dist": dict(sorted(win_stats.items())), "target": 0.825}
    with open(os.path.join(OUT, "exp_p2_temporal_voting.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE → outputs/exp_p2_temporal_voting.json")


if __name__ == "__main__":
    main()
