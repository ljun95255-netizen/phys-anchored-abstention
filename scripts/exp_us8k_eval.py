"""exp_us8k_eval.py — UrbanSound8K 评估（2026-08-06）

对 fold-10 test × {-25,-15,-5,5,15}dB × 3 kinds 单 pass:
  detector-tier: B0 / B11 / oracle / B11a（aof.evaluate, AFRule() 默认 α_k=α/10 → 与论文同边界）
  system-tier: B12(τ 扫描) / B12a(真 SNR 阈值) / B13(τ=0.0) — 自带 11 类循环
    （aof.evaluate_system 硬编码 labels < C.N_CLASSES-1=9, US8K unknown@10 不可用）
  附加列 viol = P(decide ∧ r_true < r_min)（统一风险补丁: detector 行语义 = 阈值违例,
    system 行语义 = 分类错误率, viol 使 system 行也可审计\"物理违规决策\"）
  逐 kind 消融 / per-class × SNR 热图 / 干净域 ceiling / top-k 命中
"""
import json
import os
import sys
import glob
import math
from collections import defaultdict

import numpy as np
import torch
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.baselines import EnergyDetectorB0, SPAnchorB11
from aof.data import log_mel
from aof.evaluate import evaluate_detector, evaluate_detector_oracle_threshold
from aof.metrics import operating_gap, coverage
from aof.model import SONTRA_A
from aof.wsosim import corrupt
from exp_us8k import US8K_NAMES, load_us8k_index, win_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
CKPT_DIR = os.path.join(OUT, "checkpoints_us8k")
KINDS = ["wind", "occlusion", "self_motion"]
SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
N_REAL = 10


def load_te_clips():
    index = load_us8k_index()
    te = [e for e in index if e[3] == "test"]
    clips, dropped = win_clips(te)
    print(f"test clips: {len(clips)} (dropped {dropped})", flush=True)
    return clips


def apply_rule(probs, snr_db, r_true, labels_list, tau, use_true_snr, rule):
    """numpy 版 AF-Rule（与 af_rule.decide 逐行一致; snr_db 为 [N,11] 逐类向量）。"""
    N = probs.shape[0]
    p = probs.copy()
    p[:, -1] = 0.0
    cand = p > tau
    snr = np.full_like(snr_db, np.nan)
    if use_true_snr:
        snr[:] = r_true[:, None]
    else:
        snr[:] = snr_db
    r = 10.0 ** (snr / 10.0)
    d = r * math.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    pd = 0.5 * (1.0 + erf(d / math.sqrt(2.0)))
    elig = cand & (pd >= 1.0 - rule.alpha_k)
    decide = elig.any(axis=1)
    pred = np.where(decide, (elig * p).argmax(axis=1), -1)
    correct = np.array([bool(decide[i]) and pred[i] in labels_list[i] for i in range(N)])
    return decide, correct, pred


def summ(decide, correct, alpha=C.ALPHA):
    gap, risk = operating_gap(decide, correct, alpha)
    acc = float(correct[decide].mean()) if decide.any() else float("nan")
    return {"gap": round(gap, 3), "risk": round(risk, 3),
            "coverage": round(coverage(decide), 3),
            "acc_at_dec": round(acc, 3), "n_decide": int(decide.sum())}


def main():
    rule = AFRule()                       # n_classes=10 → α_k=α/10, 与论文同边界
    rmin = rule.r_min
    rmin_db = 10 * math.log10(rmin)
    print(f"r_min = {rmin_db:.2f} dB (α_k={rule.alpha_k:.4f}, n={rule.n})", flush=True)

    clips = load_te_clips()
    rng = np.random.default_rng(C.SEED)

    # ---------- detector-tier ----------
    det = {}
    for name, d in [("B0_energy", EnergyDetectorB0()), ("B11_sp", SPAnchorB11())]:
        recs = evaluate_detector(d, clips, KINDS, SNRS)
        dec = np.array([r["decide"] for r in recs])
        r_true = np.array([r["r_true_db"] for r in recs])
        correct = ~dec | (r_true >= rmin_db)
        gap, risk = operating_gap(dec, correct, C.ALPHA)
        det[name] = {"gap": round(gap, 3), "risk": round(risk, 3),
                     "coverage": round(coverage(dec), 3),
                     "c_phys": round(float((r_true >= rmin_db).mean()), 3),
                     "viol": round(float((dec & (r_true < rmin_db)).mean()), 3)}
        print(f"[{name}] {det[name]}", flush=True)

    recs_or = evaluate_detector(EnergyDetectorB0(), clips, KINDS, SNRS)
    r_true = np.array([r["r_true_db"] for r in recs_or])
    dec = r_true >= rmin_db
    correct = ~dec | (r_true >= rmin_db)
    gap, risk = operating_gap(dec, correct, C.ALPHA)
    det["oracle"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                     "coverage": round(coverage(dec), 3), "viol": 0.0}
    print(f"[oracle] {det['oracle']}", flush=True)

    recs_b11a = evaluate_detector_oracle_threshold(SPAnchorB11(), clips, KINDS, SNRS)
    dec = np.array([r["decide"] for r in recs_b11a])
    correct = ~dec | (r_true >= rmin_db)
    gap, risk = operating_gap(dec, correct, C.ALPHA)
    det["B11a_sp_oracle_thr"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                                 "coverage": round(coverage(dec), 3), "viol": 0.0}
    print(f"[B11a] {det['B11a_sp_oracle_thr']}", flush=True)

    # ---------- system-tier 缓存 ----------
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")), key=os.path.getmtime)
    assert ckpts, f"no checkpoint in {CKPT_DIR}"
    ckpt = ckpts[-1]
    model = SONTRA_A(n_classes=11).to("mps").eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    print(f"ckpt: {os.path.basename(ckpt)}", flush=True)

    cache = []
    for x, y, fname in clips:
        labels = {i for i, v in enumerate(y) if v and i < N_REAL}
        for kind in KINDS:
            for snr_db in SNRS:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                mel = log_mel(torch.from_numpy(xc).unsqueeze(0).float().to("mps"))
                with torch.no_grad():
                    out = model(mel)
                cache.append({
                    "probs": out["event_probs"].cpu().numpy()[0],
                    "snr_vec": out["snr_db"].cpu().numpy()[0],
                    "labels": labels, "r_true": float(r_true),
                    "kind": kind, "snr_db": float(snr_db), "fname": fname})
    print(f"cached {len(cache)} windows", flush=True)

    probs = np.stack([c["probs"] for c in cache])
    snr_vec = np.stack([c["snr_vec"] for c in cache])
    r_true = np.array([c["r_true"] for c in cache])
    labels_list = [c["labels"] for c in cache]
    kinds = np.array([c["kind"] for c in cache])
    snrs = np.array([c["snr_db"] for c in cache])

    # ---------- B12 / B12a / B13 τ 扫描 ----------
    sys_tbl = {}
    for tau in TAUS:
        dec, corr, _ = apply_rule(probs, snr_vec, r_true, labels_list, tau, False, rule)
        s = summ(dec, corr)
        viol = float((dec & (r_true < rmin_db)).mean())
        sys_tbl[f"B12_tau{tau:.2f}"] = {**s, "viol": round(viol, 3)}
    dec_a, corr_a, _ = apply_rule(probs, snr_vec, r_true, labels_list, 0.5, True, rule)
    sys_tbl["B12a_true_snr"] = {**summ(dec_a, corr_a),
                                "viol": round(float((dec_a & (r_true < rmin_db)).mean()), 3)}
    dec_b13, corr_b13, _ = apply_rule(probs, snr_vec, r_true, labels_list, 0.0, False, rule)
    sys_tbl["B13_phys_only"] = {**summ(dec_b13, corr_b13),
                                "viol": round(float((dec_b13 & (r_true < rmin_db)).mean()), 3)}
    for k in ["B12_tau0.50", "B12_tau0.95", "B12a_true_snr", "B13_phys_only"]:
        print(f"[{k}] {sys_tbl[k]}", flush=True)

    # ---------- 逐 kind 消融 (B12 τ=0.5) ----------
    ab = {}
    for kind in KINDS:
        m = kinds == kind
        dec, corr, _ = apply_rule(probs[m], snr_vec[m], r_true[m],
                                  [labels_list[i] for i in np.nonzero(m)[0]], 0.5, False, rule)
        ab[kind] = summ(dec, corr)
    print(f"per-kind: {ab}", flush=True)

    # ---------- per-class × SNR 热图 (B12 τ=0.5, 决策域) ----------
    dec05, corr05, pred05 = apply_rule(probs, snr_vec, r_true, labels_list, 0.5, False, rule)
    heat = {}
    for k in range(N_REAL):
        cls_rows = {}
        for s in SNRS:
            m = np.array([k in labels_list[i] and snrs[i] == s for i in range(len(cache))])
            if m.sum() == 0:
                cls_rows[str(int(s))] = {"n": 0}
                continue
            dk, ck = dec05[m], corr05[m]
            cls_rows[str(int(s))] = {
                "n": int(m.sum()),
                "acc_at_dec": round(float(ck[dk].mean()), 3) if dk.any() else None,
                "n_decide": int(dk.sum())}
        heat[US8K_NAMES[k]] = cls_rows
    print(f"heatmap done", flush=True)

    # ---------- 干净域 ceiling + top-k ----------
    clean_top1, clean_topk = [], {1: 0, 2: 0, 3: 0}
    clean_n = 0
    for x, y, fname in clips:
        labels = {i for i, v in enumerate(y) if v and i < N_REAL}
        mel = log_mel(torch.from_numpy(x).unsqueeze(0).float().to("mps"))
        with torch.no_grad():
            out = model(mel)
        p = out["event_probs"].cpu().numpy()[0][:N_REAL]
        order = np.argsort(p)[::-1]
        clean_top1.append(1.0 if order[0] in labels else 0.0)
        for kk in (1, 2, 3):
            clean_topk[kk] += 1.0 if len(set(order[:kk]) & labels) > 0 else 0.0
        clean_n += 1
    clean = {"top1": round(float(np.mean(clean_top1)), 3),
             "top2": round(clean_topk[2] / clean_n, 3),
             "top3": round(clean_topk[3] / clean_n, 3), "n": clean_n}
    print(f"clean ceiling: {clean}", flush=True)

    # 腐蚀决策域 top-k（P3 类比）
    phys_m = r_true >= rmin_db
    pk = {1: 0, 2: 0, 3: 0}
    for i in np.nonzero(phys_m)[0]:
        order = np.argsort(probs[i][:N_REAL])[::-1]
        for kk in (1, 2, 3):
            pk[kk] += 1.0 if len(set(order[:kk]) & labels_list[i]) > 0 else 0.0
    topk_corr = {f"top{kk}": round(pk[kk] / max(phys_m.sum(), 1), 3) for kk in (1, 2, 3)}
    print(f"decidable top-k: {topk_corr}", flush=True)

    out = {"ckpt": os.path.basename(ckpt), "r_min_db": round(rmin_db, 2),
           "detectors": det, "systems": sys_tbl, "per_kind": ab,
           "heatmap": heat, "clean_ceiling": clean, "topk_decidable": topk_corr,
           "n_windows": len(cache), "n_clips": len(clips)}
    with open(os.path.join(OUT, "exp_us8k_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE → outputs/exp_us8k_eval.json")


if __name__ == "__main__":
    main()
