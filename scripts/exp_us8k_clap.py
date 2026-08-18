"""exp_us8k_clap.py — CLAP zero-shot 探针（US8K, 2026-08-06）

问题: 预训练音频表示在 US8K 物理决策域的 acc@dec 是多少?（对照 SONTRA-A / 经典锚）
架构: CLAP(htsat-unfused) 分类 + B11(SP anchor) SNR̂ → AF-Rule 弃权
  （CLAP 无 SNR 头, 用经典锚提供物理门控 — 与 SONTRA-A 的 A-Head 对照）
输出: zero-shot top-1（全域/物理域）+ 决策域 gap/risk/cov（τ 扫描）+ per-class
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.baselines import SPAnchorB11
from aof.metrics import operating_gap, coverage
from aof.wsosim import _wind, corrupt
from exp_us8k import US8K_NAMES, load_us8k_index, win_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
SR_TARGET = 48000
KINDS = ["wind", "occlusion", "self_motion"]
SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
N_REAL = 10

CLASS_PROMPTS = [
    "an air conditioner running",
    "a car horn honking",
    "children playing",
    "a dog barking",
    "drilling noise",
    "an engine idling",
    "a gun shot",
    "a jackhammer operating",
    "a siren wailing",
    "street music playing",
]


def to_48k(x, fs=16000):
    from scipy.signal import resample_poly
    return resample_poly(x, SR_TARGET, fs).astype(np.float32)


def main():
    from transformers.models.clap import ClapModel, ClapProcessor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ClapModel.from_pretrained(MODEL_ID).to(dev).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    print(f"CLAP loaded on {dev}", flush=True)

    texts = CLASS_PROMPTS
    text_inputs = proc(text=texts, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        t_emb = model.get_text_features(**text_inputs)
        t_emb = F.normalize(t_emb.pooler_output, dim=-1).cpu()
    print(f"text embeddings: {t_emb.shape}", flush=True)

    index = load_us8k_index()
    te = [e for e in index if e[3] == "test"]
    clips, dropped = win_clips(te)
    print(f"test clips: {len(clips)} (dropped {dropped})", flush=True)

    rule = AFRule()
    b11 = SPAnchorB11()
    rng = np.random.default_rng(C.SEED)
    recs = []
    for x, y, fname in clips:
        labels = {i for i, v in enumerate(y) if v and i < N_REAL}
        for kind in KINDS:
            for snr_db in SNRS:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                w_ref = _wind(xc.shape[0], C.SAMPLE_RATE, seed + 1)[0].astype(np.float32)
                snr_hat_db = b11.snr_db(xc, w_ref)
                wav48 = to_48k(xc)
                audio = proc(audio=wav48, sampling_rate=SR_TARGET, return_tensors="pt").to(dev)
                with torch.no_grad():
                    a_emb = model.get_audio_features(**audio)
                    a_emb = F.normalize(a_emb.pooler_output, dim=-1).cpu()
                sim = a_emb @ t_emb.T  # [1,10]
                recs.append({"fname": fname, "kind": kind, "snr_db": snr_db,
                             "r_true_db": float(r_true), "snr_hat_db": float(snr_hat_db),
                             "labels": labels, "sim": sim[0].numpy()})
    print(f"recs: {len(recs)}", flush=True)

    rmin_db = 10 * np.log10(rule.r_min)
    r_true = np.array([r["r_true_db"] for r in recs])
    sims = np.stack([r["sim"] for r in recs])
    snr_hat = np.array([r["snr_hat_db"] for r in recs])
    labels_list = [r["labels"] for r in recs]

    # zero-shot top-1（全域 / 物理域）
    top1_all = np.array([int(np.argmax(sims[i])) in labels_list[i] for i in range(len(recs))])
    phys_m = r_true >= rmin_db
    print(f"ZS top-1 all: {top1_all.mean():.3f}  phys: {top1_all[phys_m].mean():.3f} "
          f"(n={phys_m.sum()})", flush=True)

    # 决策域: B11 SNR̂ + AF-Rule + sim 阈值 τ
    r = 10.0 ** (snr_hat / 10.0)
    d = r * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    pd = 0.5 * (1.0 + erf(d / np.sqrt(2.0)))
    pd_ok = pd >= (1.0 - rule.alpha_k)
    best_sim = sims.max(axis=1)

    rows = {}
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        decide = pd_ok & (best_sim > tau)
        pred = np.where(decide, sims.argmax(axis=1), -1)
        correct = np.array([decide[i] and pred[i] in labels_list[i] for i in range(len(recs))])
        gap, risk = operating_gap(decide, correct, C.ALPHA)
        acc = float(correct[decide].mean()) if decide.any() else float("nan")
        rows[f"tau{tau:.1f}"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                                 "coverage": round(coverage(decide), 3),
                                 "acc_at_dec": round(acc, 3)}
        print(f"[CLAP+B11 tau={tau:.1f}] {rows[f'tau{tau:.1f}']}", flush=True)

    # per-class ZS（物理域）
    cls = defaultdict(lambda: [0, 0])
    for i in np.nonzero(phys_m)[0]:
        for k in labels_list[i]:
            cls[k][1] += 1
            cls[k][0] += 1 if int(np.argmax(sims[i])) in labels_list[i] else 0
    per_class = {US8K_NAMES[k]: {"hit": v[0], "n": v[1],
                                 "acc": round(v[0] / v[1], 3) if v[1] else None}
                 for k, v in sorted(cls.items())}
    print(f"per-class: {json.dumps(per_class)}", flush=True)

    out = {"zero_shot": {"top1_all": round(float(top1_all.mean()), 3),
                         "top1_phys": round(float(top1_all[phys_m].mean()), 3),
                         "n_phys": int(phys_m.sum())},
           "rule_rows": rows, "per_class_phys": per_class}
    with open(os.path.join(OUT, "exp_us8k_clap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE → outputs/exp_us8k_clap.json")


if __name__ == "__main__":
    main()
