"""exp_clap_snr_head_calib.py — learned SNR 头偏置校准（2026-08-16, decisive check #7）

问题: learned SNR 头 MAE 10.4 dB 且 coverage 0.571 > oracle 0.46 —— SNR̂ 系统性
      高估, 门放行 r_true < r_min 的窗口, risk 0.132 > α（NEAR_ALPHA）。
      本脚本用论文自带的 IsotonicCalibrator（aof/calibration.py, tau-Cal 血统）
      在校准集上对 learned SNR̂ 做单调校正, 检验校准后门是否闭合 α 线。

协议:
  校准集: fold 1-9（train 域）抽 300 clips（种子 C.SEED+2）, 损坏网格
          {-20,-5,10}dB × 3 family（种子 C.SEED+7 派生, 与 SNR 头训练一致）,
          收集 (learned SNR̂, r_true) 对 → IsotonicCalibrator.fit。
  评估: fold-10 test, 全网格 {-25,-15,-5,5,15} × 3（种子 C.SEED+11）,
        门 = P_D(corrected SNR̂) ≥ 1−α_k, τ 扫描 + Wilson CI + 单侧 p。
  对比: 未校准 learned（复现 0.132）/ 校准后 learned / oracle（诊断上界）。
判定（stdout VERDICT）:
  CALIBRATED_GATE_ATTAINS_ALPHA: 校准后存在 τ 使 risk < α 且 p < 0.05
                                 （可部署正结果: 全 10 类 + learned 门）
  CALIBRATED_GATE_NEAR_ALPHA:    risk < α 但 p ≥ 0.05
  CALIBRATED_GATE_BELOW:         无 τ 达 risk < α（诚实报告）

用法: python scripts/exp_clap_snr_head_calib.py
      [--ckpt outputs/checkpoints_clap_finetune/clap_ft_us8k_20260816.pt
       --snr-head outputs/checkpoints_clap_finetune/snr_head_us8k_20260816b.pt
       --n-cal 300 --device mps --tag 20260816]
输出: outputs/exp_clap_snr_head_calib_{tag}.json
注意: 新增实验记录; 校准集与测试集（fold 10）不重叠; 校准集与 SNR 头训练集
      同域（fold 1-9）——诚实披露于 JSON。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.baselines import SPAnchorB11
from aof.calibration import IsotonicCalibrator
from aof.metrics import operating_gap, coverage
from aof.wsosim import _wind, corrupt
from exp_clap_finetune import ClapFT, to_48k
from exp_clap_snr_head import SNRHead, wilson_ci, pd_from_snr
from exp_us8k import load_us8k_index, win_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
SR_TARGET = 48000
TRAIN_SNRS = [-20.0, -5.0, 10.0]
EVAL_SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
KINDS = ["wind", "occlusion", "self_motion"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]


def snr_of(model, proc, dev, head, xc):
    audio = proc(audio=[to_48k(xc)], sampling_rate=SR_TARGET, return_tensors="pt").to(dev)
    with torch.no_grad():
        emb = model.get_audio_features(**audio).pooler_output
        emb = F.normalize(emb, dim=-1)
        return float(head(emb).squeeze().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "checkpoints_clap_finetune",
                                                   "clap_ft_us8k_20260816.pt"))
    ap.add_argument("--snr-head", default=os.path.join(OUT, "checkpoints_clap_finetune",
                                                       "snr_head_us8k_20260816b.pt"))
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="20260816")
    args = ap.parse_args()

    dev = args.device if (args.device == "cpu" or torch.backends.mps.is_available()) else "cpu"
    if args.device != dev:
        print(f"WARN: {args.device} 不可用, 回退 {dev}", flush=True)

    from transformers.models.clap import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained(MODEL_ID).to(dev).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    for p in clap.parameters():
        p.requires_grad = False
    model_ft = ClapFT(clap, n_out=10, freeze_blocks=ckpt.get("freeze_blocks", 8)).to(dev)
    model_ft.load_state_dict(ckpt["state_dict"])
    model_ft.eval()
    head = SNRHead().to(dev)
    head.load_state_dict(torch.load(args.snr_head, map_location="cpu")["state_dict"])
    head.eval()
    print(f"ckpts loaded: {os.path.basename(args.ckpt)} + {os.path.basename(args.snr_head)}",
          flush=True)

    # ---- 校准集（fold 1-9, 300 clips, 种子固定）----
    idx = [e for e in load_us8k_index() if e[3] == "train"]
    clips, _ = win_clips(idx)
    rng_cal = np.random.default_rng(C.SEED + 2)
    perm = rng_cal.permutation(len(clips))[: args.n_cal]
    rng_cor = np.random.default_rng(C.SEED + 7)
    cal_snr_hat, cal_r_true = [], []
    for j in perm:
        x, y, fname = clips[j]
        for kind in KINDS:
            for snr_db in TRAIN_SNRS:
                seed = int(rng_cor.integers(1 << 31))
                xc, r_true, _ = corrupt(x, kind, float(snr_db), seed)
                if xc is None:
                    continue
                cal_snr_hat.append(snr_of(clap, proc, dev, head, xc))
                cal_r_true.append(float(r_true))
    cal_snr_hat = np.array(cal_snr_hat)
    cal_r_true = np.array(cal_r_true)
    iso = IsotonicCalibrator()
    iso.fit(cal_snr_hat, cal_r_true)
    mae_cal = iso.mae_before_after(cal_snr_hat, cal_r_true)
    print(f"校准集: n={len(cal_snr_hat)}, MAE before/after: {mae_cal}", flush=True)

    # ---- 评估（fold 10, 全网格）----
    te = [e for e in load_us8k_index() if e[3] == "test"]
    clips, dropped = win_clips(te)
    rule = AFRule()
    b11 = SPAnchorB11()
    rng = np.random.default_rng(C.SEED + 11)
    recs = []
    for x, y, fname in clips:
        labels = {int(np.argmax(y[:10]))}
        for kind in KINDS:
            for snr_db in EVAL_SNRS:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                w_ref = _wind(xc.shape[0], C.SAMPLE_RATE, seed + 1)[0].astype(np.float32)
                snr_hat_b11 = b11.snr_db(xc, w_ref)
                snr_hat_l = snr_of(clap, proc, dev, head, xc)
                snr_hat_c = float(iso.correct(np.array([snr_hat_l]))[0])
                audio = proc(audio=[to_48k(xc)], sampling_rate=SR_TARGET,
                             return_tensors="pt").to(dev)
                with torch.no_grad():
                    logits = model_ft(audio["input_features"], audio["is_longer"])
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                recs.append({"fname": fname, "kind": kind, "snr_db": snr_db,
                             "r_true_db": float(r_true), "snr_hat_b11": float(snr_hat_b11),
                             "snr_hat_learned": snr_hat_l, "snr_hat_calib": snr_hat_c,
                             "labels": labels, "probs": p})
    print(f"windows: {len(recs)}", flush=True)

    rmin_db = 10 * np.log10(rule.r_min)
    r_true = np.array([r["r_true_db"] for r in recs])
    probs = np.stack([r["probs"] for r in recs])
    labels_list = [r["labels"] for r in recs]
    n = len(recs)
    best = probs.max(axis=1)
    argmax = probs.argmax(axis=1)

    gates = {
        "learned_raw": pd_from_snr(np.array([r["snr_hat_learned"] for r in recs]), rule) >= (1.0 - rule.alpha_k),
        "learned_calib": pd_from_snr(np.array([r["snr_hat_calib"] for r in recs]), rule) >= (1.0 - rule.alpha_k),
        "oracle": r_true >= rmin_db,
    }
    mae_test = {"learned_raw": float(np.mean(np.abs(np.array([r["snr_hat_learned"] for r in recs]) - r_true))),
                "learned_calib": float(np.mean(np.abs(np.array([r["snr_hat_calib"] for r in recs]) - r_true)))}

    rows_all, hit, near = {}, [], []
    for gname, gate in gates.items():
        rows = {}
        for tau in TAUS:
            decide = gate & (best > tau)
            correct = np.array([decide[i] and argmax[i] in labels_list[i] for i in range(n)])
            gap, risk = operating_gap(decide, correct, C.ALPHA)
            nd = int(decide.sum())
            ci, pval = wilson_ci(int(correct.sum()), nd) if nd else (None, None)
            rows[f"tau{tau:.1f}"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                                     "coverage": round(coverage(decide), 3),
                                     "acc_at_dec": round(float(correct[decide].mean()), 3)
                                     if nd else None,
                                     "n_decide": nd, "wilson_ci": ci, "p_onesided": pval}
            print(f"[{gname} tau={tau:.1f}] {rows[f'tau{tau:.1f}']}", flush=True)
        rows_all[gname] = rows
        if gname == "learned_calib":
            hit = [(t, r) for t, r in rows.items()
                   if r["risk"] is not None and r["risk"] < C.ALPHA
                   and r["p_onesided"] is not None and r["p_onesided"] < 0.05]
            near = [(t, r) for t, r in rows.items()
                    if r["risk"] is not None and r["risk"] < C.ALPHA
                    and (r["p_onesided"] is None or r["p_onesided"] >= 0.05)]
    verdict = ("CALIBRATED_GATE_ATTAINS_ALPHA" if hit else
               ("CALIBRATED_GATE_NEAR_ALPHA" if near else "CALIBRATED_GATE_BELOW"))

    out = {"tag": args.tag, "n_cal": args.n_cal, "seed": C.SEED,
           "n_windows": n, "mae_calibration_split": mae_cal, "mae_test": mae_test,
           "verdict": verdict,
           "note": ("校准集 = fold 1-9 抽 300 clips（与 SNR 头训练集同域, 诚实披露）;"
                    " 测试集 fold 10 完全 hold-out; isotonic 校正 monotone 保序"),
           "interpretation": ("ATTAINS = 校准后 learned 门 risk < α 且 p < 0.05"
                              " → 全 10 类可部署正结果"),
           "rows": rows_all}
    with open(os.path.join(OUT, f"exp_clap_snr_head_calib_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}  (MAE test: {mae_test})", flush=True)
    print(f"DONE → outputs/exp_clap_snr_head_calib_{args.tag}.json")


if __name__ == "__main__":
    main()
