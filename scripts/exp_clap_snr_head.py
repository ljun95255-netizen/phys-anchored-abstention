"""exp_clap_snr_head.py — CLAP 专用 SNR 头（2026-08-16, decisive check #6）

问题: oracle 门（r_true）证明 US8K 残余失败是门失配, 但 oracle 门是诊断上界,
      部署时不可得。本脚本把诊断变成可部署系统: 给 CLAP 嵌入训练一个 SNR 回归头
      （CFAL 式监督, 标签 = WSO-Sim 注入的 r_true）, 评估 CLAP-FT + learned SNR 门
      是否达到 α 线（三门对比: B11 / learned / oracle）。

协议:
  训练: US8K fold 1-9, 损坏网格 {-20,-5,10}dB × 3 family, 嵌入 = CLAP audio
        features (pooler, L2 归一化), 头 = MLP(512→128→1) 回归 SNR(dB),
        MSE 损失, AdamW, ~4 epochs（种子 C.SEED 派生）。
  评估: fold-10 test, 全网格 {-25,-15,-5,5,15} × 3, τ 扫描;
        门 = P_D(learned SNR̂) ≥ 1−α_k（与 AF-Rule 相同构造, 只是 SNR̂ 换源）;
        Wilson CI + 单侧检验 H0: risk ≥ α → VERDICT:
  LEARNED_GATE_ATTAINS_ALPHA: 存在 τ 使 risk < α 且 p < 0.05（可部署正结果）
  LEARNED_GATE_NEAR_ALPHA:   risk < α 但 p ≥ 0.05
  LEARNED_GATE_BELOW:        无 τ 达 risk < α（诚实报告, 保留 oracle 诊断）

用法: python scripts/exp_clap_snr_head.py
      [--ckpt outputs/checkpoints_clap_finetune/clap_ft_us8k_20260816.pt
       --epochs 4 --device mps --tag 20260816]
输出: outputs/exp_clap_snr_head_{tag}.json + checkpoints_clap_finetune/snr_head_us8k_{tag}.pt
注意: 新增实验记录; learned SNR 头是独立模型组件（不覆盖冻结 ckpt）。
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
from aof.metrics import operating_gap, coverage
from aof.wsosim import _wind, corrupt
from exp_clap_finetune import ClapFT, to_48k
from exp_us8k import load_us8k_index, win_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
SR_TARGET = 48000
TRAIN_SNRS = [-20.0, -5.0, 10.0]
EVAL_SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
KINDS = ["wind", "occlusion", "self_motion"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]


class SNRHead(nn.Module):
    def __init__(self, d_in=512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, emb):
        return self.mlp(emb).squeeze(-1)


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI + 单侧 p（H0: risk >= 0.1, 对错误率 (n-k)/n 检验）。"""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    err = 1.0 - p
    se = math.sqrt(0.1 * 0.9 / n)
    zstat = (err - 0.1) / se
    pval = 0.5 * (1 + math.erf(zstat / math.sqrt(2)))
    return (round(center - half, 3), round(center + half, 3)), round(pval, 3)


def pd_from_snr(snr_db, rule):
    r = 10.0 ** (np.asarray(snr_db) / 10.0)
    d = r * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    return 0.5 * (1.0 + erf(d / np.sqrt(2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "checkpoints_clap_finetune",
                                                   "clap_ft_us8k_20260816.pt"))
    ap.add_argument("--epochs", type=int, default=4)
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
    # CLAP-FT 分类模型（冻结 ckpt, 循环外只构建一次）
    model_ft = ClapFT(clap, n_out=10, freeze_blocks=ckpt.get("freeze_blocks", 8)).to(dev)
    model_ft.load_state_dict(ckpt["state_dict"])
    model_ft.eval()
    print(f"CLAP loaded (frozen), head epochs={args.epochs}", flush=True)

    # ---- 训练 SNR 头 ----
    idx = [e for e in load_us8k_index() if e[3] == "train"]
    clips, _ = win_clips(idx)
    head = SNRHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED + 7)
    print(f"train clips: {len(clips)}", flush=True)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(clips)).tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(clips) - len(clips) % 16, 16):
            batch = [clips[j] for j in perm[i:i + 16]]
            wavs, snrs = [], []
            for x, y, f in batch:
                kind = KINDS[rng.integers(len(KINDS))]
                snr = TRAIN_SNRS[rng.integers(len(TRAIN_SNRS))]
                seed = int(rng.integers(1 << 31))
                xc, r_true, _ = corrupt(x, kind, float(snr), seed)
                wavs.append(xc if xc is not None else x)
                snrs.append(float(r_true))
            audio = proc(audio=[to_48k(w) for w in wavs], sampling_rate=SR_TARGET,
                         return_tensors="pt").to(dev)
            with torch.no_grad():
                emb = clap.get_audio_features(**audio).pooler_output
                emb = F.normalize(emb, dim=-1)
            pred = head(emb)
            loss = F.mse_loss(pred, torch.tensor(snrs, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        print(f"  SNR head epoch {ep}: mse={tot/max(nb,1):.4f}", flush=True)
    head.eval()

    os.makedirs(os.path.join(OUT, "checkpoints_clap_finetune"), exist_ok=True)
    head_path = os.path.join(OUT, "checkpoints_clap_finetune", f"snr_head_us8k_{args.tag}.pt")
    torch.save({"state_dict": head.state_dict(), "seed": C.SEED, "epochs": args.epochs},
               head_path)

    # ---- 评估 ----
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
                audio = proc(audio=[to_48k(xc)], sampling_rate=SR_TARGET,
                             return_tensors="pt").to(dev)
                with torch.no_grad():
                    logits = model_ft(audio["input_features"], audio["is_longer"])
                    emb = clap.get_audio_features(**audio).pooler_output
                    emb = F.normalize(emb, dim=-1)
                    snr_hat_learned = float(head(emb).squeeze().cpu().numpy())
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                recs.append({"fname": fname, "kind": kind, "snr_db": snr_db,
                             "r_true_db": float(r_true), "snr_hat_b11": float(snr_hat_b11),
                             "snr_hat_learned": snr_hat_learned,
                             "labels": labels, "probs": p})
    print(f"windows: {len(recs)}", flush=True)

    rmin_db = 10 * np.log10(rule.r_min)
    r_true = np.array([r["r_true_db"] for r in recs])
    probs = np.stack([r["probs"] for r in recs])
    labels_list = [r["labels"] for r in recs]
    n = len(recs)
    best = probs.max(axis=1)
    argmax = probs.argmax(axis=1)
    phys_true = r_true >= rmin_db

    gates = {
        "b11": pd_from_snr(np.array([r["snr_hat_b11"] for r in recs]), rule) >= (1.0 - rule.alpha_k),
        "learned": pd_from_snr(np.array([r["snr_hat_learned"] for r in recs]), rule) >= (1.0 - rule.alpha_k),
        "oracle": phys_true,
    }
    snr_mae = {"b11": float(np.mean(np.abs(np.array([r["snr_hat_b11"] for r in recs]) - r_true))),
               "learned": float(np.mean(np.abs(np.array([r["snr_hat_learned"] for r in recs]) - r_true)))}

    rows_all, verdict = {}, None
    hit, near = [], []
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
        if gname == "learned":
            hit = [(t, r) for t, r in rows.items()
                   if r["risk"] is not None and r["risk"] < C.ALPHA
                   and r["p_onesided"] is not None and r["p_onesided"] < 0.05]
            near = [(t, r) for t, r in rows.items()
                    if r["risk"] is not None and r["risk"] < C.ALPHA
                    and (r["p_onesided"] is None or r["p_onesided"] >= 0.05)]
    verdict = ("LEARNED_GATE_ATTAINS_ALPHA" if hit else
               ("LEARNED_GATE_NEAR_ALPHA" if near else "LEARNED_GATE_BELOW"))

    out = {"tag": args.tag, "ckpt": os.path.basename(args.ckpt), "seed": C.SEED,
           "epochs": args.epochs, "n_windows": n, "snr_mae_db": snr_mae,
           "verdict": verdict,
           "interpretation": ("learned SNR 头（可部署）vs B11 vs oracle 三门对比;"
                              " ATTAINS = risk < α 且 p < 0.05 的可部署正结果"),
           "rows": rows_all}
    with open(os.path.join(OUT, f"exp_clap_snr_head_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}  (SNR MAE: {snr_mae})", flush=True)
    print(f"DONE → outputs/exp_clap_snr_head_{args.tag}.json")


if __name__ == "__main__":
    main()
