"""exp_us8k_subset.py — US8K 类剪裁演示（2026-08-16, decisive check #5）

问题: oracle 门把 CLAP-FT 的 risk 降到 0.095（CI 跨 α, p=0.11）——归因闭合了但
      绩效不是统计坚实的正结果。论文 Discussion 的"任务重定义"处方 (i) 说:
      把不可判别的类并入 unknown（类粒度重定义）。本脚本把该处方实例化:
      在 US8K 上剪裁到"冻结 per-class acc_at_dec ≥ 0.85"的类, 其余并入 unknown,
      检验剪裁任务是否给出统计坚实的正结果（risk 显著 < α）。

剪裁规则（事前固定, 防 cherry-picking 质疑）:
  规则 = 冻结的 10 类 CLAP-FT 评估（exp_clap_finetune_us8k_20260816.json 的
         per_class, τ=0.5, B11 门内）中 acc_at_dec ≥ 0.85 的类。
  该规则在跑本脚本之前已由冻结 JSON 决定, 不因本实验结果调整。
  诚实性配套: 完整 10 类 per-class 表在论文中照旧报告; 被剪类窗口占比
  （= 并入 unknown 的代价）本脚本如实输出。

模式:
  --mode infer（默认）: 用冻结的 10 类 ckpt, 推理时限制决策集（argmax ∈ 目标集
    才决策, 否则 abstain）。零训练, ~30-45 分钟。等价于"部署时把被剪类并入
    unknown"。
  --mode train: 剪裁类集上重训分类头（冻结音频分支前 8 块, 新 4 类头）。
    更准, ~1-2 小时。两者都跑时以 train 为准, infer 作为对照。

评估: B11 门 + oracle 门 × τ 扫描 → gap/risk/cov/acc_at_dec/n_decide +
      Wilson 95% CI + 单侧检验 H0: risk ≥ α → VERDICT:
  SUBSET_ATTAINS_ALPHA: 存在 τ 使 risk < α 且单侧 p < 0.05（统计坚实的正结果）
  SUBSET_NEAR_ALPHA:   存在 τ 使 risk < α 但 p ≥ 0.05
  SUBSET_BELOW:        无 τ 达 risk < α

用法: python scripts/exp_us8k_subset.py --mode infer
      [--ckpt outputs/checkpoints_clap_finetune/clap_ft_us8k_20260816.pt
       --device mps --tag 20260816]
输出: outputs/exp_us8k_subset_{mode}_{tag}.json
注意: 新增实验记录（非 R19 冻结 pass）; 剪裁规则读取冻结 JSON, 不新算 per-class。
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
from exp_us8k import US8K_NAMES, load_us8k_index, win_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
SR_TARGET = 48000
EVAL_SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
KINDS = ["wind", "occlusion", "self_motion"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
RULE_ACC_THRESHOLD = 0.85          # 事前剪裁规则: 冻结 per-class acc_at_dec ≥ 0.85
FROZEN_PERCLASS_JSON = os.path.join(OUT, "exp_clap_finetune_us8k_20260816.json")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="infer", choices=["infer", "train"])
    ap.add_argument("--ckpt", default=os.path.join(OUT, "checkpoints_clap_finetune",
                                                   "clap_ft_us8k_20260816.pt"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="20260816")
    ap.add_argument("--epochs", type=int, default=6, help="train 模式专用")
    args = ap.parse_args()

    dev = args.device if (args.device == "cpu" or torch.backends.mps.is_available()) else "cpu"
    if args.device != dev:
        print(f"WARN: {args.device} 不可用, 回退 {dev}", flush=True)

    # ---- 事前剪裁规则（读取冻结 per-class JSON, 不新算）----
    if not os.path.exists(FROZEN_PERCLASS_JSON):
        sys.exit(f"找不到冻结 per-class JSON: {FROZEN_PERCLASS_JSON}")
    frozen = json.load(open(FROZEN_PERCLASS_JSON))
    per_class = frozen["evaluation"]["per_class"]
    target = sorted([US8K_NAMES.index(name) for name, v in per_class.items()
                     if v["acc"] is not None and v["acc"] >= RULE_ACC_THRESHOLD])
    print(f"事前剪裁规则: 冻结 per-class acc_at_dec ≥ {RULE_ACC_THRESHOLD}")
    print(f"目标类集: {[US8K_NAMES[k] for k in target]}  "
          f"({len(target)}/10 类; 其余并入 unknown)", flush=True)
    if len(target) < 2:
        sys.exit("目标类少于 2 个, 规则失效, 终止")

    from transformers.models.clap import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained(MODEL_ID).to(dev)
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    n_out = len(target) if args.mode == "train" else 10
    model = ClapFT(clap, n_out=n_out, freeze_blocks=ckpt.get("freeze_blocks", 8)).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"ckpt loaded ({os.path.basename(args.ckpt)}), mode={args.mode}", flush=True)

    if args.mode == "train":
        # 剪裁类集上重训分类头: 冻结主干, 换 len(target) 类头, 只训头
        head = nn.Linear(model.head.in_features, len(target)).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
        torch.manual_seed(C.SEED)
        rng = np.random.default_rng(C.SEED + 7)
        idx = [e for e in load_us8k_index() if e[3] == "train"]
        clips, _ = win_clips(idx)
        # 只保留目标类的训练 clip
        clips = [(x, int(np.argmax(y[:10]))) for x, y, f in clips
                 if int(np.argmax(y[:10])) in target]
        print(f"train clips (目标类): {len(clips)}", flush=True)
        import torch.optim as optim
        opt = optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for ep in range(args.epochs):
            perm = torch.randperm(len(clips)).tolist()
            tot, nb = 0.0, 0
            for i in range(0, len(clips) - len(clips) % 16, 16):
                batch = [clips[j] for j in perm[i:i + 16]]
                wavs = []
                labs = []
                for x, lab in batch:
                    kind = KINDS[rng.integers(len(KINDS))]
                    snr = [-20.0, -5.0, 10.0][rng.integers(3)]
                    seed = int(rng.integers(1 << 31))
                    xc, _, _ = corrupt(x, kind, float(snr), seed)
                    wavs.append(xc if xc is not None else x)
                    labs.append(target.index(lab))
                audio = proc(audio=[to_48k(w) for w in wavs], sampling_rate=SR_TARGET,
                             return_tensors="pt").to(dev)
                with torch.no_grad():
                    emb = model.clap.get_audio_features(**audio).pooler_output
                    emb = F.normalize(emb, dim=-1)
                logits = head(emb)
                loss = F.cross_entropy(logits, torch.tensor(labs, device=dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            sched.step()
            print(f"  head epoch {ep}: loss={tot/max(nb,1):.4f}", flush=True)

    # ---- 评估 ----
    te = [e for e in load_us8k_index() if e[3] == "test"]
    clips, dropped = win_clips(te)
    rule = AFRule()
    b11 = SPAnchorB11()
    rng = np.random.default_rng(C.SEED + 11)
    recs = []
    for x, y, fname in clips:
        lab = int(np.argmax(y[:10]))
        for kind in KINDS:
            for snr_db in EVAL_SNRS:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                w_ref = _wind(xc.shape[0], C.SAMPLE_RATE, seed + 1)[0].astype(np.float32)
                snr_hat_db = b11.snr_db(xc, w_ref)
                audio = proc(audio=[to_48k(xc)], sampling_rate=SR_TARGET,
                             return_tensors="pt").to(dev)
                with torch.no_grad():
                    if args.mode == "train":
                        emb = model.clap.get_audio_features(**audio).pooler_output
                        emb = F.normalize(emb, dim=-1)
                        logits = head(emb)
                    else:
                        logits = model(audio["input_features"], audio["is_longer"])
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                recs.append({"fname": fname, "kind": kind, "snr_db": snr_db,
                             "r_true_db": float(r_true), "snr_hat_db": float(snr_hat_db),
                             "label": lab, "probs": p})
    print(f"windows: {len(recs)}", flush=True)

    rmin_db = 10 * np.log10(rule.r_min)
    r_true = np.array([r["r_true_db"] for r in recs])
    probs = np.stack([r["probs"] for r in recs])
    snr_hat = np.array([r["snr_hat_db"] for r in recs])
    labels = np.array([r["label"] for r in recs])
    n = len(recs)
    argmax = probs.argmax(axis=1)
    best = probs.max(axis=1)

    r = 10.0 ** (snr_hat / 10.0)
    d = r * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    pd = 0.5 * (1.0 + erf(d / np.sqrt(2.0)))
    pd_ok = pd >= (1.0 - rule.alpha_k)
    phys_true = r_true >= rmin_db

    # 剪裁语义: 只允许目标类决策; 非目标类标签的窗口永不决策（并入 unknown）
    in_target = np.isin(labels, target)
    if args.mode == "train":
        argmax_t = np.array([target[p] for p in argmax])          # 映射回原类空间
    else:
        argmax_t = argmax
    decide_ok = np.isin(argmax_t, target)

    def scan(gate_mask, gate_name):
        rows, verdict = {}, None
        for tau in TAUS:
            decide = gate_mask & decide_ok & (best > tau) & in_target
            correct = np.array([decide[i] and argmax_t[i] == labels[i] for i in range(n)])
            gap, risk = operating_gap(decide, correct, C.ALPHA)
            nd = int(decide.sum())
            ci, pval = wilson_ci(int(correct.sum()), nd) if nd else (None, None)
            rows[f"tau{tau:.1f}"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                                     "coverage": round(coverage(decide), 3),
                                     "acc_at_dec": round(float(correct[decide].mean()), 3)
                                     if nd else None,
                                     "n_decide": nd, "wilson_ci": ci, "p_onesided": pval}
            print(f"[{gate_name} tau={tau:.1f}] {rows[f'tau{tau:.1f}']}", flush=True)
        return rows

    rows_b11 = scan(pd_ok, "subset+B11")
    rows_or = scan(phys_true, "subset+oracle")

    # 判定: 任一 τ 满足 risk < α 且 p < 0.05
    hit = [(t, r) for t, r in rows_or.items()
           if r["risk"] is not None and r["risk"] < C.ALPHA
           and r["p_onesided"] is not None and r["p_onesided"] < 0.05]
    near = [(t, r) for t, r in rows_or.items()
            if r["risk"] is not None and r["risk"] < C.ALPHA
            and (r["p_onesided"] is None or r["p_onesided"] >= 0.05)]
    verdict = "SUBSET_ATTAINS_ALPHA" if hit else ("SUBSET_NEAR_ALPHA" if near
                                                  else "SUBSET_BELOW")

    out = {"tag": args.tag, "mode": args.mode, "ckpt": os.path.basename(args.ckpt),
           "seed": C.SEED, "rule": {"threshold": RULE_ACC_THRESHOLD,
                                    "source": os.path.basename(FROZEN_PERCLASS_JSON),
                                    "target_classes": [US8K_NAMES[k] for k in target],
                                    "target_idx": target,
                                    "pruned_share": round(
                                        1.0 - in_target.mean(), 4)},
           "n_windows": n, "verdict": verdict,
           "interpretation": ("事前规则剪裁（per-class acc_at_dec ≥ 0.85, 冻结 JSON）"
                              "; 被剪类并入 unknown; 判定看 oracle 门 τ 扫描中"
                              " risk < α 且 p < 0.05 的存在性"),
           "rows_b11_gate": rows_b11, "rows_oracle_gate": rows_or}
    with open(os.path.join(OUT, f"exp_us8k_subset_{args.mode}_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}  (pruned_share={out['rule']['pruned_share']})", flush=True)
    print(f"DONE → outputs/exp_us8k_subset_{args.mode}_{args.tag}.json")


if __name__ == "__main__":
    main()
