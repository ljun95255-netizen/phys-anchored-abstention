"""exp_clap_gate_probe.py — CLAP-FT 门探针（2026-08-16, 残余归因 decisive check #4）

问题: CLAP-FT 在 US8K 上把 acc_at_dec 抬到 0.864（τ=0.5, coverage 0.303）, 但
      risk 0.136 > α=0.1 到不了 α 线。残余失败有两个候选解释:
      (a) 物理门失配: B11 SNR̂ 门是为 SONTRA-A 通道设计的, 与 CLAP 的鲁棒性不匹配,
          挡掉了 CLAP 本可正确决策的窗口（coverage 0.303 << 物理域 0.600）;
      (b) 残余任务结构: 即使门完美, 可决策域内仍存在分类器达不到 α 的窗口。
      本脚本把门换成 oracle（r_true ≥ r_min, 真值 SNR 界定物理域）来区分两者。

判定（stdout VERDICT + JSON）:
  ORACLE_GATE_ATTAINS_ALPHA: oracle 门下存在 τ 使 risk ≤ α 且 coverage 明显 > 0.303
    → 残余失败是门失配（工程可修: 给 CLAP 配专用 SNR 头/门）, US8K 正结果成立
  ORACLE_GATE_BELOW_ALPHA: oracle 门下仍无 τ 达 risk ≤ α
    → 残余失败是任务结构（诚实报告, 维持 4.3 的"awaiting attribution"表述）
  另输出 B11 门复现行（与 exp_clap_finetune_us8k 的 τ 扫描对照, 验证管线一致）。

用法: python scripts/exp_clap_gate_probe.py --dataset us8k --ckpt outputs/checkpoints_clap_finetune/clap_ft_us8k_20260816.pt
      [--device mps --tag 20260816]
输出: outputs/exp_clap_gate_probe_{dataset}_{tag}.json
注意: 新增实验记录（非 R19 冻结 pass, 不重跑任何冻结数字）; 复用已冻结 ckpt 推理。
"""
import argparse
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
from exp_clap_finetune import ClapFT, to_48k
from exp_us8k import US8K_NAMES, load_us8k_index, win_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
SR_TARGET = 48000
EVAL_SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
KINDS = ["wind", "occlusion", "self_motion"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="us8k", choices=["us8k"])
    ap.add_argument("--ckpt", default=os.path.join(OUT, "checkpoints_clap_finetune",
                                                   "clap_ft_us8k_20260816.pt"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="20260816")
    args = ap.parse_args()

    dev = args.device if (args.device == "cpu" or torch.backends.mps.is_available()) else "cpu"
    if args.device != dev:
        print(f"WARN: {args.device} 不可用, 回退 {dev}", flush=True)

    from transformers.models.clap import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained(MODEL_ID).to(dev)
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapFT(clap, n_out=10, freeze_blocks=8).to(dev)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"ckpt loaded: {os.path.basename(args.ckpt)} (freeze_blocks={ckpt.get('freeze_blocks')})",
          flush=True)

    te = [e for e in load_us8k_index() if e[3] == "test"]
    clips, dropped = win_clips(te)
    print(f"test clips: {len(clips)} (dropped {dropped})", flush=True)

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
                snr_hat_db = b11.snr_db(xc, w_ref)
                audio = proc(audio=[to_48k(xc)], sampling_rate=SR_TARGET,
                             return_tensors="pt").to(dev)
                with torch.no_grad():
                    logits = model(audio["input_features"], audio["is_longer"])
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                recs.append({"fname": fname, "kind": kind, "snr_db": snr_db,
                             "r_true_db": float(r_true), "snr_hat_db": float(snr_hat_db),
                             "labels": labels, "probs": p})
    print(f"windows: {len(recs)}", flush=True)

    rmin_db = 10 * np.log10(rule.r_min)
    r_true = np.array([r["r_true_db"] for r in recs])
    probs = np.stack([r["probs"] for r in recs])
    snr_hat = np.array([r["snr_hat_db"] for r in recs])
    labels_list = [r["labels"] for r in recs]
    n = len(recs)

    # B11 门（复现基准）
    r = 10.0 ** (snr_hat / 10.0)
    d = r * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    pd = 0.5 * (1.0 + erf(d / np.sqrt(2.0)))
    pd_ok = pd >= (1.0 - rule.alpha_k)
    # oracle 门
    phys_true = r_true >= rmin_db
    best = probs.max(axis=1)
    argmax = probs.argmax(axis=1)

    def scan(gate_mask, gate_name):
        rows = {}
        for tau in TAUS:
            decide = gate_mask & (best > tau)
            correct = np.array([decide[i] and argmax[i] in labels_list[i] for i in range(n)])
            gap, risk = operating_gap(decide, correct, C.ALPHA)
            rows[f"tau{tau:.1f}"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                                     "coverage": round(coverage(decide), 3),
                                     "acc_at_dec": round(float(correct[decide].mean()), 3)
                                     if decide.any() else None}
            print(f"[{gate_name} tau={tau:.1f}] {rows[f'tau{tau:.1f}']}", flush=True)
        return rows

    rows_b11 = scan(pd_ok, "B11")
    rows_oracle = scan(phys_true, "oracle")

    # 判定: oracle 门下是否存在 τ 使 risk ≤ α 且 coverage > 0.303（B11 τ=0.5 的覆盖）
    attain = [(tau, r) for tau, r in rows_oracle.items()
              if r["risk"] is not None and r["risk"] <= C.ALPHA and r["coverage"] > 0.303]
    verdict = "ORACLE_GATE_ATTAINS_ALPHA" if attain else "ORACLE_GATE_BELOW_ALPHA"

    out = {"tag": args.tag, "dataset": args.dataset, "ckpt": os.path.basename(args.ckpt),
           "seed": C.SEED, "r_min_db": round(rmin_db, 3), "n_windows": n,
           "verdict": verdict,
           "interpretation": ("oracle 门（r_true ≥ r_min）替代 B11 SNR̂ 门; "
                              "ATTAINS → 残余失败是门失配（工程可修）; "
                              "BELOW → 残余失败是任务结构"),
           "reference": {"b11_tau05": rows_b11.get("tau0.5"),
                         "b11_tau0": rows_b11.get("tau0.0")},
           "rows_b11_gate": rows_b11,
           "rows_oracle_gate": rows_oracle}
    with open(os.path.join(OUT, f"exp_clap_gate_probe_{args.dataset}_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"DONE → outputs/exp_clap_gate_probe_{args.dataset}_{args.tag}.json")


if __name__ == "__main__":
    main()
