"""exp_us8k_fold_recheck.py — US8K 任务重定义 3 折复核（2026-08-18, optimization #6）

正文头条: 类级重定义（剪 per-class acc<0.85 类入 unknown, B11 门 τ=0.5）
          fold-10 测试 risk 0.005<α @ cov 0.158（p<0.001）。
本脚本把同一协议在旋转测试折 {7,8,9} 上重跑（train=其余 9 折, 同冻结配方）:
  1. CLAP-FT 重训（epochs=6, lr=1e-4, batch=16, freeze_blocks=8, WSO-Sim 网格）
  2. B11 门评估（τ 扫 + per-class acc@dec @ τ=0.5 门内）
  3. 事前规则（0.85, 逐 pass 冻结 per-class 表）剪裁 → 重定义 risk/coverage/acc
     + Wilson CI + 单侧 p（H0: risk ≥ α）
模式:
  --validate: 用冻结 ckpt 在 fold-10 复现冻结 rows 与 subset 0.005@0.158（管线等价校验）
  默认:       fold ∈ {7,8,9} 逐折训练+评估+重定义
输出: outputs/exp_us8k_fold_recheck_{tag}.json（含逐折汇总）+ checkpoints_clap_finetune/clap_ft_us8k_fold{fold}_{tag}.pt
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))

from aof import config as C
from aof.af_rule import AFRule
from aof.baselines import SPAnchorB11
from aof.wsosim import _wind, corrupt
import exp_clap_finetune as E
from exp_clap_finetune import ClapFT, to_48k, corrupt_batch
from exp_us8k import US8K_NAMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
CKPT_DIR = os.path.join(OUT, "checkpoints_clap_finetune")
MODEL_ID = "laion/clap-htsat-unfused"
SR_TARGET = 48000
EVAL_SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
KINDS = ["wind", "occlusion", "self_motion"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
RULE_ACC = 0.85


def rotated_index(test_fold, limit=None):
    idx = E.load_us8k_index(limit)
    return [(fname, fold, y, "test" if fold == test_fold else "train")
            for fname, fold, y, _ in idx]


def train_and_eval(test_fold, args, dev, proc):
    E.load_us8k_index = (lambda limit=None: rotated_index(test_fold, limit))
    train_clips = E.build_train_clips("us8k")
    test_clips = E.build_test_clips("us8k")
    print(f"[fold {test_fold}] train clips: {len(train_clips)}  test clips: {len(test_clips)}",
          flush=True)

    from transformers.models.clap import ClapModel
    clap = ClapModel.from_pretrained(MODEL_ID).to(dev)
    model = ClapFT(clap, n_out=10, freeze_blocks=8).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED + 7)
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_clips)).tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(train_clips) - len(train_clips) % args.batch, args.batch):
            idx = perm[i: i + args.batch]
            wavs, labels = corrupt_batch([train_clips[j] for j in idx], rng)
            audio = proc(audio=[to_48k(w) for w in wavs], sampling_rate=SR_TARGET,
                         return_tensors="pt").to(dev)
            logits = model(audio["input_features"], audio["is_longer"])
            loss = F.cross_entropy(logits, torch.tensor(labels, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1; step += 1
            if step % 200 == 0:
                print(f"  ep{ep} step{step} loss={tot/nb:.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        sched.step()
        print(f"  epoch {ep} done: loss={tot/max(nb,1):.4f}", flush=True)

    ckpt_path = os.path.join(CKPT_DIR, f"clap_ft_us8k_fold{test_fold}_{args.tag}.pt")
    torch.save({"state_dict": model.state_dict(), "freeze_blocks": 8,
                "seed": C.SEED, "n_train_clips": len(train_clips)}, ckpt_path)
    return model, ckpt_path, test_clips


def window_loop(model, proc, dev, test_clips):
    """与 exp_us8k_subset.py 相同的逐窗评估循环; 返回数组。"""
    rule = AFRule()
    b11 = SPAnchorB11()
    rng = np.random.default_rng(C.SEED + 11)
    recs = []
    for x, y, fname in test_clips:
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
                    logits = model(audio["input_features"], audio["is_longer"])
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                recs.append({"snr_hat_db": float(snr_hat_db), "label": lab, "probs": p})
    probs = np.stack([r["probs"] for r in recs])
    snr_hat = np.array([r["snr_hat_db"] for r in recs])
    labels = np.array([r["label"] for r in recs])
    rule = AFRule()
    r = 10.0 ** (snr_hat / 10.0)
    d = r * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    pd = 0.5 * (1.0 + erf(d / np.sqrt(2.0)))
    pd_ok = pd >= (1.0 - rule.alpha_k)
    return probs, labels, pd_ok, len(recs)


def rows_from(probs, labels, pd_ok, n, target=None):
    best = probs.max(axis=1)
    argmax = probs.argmax(axis=1)
    if target is not None:
        decide_ok = np.isin(argmax, target)
    else:
        decide_ok = np.ones(n, dtype=bool)
    rows = {}
    for tau in TAUS:
        decide = pd_ok & (best > tau) & decide_ok
        n_dec = int(decide.sum())
        if n_dec == 0:
            rows[f"tau{tau:.1f}"] = {"risk": None, "coverage": 0.0, "acc_at_dec": None,
                                     "n_decide": 0, "wilson_ci": None, "p_onesided": None}
            continue
        correct = decide & (argmax == labels)
        acc = correct[decide].mean()
        risk = 1.0 - acc
        k_err = n_dec - int(correct[decide].sum())
        z = 1.96
        p = k_err / n_dec
        denom = 1 + z * z / n_dec
        center = (p + z * z / (2 * n_dec)) / denom
        half = z * math.sqrt(p * (1 - p) / n_dec + z * z / (4 * n_dec * n_dec)) / denom
        err = 1.0 - p
        se = math.sqrt(0.1 * 0.9 / n_dec)
        zstat = (err - 0.1) / se
        pval = 0.5 * (1 + math.erf(zstat / math.sqrt(2)))
        rows[f"tau{tau:.1f}"] = {"risk": round(risk, 3), "coverage": round(decide.mean(), 3),
                                 "acc_at_dec": round(acc, 3), "n_decide": n_dec,
                                 "wilson_ci": [round(center - half, 3), round(center + half, 3)],
                                 "p_onesided": round(pval, 3)}
    return rows


def per_class_at_tau(probs, labels, pd_ok, tau=0.5):
    best = probs.max(axis=1)
    argmax = probs.argmax(axis=1)
    decide = pd_ok & (best > tau)
    pc = {}
    for k in range(10):
        m = decide & (argmax == k)
        n = int(m.sum())
        pc[US8K_NAMES[k]] = {"acc": round(float((argmax[m] == labels[m]).mean()), 3) if n else None,
                             "n": n}
    return pc


def evaluate_fold(model, proc, dev, test_clips, tag, test_fold):
    probs, labels, pd_ok, n = window_loop(model, proc, dev, test_clips)
    full_rows = rows_from(probs, labels, pd_ok, n)
    per_class = per_class_at_tau(probs, labels, pd_ok)
    target = sorted([US8K_NAMES.index(name) for name, v in per_class.items()
                     if v["acc"] is not None and v["acc"] >= RULE_ACC])
    subset_rows = None
    if len(target) >= 2:
        subset_rows = rows_from(probs, labels, pd_ok, n, target=target)
    return {"n_windows": n, "full_10class": full_rows, "per_class": per_class,
            "target_classes": [US8K_NAMES[k] for k in target],
            "pruned_share": round(1.0 - float(np.mean(np.isin(labels, target))), 4)
            if len(target) else None,
            "subset": subset_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260818")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--validate", action="store_true",
                    help="用冻结 ckpt 在 fold-10 复现冻结数字（管线等价校验）")
    ap.add_argument("--folds", default="7,8,9")
    args = ap.parse_args()

    dev = args.device if torch.backends.mps.is_available() else "cpu"
    from transformers.models.clap import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained(MODEL_ID).to(dev)
    proc = ClapProcessor.from_pretrained(MODEL_ID)

    out = {"tag": args.tag, "rule": {"threshold": RULE_ACC,
                                     "note": "事前固定; 逐 pass 冻结 per-class 表（τ=0.5, B11 门内）"},
           "folds": {}}

    if args.validate:
        E.load_us8k_index = (lambda limit=None: rotated_index(10, limit))
        test_clips = E.build_test_clips("us8k")
        ckpt = torch.load(os.path.join(CKPT_DIR, "clap_ft_us8k_20260816.pt"), map_location="cpu")
        model = ClapFT(clap, n_out=10, freeze_blocks=ckpt.get("freeze_blocks", 8)).to(dev)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        ev = evaluate_fold(model, proc, dev, test_clips, args.tag, 10)
        frozen = json.load(open(os.path.join(OUT, "exp_clap_finetune_us8k_20260816.json")))
        fr = frozen["evaluation"]["rows"]
        ok = all(abs(ev["full_10class"][f"tau{t:.1f}"]["risk"] - fr[f"tau{t:.1f}"]["risk"]) < 0.002
                 for t in (0.0, 0.3, 0.5, 0.7))
        sub_frozen = json.load(open(os.path.join(OUT, "exp_us8k_subset_infer_20260816b.json")))
        sr = sub_frozen["rows_b11_gate"]["tau0.5"]
        sub_ok = (ev["subset"] is not None and
                  abs(ev["subset"]["tau0.5"]["risk"] - sr["risk"]) < 0.002 and
                  abs(ev["subset"]["tau0.5"]["coverage"] - sr["coverage"]) < 0.002)
        print(f"validate fold-10: rows {'OK' if ok else 'MISMATCH'} (τ=0.5 risk "
              f"{ev['full_10class']['tau0.5']['risk']} vs frozen {fr['tau0.5']['risk']})", flush=True)
        print(f"validate subset:  {'OK' if sub_ok else 'MISMATCH'} (τ=0.5 "
              f"{ev['subset']['tau0.5'] if ev['subset'] else None} vs frozen {sr})", flush=True)
        if not (ok and sub_ok):
            raise SystemExit("管线等价校验失败")
        out["validate"] = {"status": "OK", "fold10_full_tau0.5": ev["full_10class"]["tau0.5"],
                           "fold10_subset_tau0.5": ev["subset"]["tau0.5"]}
        print("VALIDATE OK", flush=True)

    for fold in [int(f) for f in args.folds.split(",")]:
        print(f"===== fold {fold} 训练 =====", flush=True)
        model, ckpt_path, test_clips = train_and_eval(fold, args, dev, proc)
        model.eval()
        ev = evaluate_fold(model, proc, dev, test_clips, args.tag, fold)
        out["folds"][str(fold)] = {**ev, "ckpt": os.path.basename(ckpt_path)}
        sub = ev["subset"]
        if sub is not None:
            r = sub["tau0.5"]
            print(f"fold {fold}: 重定义 τ=0.5 risk={r['risk']} cov={r['coverage']} "
                  f"n={r['n_decide']} p={r['p_onesided']} | 目标类 {ev['target_classes']}", flush=True)
        else:
            print(f"fold {fold}: 目标类 < 2, 规则退化, 重定义不可测", flush=True)
        print(f"fold {fold}: 全 10 类 τ=0.5 risk={ev['full_10class']['tau0.5']['risk']} "
              f"cov={ev['full_10class']['tau0.5']['coverage']}", flush=True)

    with open(os.path.join(OUT, f"exp_us8k_fold_recheck_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDONE → outputs/exp_us8k_fold_recheck_{args.tag}.json")


if __name__ == "__main__":
    main()
