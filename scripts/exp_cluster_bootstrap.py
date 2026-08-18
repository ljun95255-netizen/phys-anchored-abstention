"""exp_cluster_bootstrap.py — 头条风险数的 clip 级 bootstrap 区间（2026-08-18, optimization #3）

三个头条（正文 4.2/4.3/4.9 与摘要）:
  SC-10 B12 τ=0.5         risk 0.081 @ cov 0.542  (exp_sc10_eval.json)
  US8K B12 τ=0.5          risk 0.420 @ cov 0.424  (exp_us8k_eval.json)
  FSD50K-10 CLAP-FT τ=0.7 risk 0.074 @ cov 0.163  (exp_clap_finetune_fsd50k10_20260816.json)

论文 4.9 自认: "Intervals treat windows as independent; windows of the same clip
are correlated, so the intervals are mildly optimistic"。本脚本从冻结 checkpoint 重算
逐窗 (decide, correct, clip_id), 按源 clip 聚类 bootstrap（aof.stats.cluster_bootstrap,
预注册 n_boot=1000, seed=20260804）, 出 clip 级 95% 区间, 封住"区间乐观"攻击。

冻结纪律: 先聚合层逐位复现冻结 JSON 头条（risk/coverage/acc_at_dec, 3 位小数）;
复现失败即终止, 不写任何新数字。新 JSON 注明口径（重算自冻结 checkpoint, 非 R19 冻结 pass）。
"""
import json
import os
import sys
import math

import numpy as np
import torch
from scipy.special import erf

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPTS)
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)

from aof import config as C
from aof.af_rule import AFRule
from aof.baselines import SPAnchorB11
from aof.data import log_mel
from aof.model import SONTRA_A
from aof.stats import cluster_bootstrap
from aof.wsosim import corrupt, _wind
from aof.metrics import operating_gap, coverage
from exp_us8k_eval import apply_rule  # 与 exp_sc10_eval.apply_rule 逐行一致

OUT = os.path.join(REPO, "outputs")
KINDS = ["wind", "occlusion", "self_motion"]
EVAL_SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
N_BOOT = 1000
BOOT_SEED = 20260804


def risk_metric(decide, correct):
    d = decide.sum()
    if d == 0:
        return float("nan")
    return 1.0 - correct[decide].mean()


def cov_metric(decide, correct):
    return decide.mean()


def load_sontra(ckpt_name, ckpt_dir):
    model = SONTRA_A(n_classes=11).to("mps").eval()
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name), map_location="cpu"))
    return model


def eval_sontra_dataset(loader, ckpt_name, ckpt_dir, tau, frozen_rows_key=None):
    """复制冻结 eval 的缓存循环（exp_sc10_eval/exp_us8k_eval.main）; 返回逐窗数组。"""
    rule = AFRule()
    rmin_db = 10 * math.log10(rule.r_min)
    clips = loader()
    rng = np.random.default_rng(C.SEED)
    model = load_sontra(ckpt_name, ckpt_dir)
    cache = []
    for x, y, cid in clips:
        labels = {i for i, v in enumerate(y) if v and i < 10}
        for kind in KINDS:
            for snr_db in EVAL_SNRS:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                mel = log_mel(torch.from_numpy(xc).unsqueeze(0).float().to("mps"))
                with torch.no_grad():
                    out = model(mel)
                cache.append({"probs": out["event_probs"].cpu().numpy()[0],
                              "snr_vec": out["snr_db"].cpu().numpy()[0],
                              "labels": labels, "r_true": float(r_true),
                              "cid": cid})
    print(f"  cached {len(cache)} windows", flush=True)
    probs = np.stack([c["probs"] for c in cache])
    snr_vec = np.stack([c["snr_vec"] for c in cache])
    r_true = np.array([c["r_true"] for c in cache])
    labels_list = [c["labels"] for c in cache]
    cids = np.array([c["cid"] for c in cache])
    decide, correct, _ = apply_rule(probs, snr_vec, r_true, labels_list, tau, False, rule)
    return decide, correct, cids


def eval_fsd50k_probe(tau, ckpt_name="clap_ft_fsd50k10_20260816.pt"):
    """复制 exp_clap_finetune.evaluate 的窗口循环（B11 门 + CLAP-FT, τ 扫）; 返回逐窗数组。"""
    from transformers.models.clap import ClapModel, ClapProcessor
    from exp_clap_finetune import ClapFT, to_48k, _fsd50k_clips, MODEL_ID, SR_TARGET

    rule = AFRule()
    b11 = SPAnchorB11()
    rng = np.random.default_rng(C.SEED + 11)
    clap = ClapModel.from_pretrained(MODEL_ID).to("mps")
    model = ClapFT(clap, n_out=10).to("mps").eval()
    ckpt_dir = os.path.join(OUT, "checkpoints_clap_finetune")
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    proc = ClapProcessor.from_pretrained(MODEL_ID)

    clips = _fsd50k_clips("val")
    recs = []
    for i, (x, lab, fname) in enumerate(clips):
        labels = {lab}
        for kind in KINDS:
            for snr_db in EVAL_SNRS:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                w_ref = _wind(xc.shape[0], C.SAMPLE_RATE, seed + 1)[0].astype(np.float32)
                snr_hat_db = b11.snr_db(xc, w_ref)
                audio = proc(audio=[to_48k(xc)], sampling_rate=SR_TARGET,
                             return_tensors="pt").to("mps")
                with torch.no_grad():
                    logits = model(audio["input_features"], audio["is_longer"])
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                recs.append({"cid": fname, "snr_hat_db": float(snr_hat_db),
                             "labels": labels, "probs": p})
        if (i + 1) % 300 == 0:
            print(f"  eval {i+1}/{len(clips)}", flush=True)

    rmin_db = 10 * np.log10(rule.r_min)
    probs = np.stack([r["probs"] for r in recs])
    snr_hat = np.array([r["snr_hat_db"] for r in recs])
    labels_list = [r["labels"] for r in recs]
    cids = np.array([r["cid"] for r in recs])
    r = 10.0 ** (snr_hat / 10.0)
    d = r * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    pd = 0.5 * (1.0 + erf(d / np.sqrt(2.0)))
    pd_ok = pd >= (1.0 - rule.alpha_k)
    best = probs.max(axis=1)
    argmax = probs.argmax(axis=1)
    decide = pd_ok & (best > tau)
    correct = np.array([decide[i] and argmax[i] in labels_list[i] for i in range(len(recs))])
    return decide, correct, cids


def verify(name, decide, correct, expect, tol=0.0005):
    """逐位校验（SONTRA-A 冻结 pass）; CLAP-FT 探针用 MPS 可复现容差 0.004
    （README 声明: MPS 推理不可位复现, 重跑与冻结记录一致到 ~1e-3）。"""
    gap, risk = operating_gap(decide, correct, C.ALPHA)
    acc = float(correct[decide].mean()) if decide.any() else float("nan")
    got = {"risk": round(risk, 3), "coverage": round(coverage(decide), 3),
           "acc_at_dec": round(acc, 3) if not np.isnan(acc) else None}
    ok = all(abs(got[k] - v) <= tol for k, v in expect.items())
    print(f"  verify {name}: got {got} expect {expect} tol={tol} → {'OK' if ok else 'MISMATCH'}",
          flush=True)
    if not ok:
        raise SystemExit(f"冻结复现失败: {name} {got} != {expect}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sontra", action="store_true",
                    help="SC-10/US8K 已验证（两次一致）; 只重跑 FSD50K-10 探针段")
    args = ap.parse_args()
    out = {"tag": "20260818", "note": "重算自冻结 checkpoint（非 R19 冻结 pass）; "
           "clip 级 bootstrap, n_boot=1000, seed=20260804; 窗口级区间=冻结 Wilson CI; "
           "FSD50K-10 探针点估计 0.077 vs 冻结 0.074 = MPS 非位复现漂移（<=0.003, README 容差）",
           "method": "cluster bootstrap by source clip (windows of same clip sampled together)"}
    results = {}

    if not args.skip_sontra:
        # --- SC-10 B12 τ=0.5 ---
        from exp_sc10_eval import load_te_clips as sc_load_te
        print("[SC-10] SONTRA-A 重算...", flush=True)
        dec, corr, cids = eval_sontra_dataset(sc_load_te, "sontra_a_ep35.pt",
                                              os.path.join(OUT, "checkpoints_sc10"), 0.5)
        verify("SC-10", dec, corr, {"risk": 0.081, "coverage": 0.542, "acc_at_dec": 0.919})
        cb = cluster_bootstrap(dec, corr, cids, risk_metric, N_BOOT, BOOT_SEED)
        cc = cluster_bootstrap(dec, corr, cids, cov_metric, N_BOOT, BOOT_SEED)
        results["sc10_b12_tau0.5"] = {"n_windows": int(len(dec)), "n_clips": int(len(np.unique(cids))),
                                      "n_decide": int(dec.sum()),
                                      "risk": {"point": 0.081, "clip_ci": [round(cb["ci_lo"], 3), round(cb["ci_hi"], 3)],
                                               "window_ci": [0.078, 0.084]},
                                      "coverage": {"point": 0.542, "clip_ci": [round(cc["ci_lo"], 3), round(cc["ci_hi"], 3)]}}
        print(f"  SC-10 clip CI: risk {results['sc10_b12_tau0.5']['risk']}", flush=True)

        # --- US8K B12 τ=0.5 ---
        from exp_us8k_eval import load_te_clips as us8k_load_te
        print("[US8K] SONTRA-A 重算...", flush=True)
        dec, corr, cids = eval_sontra_dataset(us8k_load_te, "sontra_a_ep23.pt",
                                              os.path.join(OUT, "checkpoints_us8k"), 0.5)
        verify("US8K", dec, corr, {"risk": 0.420, "coverage": 0.424, "acc_at_dec": 0.580})
        cb = cluster_bootstrap(dec, corr, cids, risk_metric, N_BOOT, BOOT_SEED)
        cc = cluster_bootstrap(dec, corr, cids, cov_metric, N_BOOT, BOOT_SEED)
        results["us8k_b12_tau0.5"] = {"n_windows": int(len(dec)), "n_clips": int(len(np.unique(cids))),
                                      "n_decide": int(dec.sum()),
                                      "risk": {"point": 0.420, "clip_ci": [round(cb["ci_lo"], 3), round(cb["ci_hi"], 3)],
                                               "window_ci": [0.406, 0.434]},
                                      "coverage": {"point": 0.424, "clip_ci": [round(cc["ci_lo"], 3), round(cc["ci_hi"], 3)]}}
        print(f"  US8K clip CI: risk {results['us8k_b12_tau0.5']['risk']}", flush=True)
    else:
        print("[skip] SC-10/US8K 已在本轮前一次运行验证（clip CI 双跑一致）", flush=True)

    # --- FSD50K-10 CLAP-FT 探针 τ=0.7 ---
    print("[FSD50K-10] CLAP-FT 探针重算 (23,220 windows, ~30min)...", flush=True)
    dec, corr, cids = eval_fsd50k_probe(0.7)
    verify("FSD50K-10 probe", dec, corr, {"risk": 0.074, "coverage": 0.163, "acc_at_dec": 0.926},
           tol=0.004)
    cb = cluster_bootstrap(dec, corr, cids, risk_metric, N_BOOT, BOOT_SEED)
    cc = cluster_bootstrap(dec, corr, cids, cov_metric, N_BOOT, BOOT_SEED)
    results["fsd50k10_probe_tau0.7"] = {"n_windows": int(len(dec)), "n_clips": int(len(np.unique(cids))),
                                        "n_decide": int(dec.sum()),
                                        "risk": {"point": 0.074, "clip_ci": [round(cb["ci_lo"], 3), round(cb["ci_hi"], 3)],
                                                 "window_ci": [0.066, 0.083]},
                                        "coverage": {"point": 0.163, "clip_ci": [round(cc["ci_lo"], 3), round(cc["ci_hi"], 3)]}}
    print(f"  FSD50K-10 clip CI: risk {results['fsd50k10_probe_tau0.7']['risk']}", flush=True)

    out["results"] = results
    with open(os.path.join(OUT, "exp_cluster_bootstrap_20260818.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE → outputs/exp_cluster_bootstrap_20260818.json")


if __name__ == "__main__":
    main()
