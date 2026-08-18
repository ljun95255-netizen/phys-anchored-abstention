"""
evaluate.py — 主评估驱动（v4.1 §7）: detector-tier（B0/B11/B11a/Oracle）与 system-tier（B12 等）
对 D2 官方 test 的腐蚀版本计算: gap(α), coverage, c_phys, rank-AUC, selective-ECE
"""
import numpy as np
import torch

from . import config as C
from .af_rule import AFRule, n_eff
from .metrics import (operating_gap, coverage, physical_coverage,
                      auroc_from_scores, rank_auc, selective_ece)
from .baselines import EnergyDetectorB0, SPAnchorB11
from .wsosim import corrupt


def evaluate_detector(det, clips_test, kinds, snr_grid, rng=None):
    """detector-tier: 决策 = r̂ ≥ r_min（真实标签事件存在性）。返回逐样本记录。"""
    rule = AFRule()
    rng = np.random.default_rng(C.SEED) if rng is None else rng
    rows = []
    for x, target, fname in clips_test:
        for kind in kinds:
            for snr_db in snr_grid:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                w_ref = _noise_ref(x.shape[0], seed + 1)
                r_hat_db = det.snr_db(xc, w_ref)
                r_hat = 10 ** (r_hat_db / 10)                 # 带内功率比, 与 rule.r_min 同口径
                decide = r_hat >= rule.r_min
                rows.append({
                    "decide": decide, "snr_hat_db": r_hat_db, "r_true_db": r_true,
                    "has_event": 1.0, "kind": kind,
                })
    return rows


def _noise_ref(n, seed):
    from .wsosim import _wind
    w, _ = _wind(n, C.SAMPLE_RATE, seed)
    return w.astype(np.float32)


@torch.no_grad()
def evaluate_system(model, rule, clips_test, kinds, snr_grid, device="cpu",
                    mc_dropout=False, rng=None, tau=0.5, use_true_snr=False):
    """system-tier: 模型输出 + AF-Rule → gap 矩阵记录。
    v3 增量: 每条记录含 r_true_db（外生真值）——域偏移判据 ① 的留出域 SNR MAE 依赖它。
    use_true_snr=True → B12a 变体: 决策阈值用真实 SNR（oracle 阈值下的系统表现）。"""
    rng = np.random.default_rng(C.SEED) if rng is None else rng
    model = model.to(device).eval()
    recs = []
    for x, target, fname in clips_test:
        for kind in kinds:
            for snr_db in snr_grid:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                mel = torch.from_numpy(xc).unsqueeze(0).float().to(device)
                mel = _to_mel(mel)
                out = model(mel)
                probs = out["event_probs"].cpu()
                snr_in = out["snr_db"].cpu()
                if use_true_snr:
                    snr_in = torch.full_like(snr_in, float(r_true))
                decide, pred, _ = rule.decide(probs, snr_in, tau=tau)
                if isinstance(target, (list, tuple)):
                    labels = {i for i, v in enumerate(target) if v and i < C.N_CLASSES - 1}
                else:
                    labels = {target}
                correct = bool(pred.item() in labels and decide.item())
                # raw_acc 排除 unknown 类（无训练信号, argmax 会被其 0.5 概率霸占）
                raw_correct = bool(int(out["event_probs"].cpu().numpy()[0][:C.N_CLASSES - 1].argmax()) in labels)
                recs.append({
                    "decide": bool(decide.item()), "correct": correct,
                    "raw_correct": raw_correct,
                    "snr_hat_db": float(out["snr_db"].cpu().numpy().max()),
                    "snr_hat_mean_db": float(out["snr_db"].cpu().numpy().mean()),
                    "r_true_db": float(r_true), "kind": kind, "snr_db": float(snr_db),
                    "fname": fname,
                    "labels": sorted(labels),
                    "event_probs": out["event_probs"].cpu().numpy()[0],
                })
    return recs


def evaluate_detector_oracle_threshold(det, clips_test, kinds, snr_grid, rng=None):
    """B11a: SP 估计器 + **真实 SNR 阈值**（经典可达上界）——决策基于真实可听性,
    SNR̂ 由估计器输出（报告估计质量）。与 oracle（B0 估计器）区分: 估计器不同。"""
    rule = AFRule()
    rng = np.random.default_rng(C.SEED) if rng is None else rng
    rows = []
    for x, target, fname in clips_test:
        for kind in kinds:
            for snr_db in snr_grid:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                w_ref = _noise_ref(x.shape[0], seed + 1)
                r_hat_db = det.snr_db(xc, w_ref)
                r_true_lin = 10 ** (r_true / 10)
                decide = r_true_lin >= rule.r_min          # 真实阈值
                rows.append({
                    "decide": decide, "snr_hat_db": r_hat_db, "r_true_db": r_true,
                    "has_event": 1.0, "kind": kind,
                })
    return rows


def _to_mel(x):
    from .data import log_mel
    return log_mel(x)


def summary(recs, alpha=C.ALPHA):
    decide = np.array([r["decide"] for r in recs], dtype=bool)
    correct = np.array([r["correct"] for r in recs], dtype=bool) if "correct" in recs[0] else None
    gap = risk = float("nan")
    if correct is not None:
        gap, risk = operating_gap(decide, correct, alpha)
    cov = coverage(decide)
    snr_hat = np.array([r["snr_hat_db"] for r in recs])
    return {"gap": gap, "risk": risk, "coverage": cov,
            "rank_auc": rank_auc(snr_hat, correct) if correct is not None else float("nan")}
