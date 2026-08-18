"""exp_tau_scan_frozen.py — FSD50K 冻结 B12 τ 扫描（2026-08-07 为期刊版 fig3 重建曲线）
与 exp_round2_fixes.py 完全同管线（ckpt ep22, val 500, C.SEED rng, mps）。
一次前向, 复用 logits 对 τ∈{0.05..0.95} 计算 decide/correct。
确定性检查: τ=0.5 必须逐位复现冻结值 (0.346, 0.446, 0.495), 否则告警。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.evaluate import _to_mel
from aof.metrics import operating_gap, coverage
from aof.model import SONTRA_A
from aof.wsosim import corrupt
from run_main import load_fsd50k_clips, sample_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

FROZEN_T05 = (0.346, 0.446, 0.495)


def main():
    ckpt = os.path.join(OUT, "checkpoints", "sontra_a_ep22.pt")
    snrs = [-25.0, -15.0, -5.0, 5.0, 15.0]
    kinds = ["wind", "occlusion", "self_motion"]

    from aof.mapping import build_dev_index
    index = build_dev_index()
    val_sel = sample_clips([r for r in index if r[2] == "val"], 500, C.SEED + 1)
    idx_map = {f: y for f, y, s in index}
    va_clips = [(x, f) for x, f in load_fsd50k_clips([f for f, _ in val_sel])]
    from aof.cf_sampler import CFSampler
    sampler = CFSampler([])
    va_clips = [(sampler._best_window(x, C.WINDOW_SAMPLES), idx_map[f], f)
                for x, f in va_clips
                if sampler._best_window(x, C.WINDOW_SAMPLES) is not None]

    model = SONTRA_A()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model = model.to("mps").eval()
    rule = AFRule()
    rng = np.random.default_rng(C.SEED)
    print(f"ckpt: {os.path.basename(ckpt)}  clips: {len(va_clips)}", flush=True)

    taus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    acc = {t: {"dec": [], "corr": []} for t in taus}

    for x, target, fname in va_clips:
        for kind in kinds:
            for snr_db in snrs:
                seed = int(rng.integers(1 << 31))
                xc, r_true, meta = corrupt(x, kind, snr_db, seed)
                if xc is None:
                    continue
                mel = torch.from_numpy(xc).unsqueeze(0).float().to("mps")
                mel = _to_mel(mel)
                out = model(mel)
                probs = out["event_probs"].cpu()
                snr_in = out["snr_db"].cpu()
                labels = {i for i, v in enumerate(target) if v and i < C.N_CLASSES - 1}
                for t in taus:
                    decide, pred, _ = rule.decide(probs, snr_in, tau=t)
                    acc[t]["dec"].append(bool(decide.item()))
                    acc[t]["corr"].append(bool(pred.item() in labels and decide.item()))

    res = {}
    ok = True
    for t in taus:
        dec = np.array(acc[t]["dec"])
        corr = np.array(acc[t]["corr"])
        gap, risk = operating_gap(dec, corr, C.ALPHA)
        cov = coverage(dec)
        accdec = float(corr[dec].mean()) if dec.any() else float("nan")
        res[f"B12_tau{t:g}"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                                "coverage": round(cov, 3), "acc_at_dec": round(accdec, 3)}
        if abs(t - 0.5) < 1e-9:
            frozen_ok = (abs(gap - FROZEN_T05[0]) < 1e-9 and abs(risk - FROZEN_T05[1]) < 1e-9
                         and abs(cov - FROZEN_T05[2]) < 1e-9)
            ok = frozen_ok
            print(f"  τ=0.5 determinism: raw got ({gap:.6f},{risk:.6f},{cov:.6f}) "
                  f"frozen ({FROZEN_T05[0]:.6f},{FROZEN_T05[1]:.6f},{FROZEN_T05[2]:.6f}) "
                  f"→ {'MATCH' if frozen_ok else 'DRIFT!'}", flush=True)
    if not ok:
        print("WARNING: τ=0.5 did not reproduce frozen values — curve NOT frozen-equivalent!", flush=True)
    with open(os.path.join(OUT, "exp_tau_scan_frozen.json"), "w") as f:
        json.dump({"frozen_equivalent": bool(ok), **res}, f, indent=1)
    print("DONE → outputs/exp_tau_scan_frozen.json", flush=True)


if __name__ == "__main__":
    main()
