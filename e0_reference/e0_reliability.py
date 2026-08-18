"""
e0_reliability.py — E0 一致性验证 v3（最终形式）：逐 (r, T) 单元可靠性图

统计量 = 能量比: z = sum(x_band^2) / sum(g*w_ref_band^2)   [窗内事件能量/参考噪声能量]
  E[z0]=1, Var=2/n (oracle 分母) 或 4/n (参考窗分母, 噪声估计损耗)
  r = sum(e_band^2)/sum(g*w_ref_band^2)  (增益无关, 单位自洽)

双重预测:
  pred_oracle    = 理想理论 (分母精确已知): pe_theory_rn(r,n)
  pred_realistic = 参考窗噪声估计 (分母独立波动): 高斯近似 s0=2/sqrt(n), s1=2(1+r)/sqrt(n)
实测 Pe = (fp+miss)/2n_trials  —— 与 pred_realistic 的偏差 = 残余模型误差(AM/非高斯)

网格: r = E_s/P_n ∈ {0.05, 0.15, 0.5} × T ∈ {0.32, 0.64, 1.28}s
运行: python e0_reliability.py --source synthetic
"""
import argparse, csv, io, math, os
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavf
from energy_detector import pe_theory_rn, ppf, phi
import sp_anchor

FS = 16000
F_LO, F_HI = 1000.0, 4000.0
B_EFF = F_HI - F_LO
RATE2 = 2 * B_EFF
R_GRID = [0.05, 0.15, 0.5]
T_GRID = [0.32, 0.64, 1.28]
ALPHA = 0.1
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)

def to_band(x):
    b, a = sig.butter(4, [F_LO / (FS / 2), F_HI / (FS / 2)], btype="band")
    return sig.resample_poly(sig.lfilter(b, a, x), 3, 8)

def pe_realistic(r, n, alpha):
    """参考窗噪声估计下的预测 Pe（双方差高斯，s0=2/sqrt(n), s1=2(1+r)/sqrt(n)）。"""
    thr = 1.0 + 2.0 / math.sqrt(n) * ppf(1.0 - alpha)
    s0, s1 = 2.0 / math.sqrt(n), 2.0 * (1.0 + r) / math.sqrt(n)
    pfa = 1.0 - phi((thr - 1.0) / s0)
    miss = phi((thr - (1.0 + r)) / s1)
    return 0.5 * (pfa + miss)

def run_cell(events, r_target, T, seed, detector="energy"):
    """detector: energy=参考窗能量比（oracle 口径）; sp=B11a PSD 估计噪声底。"""
    n_eff = int(round(RATE2 * T))
    snr_db = 10.0 * math.log10(r_target / n_eff)   # 带内每样本等效 SNR（叙述用）
    rng = np.random.default_rng(seed)
    n_t = 0
    z0s, z1s = [], []
    thr = 1.0 + 2.0 / math.sqrt(n_eff) * ppf(1.0 - ALPHA)
    for ev in events:
        n_src = int(round(T * FS))
        if ev.shape[0] < n_src:
            continue
        for start in range(0, ev.shape[0] - n_src + 1, n_src):   # 非重叠全窗口
            n_t += 1
            seg = ev[start:start + n_src]
            w_ref, _ = _wind(n_src, seed=seed + 2 * n_t)      # 参考窗（噪声估计源）
            w_tst, _ = _wind(n_src, seed=seed + 2 * n_t + 1)  # 测试窗噪声（独立）
            eb = to_band(seg)
            e_energy = float(np.sum(eb ** 2))
            if e_energy < 1e-9 * n_eff:      # 静音窗口: 无事件可检测, 跳过
                continue
            wrb = to_band(w_ref)
            wtb = to_band(w_tst)
            denom_true = float(np.sum(wrb ** 2)) + 1e-15
            if detector == "sp":             # B11a: PSD 估计噪声能量（谱减口径）
                denom = max(sp_anchor.band_noise_energy(w_ref, FS, F_LO, F_HI, n_eff), 1e-15)
            else:
                denom = denom_true
            g = math.sqrt(e_energy / (denom_true * r_target))
            z0s.append(float(np.sum(wtb ** 2)) / denom)
            # z1 以混合内实际噪声能量 g²·denom 归一化（E[z1]=1+r_target，方差紧）
            z1s.append(float(np.sum(to_band(seg + g * w_tst) ** 2)) / (g * g * denom))
    if n_t == 0:
        return None
    z0, z1 = np.array(z0s), np.array(z1s)
    emp = 0.5 * (np.mean(z0 > thr) + np.mean(z1 < thr))
    # 实测方差校准预测（高斯拟合实测 z0/z1 分布）
    m0, s0 = float(np.mean(z0)), float(np.std(z0))
    m1, s1 = float(np.mean(z1)), float(np.std(z1))
    pfa_c = 1.0 - phi((thr - m0) / max(s0, 1e-9))
    miss_c = phi((thr - m1) / max(s1, 1e-9))
    pred_calib = 0.5 * (pfa_c + miss_c)
    r_exact = r_target   # 由 g 构造保证
    return {"snr_db": snr_db, "T": T, "BT": B_EFF * T, "n_eff": n_eff, "r": r_exact,
            "pred_oracle": pe_theory_rn(r_exact, n_eff),
            "pred_real": pe_realistic(r_exact, n_eff, ALPHA),
            "pred_calib": pred_calib,
            "sig0_infl": s0 / (math.sqrt(2.0 / n_eff)),   # 实测 vs 理论标准差膨胀比
            "sig1_infl": s1 / (2.0 * (1.0 + r_exact) / math.sqrt(n_eff)),
            "emp_pe": emp, "n": n_t}

def _wind(n, seed):
    from wind_noise import make_wind
    return make_wind(n, FS, seed=seed)

def load_esc50_wavs(parquet_dir, classes):
    """从 ashraq/esc50 parquet 提取指定类别的 16k 音频（文件名可能是 blob 哈希）。"""
    import pyarrow.parquet as pq
    out = []
    for fn in sorted(os.listdir(parquet_dir)):
        p = os.path.join(parquet_dir, fn)
        if os.path.getsize(p) < 1_000_000:   # 跳过小文件(README/json)
            continue
        try:
            df = pq.read_table(p).to_pandas()
        except Exception:
            continue
        for _, row in df.iterrows():
            if row.get("target") in classes:
                try:
                    sr, arr = wavf.read(io.BytesIO(row["audio"]["bytes"]))
                except Exception:
                    continue
                x = np.asarray(arr, dtype=np.float64)
                if sr != FS:
                    x = sig.resample_poly(x, FS, sr)
                out.append(x)
                if len(out) >= 40 * len(classes):
                    return out
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "esc50"], default="synthetic")
    ap.add_argument("--detector", choices=["energy", "sp"], default="energy")
    ap.add_argument("--esc50-dir", default=os.environ.get("ESC50_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "esc50")))
    ap.add_argument("--classes", default="3,7")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    if args.source == "synthetic":
        rng = np.random.default_rng(args.seed)
        events = []
        for _ in range(400):
            n = int(1.28 * FS)
            b, a = sig.butter(4, [F_LO / (FS / 2), F_HI / (FS / 2)], btype="band")
            ev = sig.lfilter(b, a, rng.standard_normal(n))
            events.append(ev / (np.std(ev) + 1e-12) * 0.5)
    else:
        classes = [int(c) for c in args.classes.split(",")]
        events = load_esc50_wavs(args.esc50_dir, classes)
        print(f"[esc50] loaded {len(events)} clips for classes {classes}")

    rows = []
    print(f"{'r_tgt':>6} {'SNR(dB)':>8} {'T(s)':>6} {'BT':>7} "
          f"{'oracle':>8} {'real':>8} {'calib':>8} {'emp':>8} {'σ0×':>6} {'|emp-cal|':>9}")
    for r0 in R_GRID:
        for T in T_GRID:
            res = run_cell(events, r0, T, args.seed, detector=args.detector)
            if res is None:
                print(f"{r0:>6.2f} {T:>6.2f}  NO EVENTS")
                continue
            d = abs(res["emp_pe"] - res["pred_calib"])
            print(f"{r0:>6.2f} {res['snr_db']:>8.1f} {T:>6.2f} {res['BT']:>7.0f} "
                  f"{res['pred_oracle']:>8.4f} {res['pred_real']:>8.4f} {res['pred_calib']:>8.4f} "
                  f"{res['emp_pe']:>8.4f} {res['sig0_infl']:>6.2f} {d:>9.4f}")
            rows.append(res)
    with open(os.path.join(OUT, f"e0_reliability_{args.source}_{args.detector}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "T", "BT", "n_eff", "r",
                                          "pred_oracle", "pred_real", "pred_calib",
                                          "sig0_infl", "sig1_infl", "emp_pe", "n"])
        w.writeheader()
        for r_ in rows:
            w.writerow({k: r_[k] for k in w.fieldnames})
    print(f"\n[written] {os.path.join(OUT, f'e0_reliability_{args.source}_{args.detector}.csv')}")

if __name__ == "__main__":
    main()
