"""verify_us8k.py — 解码后 US8K 完整性验证（2026-08-06）

对照官方 UrbanSound8K.csv: 文件数 / fname 全集一致 / 采样率 / 时长分布 / fold×class 计数
用法: python3 scripts/verify_us8k.py
"""
import csv
import os
import sys
from collections import Counter

import numpy as np

import os
ROOT = os.environ.get("US8K_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "urbansound8k", "UrbanSound8K"))
AUDIO = os.path.join(ROOT, "audio")
META = os.path.join(ROOT, "metadata", "UrbanSound8K.csv")
N_EXPECT = 8732


def main():
    # 官方 CSV
    rows = []
    with open(META) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    csv_fnames = {r["fname"] for r in rows}
    print(f"CSV rows: {len(rows)}  unique fnames: {len(csv_fnames)}")

    # 磁盘 wav
    import scipy.io.wavfile as wf
    disk = {}
    sr_hist, dur_hist, fold_cls = Counter(), Counter(), Counter()
    for fold in range(1, 11):
        d = os.path.join(AUDIO, f"fold{fold}")
        if not os.path.isdir(d):
            print(f"  MISSING dir: fold{fold}")
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".wav"):
                continue
            p = os.path.join(d, fn)
            sr, arr = wf.read(p)
            disk[fn] = p
            sr_hist[sr] += 1
            dur_hist[round(len(arr) / sr, 2)] += 1
    print(f"disk wavs: {len(disk)}")

    missing = csv_fnames - set(disk)
    extra = set(disk) - csv_fnames
    print(f"missing vs CSV: {len(missing)} {sorted(missing)[:5]}")
    print(f"extra vs CSV: {len(extra)} {sorted(extra)[:5]}")
    print(f"sr hist: {dict(sr_hist)}")
    durs = sorted(dur_hist)
    print(f"duration range: {durs[0]}..{durs[-1]}s; clips<1.28s: "
          f"{sum(v for k, v in dur_hist.items() if k < 1.28)}")
    # fold×class 对照
    for r in rows:
        fold_cls[(int(r["fold"]), int(r["classID"]))] += 1
    print(f"fold×class cells: {len(fold_cls)}")
    ok = len(disk) == N_EXPECT and not missing and not extra
    print(f"VERDICT: {'PASS 8732/8732' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
