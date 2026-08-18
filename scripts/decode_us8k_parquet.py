"""decode_us8k_parquet.py — danavery/urbansound8K parquet → 官方布局 wav（2026-08-06）

用法:
  --dry-run     只打印 schema + 行数, 不写文件
  --decode      解码全部 parquet → datasets/urbansound8k/UrbanSound8K/audio/foldN/*.wav
                + metadata/UrbanSound8K.csv（按官方列序）
"""
import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

OUT_ROOT = os.environ.get("US8K_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "urbansound8k", "UrbanSound8K"))
SRC = os.environ.get("US8K_PARQUET_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "us8k_hf", "*.parquet"))
OFFICIAL_COLS = ["fname", "fsID", "start", "end", "salience", "fold", "classID", "class"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    files = sorted(glob.glob(SRC))
    print(f"parquet files: {len(files)}", flush=True)
    assert files, "no parquet found"

    schema = pq.read_schema(files[0])
    print("schema:", schema, flush=True)

    total_rows = 0
    cols_union = Counter()
    for f in files:
        pf = pq.ParquetFile(f)
        total_rows += pf.metadata.num_rows
        for name in pf.schema.names:
            cols_union[name] += 1
    print(f"rows total: {total_rows}  cols: {dict(cols_union)}", flush=True)
    if args.dry_run:
        return

    audio_dir = os.path.join(OUT_ROOT, "audio")
    meta_dir = os.path.join(OUT_ROOT, "metadata")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    import scipy.io.wavfile as wavf

    rows_out = []
    n_written = 0
    sr_hist = Counter()
    for f in files:
        t = pq.read_table(f)
        d = t.to_pydict()
        n = len(d[list(d.keys())[0]])
        for i in range(n):
            audio_row = d.get("audio")
            if audio_row is not None:
                raw = audio_row[i].get("bytes") if isinstance(audio_row[i], dict) else audio_row[i][0]
            else:
                raw = d.get("bytes", [None] * n)[i]
            if raw is None:
                continue
            fname = None
            for k in ("slice_file_name", "audio.path", "fname"):
                if k in d and d[k][i]:
                    fname = str(d[k][i]).split("/")[-1]
                    break
            if fname is None:
                # 从 fsID/classID/instance/slice 推导
                raise SystemExit("cannot determine fname; add column mapping")
            fold = int(d["fold"][i]) if "fold" in d else 0
            class_id = int(d["classID"][i]) if "classID" in d else -1
            cls_name = str(d["class"][i]) if "class" in d else ""
            start = d["start"][i] if "start" in d else None
            end = d["end"][i] if "end" in d else None
            salience = d["salience"][i] if "salience" in d else None
            fsid = d["fsID"][i] if "fsID" in d else None

            import io
            import soundfile as sf
            arr, sr = sf.read(io.BytesIO(bytes(raw)), dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            sr_hist[sr] += 1
            fold_dir = os.path.join(audio_dir, f"fold{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            wavf.write(os.path.join(fold_dir, fname), sr, (arr * 32767.0).astype(np.int16))
            n_written += 1
            rows_out.append([fname, fsid, start, end, salience, fold, class_id, cls_name])
        print(f"  {os.path.basename(f)} done ({n_written} total)", flush=True)

    # 官方列序 CSV
    with open(os.path.join(meta_dir, "UrbanSound8K.csv"), "w") as f:
        f.write(",".join(OFFICIAL_COLS) + "\n")
        for r in rows_out:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print(f"wrote {n_written} wavs; sr hist: {dict(sr_hist)}; csv rows: {len(rows_out)}", flush=True)


if __name__ == "__main__":
    main()
