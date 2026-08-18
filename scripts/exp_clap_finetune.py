"""exp_clap_finetune.py — CLAP fine-tune decisive check（2026-08-16, decisive check #1）

问题: 论文的可决策域天花板归因（task-definition ceiling）目前依赖 1.29M 参数的
      SONTRA-A; 审稿人最可能的攻击是"你的模型太弱, 大模型 fine-tune 后就赢了"。
      zero-shot CLAP 探针（13.2% phys top-1）不约束 fine-tuned 情形（论文明说）。
      本脚本把 CLAP(htsat-unfused) 在 WSO-Sim 损坏任务上 fine-tune, 回答:
      强预训练表示 + 任务适配能否抬起可决策域天花板?

协议（对齐论文评估契约）:
  训练: 官方 train split（US8K fold 1-9 / SC-10 train）; 在线损坏网格
        {-20,-5,10}dB × {wind,occlusion,self_motion}（与 exp_sc10 训练网格一致）;
        CE 监督（单标签）; 冻结音频分支前 --freeze-blocks 个 transformer 块,
        微调其余块 + 音频投影 + 新分类头; AdamW + 余弦退火; 种子 C.SEED 派生。
  评估: 官方 test split, 网格 {-25,-15,-5,5,15}dB × 3 family;
        AF-Rule 物理门（B11 SP anchor 提供 SNR̂, 同 exp_us8k_clap.py）;
        τ 扫描 → gap/risk/cov/acc_at_dec/viol; clean 域 top-1 天花板;
        逐类 acc_at_dec（可决策域）。
  判定（输出中打印）:
    acc_at_dec(US8K, 门内) ≥ 82.2% → 天花板被抬起（头条从负转正, 重写实验/讨论）
    < 82.2% → 任务定义天花板获强表示级证据（论文叙事加强, 攻击永久死亡）

用法: python scripts/exp_clap_finetune.py --dataset us8k
      [--epochs 6 --lr 1e-4 --batch 16 --freeze-blocks 8 --device mps --smoke 0 --tag 20260816]
      --smoke N: 只跑 N 个 batch 的训练 + 20 clips 评估（快速验证管线）
输出: outputs/exp_clap_finetune_{dataset}_{tag}.json
      outputs/checkpoints_clap_finetune/clap_ft_{dataset}_{tag}.pt
注意: 新增实验记录（非 R19 冻结 pass）; 参数/种子写入 JSON（冻结纪律）。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aof import config as C
from aof.af_rule import AFRule
from aof.baselines import SPAnchorB11
from aof.mapping import CLASS_NAMES as FSD50K_NAMES, build_dev_index
from aof.metrics import operating_gap, coverage
from aof.wsosim import _wind, corrupt
from exp_sc10 import SC_NAMES, load_sc_index, load_sc_wav
from exp_us8k import US8K_NAMES, load_us8k_index, win_clips
from run_main import load_fsd50k_clips

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
CKPT_DIR = os.path.join(OUT, "checkpoints_clap_finetune")
MODEL_ID = os.environ.get("CLAP_MODEL_ID", "laion/clap-htsat-unfused")
SR_TARGET = 48000
TRAIN_SNRS = [-20.0, -5.0, 10.0]
EVAL_SNRS = [-25.0, -15.0, -5.0, 5.0, 15.0]
KINDS = ["wind", "occlusion", "self_motion"]


def to_48k(x, fs=16000):
    from scipy.signal import resample_poly
    return resample_poly(x, SR_TARGET, fs).astype(np.float32)


class ClapFT(nn.Module):
    """CLAP 音频分支微调: 冻结文本侧 + 前 freeze_blocks 个 htsat 块, 头接在 pooler 上。"""

    def __init__(self, clap, n_out, freeze_blocks=8):
        super().__init__()
        self.clap = clap
        for p in self.clap.text_model.parameters():
            p.requires_grad = False
        if hasattr(self.clap, "text_projection"):
            for p in self.clap.text_projection.parameters():
                p.requires_grad = False
        # htsat 块路径在 transformers 版本间有差异（旧版 audio_model.htsat.blocks,
        # 新版 audio_encoder.layers 是 [2,2,6,2] 的 stage 列表）; 找不到就冻结整个
        # audio_model 主干（保守方案, 打印 WARN 说明）。
        blocks = None
        for path in ("audio_model.htsat.blocks", "audio_encoder.layers"):
            try:
                blocks = self.clap.audio_model.get_submodule(path)
                break
            except AttributeError:
                continue
        if blocks is None:
            print("WARN: htsat 块路径未找到; 回退为冻结全部 audio_model 主干", flush=True)
            for p in self.clap.audio_model.parameters():
                p.requires_grad = False
        else:
            blks = [b for st in blocks for b in st.blocks] if hasattr(blocks[0], "blocks") else list(blocks)
            n_frozen = 0
            for i, b in enumerate(blks):
                if i < freeze_blocks:
                    for p in b.parameters():
                        p.requires_grad = False
                    n_frozen += 1
            print(f"freeze: {n_frozen}/{len(blks)} htsat blocks", flush=True)
        self.head = nn.Linear(self.clap.config.projection_dim, n_out)

    def forward(self, input_features, is_longer):
        out = self.clap.get_audio_features(input_features=input_features, is_longer=is_longer)
        emb = F.normalize(out.pooler_output, dim=-1)
        return self.head(emb)


def _fsd50k_clips(split):
    """FSD50K-10: dev.csv 官方 split; 主标签 = 第一个非 unknown 本体类。
    口径警示（第八轮）: 探针训练+评估为 single-label 严格匹配（argmax==主标签），
    而论文 SONTRA-A 的 FSD50K-10 数字（含 43.5% clean ceiling）为 §4.2 multi-label hits。
    探针口径更严，论文已披露（4.3 Scoring convention / limitation (ii)），结论保守成立。"""
    index = build_dev_index()
    clips = []
    for fname, y, s in index:
        if s != split:
            continue
        cls = [i for i, v in enumerate(y) if v and i < C.N_CLASSES - 1]
        if not cls:
            continue
        loaded = load_fsd50k_clips([fname])
        if not loaded:
            continue
        clips.append((loaded[0][0], cls[0], fname))
    return clips


def build_train_clips(dataset):
    """返回 [(x @16k, label int)]。"""
    clips = []
    if dataset == "us8k":
        idx = [e for e in load_us8k_index() if e[3] == "train"]
        for x, y, fname in win_clips(idx)[0]:
            clips.append((x, int(np.argmax(y[:10]))))
    elif dataset == "fsd50k10":
        clips = [(x, lab) for x, lab, _ in _fsd50k_clips("train")]
    else:
        for rel, y, split in load_sc_index():
            if split != "train":
                continue
            x = load_sc_wav(rel)
            if x is None:
                continue
            clips.append((x, int(np.argmax(y[:10]))))
    return clips


def build_test_clips(dataset):
    """返回 [(x @16k, label int, fname)]。"""
    clips = []
    if dataset == "us8k":
        idx = [e for e in load_us8k_index() if e[3] == "test"]
        for x, y, fname in win_clips(idx)[0]:
            clips.append((x, int(np.argmax(y[:10])), fname))
    elif dataset == "fsd50k10":
        clips = _fsd50k_clips("val")
    else:
        for rel, y, split in load_sc_index():
            if split != "test":
                continue
            x = load_sc_wav(rel)
            if x is None:
                continue
            clips.append((x, int(np.argmax(y[:10])), rel))
    return clips


def corrupt_batch(batch_clips, rng):
    """逐 clip 在线损坏; 返回 (wavs list, labels list)。损坏失败则回退干净窗。"""
    wavs, labels = [], []
    for x, lab in batch_clips:
        kind = KINDS[rng.integers(len(KINDS))]
        snr = TRAIN_SNRS[rng.integers(len(TRAIN_SNRS))]
        seed = int(rng.integers(1 << 31))
        xc, _, meta = corrupt(x, kind, float(snr), seed)
        wavs.append(xc if xc is not None else x)
        labels.append(lab)
    return wavs, labels


def evaluate(model, proc, dev, clips, tag, dataset, names):
    """冻结纪律评估: 全网格损坏 + B11 SNR̂ 物理门 + τ 扫描; 另报 clean 天花板与逐类。"""
    rule = AFRule()
    b11 = SPAnchorB11()
    rng = np.random.default_rng(C.SEED + 11)
    recs = []
    t0 = time.time()
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
                             return_tensors="pt").to(dev)
                with torch.no_grad():
                    logits = model(audio["input_features"], audio["is_longer"])
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                recs.append({"fname": fname, "kind": kind, "snr_db": snr_db,
                             "r_true_db": float(r_true), "snr_hat_db": float(snr_hat_db),
                             "labels": labels, "probs": p})
        if (i + 1) % 100 == 0:
            print(f"    eval {i+1}/{len(clips)} ({time.time()-t0:.0f}s)", flush=True)

    rmin_db = 10 * np.log10(rule.r_min)
    r_true = np.array([r["r_true_db"] for r in recs])
    probs = np.stack([r["probs"] for r in recs])
    snr_hat = np.array([r["snr_hat_db"] for r in recs])
    labels_list = [r["labels"] for r in recs]
    n = len(recs)

    # clean 域天花板（原始 clip 窗, 不损坏）
    clean_hits = 0
    for x, lab, _ in clips:
        audio = proc(audio=[to_48k(x)], sampling_rate=SR_TARGET, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(audio["input_features"], audio["is_longer"])
        clean_hits += int(int(logits.argmax(-1).item()) == lab)
    clean_ceiling = clean_hits / len(clips) if clips else None

    # 物理门（B11 SNR̂）: 同 exp_us8k_clap.py
    r = 10.0 ** (snr_hat / 10.0)
    d = r * np.sqrt(rule.n / 2) / (1.0 + np.sqrt(1.0 + 2.0 * r))
    pd = 0.5 * (1.0 + erf(d / np.sqrt(2.0)))
    pd_ok = pd >= (1.0 - rule.alpha_k)
    best = probs.max(axis=1)
    argmax = probs.argmax(axis=1)

    rows = {}
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
        decide = pd_ok & (best > tau)
        correct = np.array([decide[i] and argmax[i] in labels_list[i] for i in range(n)])
        gap, risk = operating_gap(decide, correct, C.ALPHA)
        acc = float(correct[decide].mean()) if decide.any() else float("nan")
        rows[f"tau{tau:.1f}"] = {"gap": round(gap, 3), "risk": round(risk, 3),
                                 "coverage": round(coverage(decide), 3),
                                 "acc_at_dec": round(acc, 3)}
    best_tau = max(rows, key=lambda k: rows[k]["acc_at_dec"]) if rows else None

    # 逐类 acc_at_dec（门内, τ=0.5 同论文口径）
    tau = 0.5
    decide = pd_ok & (best > tau)
    cls = defaultdict(lambda: [0, 0])
    for i in np.nonzero(decide)[0]:
        k = int(argmax[i])
        if k < len(names):
            cls[k][1] += 1
            cls[k][0] += 1 if k in labels_list[i] else 0
    per_class = {names[k]: {"acc": round(v[0] / v[1], 3), "n": v[1]} if v[1] else {"acc": None, "n": 0}
                 for k, v in sorted(cls.items())}

    return {"clean_ceiling": round(clean_ceiling, 3) if clean_ceiling else None,
            "rows": rows, "best_tau": best_tau, "per_class": per_class,
            "n_windows": n, "n_clips": len(clips)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="us8k", choices=["us8k", "sc10", "fsd50k10"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze-blocks", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", type=int, default=0, help="只跑 N batch 训练 + 20 clips 评估")
    ap.add_argument("--tag", default="20260816")
    args = ap.parse_args()

    dev = args.device if (args.device == "cpu" or torch.backends.mps.is_available()) else "cpu"
    if args.device != dev:
        print(f"WARN: {args.device} 不可用, 回退 {dev}", flush=True)

    from transformers.models.clap import ClapModel, ClapProcessor
    clap = ClapModel.from_pretrained(MODEL_ID).to(dev)
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapFT(clap, n_out=10, freeze_blocks=args.freeze_blocks).to(dev)
    names = (FSD50K_NAMES if args.dataset == "fsd50k10"
             else (SC_NAMES if args.dataset == "sc10" else US8K_NAMES))
    print(f"CLAP-FT on {dev}; dataset={args.dataset}; freeze_blocks={args.freeze_blocks}",
          flush=True)

    train_clips = build_train_clips(args.dataset)
    test_clips = build_test_clips(args.dataset)
    print(f"train clips: {len(train_clips)}  test clips: {len(test_clips)}", flush=True)
    if args.smoke:
        train_clips = train_clips[: args.smoke * args.batch]
        test_clips = test_clips[:20]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED + 7)

    t0 = time.time()
    step, best_loss = 0, None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_clips)).tolist()
        tot, nbatch = 0.0, 0
        for i in range(0, len(train_clips) - len(train_clips) % args.batch, args.batch):
            idx = perm[i: i + args.batch]
            wavs, labels = corrupt_batch([train_clips[j] for j in idx], rng)
            audio = proc(audio=[to_48k(w) for w in wavs], sampling_rate=SR_TARGET,
                         return_tensors="pt").to(dev)
            logits = model(audio["input_features"], audio["is_longer"])
            loss = F.cross_entropy(logits, torch.tensor(labels, device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nbatch += 1
            step += 1
            if step % 50 == 0:
                el = (time.time() - t0) / step
                print(f"  ep{ep} step{step} loss={tot/nbatch:.4f} "
                      f"({el:.2f}s/step, ETA {(len(train_clips)//args.batch - i//args.batch - 1)*el/60:.0f}m)",
                      flush=True)
        sched.step()
        print(f"epoch {ep} done: loss={tot/nbatch:.4f}", flush=True)
        if args.smoke:
            break

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"clap_ft_{args.dataset}_{args.tag}.pt")
    torch.save({"state_dict": model.state_dict(),
                "freeze_blocks": args.freeze_blocks, "seed": C.SEED,
                "n_train_clips": len(train_clips)}, ckpt_path)

    model.eval()
    ev = evaluate(model, proc, dev, test_clips, args.tag, args.dataset, names)
    print(f"clean ceiling: {ev['clean_ceiling']}", flush=True)
    print(f"rows: {json.dumps(ev['rows'])}", flush=True)

    # 判定
    acc_at_dec = None
    for tau, row in ev["rows"].items():
        if row["coverage"] and 0.35 <= row["coverage"] <= 0.62:
            acc_at_dec = row["acc_at_dec"]
            break
    if acc_at_dec is None:
        acc_at_dec = ev["rows"].get("tau0.5", {}).get("acc_at_dec")
    verdict = ("CEILING_LIFTED" if (acc_at_dec is not None and acc_at_dec >= 0.822)
               else "CEILING_CONFIRMED")
    out = {"tag": args.tag, "dataset": args.dataset, "model": MODEL_ID,
           "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
           "freeze_blocks": args.freeze_blocks, "seed": C.SEED, "device": dev,
           "n_train_clips": len(train_clips), "ckpt": ckpt_path,
           "verdict": verdict,
           "interpretation": ("acc_at_dec ≥ 0.822 → 天花板抬起, 头条转正, 重写实验/讨论;"
                              " < 0.822 → 任务定义天花板获强表示级证据, 叙事加强"),
           "counterfactual": {"to_beat_anchor": 0.822, "to_reach_alpha": 0.90,
                              "sontra_a_ref": 0.580, "clap_zs_ref": 0.181},
           "evaluation": ev}
    with open(os.path.join(OUT, f"exp_clap_finetune_{args.dataset}_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: {verdict}  (acc_at_dec≈{acc_at_dec})", flush=True)
    print(f"DONE → outputs/exp_clap_finetune_{args.dataset}_{args.tag}.json")


if __name__ == "__main__":
    main()
