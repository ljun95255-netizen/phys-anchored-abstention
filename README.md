# Physically Anchored Abstention: Decomposing the Operating Gap in Low-SNR Acoustic Recognition

Code accompanying the manuscript for *Engineering Applications of Artificial Intelligence* (single author: Fan Feiyu). Released under the Apache License 2.0 (see `LICENSE`).

> **Submission status (2026-08-16):** this repository contains no submission records (no manuscript number, no submission timestamp), so the header claim above is intentionally neutral. Confirm the actual status at Elsevier Editorial Manager; if not yet submitted, work through `paper/submission_checklist.md` before submitting.

This repository implements **WSO-Sim** (a scripted wind/occlusion/self-motion corruption simulator with exogenous SNR labels), **AOF** (the acoustic observability frontier of classical detection theory), **SONTRA-A** (a 1.29M-parameter SNR-anchored estimator with an abstention rule), the classical energy/spectral-subtraction anchors, and the full training/evaluation pipelines behind every reported number.

## Repository layout

```
project/
├── aof/                  # Core package
│   ├── config.py         # Hyper-parameters, model contract, data paths (env-overridable)
│   ├── model.py          # SONTRA-A: gated separation + MobileNetV2-style backbone + belief state + dual heads
│   ├── wsosim.py         # WSO-Sim corruption families (wind / occlusion / self-motion) + exogenous r labels
│   ├── af_rule.py        # AF-Rule (Neyman-Pearson dual constraint, Bonferroni alpha_k) + r_min + SNR wall
│   ├── frontiers.py      # ED main frontier + MF lower bound (per-sample dB convention)
│   ├── baselines.py      # B0 energy detector, B11/B11a spectral-subtraction anchor, oracle
│   ├── evaluate.py       # Shared-pass evaluation drivers (detector-tier / system-tier)
│   ├── metrics.py        # operating gap, coverage, c_phys, rank-AUC, SNR-MAE, ...
│   ├── cf_sampler.py     # (clean, corrupted) counterfactual pair sampler
│   ├── data.py           # WAV -> log-mel, FSD50K loading
│   ├── losses.py         # CFAL loss (BCE + separation MSE + masked Huber + L2)
│   ├── train.py          # Training loop (AdamW, warmup, cosine, early stop)
│   ├── train_variants.py # SelectiveNet / Deep Gamblers / shift-aware variants
│   ├── conformal_rc.py   # Split-conformal family (simplified implementations, see paper)
│   ├── baselines_extra.py# Deep ensemble, spectral uncertainty, Sohn NP-LRT
│   ├── stats.py          # Bootstrap risk tests, cluster bootstrap, ECE, MMD
│   ├── wind_inflation.py # AM wind-noise inflation closure
│   ├── calibration.py    # tau-Cal: isotonic SNR residual correction + event-probability calibration
│   └── mapping.py        # FSD50K -> 10-class mapping (65 raw labels), class names
├── e0_reference/         # Independent energy-detection reference + Monte-Carlo crosscheck
├── run_main.py           # FSD50K-10 subset building, SONTRA-A training, gap main matrix
├── scripts/              # Paper pipelines: tau sweep, cross-dataset (SC-10/US8K), probes, figures, tables
├── tests/                # Unit tests (model, rule/metrics, v3 tools) — all pass
├── outputs/              # Frozen evaluation records (JSON) + frozen SONTRA-A checkpoints (seed
│                         #   20260803); CLAP fine-tune checkpoints (~2.3 GB) are excluded from
│                         #   this repository and reproducible via scripts/exp_clap_finetune.py
└── data/                 # Dataset download instructions (datasets are NOT bundled)
```

## Requirements

Tested with Python 3.13 on Apple Silicon (MPS); CUDA should work with a corresponding PyTorch build.

```bash
pip install -r requirements.txt
```

`transformers` is only needed for the CLAP zero-shot probe (`scripts/exp_us8k_clap.py`).

## Datasets

The three public datasets are **not bundled**. See [`data/README.md`](data/README.md) for download URLs and expected directory layouts. Set the environment variables to your locations (defaults point to `data/` inside this repo):

| Variable | Default |
|---|---|
| `FSD50K_DIR` | `data/fsd50k` |
| `SC_ROOT` | `data/speech_commands` |
| `US8K_ROOT` | `data/urbansound8k/UrbanSound8K` |
| `ESC50_DIR` | `data/esc50` (e0 reliability grid only) |
| `CLAP_MODEL_ID` | `laion/clap-htsat-unfused` (CLAP probe only) |

## Quick start

```bash
python -m pytest tests/ -q                    # all unit tests
python scripts/mc_unified.py                  # Monte-Carlo validation of the frontier formula (Table in paper)
```

## Reproducing the paper

All reported numbers are produced by **fixed evaluation passes** whose aggregate records are in `outputs/` (the JSON files are the reference; re-running on MPS may drift in the 3rd decimal, see Reproducibility note).

### Frozen checkpoints (seed 20260803)

| Checkpoint | Dataset | Trained on |
|---|---|---|
| `outputs/checkpoints/sontra_a_ep22.pt` | FSD50K-10 | 6,676 clips, balanced quota 1,200/class, 45 epochs |
| `outputs/checkpoints_sc10/sontra_a_ep35.pt` | Speech Commands 10-core | same recipe |
| `outputs/checkpoints_us8k/sontra_a_ep23.pt` | UrbanSound8K (10 native classes) | same recipe |

### Main gap matrix (FSD50K-10, 6,765-window shared pass)

```bash
python scripts/exp_tau_scan_frozen.py          # B12 tau sweep + detector tier (Table II, Fig. 5)
python scripts/exp_b13_snr_only.py             # B13 physical-only variant
python scripts/exp_conformal_baselines.py --ckpt outputs/checkpoints/sontra_a_ep22.pt   # AGRC/SCRC
```

Key frozen operating points (gap / risk / coverage):

| System | FSD50K-10 | SC-10 | US8K |
|---|---|---|---|
| oracle | −0.100 / 0 / 0.600 | −0.100 / 0 / 0.600 | −0.100 / 0 / 0.600 |
| B0 energy | +0.075 / 0.175 / 0.427 | +0.229 / 0.329 / 0.128 | +0.085 / 0.185 / 0.425 |
| B11 SP anchor | +0.128 / 0.228 / 0.468 | +0.308 / 0.408 / 0.156 | +0.134 / 0.234 / 0.471 |
| B12 (τ=0.5) | +0.346 / 0.446 / 0.495 | −0.019 / 0.081 / 0.542 | +0.320 / 0.420 / 0.424 |
| B12 (τ=0.95) | +0.184 / 0.284 / 0.183 | −0.077 / 0.023 / 0.445 | +0.125 / 0.225 / 0.204 |
| B13 physical-only | +0.394 / 0.494 / 0.612 | — | +0.400 / 0.500 / 0.604 |
| AGRC (conformal) | −0.017 / 0.083 / 0.011 | — | — |

The gap decomposes as physical bound (0.400) + threshold-estimation cost (B12a−B12 = 0.003) + classifier cost (B12−B11), localizing the learned system's deficit to classifier accuracy in the decidable region.

### Cross-dataset training and evaluation

```bash
python scripts/exp_sc10.py                     # train SONTRA-A on Speech Commands 10-core
python scripts/exp_sc10_eval.py                # SC-10 frozen evaluation (Table III)
python scripts/exp_us8k.py                     # train SONTRA-A on UrbanSound8K (fold 1–9)
python scripts/exp_us8k_eval.py                # US8K fold-10 evaluation
python scripts/exp_us8k_clap.py                # CLAP zero-shot probe (needs transformers + HF download)
```

### Ceiling probes (task-definition ceiling, Section V-C)

```bash
python scripts/exp_p1_class_snr_heatmap.py     # per-class × SNR recall heatmap
python scripts/exp_p2_temporal_voting.py       # multi-window voting probe (negative result)
python scripts/exp_p3_multilabel.py            # multi-label decomposition, top-k hit rates
python scripts/exp_p4_clean_ceiling.py         # clean-domain classifier ceiling
```

### Figures and tables

```bash
python scripts/figs_cas.py                     # Fig. 1a / Fig. 2 (STIX fonts, 300 dpi) -> outputs/
python scripts/paper_stats.py                  # statistical tests cited in the paper
python scripts/paper_extra_stats.py            # per-class error analysis, dataset cards
python scripts/make_xdata_table.py             # LaTeX table rows from evaluation JSONs
```

### Training from scratch (FSD50K-10)

```bash
python run_main.py --balanced 1200 --n-val 500 --epochs 45
```

Balanced per-class quota (1,200/class, rare classes exhausted → 6,676 clips) is the protocol that produced the frozen checkpoint. The training grid is {−20, −5, +10} dB × {wind, occlusion, self_motion}; the evaluation grid is {−25, −15, −5, +5, +15} dB (straddling the physical frontier). Use `python -u` for long runs (stdout buffering can hide tracebacks on crash).

## Hardware and runtime notes

- Tested on Apple Silicon MPS (16 GB). Batch 64 is the recommended default; batch 128 degrades ~3× on MPS.
- Training FSD50K-10 (6,676 clips, 45 epochs) takes several hours on the M4 MPS; SC-10/US8K about 1.5–2 h each.
- The unit tests run on CPU and do not require datasets.

## Reproducibility note

MPS inference is not bit-deterministic: re-running an evaluation with the same code and seed reproduces results to ~1e-3 (e.g., B12 τ=0.5 risk 0.446 vs 0.4456). The frozen JSON records in `outputs/` are the authoritative aggregate records behind the paper's tables; a re-run should match them within this tolerance. The physical frontier (`r_min`, `c_phys`) and the MC validation table are deterministic.

## Decisive-check experiments (added 2026-08-16)

Three pre-submission experiments targeting the paper's three most predictable
reviewer attacks (see `paper/integration_drafts.md` for the conditional write-up
drafts and go/no-go rules). These are **new** experiment records — they do not
re-run or modify any frozen R19 number.

| Script | Question | Run time (M4 MPS) |
|---|---|---|
| `scripts/exp_heterogeneity.py` | Does the embedding-space class-distance ratio rank SC-10 < US8K ≈ FSD50K-10, matching the ceiling? | ~10–20 min |
| `scripts/exp_clap_finetune.py` | Does fine-tuned CLAP lift the decidable-region ceiling (acc@dec ≥ 82.2%)? | ~1–4 h per dataset |
| `scripts/exp_conformal_hybrid.py` | Does conformal risk control regain coverage when calibrated within the decidable region? | ~30–60 min |

```bash
python scripts/exp_heterogeneity.py --device mps
python scripts/exp_clap_finetune.py --dataset us8k --smoke 16   # 管线自检
python scripts/exp_clap_finetune.py --dataset us8k --epochs 6 --freeze-blocks 8
python scripts/exp_conformal_hybrid.py --device mps
```

Each script prints a `VERDICT:` line and writes its own JSON to `outputs/`; results
are only folded into the paper if they confirm (rules in `integration_drafts.md`).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
