"""make_tau_tables.py — 补充材料 τ 全扫表 + Wilson 区间（2026-08-18, optimization #2/#8）

输入（全部冻结 JSON, 不重跑）:
  exp_clap_finetune_{fsd50k10,sc10,us8k}_20260816.json  探针 τ 扫（rows, per_class）
  exp_tau_scan_frozen.json                               FSD50K-10 B12 τ 扫（frozen_equivalent=false）
  exp_{sc10,us8k}_eval.json                              B12 τ 扫（systems, 含精确 n_decide）
  paper_stats.json                                       部分行精确 (n, k) 交叉验证
输出:
  outputs/exp_tau_tables_20260818.json                   恢复的精确计数 + Wilson CI + z + p
  ../paper/supp_tables.tex                               可直接 input 的 LaTeX 表片段

计数口径:
  - systems 表: n_decide 精确（JSON 字段）; k = round(risk × n_decide)
  - 探针 rows: n_decide = round(coverage × n_windows); k = round(risk × n_decide)
  - per_class: n 精确（JSON 字段）; k = round(acc × n)
  恢复误差: 冻结 JSON 的 risk/coverage/acc 为 3 位小数, 恢复计数与真实计数最多差 ±1,
  Wilson 界随之漂移 ≤0.004（对表脚注披露; 正文已报的 CI 用正文值）。
检验: 复算 paper_stats 精确行与正文 CI（FSD50K-10 探针 τ=0.7 [0.066,0.083] 等）。
z 约定（与正文一致）: z = (α − risk)/sqrt(risk(1−risk)/n); p = 1 − Φ(z)（H0: risk ≥ α）。
"""
import json
import math
import os
import sys
from scipy.stats import norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
PAPER = os.path.join(REPO, "..", "paper")
ALPHA = 0.1
Z = 1.959963984540054

TAU_KEYS = {"probe": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
            "b12": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]}


def wilson(k, n):
    if n == 0:
        return None
    p = k / n
    c = (k + Z * Z / 2) / (n + Z * Z)
    se = math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return max(0.0, c - Z * se), min(1.0, c + Z * se)


def wald_z_p(risk, n):
    if n == 0:
        return None, None
    se = math.sqrt(risk * (1 - risk) / n)
    z = (ALPHA - risk) / se
    return z, 1.0 - norm.cdf(z)


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def probe_rows(ds, n_windows):
    d = load(f"exp_clap_finetune_{ds}_20260816.json")
    rows = d["evaluation"]["rows"]
    out = []
    for t in TAU_KEYS["probe"]:
        r = rows[f"tau{t:.1f}"]
        n_dec = round(r["coverage"] * n_windows)
        k = round(r["risk"] * n_dec) if n_dec else 0
        out.append((t, r, n_dec, k))
    return out


def b12_rows(source_json, key_prefix, n_windows, exact_n=False, key_fmt="{:.2f}", bucket=None):
    d = load(source_json)
    if bucket:
        d = d[bucket]
    out = []
    for t in TAU_KEYS["b12"]:
        r = d[f"{key_prefix}{key_fmt.format(t)}"]
        n_dec = r.get("n_decide") if exact_n else round(r["coverage"] * n_windows)
        k = round(r["risk"] * n_dec) if n_dec else 0
        out.append((t, r, n_dec, k))
    return out


def fmt_p(p):
    if p is None:
        return "--"
    if p == 0.0 or p < 1e-6:
        return "$<10^{-6}$"
    return f"{p:.3f}"


def fmt_ci(ci):
    if ci is None:
        return "--"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def build_table(name, rows, caption, source_note, n_windows):
    jrows = []
    for t, r, n_dec, k in rows:
        ci = wilson(k, n_dec) if n_dec else None
        z, p = wald_z_p(r["risk"], n_dec) if n_dec else (None, None)
        jrows.append({"tau": t, "gap": r["gap"], "risk": r["risk"],
                      "coverage": r["coverage"], "acc_at_dec": r["acc_at_dec"],
                      "n_decide": n_dec, "k_err": k,
                      "wilson_ci": list(ci) if ci else None,
                      "z": round(z, 2) if z is not None else None,
                      "p": p if p is not None else None})
        print(f"  τ={t:4.2f} risk={r['risk']} n={n_dec} k={k} CI={fmt_ci(ci)} "
              f"z={z:.2f} p={fmt_p(p)}" if z else f"  τ={t:4.2f} risk={r['risk']} n=0")
    # LaTeX
    lines = [
        "\\begin{table}[!htb]", "\\centering", "\\small",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lccccrll}",
        "\\toprule",
        "$\\tau$ & gap & risk & cov & acc@dec & $n_{\\rm dec}$ & Wilson 95\\% CI & $z$ / $p$\\\\",
        "\\midrule",
    ]
    for t, r, n_dec, k in rows:
        ci = wilson(k, n_dec) if n_dec else None
        z, p = wald_z_p(r["risk"], n_dec) if n_dec else (None, None)
        if n_dec == 0:
            lines.append(f"{t:.2f} & {r['gap']:.3f} & {r['risk']:.3f} & {r['coverage']:.3f} "
                         f"& -- & 0 & -- & --\\\\")
        else:
            lines.append(f"{t:.2f} & {r['gap']:.3f} & {r['risk']:.3f} & {r['coverage']:.3f} "
                         f"& {r['acc_at_dec']:.3f} & {n_dec:,} & {fmt_ci(ci)} "
                         f"& ${z:.2f}$ / {fmt_p(p)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\par\\vspace{{2pt}}{{\\footnotesize {source_note}}}",
              "\\end{table}"]
    return jrows, "\n".join(lines)


def per_class_table(ds, names):
    d = load(f"exp_clap_finetune_{ds}_20260816.json")
    pc = d["evaluation"]["per_class"]
    jrows = []
    for cls in names:
        if cls not in pc or pc[cls]["n"] == 0:
            continue
        acc, n = pc[cls]["acc"], pc[cls]["n"]
        k = round(acc * n) if acc is not None else 0
        ci = wilson(k, n) if n else None
        jrows.append({"class": cls, "acc": acc, "n": n, "k_err": k, "wilson_ci": list(ci) if ci else None})
    return jrows


def main():
    out = {"tag": "20260818", "alpha": ALPHA,
           "note": "全部来自冻结 JSON 的恢复计数; n_decide 精确（systems）或 round(cov×n_windows)"
                   "（探针）; k=round(risk×n_decide); Wilson 界对 3 位小数恢复值有 ≤0.004 漂移",
           "tables": {}, "per_class": {}}

    print("=== S1: CLAP-FT 探针 τ 扫（FSD50K-10, 23,220 windows） ===")
    rows = probe_rows("fsd50k10", 23220)
    jr, tex = build_table("probe_fsd50k10", rows,
                          "Full $\\tau$ sweep of the fine-tuned CLAP probe on FSD50K-10"
                          " (deployable B11 gate, 23{,}220-window pass; risk = classification"
                          " error on decided windows; Wilson interval and one-sided $z$/$p$"
                          " against $H_0{:}\\,$risk$\\ge\\alpha$, $\\alpha{=}0.1$).",
                          "Source: frozen record exp\\_clap\\_finetune\\_fsd50k10\\_20260816.json;"
                          " counts recovered from rounded coverage/risk (3rd-decimal drift possible).",
                          23220)
    out["tables"]["probe_fsd50k10"] = jr
    tables = [tex]

    print("=== S2: CLAP-FT 探针 τ 扫（SC-10 对照, 61,110 windows） ===")
    rows = probe_rows("sc10", 61110)
    jr, tex = build_table("probe_sc10", rows,
                          "Full $\\tau$ sweep of the fine-tuned CLAP probe on SC-10 (control arm"
                          "; deployable B11 gate, 61{,}110-window pass).",
                          "Source: frozen record exp\\_clap\\_finetune\\_sc10\\_20260816.json;"
                          " counts recovered from rounded coverage/risk.",
                          61110)
    out["tables"]["probe_sc10"] = jr
    tables.append(tex)

    print("=== S3: CLAP-FT 探针 τ 扫（US8K, 11,775 windows） ===")
    rows = probe_rows("us8k", 11775)
    jr, tex = build_table("probe_us8k", rows,
                          "Full $\\tau$ sweep of the fine-tuned CLAP probe on UrbanSound8K"
                          " (deployable B11 gate, 11{,}775-window pass; $\\tau\\ge0.7$ rows have"
                          " zero decisions: the probe's confidence never exceeds 0.7 on"
                          " corrupted windows).",
                          "Source: frozen record exp\\_clap\\_finetune\\_us8k\\_20260816.json;"
                          " counts recovered from rounded coverage/risk.",
                          11775)
    out["tables"]["probe_us8k"] = jr
    tables.append(tex)

    print("=== S4: SONTRA-A B12 τ 扫（FSD50K-10, 6,765 windows, frozen_equivalent=false） ===")
    rows = b12_rows("exp_tau_scan_frozen.json", "B12_tau", 6765, key_fmt="{:g}")
    jr, tex = build_table("b12_fsd50k10", rows,
                          "Full $\\tau$ sweep of SONTRA-A B12 on FSD50K-10 (6{,}765-window pass).",
                          "Source: exp\\_tau\\_scan\\_frozen.json; $\\tau{=}0.5$ and $0.95$ are"
                          " bit-identical to the frozen pass, other rows carry 3rd-decimal"
                          " drift (recomputed from the frozen checkpoint).",
                          6765)
    out["tables"]["b12_fsd50k10"] = jr
    tables.append(tex)

    print("=== S5: SONTRA-A B12 τ 扫（SC-10, 61,110 windows, 精确计数） ===")
    rows = b12_rows("exp_sc10_eval.json", "B12_tau", 61110, exact_n=True, bucket="systems")
    jr, tex = build_table("b12_sc10", rows,
                          "Full $\\tau$ sweep of SONTRA-A B12 on SC-10 (deployable B11 gate,"
                          " 61{,}110-window pass; exact decision counts).",
                          "Source: frozen record exp\\_sc10\\_eval.json (exact $n_{\\rm dec}$);"
                          " $k$ recovered from rounded risk.",
                          61110)
    out["tables"]["b12_sc10"] = jr
    tables.append(tex)

    print("=== S6: SONTRA-A B12 τ 扫（US8K, 11,775 windows, 精确计数） ===")
    rows = b12_rows("exp_us8k_eval.json", "B12_tau", 11775, exact_n=True, bucket="systems")
    jr, tex = build_table("b12_us8k", rows,
                          "Full $\\tau$ sweep of SONTRA-A B12 on UrbanSound8K (deployable B11"
                          " gate, 11{,}775-window pass; exact decision counts).",
                          "Source: frozen record exp\\_us8k\\_eval.json (exact $n_{\\rm dec}$);"
                          " $k$ recovered from rounded risk.",
                          11775)
    out["tables"]["b12_us8k"] = jr
    tables.append(tex)

    # ---- per-class (probe, 门内 τ=0.5; 与正文/表 3 同口径) ----
    names_f = ["vehicle", "siren", "impact", "mechanical_anomaly", "human_activity",
               "tire_squeal", "bicycle", "horn", "construction", "motorcycle"]
    print("=== S7: per-class acc@dec + Wilson（探针, 门内 τ=0.5） ===")
    for ds, names, lab in (("fsd50k10", names_f, "FSD50K-10"),
                           ("sc10", ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"], "SC-10"),
                           ("us8k", ["air_conditioner", "car_horn", "children_playing", "dog_bark",
                                     "drilling", "engine_idling", "jackhammer", "siren", "street_music"], "US8K")):
        jr = per_class_table(ds, names)
        out["per_class"][ds] = jr
        rows_tex = []
        for r in jr:
            ci = r["wilson_ci"]
            rows_tex.append(f"{r['class']} & {r['acc']:.3f} & {r['n']:,} & {fmt_ci(ci)}\\\\")
            print(f"  {r['class']}: acc={r['acc']} n={r['n']} CI={fmt_ci(ci)}")
        tables.append(
            "\\begin{table}[!htb]\n\\centering\n\\small\n"
            f"\\caption{{Per-class accuracy-at-decision of the fine-tuned CLAP probe "
            f"(gate, $\\tau{{=}}0.5$) on {lab}, with Wilson 95\\% intervals "
            f"(exact $n$ from the frozen record; $k$ recovered from rounded acc).}}\n"
            f"\\label{{tab:supp_pc_{ds}}}\n"
            "\\begin{tabular}{lccc}\n\\toprule\nClass & acc@dec & $n_{\\rm dec}$ & Wilson 95\\% CI\\\\\n\\midrule\n"
            + "\n".join(rows_tex) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}")

    with open(os.path.join(OUT, "exp_tau_tables_20260818.json"), "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    os.makedirs(PAPER, exist_ok=True)
    with open(os.path.join(PAPER, "supp_tables.tex"), "w") as f:
        f.write("% Auto-generated by scripts/make_tau_tables.py (2026-08-18); do not edit by hand.\n\n")
        f.write("\n\n".join(tables))
        f.write("\n")
    print(f"\nDONE → outputs/exp_tau_tables_20260818.json + paper/supp_tables.tex")

    # ---- 复算校验 vs 正文已知 CI ----
    print("\n=== 校验 ===")
    chk = [
        ("FSD50K-10 探针 τ=0.7 正文 CI", out["tables"]["probe_fsd50k10"][6]["wilson_ci"], [0.066, 0.083]),
        ("FSD50K-10 B12 τ=0.5 paper_stats", out["tables"]["b12_fsd50k10"][5]["wilson_ci"], [0.4293, 0.4630]),
        ("SC-10 B12 τ=0.5 paper_stats", out["tables"]["b12_sc10"][5]["wilson_ci"], [0.0781, 0.0840]),
        ("US8K B12 τ=0.5 paper_stats", out["tables"]["b12_us8k"][5]["wilson_ci"], [0.4060, 0.4340]),
    ]
    for name, got, exp in chk:
        ok = got is not None and abs(got[0] - exp[0]) < 0.001 and abs(got[1] - exp[1]) < 0.001
        print(f"  {name}: got {got} expect {exp} → {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("校验失败")


if __name__ == "__main__":
    main()
