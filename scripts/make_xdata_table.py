"""make_xdata_table.py — 跨数据集评估 JSON → 论文 LaTeX 表（2026-08-06）

用法: python3 scripts/make_xdata_table.py <exp_sc10_eval.json|exp_us8k_eval.json>
输出: LaTeX tabular 行（镜像主表 gap/risk/cov/c_phys + viol + acc@dec）
"""
import json
import os
import sys


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "exp_sc10_eval.json")
    d = json.load(open(p))
    det = d["detectors"]
    sys_ = d["systems"]
    print(f"# {os.path.basename(p)}  r_min={d.get('r_min_db')}dB  "
          f"n_windows={d.get('n_windows')}  n_clips={d.get('n_clips')}\n")
    rows = [
        ("oracle", det["oracle"]),
        ("B11a (SP + true thr.)", det["B11a_sp_oracle_thr"]),
        ("B0 (energy detector)", det["B0_energy"]),
        ("B11 (SP anchor)", det["B11_sp"]),
        ("B12a (A-Head + true thr.)", sys_["B12a_true_snr"]),
        ("B13 (A-Head + phys. thr.)", sys_["B13_phys_only"]),
        ("B12 (CAE, $\\tau{=}0.5$)", sys_["B12_tau0.50"]),
        ("B12 (CAE, $\\tau{=}0.95$)", sys_["B12_tau0.95"]),
    ]
    print("System & gap & risk & cov & $c_{phys}$ & viol & acc@dec \\\\")
    print("\\midrule")
    for name, r in rows:
        gap = r["gap"]
        gap_s = f"{gap:+.3f}"
        cphys = r.get("c_phys", "-")
        viol = r.get("viol", "-")
        acc = r.get("acc_at_dec", "-")
        try:
            acc_s = "-" if acc is None or (isinstance(acc, str) and acc == "nan") else f"{float(acc):.3f}"
        except (TypeError, ValueError):
            acc_s = "-"
        print(f"{name} & {gap_s} & {r['risk']:.3f} & {r['coverage']:.3f} "
              f"& {cphys} & {viol} & {acc_s} \\\\")
    print("\n# τ 扫描 (B12)")
    for k in sorted(sys_):
        if k.startswith("B12_tau"):
            r = sys_[k]
            print(f"  {k}: gap={r['gap']:+.3f} risk={r['risk']:.3f} cov={r['coverage']:.3f} "
                  f"acc@dec={r['acc_at_dec']} viol={r['viol']}")
    print("\n# per-kind (B12 τ=0.5)")
    for k, r in d.get("per_kind", {}).items():
        print(f"  {k}: gap={r['gap']:+.3f} risk={r['risk']:.3f} cov={r['coverage']:.3f} acc@dec={r['acc_at_dec']}")
    print("\n# clean ceiling:", d.get("clean_ceiling"))
    print("# topk decidable:", d.get("topk_decidable"))


if __name__ == "__main__":
    main()
