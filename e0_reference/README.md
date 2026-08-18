# E0 一致性验证 — 参考实现与结果（2026-08-04 首跑）

## 目的
验证 AOF 前沿（v4.1 能量检测公式）在真实音频管线上的决策级行为：
推导阈值是否实现目标风险？退化（AM 风噪 + 参考窗噪声估计）造成多少损耗？

## 文件
| 文件 | 作用 |
|---|---|
| `energy_detector.py` | (r,n) 核心理论 + MC 校验 + 论文形式换算 |
| `wind_noise.py` | WSO-Sim 风噪合成（低频整形 + 5-15Hz AM，确定性种子） |
| `mc_crosscheck.py` | 前沿公式 Monte Carlo 交叉校验（论文附录证据） |
| `e0_reliability.py` | 逐 (r,T) 单元可靠性图（能量比统计量 + 双重预测 + 实测校准） |

## 运行
```bash
cd e0_reference
env -u PYTHONPATH /opt/miniconda3/bin/python3 mc_crosscheck.py
env -u PYTHONPATH /opt/miniconda3/bin/python3 e0_reliability.py --source synthetic
env -u PYTHONPATH /opt/miniconda3/bin/python3 e0_reliability.py --source esc50 --classes 19,9,37
```
（Python: conda 3.13 + numpy 2.5.1 + scipy 1.18 + pyarrow 24；`env -u PYTHONPATH` 规避 hermes venv 污染）

## 统计量与单位（最终形式，已修三处单位 bug）
- 统计量 = 能量比 `z = sum(x_band²)/sum((g·w_ref)²)`，E[z0]=1，Var=2/n（oracle）/ 4/n（参考窗估计）
- `r = E_s/P_n`（窗内事件能量/噪声能量），理论 `Pe = Φ(−r√(n/2)/(1+√(1+2r)))`，n=2BT
- 阈值：`thr = 1 + 2·Φ⁻¹(1−α)/√n`（α=0.1；2/√n 已含参考窗估计方差）
- H0/H1 用独立风噪实现（参考窗 w_ref 估计噪声底，测试窗 w_tst 统计）

## 首跑结果（种子 20260803）

### MC 交叉校验（论文附录证据）
BT=256/1024 × 10 个 SNR 点：`|Pe_emp − Pe_theory| ≤ 0.0057`（10000 trials）。
SNR_min(0.1)：BT=256 → −31.70dB（渐近 −29.89，MF oracle −17.64）；BT=1024 → −40.90dB。

### 可靠性网格（ESC-50 真实音频, 72 clips: thunderstorm/crow/clock_alarm, 非重叠全窗口）
| r | SNR(dB)@T | T | oracle Pe | realistic | calib | **emp** | σ0× | |emp−calib| |
|---|---|---|---|---|---|---|---|---|
| 0.05 | −45.8 | 0.32 | 0.2248 | 0.3148 | 0.4222 | **0.4163** | 3.89 | 0.0058 |
| 0.05 | −51.9 | 1.28 | 0.0652 | 0.1427 | 0.3583 | **0.3396** | 4.24 | 0.0187 |
| 0.15 | −41.1 | 0.32 | 0.0149 | 0.0864 | 0.2851 | **0.2722** | 3.89 | 0.0128 |
| 0.50 | −38.9 | 0.64 | 0.0000 | 0.0735 | 0.1782 | **0.1766** | 3.68 | 0.0017 |
（完整 9 单元见 `../outputs/e0_reliability_esc50.csv` 与 synthetic 版）

## 结论（首跑，写入实验卡）
1. **管线统计自洽**：9/9 单元 `|emp−calib| ≤ 0.0313`（合成 0.0185）→ 能量比统计量 + 高斯近似在真实音频上成立。
2. **量化发现（论文素材）**：AM 风噪（m=0.5, f=10Hz）使 H0 统计量标准差膨胀 **σ0× ≈ 3.7-4.2**（方差 ×14-18）→ oracle 理论阈值虚警失控：r=0.05/T=1.28 处 oracle Pe=0.065 → 实测 0.340。即"风噪摧毁能量检测器"的定量证据（2110.05632 现象）。
3. **解释**：参考窗与测试窗的 AM 包络相位失配 → 窗能量波动 ≫ χ² 理论 → 等效噪声底方差 ×14-18。真实系统对策：① 风噪 PSD 跟踪/分窗估计（SP 锚升级）；② 阈值按实测方差校准（τ-Cal）；③ A-Head 学习鲁棒 SNR 估计（L3 层）。
4. **一致性判据（实验卡）**：|emp−calib| ≤ 0.05 → **PASS**（所有单元）。oracle 与实测的差距 = 退化损耗（论文的 Operating Gap 组分），非失败。

## 已知限制
- 事件 = 整窗能量（τ=T）；真实事件子窗（τ<T）将降低有效 r —— 已列入 G1 实验。
- 风噪为合成（WSO-Sim v1）；真实风噪域验证 = D5（8/8-8/15 录音窗口）。
- ESC-50 仅用 shard1 三类的 72 clips；shard2 到达后扩至 10 类映射所需类别。
