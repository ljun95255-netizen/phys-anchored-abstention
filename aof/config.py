"""config.py — ICASSP 2027 v4.1 实验配置（对齐 ICASSP_idea.html §6.2 + Swift 模型契约）"""
from dataclasses import dataclass, field

# ---- 音频/特征（Swift 模型契约: mel 24 带, 16kHz, 1.28s 窗）----
SAMPLE_RATE = 16_000
WINDOW_SEC = 1.28
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)          # 20480
N_MELS = 24          # 与 Swift 模型 stem 输入一致（论文 §6.2 的 64 由模型契约覆盖，写入时统一）
N_FFT = 512
HOP = int(0.01 * SAMPLE_RATE)                            # 160
N_FRAMES = WINDOW_SAMPLES // HOP                        # 128
MEL_FMIN, MEL_FMAX = 40.0, 7600.0

# ---- 模型（Swift 架构忠实移植）----
ENC_CHANNELS = 96
# 窗口化变体: 编码器后 T'=32 步, dilation 32/64 的感受野(65/129)超出序列长度 →
# 因果卷积几乎只看填充, temporal 输出退化 [R11 实证]. 流式全 dilations 保留于 Swift 侧.
TEMPORAL_DILATIONS = [1, 2, 4, 8, 16]
N_CLASSES = 10          # 车辆/自行车/喇叭/警笛/轮胎打滑/撞击/施工/机械异常/人类活动/unknown
UNKNOWN_CLASS = 9       # unknown 掩码置 0

# ---- v4.1 超参（§6.2）----
BATCH_SIZE = 64
LR = 3e-4
WARMUP_FRAC = 0.05
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 8
GAMMA = 0.5             # 决策阈值 γ（p̂ > γ 进入候选集）
ALPHA = 0.1             # NP 双参: P_D ≥ 1−α
BETA = 0.05             # NP 双参: P_FA ≤ β
SNR_WALL_RHO_DB = 1.0   # 噪声功率估计不确定 ρ（dB）→ SNR wall

# ---- CFAL 损失权重（运行期 EMA 归一化后的相对权重, 2026-08-04 实证: 固定权重会被尺度漂移摧毁）----
LAM_EV = 1.0            # 事件损失相对权重
LAM_SEP = 0.5           # 分离损失相对权重
LAM_AUD = 1.0           # 可听性损失相对权重
LAM_REG = 1e-4          # 正则
HUBER_DELTA_DB = 2.0    # 掩码 Huber δ=2dB

# ---- WSO-Sim 参数（v4.1 §5.5）----
WIND_AM_FREQS = [5.0, 9.0, 14.0]   # 三正弦 AM（5-15Hz）
WIND_AM_DEPTH = 0.5
WIND_FC = 500.0                     # 风噪低通
OCCL_FREQS = [500.0, 1000.0, 2000.0, 4000.0]   # 遮挡低通 f_c ∈ [0.5,4]kHz
OCCL_ATTN_DB = [0.0, -6.0, -12.0]   # 遮挡衰减
SELFMOTION_FREQS = [1.0, 2.0]       # 自运动周期增益 1-2Hz
SELFMOTION_DEPTH = 0.3

# ---- 事件带（E0 同口径）----
EVENT_BAND = (1000.0, 4000.0)

# ---- 数据（env 可覆盖; 默认 <repo>/data/...）----
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
ESC50_DIR = _os.environ.get("ESC50_DIR", _os.path.join(_REPO_ROOT, "data", "esc50"))
FSD50K_DIR = _os.environ.get("FSD50K_DIR", _os.path.join(_REPO_ROOT, "data", "fsd50k"))
CACHE_DIR = _os.environ.get("CACHE_DIR", _os.path.join(_REPO_ROOT, "outputs", "cache"))

SEED = 20260803

# ---- 网格（E0 协议）----
R_GRID = [0.05, 0.15, 0.5]
T_GRID = [0.32, 0.64, 1.28]
