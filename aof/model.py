"""
model.py — SONTRA-A 的 PyTorch 移植（Swift/MLX 架构忠实复刻）

层映射（Swift L1-L7 → 本文件）:
  SafetyAcousticSeparationLayer  → SeparationLayer   (Swift L1, ~111K)
  AcousticRepresentationEncoder  → EncoderLayer      (Swift L2, ~803K)
  AcousticTemporalDynamicsLayer  → TemporalLayer      (Swift L3, ~264K)
  AcousticBeliefStateLayer       → BeliefLayer        (Swift L5, ~90K)
  AudibilityHead                 → A-Head（论文新增, 逐类 SNR̂ [B,10]）

SONTRA-A = Encoder + Separation + Temporal + Belief + A-Head（≈1.28M）
参数断言: 全栈(含 L4 fusion/L6 risk, 仅用于对照) ≈ 1,321,537 [COMPUTED 于 Swift 侧]
输入: log-mel [B, 1, N_MELS=24, 128]（16kHz, 1.28s 窗）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# ---------- 基础块 ----------

class ChannelLN(nn.Module):
    """LayerNorm over channels（对齐 MLX LayerNorm(dimensions: C) on [B,H,W,C]）。"""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):                      # [B,C,H,W]
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1)
        self.norm = ChannelLN(out_c)

    def forward(self, x):
        return self.norm(self.pw(self.dw(x)))


class MBConv(nn.Module):
    """MobileNetV2 块（Swift: expansion=12, 无 bias 的 block convs, residual 当 in==out 且 stride==1）。"""

    def __init__(self, in_c: int, out_c: int, stride=(2, 2), expansion: int = 12):
        super().__init__()
        exp = in_c * expansion
        self.expand = nn.Conv2d(in_c, exp, 1, bias=False)
        self.dw = nn.Conv2d(exp, exp, 3, stride=stride, padding=1, groups=exp, bias=False)
        self.proj = nn.Conv2d(exp, out_c, 1, bias=False)
        self.norm = ChannelLN(out_c)
        self.uses_residual = (in_c == out_c and tuple(stride) == (1, 1))

    def forward(self, x):
        x_in = x
        x = F.gelu(self.expand(x))
        x = F.gelu(self.dw(x))
        x = self.norm(self.proj(x))
        return F.gelu(x_in + x if self.uses_residual else x)


# ---------- 层 ----------

class SeparationLayer(nn.Module):
    """SafetyAcousticSeparationLayer（Swift L1, ~111K）：nuisance/relevance 门控增强。
    侧输入 signal_features [B,10] 与 quality_features [B,8] 在论文管线中缺省为 0。"""

    def __init__(self):
        super().__init__()
        self.block_1 = DepthwiseSeparableConv2D(1, 24)
        self.block_2 = DepthwiseSeparableConv2D(24, 24)
        self.block_3 = DepthwiseSeparableConv2D(24, 24)
        self.nuisance_hidden = nn.Linear(24, 1024)
        self.nuisance_output = nn.Linear(1024, 32)
        self.nuisance_mask = nn.Linear(32, 24)
        self.relevance_hidden = nn.Linear(24, 1024)
        self.relevance_output = nn.Linear(1024, 24)
        self.quality_gate = nn.Linear(8, 24)
        self.signal_gate = nn.Linear(10, 24)

    def forward(self, log_mel, signal_features=None, quality_features=None):
        B = log_mel.shape[0]
        if signal_features is None:
            signal_features = torch.zeros(B, 10, device=log_mel.device)
        if quality_features is None:
            quality_features = torch.zeros(B, 8, device=log_mel.device)
        residual_evidence = self.block_1(log_mel)                 # [B,24,H,W]
        separated = self.block_3(self.block_2(residual_evidence))
        pooled = separated.mean(dim=(2, 3))                       # [B,24]
        nuisance_state = self.nuisance_output(F.gelu(self.nuisance_hidden(pooled)))
        nuisance_gate = torch.sigmoid(self.nuisance_mask(nuisance_state))
        nuisance_gate = nuisance_gate.unsqueeze(-1).unsqueeze(-1)  # [B,24,1,1]
        relevance_gate = torch.sigmoid(
            self.relevance_output(F.gelu(self.relevance_hidden(pooled)))
            + self.quality_gate(quality_features)
            + self.signal_gate(signal_features)
        ).unsqueeze(-1).unsqueeze(-1)
        enhanced = separated * (1 - nuisance_gate) * relevance_gate + residual_evidence
        return enhanced, nuisance_state


class EncoderLayer(nn.Module):
    """AcousticRepresentationEncoder（Swift L2, ~803K）：stem + 5×MBConv(exp=12) + 频域池化 + 序列适配。"""

    def __init__(self, mel_bands: int = C.N_MELS):
        super().__init__()
        self.stem = nn.Conv2d(mel_bands, 32, 3, stride=2, padding=1)
        self.stem_normalization = ChannelLN(32)
        self.blocks = nn.ModuleList([
            MBConv(32, 48, (2, 2)),
            MBConv(48, 64, (2, 1)),
            MBConv(64, 96, (2, 1)),
            MBConv(96, 96, (1, 1)),
            MBConv(96, 96, (1, 1)),
        ])
        self.sequence_hidden = nn.Linear(96, 512)
        self.sequence_projection = nn.Linear(512, 96)
        self.sequence_normalization = nn.LayerNorm(96)

    def forward(self, enhanced_map):
        """enhanced_map [B,24,H,W] → [B,T',96]（T'=W/8）。"""
        x = F.gelu(self.stem_normalization(self.stem(enhanced_map)))
        for b in self.blocks:
            x = b(x)
        freq_pooled = x.mean(dim=2)                       # [B,96,W']
        ft = freq_pooled.permute(0, 2, 1)                 # [B,W',96]
        adapted = self.sequence_projection(F.gelu(self.sequence_hidden(ft)))
        return self.sequence_normalization(freq_pooled.permute(0, 2, 1) + adapted)  # [B,T',96]


class CausalTemporalBlock(nn.Module):
    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv1d(c, c, 3, groups=c, dilation=dilation, bias=True)  # 因果: 左填充
        self.gate_projection = nn.Linear(c, 2 * c)
        self.pointwise_projection = nn.Linear(c, c)
        self.skip_projection = nn.Linear(c, c)
        self.normalization = nn.LayerNorm(c)

    def forward(self, x):
        """x [B,T,C] → (residual, skip)。"""
        xp = F.pad(x.permute(0, 2, 1), (2 * self.dw.dilation[0], 0)).permute(0, 2, 1)  # 左填 2d
        filt = self.dw(xp.permute(0, 2, 1)).permute(0, 2, 1)                          # [B,T,C]
        a, b = self.gate_projection(filt).chunk(2, dim=-1)
        activated = torch.tanh(a) * torch.sigmoid(b)
        projected = self.pointwise_projection(activated)
        return self.normalization(x + projected), self.skip_projection(projected)


class TemporalLayer(nn.Module):
    """AcousticTemporalDynamicsLayer（Swift L3, ~264K）：7 因果块，out=LN(residual+mean(skips))。"""

    def __init__(self, c: int = C.ENC_CHANNELS, dilations=None):
        super().__init__()
        dilations = dilations or C.TEMPORAL_DILATIONS
        self.blocks = nn.ModuleList([CausalTemporalBlock(c, d) for d in dilations])
        self.output_normalization = nn.LayerNorm(c)

    def forward(self, sequence):
        residual = sequence
        skip_sum = None
        for b in self.blocks:
            residual, skip = b(residual)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        return self.output_normalization(residual + skip_sum / len(self.blocks))


class BeliefLayer(nn.Module):
    """AcousticBeliefStateLayer（Swift L5, ~90K）：共享特征 + 多头。论文用 event/open_set/observability。"""

    def __init__(self, in_c: int = C.ENC_CHANNELS, n_classes: int = C.N_CLASSES):
        super().__init__()
        self.shared_projection = nn.Linear(in_c, 64)
        self.refinement_hidden = nn.Linear(64, 640)
        self.refinement_projection = nn.Linear(640, 64)
        self.normalization = nn.LayerNorm(64)
        self.event_head = nn.Linear(64, n_classes)
        self.dynamic_head = nn.Linear(64, 7)
        self.traffic_head = nn.Linear(64, 6)
        self.pass_by_head = nn.Linear(64, 2)
        self.observability_head = nn.Linear(64, 1)
        self.open_set_head = nn.Linear(64, 1)

    def forward(self, x):
        """x [B,T,C] → 各头在时间维求均值（论文批推理口径）。"""
        projected = self.shared_projection(x)                    # [B,T,64]
        refined = self.refinement_projection(F.gelu(self.refinement_hidden(projected)))
        shared = self.normalization(projected + refined)
        pooled = shared.mean(dim=1)                              # [B,64]
        return {
            "event_logits": self.event_head(pooled),
            "event_probs": torch.sigmoid(self.event_head(pooled)),
            "open_set": torch.sigmoid(self.open_set_head(pooled)),
            "observability": torch.sigmoid(self.observability_head(pooled)),
            "features": pooled,
        }


class AudibilityHead(nn.Module):
    """A-Head（论文新增）: 逐类 SNR̂ [B,10]（dB, r=E_s/P_n 能量比口径）。
    输入 = belief 共享特征 [B,64] + encoder 池化特征 [B,96]（L1/L2 特征接入）。"""

    def __init__(self, belief_dim: int = 64, enc_dim: int = C.ENC_CHANNELS, n_classes: int = C.N_CLASSES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(belief_dim + enc_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, belief_features, enc_pooled):
        return self.mlp(torch.cat([belief_features, enc_pooled], dim=-1)).clamp(-60.0, 30.0)


class SONTRA_A(nn.Module):
    """SONTRA-A = Separation + Encoder + Temporal + Belief + A-Head（≈1.28M, 论文主模型）。"""

    def __init__(self, mel_bands: int = C.N_MELS, n_classes: int = C.N_CLASSES):
        super().__init__()
        self.n_classes = n_classes
        self.separation = SeparationLayer()
        self.encoder = EncoderLayer(mel_bands)
        self.temporal = TemporalLayer()
        self.belief = BeliefLayer(n_classes=n_classes)
        self.ahead = AudibilityHead(n_classes=n_classes)

    def forward(self, log_mel, signal_features=None, quality_features=None):
        """log_mel [B,1,24,128] → (event_probs[B,10], snr_db[B,10], open_set[B,1], observability[B,1],
        features[B,64], enc_pooled[B,96])"""
        enhanced, _ = self.separation(log_mel, signal_features, quality_features)
        seq = self.encoder(enhanced)                    # [B,T',96]
        seq = self.temporal(seq)
        belief = self.belief(seq)
        enc_pooled = seq.mean(dim=1)                    # [B,96]
        snr_db = self.ahead(belief["features"], enc_pooled)
        return {
            "event_probs": belief["event_probs"],
            "event_logits": belief["event_logits"],
            "snr_db": snr_db,
            "open_set": belief["open_set"],
            "observability": belief["observability"],
            "features": belief["features"],
            "enc_pooled": enc_pooled,
        }

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
