"""baselines_extra.py — v3 §11.4 补充基线（conformal 族之外的"头号杀手防身"项）
  DeepEnsemble      : ≥3 成员独立种子训练, 平均概率 + 分歧作为弃权分数
  SpectralUncertainty: 特征协方差谱熵（音频 OOD SOTA 线）作为弃权分数
  SohnNPLRT         : VAD NP 似然比检验（Sohn 1999 谱系）——帧级二元语音检测基线
  (B7/B8/B9 训练变体在 train_variants.py)
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


class DeepEnsemble:
    """≥3 成员（同一架构不同种子）: 平均概率决策 + 成员分歧作弃权分数。
    成员由外部训练（train_variants 或 run_main 多 seed）。"""

    def __init__(self, models: list, device="cpu"):
        if len(models) < 3:
            raise ValueError("DeepEnsemble 需要 ≥3 成员（预注册）")
        self.models = [m.to(device).eval() for m in models]
        self.device = device

    @torch.no_grad()
    def predict(self, log_mel):
        probs, snrs = [], []
        for m in self.models:
            out = m(log_mel.to(self.device))
            probs.append(out["event_probs"].cpu())
            snrs.append(out["snr_db"].cpu())
        P = torch.stack(probs)
        p_mean = P.mean(0)
        disagreement = P.var(0).sum(-1)            # 成员分歧（认知不确定性代理）
        return {"event_probs": p_mean, "snr_db": torch.stack(snrs).mean(0),
                "disagreement": disagreement}


class SpectralUncertainty:
    """特征协方差谱熵作为弃权分数（低熵=结构明确, 高熵=OOD/不可信）。
    对骨干特征 [n, d] 计算。分数 = −谱熵（高=可信, 与决策分数同向）。"""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, log_mel):
        out = self.model(log_mel.to(self.device))
        feats = out["features"].cpu().numpy()      # [B, 64]
        scores = []
        for f in feats:
            f = f - f.mean()
            cov = np.outer(f, f) + 1e-9 * np.eye(len(f))
            ev = np.linalg.eigvalsh(cov)
            ev = ev[ev > 1e-12]
            p = ev / ev.sum()
            H = -float((p * np.log(p)).sum())
            scores.append(-H)                      # 高=可信
        return np.array(scores)


class SohnNPLRT:
    """VAD NP-LRT（Sohn 1999 谱系）: 每帧能量统计量的似然比检验。
    帧级二元语音/事件检测——作为"经典决策基线"进入 detector-tier 对照。
    简化: 假设事件带内噪声白化（与 E0 同口径）, 帧长 20ms, 似然比 = 能量比。"""

    def __init__(self, band=C.EVENT_BAND, fs=C.SAMPLE_RATE, frame_ms: float = 20.0,
                 p_fa: float = C.ALPHA):
        self.band, self.fs = band, fs
        self.frame = int(fs * frame_ms / 1000)
        self.p_fa = p_fa

    def frame_snr_db(self, x: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
        from .baselines import estimate_noise_floor
        f_lo, f_hi = self.band
        from .wsosim import _band
        pn = estimate_noise_floor(x, w_ref, self.fs, self.band)
        # 帧级噪声功率（整窗 pn 按帧时长折算——帧 r 与整窗 r 口径对齐）
        pn_frame = max(pn * self.frame / max(x.shape[0], 1), 1e-15)
        xb = _band(x, self.fs, f_lo, f_hi)
        n_frames = max(xb.shape[0] // self.frame, 1)
        e_per = np.array([float(np.sum(xb[i * self.frame:(i + 1) * self.frame] ** 2))
                          for i in range(n_frames)])
        r = np.maximum(e_per / pn_frame - 1.0, 1e-9)
        return 10.0 * np.log10(r)

    def decide(self, x: np.ndarray, w_ref: np.ndarray, vote_win: int = 11,
               vote_k: int = 5) -> bool:
        """帧级 NP-LRT + 多数表决（VAD hangover 惯例）。
        孤立帧超阈不可靠——AM 风噪包络峰帧能量 +5~8dB（WSO-Sim 实证）会让 OR 聚合虚警;
        11 帧（220ms）窗内 ≥5 帧超阈才决策（覆盖 5Hz AM 半周期 100ms）。"""
        from .af_rule import r_min_theory
        f_lo, f_hi = self.band
        n_frame = max(int(round(2 * (f_hi - f_lo) * self.frame / self.fs)), 2)
        r_min_frame = r_min_theory(self.p_fa, n_frame)
        r_db = self.frame_snr_db(x, w_ref)
        hit = (10.0 ** (r_db / 10.0) >= r_min_frame).astype(int)
        if len(hit) < vote_win:
            return int(hit.sum()) >= min(vote_k, len(hit))
        conv = np.convolve(hit, np.ones(vote_win, dtype=int), "valid")
        return bool((conv >= vote_k).any())
