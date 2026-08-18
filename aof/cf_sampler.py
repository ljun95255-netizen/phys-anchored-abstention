"""
cf_sampler.py — CF-Sampler（v4.1 §5.5）: (clean, corrupted) 对 + 外生 SNR 标签 + 掩码
每 clip: 随机腐蚀类型/SNR（确定性种子）→ r_dB 实测标签（E0 口径）
mask: 该 clip 真实标签类为 1，unknown 类恒 0
"""
import numpy as np

from . import config as C
from .data import log_mel
from .wsosim import corrupt


class CFSampler:
    def __init__(self, clips, kinds=("wind", "occlusion", "self_motion"),
                 snr_grid=(-5.0, 5.0, 15.0), seed=C.SEED, n_mels=C.N_MELS):
        self.clips = clips
        self.kinds = kinds
        self.snr_grid = snr_grid
        self.rng = np.random.default_rng(seed)
        self.n_mels = n_mels

    def _best_window(self, x, window):
        from .wsosim import _band
        best, best_e = None, 0.0
        for start in range(0, x.shape[0] - window + 1, window):
            seg = x[start:start + window]
            e = float(np.sum(_band(seg, C.SAMPLE_RATE, *C.EVENT_BAND) ** 2))
            if e > best_e:
                best, best_e = seg, e
        return best if best_e > 1e-9 * window else None

    def make_pairs(self, n_pairs: int, window: int = C.WINDOW_SAMPLES):
        """→ [(x_corr[1,24,128] f32, x_clean[1,24,128] f32, y[10] f32, r_dB[10] f32, mask[10] f32)]"""
        pairs = []
        i = 0
        while len(pairs) < n_pairs:
            x, target, _ = self.clips[i % len(self.clips)]
            i += 1
            if x.shape[0] < window:
                continue
            # 选事件带能量最大的 1.28s 窗（事件定位朴素启发; τ=T 协议, 静音窗剔除）
            seg = self._best_window(x, window)
            if seg is None:
                continue
            kind = self.kinds[self.rng.integers(len(self.kinds))]
            snr = float(self.snr_grid[self.rng.integers(len(self.snr_grid))])
            seed = int(self.rng.integers(1 << 31))
            xc, r_db, meta = corrupt(seg, kind, snr, seed)
            if xc is None:
                continue
            r_db_vec = np.full(C.N_CLASSES, -60.0, dtype=np.float32)
            mask = np.zeros(C.N_CLASSES, dtype=np.float32)
            y = np.zeros(C.N_CLASSES, dtype=np.float32)
            # ESC-50 单标签; FSD50K 多标签扩展: 传入 label 集合
            lab = target if isinstance(target, (list, tuple, set)) else {target}
            for k in lab:
                if k < C.N_CLASSES - 1:
                    mask[k] = 1.0
                    r_db_vec[k] = r_db
                    y[k] = 1.0
            pairs.append((log_mel(torch_from(xc)).squeeze(0).numpy(),
                          log_mel(torch_from(x)).squeeze(0).numpy(),
                          y, r_db_vec, mask, snr))
        return pairs


def torch_from(x: np.ndarray):
    import torch
    return torch.from_numpy(x).unsqueeze(0).float()
