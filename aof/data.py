"""
data.py — 数据管线: WAV → log-mel（torch 原生, 无额外依赖）+ ESC-50/FSD50K 加载 + npy 缓存
契约: [B, 1, N_MELS=24, 128] @16kHz, 1.28s 窗（Swift 模型契约; 论文 §6.2 的 64 写入时统一为 24）
"""
import io
import os

import numpy as np
import scipy.io.wavfile as wavf
import scipy.signal as sig
import torch
import torch.nn.functional as F

from . import config as C


def _mel_filterbank(n_mels: int, fmin: float, fmax: float, fs: int, n_fft: int) -> np.ndarray:
    """标准三角 mel 滤波器组 [n_mels, n_fft//2+1]（librosa 口径）。"""
    def hz2mel(h):
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / fs).astype(int).clip(0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = (freqs[lo:mid] - freqs[lo]) / max(freqs[mid] - freqs[lo], 1e-9)
        if hi > mid:
            fb[i, mid:hi] = (freqs[hi] - freqs[mid:hi]) / max(freqs[hi] - freqs[mid], 1e-9)
    return fb


_MEL_FB = None


def log_mel(x: torch.Tensor, n_mels: int = C.N_MELS, fs: int = C.SAMPLE_RATE,
            n_fft: int = C.N_FFT, hop: int = C.HOP) -> torch.Tensor:
    """x [B, N] float32 → log-mel [B, 1, n_mels, n_frames]。n_frames 截断/填充到 128。
    center=False + 手动镜像 padding（规避 torch 2.12 stft 弃用警告）。"""
    global _MEL_FB
    if _MEL_FB is None or _MEL_FB.shape[0] != n_mels:
        fb = _mel_filterbank(n_mels, C.MEL_FMIN, C.MEL_FMAX, fs, n_fft)
        _MEL_FB = torch.tensor(fb, dtype=torch.float32)
    x = F.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft).to(x.device),
                      return_complex=True, center=False)
    power = spec.abs().pow(2)                                  # [B, n_fft//2+1, T]
    mel = torch.einsum("m f, b f t -> b m t", _MEL_FB.to(x.device), power)
    out = torch.log(mel.clamp_min(1e-10))
    target = C.N_FRAMES
    if out.shape[-1] > target:
        out = out[..., :target]
    elif out.shape[-1] < target:
        out = F.pad(out, (0, target - out.shape[-1]))
    return out.unsqueeze(1)                                    # [B,1,24,128]


def read_wav_bytes(b: bytes, fs: int = C.SAMPLE_RATE) -> np.ndarray:
    sr, arr = wavf.read(io.BytesIO(b))
    x = np.asarray(arr, dtype=np.float64)
    if sr != fs:
        x = sig.resample_poly(x, fs, sr)
    return x.astype(np.float32)


def load_esc50(dir_path: str = C.ESC50_DIR, classes=None, max_clips: int = 2000):
    """加载 ashraq/esc50 parquet（WAV 字节）→ [(x float32 16k, target int, filename)]。"""
    import pyarrow.parquet as pq
    out = []
    for fn in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, fn)
        if os.path.getsize(p) < 1_000_000:
            continue
        try:
            df = pq.read_table(p).to_pandas()
        except Exception:
            continue
        for _, row in df.iterrows():
            if classes is not None and row.get("target") not in classes:
                continue
            try:
                x = read_wav_bytes(row["audio"]["bytes"])
            except Exception:
                continue
            out.append((x, int(row["target"]), str(row["filename"])))
            if len(out) >= max_clips:
                return out
    return out


def cache_log_mel(x: np.ndarray, cache_dir: str = C.CACHE_DIR, key: str = None):
    """npy 缓存（mel 计算 ~10ms/clip, 缓存后训练提速）。"""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.npy")
    if os.path.exists(path):
        return torch.from_numpy(np.load(path))
    m = log_mel(torch.from_numpy(x).unsqueeze(0))
    np.save(path, m.numpy())
    return m


class WindowSampler:
    """把长 clip 切成 1.28s 窗（非重叠），带标签对齐。"""

    def __init__(self, clips, window: int = C.WINDOW_SAMPLES):
        self.clips = clips
        self.window = window

    def windows(self):
        for x, target, fname in self.clips:
            n = x.shape[0]
            for start in range(0, n - self.window + 1, self.window):
                yield x[start:start + self.window], target, fname
