"""calibration.py — τ-Cal 组件（v4.1 §1 命名体系; v3 §11.4 校准域分离）
τ-Cal = 估计器偏差残差修正（isotonic / SCoRE conformal 简化）。
实现:
  IsotonicCalibrator: isotonic 回归拟合 SNR̂ 残差 r_true − r̂（vs r̂）→ 修正 r̂_corr = r̂ + f(r̂)
    ——"估计器偏差残差修正"的免分布实现（sklearn IsotonicRegression, out_of_bounds=clip）
  ScoreCalibrator   : 事件概率的 isotonic 校准（τ 的校准集选择; 与 τ 扫描配合）
NOTE: SCoRE conformal 完整版超出脚本范围; isotonic 为可辩护简化（论文须写明）。
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """SNR̂ 偏差残差修正: f: r̂ → E[r_true − r̂ | r̂]（单调, 免分布）。"""

    def __init__(self, y_min: float = -60.0, y_max: float = 30.0):
        self.y_min, self.y_max = y_min, y_max
        self._iso = None

    def fit(self, snr_hat: np.ndarray, snr_true: np.ndarray, mask: np.ndarray = None):
        m = np.ones(len(snr_hat), dtype=bool) if mask is None else mask.astype(bool)
        x = np.asarray(snr_hat, dtype=float)[m]
        y = (np.asarray(snr_true, dtype=float) - x)[m]     # 残差
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(x, y)
        return self

    def correct(self, snr_hat: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("fit first")
        x = np.asarray(snr_hat, dtype=float)
        resid = self._iso.predict(x)
        return np.clip(x + resid, self.y_min, self.y_max)

    def mae_before_after(self, snr_hat, snr_true, mask=None) -> dict:
        m = np.ones(len(snr_hat), dtype=bool) if mask is None else mask.astype(bool)
        before = float(np.abs(snr_hat[m] - snr_true[m]).mean())
        after = float(np.abs(self.correct(snr_hat)[m] - snr_true[m]).mean())
        return {"mae_before": before, "mae_after": after, "improvement": before - after}


class ScoreCalibrator:
    """事件概率 isotonic 校准（τ 的校准集选择——v3 校准域分离: 校准集≠评估域）。"""

    def __init__(self):
        self._iso = None

    def fit(self, prob: np.ndarray, y: np.ndarray):
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(np.clip(prob, 0, 1), y)
        return self

    def calibrate(self, prob: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("fit first")
        return self._iso.predict(np.clip(prob, 0, 1))
