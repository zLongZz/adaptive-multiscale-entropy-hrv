"""Provides sinus interpolator functions for the ECG-to-entropy pipeline."""
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple
from modules.core.logger import log_debug, log_warning

class SinusInterpolator:

    def __init__(self, max_ectopy_ratio: float=0.2, extrapolate: bool=False):
        self.max_ectopy_ratio = max_ectopy_ratio
        self.extrapolate = extrapolate
        log_debug(f'[Layer 5] SinusInterpolator initialized: max_ectopy={max_ectopy_ratio}')

    def interpolate(self, rr: np.ndarray, ectopic_mask: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        n_ectopic = np.sum(ectopic_mask)
        ectopy_rate = n_ectopic / len(rr) if len(rr) > 0 else 0.0
        if ectopy_rate > self.max_ectopy_ratio:
            log_debug(f'[Layer 5] Ectopy rate too high ({ectopy_rate:.2%} > {self.max_ectopy_ratio:.2%}), cannot interpolate')
            return (rr, False, f'ectopy_rate_too_high ({ectopy_rate:.2%})')
        if n_ectopic == 0:
            log_debug('[Layer 5] No ectopic beats detected, no interpolation needed')
            return (rr, True, 'no_ectopy')
        normal_mask = ~ectopic_mask
        n_normal = np.sum(normal_mask)
        if n_normal < 3:
            log_debug(f'[Layer 5] Too few normal beats ({n_normal} < 3), cannot interpolate')
            return (rr, False, f'insufficient_normal_beats ({n_normal})')
        try:
            normal_idx = np.where(normal_mask)[0]
            normal_rr = rr[normal_mask]
            cs = CubicSpline(normal_idx, normal_rr, extrapolate=self.extrapolate)
            ectopic_idx = np.where(ectopic_mask)[0]
            rr_interpolated = rr.copy()
            rr_interpolated[ectopic_idx] = cs(ectopic_idx)
            nan_mask = np.isnan(rr_interpolated)
            if np.any(nan_mask):
                for idx in np.where(nan_mask)[0]:
                    nearest_normal = normal_idx[np.argmin(np.abs(normal_idx - idx))]
                    rr_interpolated[idx] = rr[nearest_normal]
            rr_interpolated = np.clip(rr_interpolated, 0.3, 2.0)
            log_debug(f'[Layer 5] Interpolated {n_ectopic} ectopic beats ({ectopy_rate:.2%})')
            return (rr_interpolated, True, 'success')
        except Exception as e:
            log_warning(f'[Layer 5] Interpolation failed: {str(e)}')
            return (rr, False, f'interpolation_error: {str(e)}')

    def check_interpolation_quality(self, rr_original: np.ndarray, rr_interpolated: np.ndarray, ectopic_mask: np.ndarray) -> dict:
        ectopic_idx = np.where(ectopic_mask)[0]
        if len(ectopic_idx) == 0:
            return {'n_interpolated': 0, 'mean_change': 0.0, 'max_change': 0.0, 'rmse': 0.0}
        changes = np.abs(rr_interpolated[ectopic_idx] - rr_original[ectopic_idx])
        return {'n_interpolated': len(ectopic_idx), 'mean_change': np.mean(changes), 'max_change': np.max(changes), 'rmse': np.sqrt(np.mean(changes ** 2))}
