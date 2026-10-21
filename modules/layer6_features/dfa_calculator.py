"""Provides dfa calculator functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import Dict, Any, Tuple, List
from scipy import stats
from modules.core.logger import log_debug, log_warning

class DFACalculator:

    def __init__(self, n_min: int=8, n_max: int=64, min_rr_length: int=128):
        self.n_min = n_min
        self.n_max = n_max
        self.min_rr_length = min_rr_length
        base_sizes = [4, 6, 8, 12, 16, 24, 32, 48, 64]
        self.box_sizes = [n for n in base_sizes if n_min <= n <= n_max]
        if n_min not in self.box_sizes and n_min <= n_max:
            self.box_sizes.insert(0, n_min)
            self.box_sizes.sort()
        log_debug(f'[Layer 6] DFACalculator initialized (OPTIMIZED): n_range=({n_min},{n_max}), box_sizes={self.box_sizes}, min_rr_length={min_rr_length}')

    def compute(self, rr: np.ndarray) -> Dict[str, Any]:
        if np.any(np.isnan(rr)):
            n_nan = np.sum(np.isnan(rr))
            log_warning(f'[Layer 6] DFA: RR contains {n_nan} NaN values, filtering...')
            rr_clean = rr[~np.isnan(rr)]
            if len(rr_clean) < self.min_rr_length:
                log_warning(f'[Layer 6] DFA: After NaN removal, RR too short ({len(rr_clean)})')
                return self._empty_dfa_result('insufficient_length_after_nan_removal')
            rr = rr_clean
        return self._compute_internal(rr)

    def _compute_internal(self, rr: np.ndarray) -> Dict[str, Any]:
        if len(rr) < self.min_rr_length:
            log_warning(f'[Layer 6] RR too short for DFA ({len(rr)} < {self.min_rr_length})')
            return self._empty_dfa_result('insufficient_length')
        try:
            rr_mean = np.mean(rr)
            y = np.cumsum(rr - rr_mean)
            alpha1 = self._fit_range(y, [4, 6, 8, 12, 16], len(rr))
            alpha2 = self._fit_range(y, [16, 24, 32, 48, 64], len(rr))
            legacy = self._fit_range(y, [8, 12, 16, 24, 32, 48, 64], len(rr))
            if alpha2 is None:
                return self._empty_dfa_result('insufficient_alpha2_boxes')
            alpha2_slope, alpha2_intercept, alpha2_r_squared = alpha2
            if alpha1 is None:
                alpha1_slope = alpha1_intercept = alpha1_r_squared = np.nan
            else:
                alpha1_slope, alpha1_intercept, alpha1_r_squared = alpha1
            if legacy is None:
                legacy_slope = legacy_intercept = legacy_r_squared = np.nan
            else:
                legacy_slope, legacy_intercept, legacy_r_squared = legacy
            if alpha2_r_squared < 0.95:
                quality = 'warning_low_r2'
            elif alpha2_slope < 0 or alpha2_slope > 2:
                quality = 'warning_out_of_range'
            else:
                quality = 'good'
            log_debug(f'[Layer 6] DFA: alpha1={alpha1_slope:.3f}, alpha2={alpha2_slope:.3f}, R2={alpha2_r_squared:.3f}, quality={quality}')
            return {'DFA_alpha1': alpha1_slope, 'DFA_alpha1_intercept': alpha1_intercept, 'DFA_alpha1_r_squared': alpha1_r_squared, 'DFA_alpha2': alpha2_slope, 'DFA_intercept': alpha2_intercept, 'DFA_r_squared': alpha2_r_squared, 'DFA_slope_8_64': legacy_slope, 'DFA_slope_8_64_intercept': legacy_intercept, 'DFA_slope_8_64_r_squared': legacy_r_squared, 'DFA_quality': quality}
        except Exception as e:
            log_warning(f'[Layer 6] DFA computation failed: {str(e)}')
            return self._empty_dfa_result(f'error: {str(e)}')

    def _fit_range(self, y: np.ndarray, requested_boxes: List[int], rr_length: int) -> Tuple[float, float, float] | None:
        max_box = min(max(requested_boxes), rr_length // 4)
        dynamic_boxes = [n for n in requested_boxes if n <= max_box]
        if len(dynamic_boxes) < 2:
            return None
        log_n = []
        log_F = []
        for n in dynamic_boxes:
            F_n = self._compute_fluctuation(y, n)
            if F_n > 0 and (not np.isnan(F_n)) and (not np.isinf(F_n)):
                log_n.append(np.log(n))
                log_F.append(np.log(F_n))
        if len(log_n) < 2:
            return None
        slope, intercept, r_value, _, _ = stats.linregress(log_n, log_F)
        return (float(slope), float(intercept), float(r_value ** 2))

    def _generate_dynamic_boxes(self, rr_length: int) -> List[int]:
        max_box = min(self.n_max, rr_length // 4)
        if max_box < self.n_min:
            return [self.n_min]
        boxes = []
        for n in self.box_sizes:
            if n <= max_box:
                boxes.append(n)
        if len(boxes) < 3:
            mid = (self.n_min + max_box) // 2
            if mid not in boxes:
                boxes.append(mid)
                boxes.sort()
        log_debug(f'[Layer 6] DFA: dynamic boxes={boxes} for rr_length={rr_length}')
        return boxes

    def _compute_fluctuation(self, y: np.ndarray, n: int) -> float:
        N = len(y)
        n_boxes = N // n
        if n_boxes == 0:
            return 0.0
        fluctuations = []
        for i in range(n_boxes):
            start = i * n
            end = (i + 1) * n
            y_box = y[start:end]
            x = np.arange(n)
            coeffs = np.polyfit(x, y_box, 1)
            y_trend = np.polyval(coeffs, x)
            residuals = y_box - y_trend
            rms = np.sqrt(np.mean(residuals ** 2))
            fluctuations.append(rms)
        F_n = np.sqrt(np.mean(np.array(fluctuations) ** 2))
        return F_n

    def _empty_dfa_result(self, reason: str) -> Dict[str, Any]:
        return {'DFA_alpha1': np.nan, 'DFA_alpha1_intercept': np.nan, 'DFA_alpha1_r_squared': np.nan, 'DFA_alpha2': np.nan, 'DFA_intercept': np.nan, 'DFA_r_squared': np.nan, 'DFA_slope_8_64': np.nan, 'DFA_slope_8_64_intercept': np.nan, 'DFA_slope_8_64_r_squared': np.nan, 'DFA_quality': reason}

    def compute_batch(self, rr_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        results = []
        for i, rr in enumerate(rr_list):
            try:
                result = self.compute(rr)
                results.append(result)
            except Exception as e:
                log_warning(f'[Layer 6] DFA batch item {i} failed: {str(e)}')
                results.append(self._empty_dfa_result(f'error: {str(e)}'))
        return results
