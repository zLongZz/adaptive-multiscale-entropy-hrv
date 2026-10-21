"""Provides afib filter functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import Tuple
from modules.core.logger import log_debug, log_warning

class AFibFilter:

    def __init__(self, extreme_lower: float=0.2, extreme_upper: float=2.5, min_remaining_ratio: float=0.7):
        self.extreme_lower = extreme_lower
        self.extreme_upper = extreme_upper
        self.min_remaining_ratio = min_remaining_ratio
        log_debug(f'[Layer 5] AFibFilter initialized: range=({extreme_lower}, {extreme_upper})s, min_remaining={min_remaining_ratio}')

    def filter_artifacts(self, rr: np.ndarray, rr_ts: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, bool, str]:
        if len(rr) == 0:
            return (rr, rr_ts if rr_ts is not None else np.array([]), True, 'empty')
        valid_mask = (rr >= self.extreme_lower) & (rr <= self.extreme_upper)
        n_removed = np.sum(~valid_mask)
        remaining_ratio = np.sum(valid_mask) / len(rr)
        rr_filtered = rr[valid_mask]
        if rr_ts is not None:
            rr_ts_filtered = rr_ts[valid_mask]
        else:
            rr_ts_filtered = np.array([])
        if remaining_ratio < self.min_remaining_ratio:
            log_warning(f'[Layer 5] Too many artifacts removed ({remaining_ratio:.2%} < {self.min_remaining_ratio:.2%}), window marked as unacceptable')
            return (rr_filtered, rr_ts_filtered, False, f'insufficient_remaining ({remaining_ratio:.2%})')
        if n_removed > 0:
            log_debug(f'[Layer 5] Removed {n_removed} artifacts ({1 - remaining_ratio:.2%})')
        return (rr_filtered, rr_ts_filtered, True, 'success')

    def get_filter_statistics(self, rr_original: np.ndarray, rr_filtered: np.ndarray) -> dict:
        n_original = len(rr_original)
        n_filtered = len(rr_filtered)
        n_removed = n_original - n_filtered
        return {'n_original': n_original, 'n_filtered': n_filtered, 'n_removed': n_removed, 'removal_ratio': n_removed / n_original if n_original > 0 else 0.0, 'remaining_ratio': n_filtered / n_original if n_original > 0 else 0.0}
