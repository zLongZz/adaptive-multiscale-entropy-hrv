"""Provides ectopy detector functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import Tuple, List
from modules.core.logger import log_debug, log_warning

class EctopyDetector:

    def __init__(self, iqr_factor: float=1.5, window_size: int=50):
        self.iqr_factor = iqr_factor
        self.window_size = window_size
        log_debug(f'[Layer 5] EctopyDetector initialized: IQR_factor={iqr_factor}, window={window_size}')

    def detect_ectopy(self, rr: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(rr) < 3:
            return (np.zeros(len(rr), dtype=bool), 0.0)
        ectopic_mask = np.zeros(len(rr), dtype=bool)
        for i in range(len(rr)):
            start = max(0, i - self.window_size // 2)
            end = min(len(rr), i + self.window_size // 2 + 1)
            local_rr = rr[start:end]
            if len(local_rr) < 3:
                continue
            q1 = np.percentile(local_rr, 25)
            q3 = np.percentile(local_rr, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_factor * iqr
            upper_bound = q3 + self.iqr_factor * iqr
            if rr[i] < lower_bound or rr[i] > upper_bound:
                ectopic_mask[i] = True
        ectopy_rate = np.sum(ectopic_mask) / len(rr)
        log_debug(f'[Layer 5] Detected {np.sum(ectopic_mask)} ectopic beats ({ectopy_rate:.2%})')
        return (ectopic_mask, ectopy_rate)

    def detect_ectopy_simple(self, rr: np.ndarray, threshold_lower: float=0.3, threshold_upper: float=2.0) -> Tuple[np.ndarray, float]:
        ectopic_mask = (rr < threshold_lower) | (rr > threshold_upper)
        ectopy_rate = np.sum(ectopic_mask) / len(rr) if len(rr) > 0 else 0.0
        return (ectopic_mask, ectopy_rate)
