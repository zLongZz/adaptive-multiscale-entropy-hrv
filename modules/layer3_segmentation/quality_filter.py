"""Provides quality filter functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import List, Dict, Any
from modules.core.logger import log_info, log_debug, log_warning

class QualityFilter:

    def __init__(self, min_beats: int=1000, rr_min: float=0.3, rr_max: float=2.0, max_outlier_ratio: float=0.2):
        self.min_beats = min_beats
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.max_outlier_ratio = max_outlier_ratio
        log_debug(f'[Layer 3] QualityFilter initialized: min_beats={min_beats}, RR_range=({rr_min},{rr_max})s, max_outlier={max_outlier_ratio}')

    def filter_windows(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(windows) == 0:
            return []
        filtered = []
        n_total = len(windows)
        n_filtered_beats = 0
        n_filtered_outliers = 0
        for win in windows:
            if win['n_beats'] < self.min_beats:
                n_filtered_beats += 1
                continue
            rr = win['rr']
            n_valid = np.sum((rr >= self.rr_min) & (rr <= self.rr_max))
            valid_ratio = n_valid / len(rr)
            if valid_ratio < 1 - self.max_outlier_ratio:
                n_filtered_outliers += 1
                log_debug(f"[Layer 3] Window filtered: lead={win.get('lead', 'N/A')}, idx={win['window_idx']}, valid_ratio={valid_ratio:.2%}")
                continue
            filtered.append(win)
        n_kept = len(filtered)
        log_info(f'[Layer 3] Quality filter: {n_total} windows → {n_kept} kept (filtered: {n_filtered_beats} beats, {n_filtered_outliers} outliers)')
        return filtered

    def check_window_quality(self, rr: np.ndarray) -> Dict[str, Any]:
        n_beats = len(rr)
        if n_beats == 0:
            return {'n_beats': 0, 'valid_ratio': 0.0, 'passed': False, 'reason': 'empty'}
        n_valid = np.sum((rr >= self.rr_min) & (rr <= self.rr_max))
        valid_ratio = n_valid / n_beats
        passed = n_beats >= self.min_beats and valid_ratio >= 1 - self.max_outlier_ratio
        reason = ''
        if n_beats < self.min_beats:
            reason = f'insufficient_beats ({n_beats}<{self.min_beats})'
        elif valid_ratio < 1 - self.max_outlier_ratio:
            reason = f'too_many_outliers ({valid_ratio:.2%}<{1 - self.max_outlier_ratio:.2%})'
        else:
            reason = 'passed'
        return {'n_beats': n_beats, 'valid_ratio': valid_ratio, 'passed': passed, 'reason': reason}

    def get_statistics(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(windows) == 0:
            return {'n_windows': 0, 'mean_beats': 0, 'median_beats': 0, 'min_beats': 0, 'max_beats': 0}
        n_beats_list = [win['n_beats'] for win in windows]
        return {'n_windows': len(windows), 'mean_beats': np.mean(n_beats_list), 'median_beats': np.median(n_beats_list), 'min_beats': np.min(n_beats_list), 'max_beats': np.max(n_beats_list)}
