"""Provides feature processor functions for the ECG-to-entropy pipeline."""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from multiprocessing import Pool
from modules.core.logger import log_info, log_debug, log_warning
from .rcmse_numba import compute_rcmse_fast
from .dfa_calculator import DFACalculator
from .spatial_complexity import SpatialComplexityCalculator, compute_spatial_complexity_simple

def _numba_worker_init():
    import numba
    numba.set_num_threads(1)

class FeatureProcessor:

    def __init__(self, rcmse_scales: range=range(1, 21), rcmse_m: int=2, rcmse_r_factor: float=0.15, rcmse_min_length: int=30, dfa_n_min: int=8, dfa_n_max: int=64, use_parallel: bool=True, n_processes: int=19):
        self.rcmse_scales = rcmse_scales
        self.rcmse_m = rcmse_m
        self.rcmse_r_factor = rcmse_r_factor
        self.rcmse_min_length = rcmse_min_length
        self.use_parallel = use_parallel
        self.n_processes = n_processes
        self.dfa_calculator = DFACalculator(n_min=dfa_n_min, n_max=dfa_n_max)
        self.spatial_calculator = SpatialComplexityCalculator(n_components=12, standardize=True)
        log_info(f'[Layer 6] FeatureProcessor initialized')
        log_debug(f'[Layer 6]   RCMSE: scales={len(rcmse_scales)}, m={rcmse_m}, r_factor={rcmse_r_factor}')
        log_debug(f'[Layer 6]   DFA: n_range=({dfa_n_min},{dfa_n_max})')
        log_debug(f'[Layer 6]   Parallel: {use_parallel}, n_processes={n_processes}')

    def process_window(self, window: Dict[str, Any]) -> Dict[str, Any]:
        rr_final = window.get('rr_final', window.get('rr'))
        if rr_final is None or len(rr_final) == 0:
            log_warning(f'[Layer 6] Window has no valid RR data')
            return self._generate_empty_features(window, 'rr_missing')
        rr_final = np.asarray(rr_final, dtype=float)
        rr_final = rr_final[np.isfinite(rr_final)]
        rr_mean = float(np.mean(rr_final)) if len(rr_final) else np.nan
        rr_std = float(np.std(rr_final)) if len(rr_final) else np.nan
        rr_cv = rr_std / rr_mean if np.isfinite(rr_mean) and rr_mean != 0 else np.nan
        rcmse_result = compute_rcmse_fast(rr_final, scales=self.rcmse_scales, m=self.rcmse_m, r_factor=self.rcmse_r_factor, min_length=self.rcmse_min_length)
        dfa_result = self.dfa_calculator.compute(rr_final)
        feature_quality = self._assess_feature_quality(rcmse_result, dfa_result)
        rcmse_quality, reject_reason = self._diagnose_quality(rr_final, rr_std, rcmse_result, dfa_result)
        features = {**window, **rcmse_result, **dfa_result, 'rr_n_input': int(len(rr_final)), 'rr_mean_final': rr_mean, 'rr_std_final': rr_std, 'rr_cv_final': rr_cv, 'rcmse_quality': rcmse_quality, 'layer6_reject_reason': reject_reason, 'feature_quality': feature_quality}
        return features

    def _diagnose_quality(self, rr_final, rr_std, rcmse_result, dfa_result):
        min_required = self.rcmse_min_length * max(self.rcmse_scales)
        reasons = []
        if len(rr_final) < min_required:
            rcmse_quality = 'insufficient_length'
            reasons.append('rr_below_rcmse_min')
        elif np.isfinite(rr_std) and rr_std < 1e-10:
            rcmse_quality = 'near_constant'
            reasons.append('rr_near_constant')
        elif not np.isfinite(rcmse_result.get('MSE_long', np.nan)):
            rcmse_quality = 'insufficient_valid_long_scales'
            reasons.append('mse_long_missing_insufficient_valid_long_scales')
        else:
            rcmse_quality = 'valid'
        dfa_alpha2 = dfa_result.get('DFA_alpha2', np.nan)
        if not np.isfinite(dfa_alpha2):
            dfa_quality = str(dfa_result.get('DFA_quality', 'unknown')).replace(' ', '_')
            reasons.append(f'dfa_alpha2_missing_{dfa_quality}')
        return (rcmse_quality, '+'.join(reasons))

    def _assess_feature_quality(self, rcmse_result: Dict, dfa_result: Dict) -> str:
        n_valid_scales = rcmse_result.get('n_valid_scales', 0)
        mse_long = rcmse_result.get('MSE_long', np.nan)
        dfa_alpha2 = dfa_result.get('DFA_alpha2', np.nan)
        dfa_quality = dfa_result.get('DFA_quality', 'unknown')
        if np.isnan(mse_long) or np.isnan(dfa_alpha2):
            return 'rejected'
        if n_valid_scales < 10:
            return 'poor'
        if n_valid_scales < 15 or 'warning' in dfa_quality:
            return 'warning'
        return 'good'

    def _generate_empty_features(self, window: Dict[str, Any], reason: str) -> Dict[str, Any]:
        features = {**window}
        for tau in self.rcmse_scales:
            features[f'MSE_{tau}'] = np.nan
        features.update({'MSE_short': np.nan, 'MSE_long': np.nan, 'CXI': np.nan, 'CXI_diff': np.nan, 'CXI_ratio': np.nan, 'n_valid_scales': 0, 'n_valid_long_scales': 0, 'n_valid_short_scales': 0, 'valid_scales': [], 'DFA_alpha1': np.nan, 'DFA_alpha1_intercept': np.nan, 'DFA_alpha1_r_squared': np.nan, 'DFA_alpha2': np.nan, 'DFA_intercept': np.nan, 'DFA_r_squared': np.nan, 'DFA_slope_8_64': np.nan, 'DFA_slope_8_64_intercept': np.nan, 'DFA_slope_8_64_r_squared': np.nan, 'DFA_quality': 'error', 'rr_n_input': 0, 'rr_mean_final': np.nan, 'rr_std_final': np.nan, 'rr_cv_final': np.nan, 'rcmse_quality': reason, 'layer6_reject_reason': reason, 'feature_quality': 'rejected'})
        return features

    def process_batch(self, windows: List[Dict[str, Any]], batch_size: int=100) -> pd.DataFrame:
        n_windows = len(windows)
        log_info(f'[Layer 6] Processing {n_windows} windows (batch_size={batch_size})')
        all_features = []
        n_batches = (n_windows + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_windows)
            batch_windows = windows[start_idx:end_idx]
            log_info(f'[Layer 6] Processing batch {batch_idx + 1}/{n_batches} ({len(batch_windows)} windows)')
            if self.use_parallel and len(batch_windows) > 1:
                with Pool(self.n_processes, initializer=_numba_worker_init) as pool:
                    batch_features = pool.map(self.process_window, batch_windows)
            else:
                batch_features = [self.process_window(w) for w in batch_windows]
            all_features.extend(batch_features)
            n_processed = len(all_features)
            n_good = sum((1 for f in all_features if f.get('feature_quality') == 'good'))
            log_info(f'[Layer 6] Progress: {n_processed}/{n_windows} windows, {n_good} good quality ({n_good / n_processed:.1%})')
        df = pd.DataFrame(all_features)
        self._report_statistics(df)
        return df

    def _report_statistics(self, df: pd.DataFrame):
        n_total = len(df)
        quality_counts = df['feature_quality'].value_counts()
        log_info(f'[Layer 6] Feature quality distribution:')
        for quality, count in quality_counts.items():
            log_info(f'[Layer 6]   {quality}: {count} ({count / n_total:.1%})')
        mse_long_valid = df['MSE_long'].dropna()
        if len(mse_long_valid) > 0:
            log_info(f'[Layer 6] MSE_long statistics:')
            log_info(f'[Layer 6]   Mean: {mse_long_valid.mean():.3f}')
            log_info(f'[Layer 6]   Median: {mse_long_valid.median():.3f}')
            log_info(f'[Layer 6]   Range: [{mse_long_valid.min():.3f}, {mse_long_valid.max():.3f}]')
        cxi_valid = df['CXI'].dropna()
        if len(cxi_valid) > 0:
            n_positive = np.sum(cxi_valid > 0)
            log_info(f'[Layer 6] CXI statistics:')
            log_info(f'[Layer 6]   Mean: {cxi_valid.mean():.3f}')
            log_info(f'[Layer 6]   Positive CXI: {n_positive}/{len(cxi_valid)} ({n_positive / len(cxi_valid):.1%})')
        dfa_valid = df['DFA_alpha2'].dropna()
        if len(dfa_valid) > 0:
            log_info(f'[Layer 6] DFA_alpha2 statistics:')
            log_info(f'[Layer 6]   Mean: {dfa_valid.mean():.3f}')
            log_info(f'[Layer 6]   Median: {dfa_valid.median():.3f}')

    def save_to_hdf5(self, df: pd.DataFrame, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_hdf(str(output_path), key='features', mode='w', complevel=9)
        log_info(f'[Layer 6] Features saved to HDF5: {output_path}')
        log_info(f'[Layer 6]   Shape: {df.shape}')
        log_info(f'[Layer 6]   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB')

    def save_to_pickle(self, df: pd.DataFrame, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(str(output_path))
        log_info(f'[Layer 6] Features saved to Pickle: {output_path}')
        log_info(f'[Layer 6]   Shape: {df.shape}')
