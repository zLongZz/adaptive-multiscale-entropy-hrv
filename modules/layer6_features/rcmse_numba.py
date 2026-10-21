"""Provides rcmse numba functions for the ECG-to-entropy pipeline."""
import numpy as np
import numba
from numba import prange
from typing import Dict, List, Any, Tuple
from modules.core.logger import log_debug, log_warning
MIN_VALID_LONG_SCALES: int = 3
MIN_VALID_SHORT_SCALES: int = 2
CXI_RATIO_DENOMINATOR_MIN: float = 1e-06

def _compute_cxi_ratio(mse_short: float, mse_long: float) -> float:
    if not np.isfinite(mse_short) or not np.isfinite(mse_long):
        return np.nan
    if mse_long <= CXI_RATIO_DENOMINATOR_MIN:
        return np.nan
    return float(mse_short / mse_long)

@numba.njit(cache=True, fastmath=True)
def _coarse_grain_offset_numba(data: np.ndarray, tau: int, offset: int) -> np.ndarray:
    n = len(data) - offset
    n_cg = n // tau
    coarse_grained = np.zeros(n_cg)
    for j in range(n_cg):
        start = j * tau + offset
        end = (j + 1) * tau + offset
        coarse_grained[j] = np.mean(data[start:end])
    return coarse_grained

@numba.njit(cache=True, parallel=True, fastmath=True)
def _count_matches_numba(data: np.ndarray, m: int, r: float) -> Tuple[int, int]:
    N = len(data)
    if N < m + 2:
        return (0, 0)
    A = 0
    B = 0
    for i in prange(N - m):
        for j in range(i + 1, N - m):
            dist = 0.0
            for k in range(m):
                d = abs(data[i + k] - data[j + k])
                if d > dist:
                    dist = d
            if dist < r:
                A += 1
    for i in prange(N - m):
        for j in range(i + 1, N - m):
            dist = 0.0
            for k in range(m + 1):
                d = abs(data[i + k] - data[j + k])
                if d > dist:
                    dist = d
            if dist < r:
                B += 1
    return (A, B)

def compute_rcmse_fast(rr: np.ndarray, scales: range=range(1, 21), m: int=2, r_factor: float=0.15, min_length: int=30) -> Dict[str, Any]:
    rr = np.asarray(rr, dtype=np.float64)
    if np.any(np.isnan(rr)):
        n_nan = int(np.sum(np.isnan(rr)))
        log_warning(f'[Layer 6] RR contains {n_nan} NaN values, filtering...')
        rr = rr[~np.isnan(rr)]
    max_scale = max(scales)
    min_rr_required = min_length * max_scale
    if len(rr) < min_rr_required:
        log_warning(f'[Layer 6] RR too short for full RCMSE: N={len(rr)} < {min_rr_required} (= min_length × max_scale = {min_length} × {max_scale})')
        return _empty_rcmse_result(scales)
    mse_dict: Dict[str, float] = {}
    valid_scales: List[int] = []
    for tau in scales:
        cg_0 = _coarse_grain_offset_numba(rr, tau, 0)
        if len(cg_0) < min_length:
            mse_dict[f'MSE_{tau}'] = np.nan
            continue
        if np.any(np.isnan(cg_0)):
            log_warning(f'[Layer 6] Scale τ={tau}: coarse-grained(offset=0) contains NaN')
            mse_dict[f'MSE_{tau}'] = np.nan
            continue
        std_cg = float(np.std(cg_0))
        if std_cg < 1e-10:
            log_warning(f'[Layer 6] Scale τ={tau}: std≈0, constant signal')
            mse_dict[f'MSE_{tau}'] = np.nan
            continue
        r_tau = r_factor * std_cg
        A_sum = 0
        B_sum = 0
        n_valid_offsets = 0
        for offset in range(tau):
            cg = _coarse_grain_offset_numba(rr, tau, offset)
            if len(cg) < min_length:
                continue
            A_k, B_k = _count_matches_numba(cg, m, r_tau)
            A_sum += A_k
            B_sum += B_k
            n_valid_offsets += 1
        if n_valid_offsets == 0 or A_sum == 0 or B_sum == 0:
            mse_dict[f'MSE_{tau}'] = np.nan
        else:
            rcmse_tau = -np.log(B_sum / A_sum)
            if np.isnan(rcmse_tau) or np.isinf(rcmse_tau):
                mse_dict[f'MSE_{tau}'] = np.nan
            else:
                mse_dict[f'MSE_{tau}'] = rcmse_tau
                valid_scales.append(tau)
    short_values = np.array([mse_dict.get(f'MSE_{t}', np.nan) for t in range(1, 6)], dtype=np.float64)
    long_values = np.array([mse_dict.get(f'MSE_{t}', np.nan) for t in range(11, 21)], dtype=np.float64)
    n_valid_short = int(np.sum(~np.isnan(short_values)))
    n_valid_long = int(np.sum(~np.isnan(long_values)))
    mse_short = float(np.nanmean(short_values)) if n_valid_short >= MIN_VALID_SHORT_SCALES else np.nan
    mse_long = float(np.nanmean(long_values)) if n_valid_long >= MIN_VALID_LONG_SCALES else np.nan
    if not np.isnan(mse_short) and (not np.isnan(mse_long)):
        cxi_diff = mse_short - mse_long
        cxi_ratio = _compute_cxi_ratio(mse_short, mse_long)
    else:
        cxi_diff = np.nan
        cxi_ratio = np.nan
    result: Dict[str, Any] = {**mse_dict, 'MSE_short': mse_short, 'MSE_long': mse_long, 'CXI': cxi_diff, 'CXI_diff': cxi_diff, 'CXI_ratio': cxi_ratio, 'n_valid_scales': len(valid_scales), 'n_valid_long_scales': n_valid_long, 'n_valid_short_scales': n_valid_short, 'valid_scales': valid_scales}
    log_debug(f'[Layer 6] RCMSE computed: {len(valid_scales)}/{max_scale} valid, long={n_valid_long}/10, short={n_valid_short}/5, MSE_long={mse_long:.3f}, CXI_diff={cxi_diff:.3f}')
    return result

def _empty_rcmse_result(scales: range) -> Dict[str, Any]:
    result: Dict[str, Any] = {f'MSE_{tau}': np.nan for tau in scales}
    result.update({'MSE_short': np.nan, 'MSE_long': np.nan, 'CXI': np.nan, 'CXI_diff': np.nan, 'CXI_ratio': np.nan, 'n_valid_scales': 0, 'n_valid_long_scales': 0, 'n_valid_short_scales': 0, 'valid_scales': []})
    return result

def compute_rcmse_batch(rr_list: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
    results = []
    for i, rr in enumerate(rr_list):
        try:
            result = compute_rcmse_fast(rr, **kwargs)
            results.append(result)
        except Exception as e:
            log_warning(f'[Layer 6] Batch item {i} failed: {str(e)}')
            results.append(_empty_rcmse_result(kwargs.get('scales', range(1, 21))))
    return results
