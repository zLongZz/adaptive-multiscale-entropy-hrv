"""Provides coverage filter functions for the ECG-to-entropy pipeline."""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_info, log_warning

class CoverageFilter:

    def __init__(self, min_windows: int=20, min_coverage: float=0.5, expected_windows: int=24, verbose: bool=True):
        self.min_windows = min_windows
        self.min_coverage = min_coverage
        self.expected_windows = expected_windows
        self.verbose = verbose
        log_info(f'[Layer 7] CoverageFilter initialized')
        log_info(f'[Layer 7]   Min windows: {min_windows}')
        log_info(f'[Layer 7]   Min coverage: {min_coverage:.1%}')
        log_info(f'[Layer 7]   Expected windows: {expected_windows}')

    def filter(self, df_subject: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        log_info(f'[Layer 7] Starting coverage filtering on {len(df_subject)} combinations')
        required_cols = ['ecg_id', 'lead', 'n_windows', 'coverage']
        missing = [c for c in required_cols if c not in df_subject.columns]
        if missing:
            raise ValueError(f'Missing required columns: {missing}')
        mask_windows = df_subject['n_windows'] >= self.min_windows
        mask_coverage = df_subject['coverage'] >= self.min_coverage
        mask_passed = mask_windows & mask_coverage
        df_passed = df_subject[mask_passed].copy()
        df_failed = df_subject[~mask_passed].copy()
        if len(df_failed) > 0:
            df_failed['filter_reason'] = self._get_filter_reasons(df_failed)
        self._report_filtering_stats(df_subject, df_passed, df_failed)
        log_info(f'[Layer 7] Filtering completed: {len(df_passed)} passed, {len(df_failed)} failed')
        return (df_passed, df_failed)

    def _get_filter_reasons(self, df_failed: pd.DataFrame) -> pd.Series:
        reasons = []
        for _, row in df_failed.iterrows():
            reason_parts = []
            if row['n_windows'] < self.min_windows:
                reason_parts.append(f"windows<{self.min_windows} ({row['n_windows']:.0f})")
            if row['coverage'] < self.min_coverage:
                reason_parts.append(f"coverage<{self.min_coverage:.0%} ({row['coverage']:.1%})")
            reasons.append('; '.join(reason_parts))
        return pd.Series(reasons, index=df_failed.index)

    def _report_filtering_stats(self, df_original: pd.DataFrame, df_passed: pd.DataFrame, df_failed: pd.DataFrame):
        n_total = len(df_original)
        n_passed = len(df_passed)
        n_failed = len(df_failed)
        pass_rate = n_passed / n_total if n_total > 0 else 0
        log_info(f'[Layer 7] Coverage filtering statistics:')
        log_info(f'[Layer 7]   Total combinations: {n_total}')
        log_info(f'[Layer 7]   Passed: {n_passed} ({pass_rate:.1%})')
        log_info(f'[Layer 7]   Failed: {n_failed} ({1 - pass_rate:.1%})')
        n_subjects_original = df_original['ecg_id'].nunique()
        n_subjects_passed = df_passed['ecg_id'].nunique()
        n_subjects_failed = n_subjects_original - n_subjects_passed
        log_info(f'[Layer 7]   Subjects (original): {n_subjects_original}')
        log_info(f'[Layer 7]   Subjects (retained): {n_subjects_passed}')
        log_info(f'[Layer 7]   Subjects (lost): {n_subjects_failed}')
        if len(df_failed) > 0 and 'filter_reason' in df_failed.columns:
            log_info(f'[Layer 7]   Failure reasons:')
            reason_counts = df_failed['filter_reason'].value_counts()
            for reason, count in reason_counts.items():
                log_info(f'[Layer 7]     {reason}: {count}')
        if n_subjects_failed > n_subjects_passed * 0.2:
            log_warning(f'[Layer 7] Warning: {n_subjects_failed} subjects lost ({n_subjects_failed / n_subjects_original:.1%})')
            log_warning(f'[Layer 7] Consider relaxing filtering criteria')

def filter_by_rhythm_quality(df_subject: pd.DataFrame, min_rhythm_ratio: float=0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if 'rhythm_ratio' not in df_subject.columns:
        log_warning('[Layer 7] rhythm_ratio column not found, skipping rhythm quality filter')
        return (df_subject, pd.DataFrame())
    mask_passed = df_subject['rhythm_ratio'] >= min_rhythm_ratio
    df_passed = df_subject[mask_passed].copy()
    df_failed = df_subject[~mask_passed].copy()
    log_info(f'[Layer 7] Rhythm quality filtering:')
    log_info(f'[Layer 7]   Passed: {len(df_passed)} ({len(df_passed) / len(df_subject):.1%})')
    log_info(f'[Layer 7]   Failed: {len(df_failed)} (mixed rhythm)')
    return (df_passed, df_failed)

def filter_by_feature_quality(df_subject: pd.DataFrame, min_quality_ratio: float=0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if 'quality_good_ratio' not in df_subject.columns:
        log_warning('[Layer 7] quality_good_ratio column not found, skipping feature quality filter')
        return (df_subject, pd.DataFrame())
    mask_passed = df_subject['quality_good_ratio'] >= min_quality_ratio
    df_passed = df_subject[mask_passed].copy()
    df_failed = df_subject[~mask_passed].copy()
    log_info(f'[Layer 7] Feature quality filtering:')
    log_info(f'[Layer 7]   Passed: {len(df_passed)} ({len(df_passed) / len(df_subject):.1%})')
    log_info(f'[Layer 7]   Failed: {len(df_failed)} (low feature quality)')
    return (df_passed, df_failed)

def get_filtering_summary(df_passed: pd.DataFrame, df_failed: pd.DataFrame) -> Dict[str, any]:
    n_total = len(df_passed) + len(df_failed)
    summary = {'n_passed': len(df_passed), 'n_failed': len(df_failed), 'pass_rate': len(df_passed) / n_total if n_total > 0 else 0.0, 'n_subjects_passed': df_passed['ecg_id'].nunique() if len(df_passed) > 0 else 0, 'n_subjects_failed': df_failed['ecg_id'].nunique() if len(df_failed) > 0 else 0}
    if len(df_failed) > 0 and 'filter_reason' in df_failed.columns:
        summary['failure_reasons'] = df_failed['filter_reason'].value_counts().to_dict()
    return summary
