"""Provides record eligibility functions for the ECG-to-entropy pipeline."""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
ELIGIBILITY_POLICY_ID = 'paper_primary_v1_20260719'

@dataclass(frozen=True)
class EligibilityThresholds:
    primary_duration_hours: float = 20.0
    sensitivity_duration_hours: float = 18.0
    min_entropy_windows: int = 20
    min_acceptance_ratio: float = 0.5
    min_clock_hours: int = 12
    sinus_predominance_ratio: float = 0.8
    legacy_min_windows: int = 20
    legacy_min_acceptance_ratio: float = 0.5

class RecordEligibilityEvaluator:

    def __init__(self, thresholds: Optional[EligibilityThresholds]=None):
        self.thresholds = thresholds or EligibilityThresholds()

    def summarize(self, window_audit: pd.DataFrame, *, hea_duration_seconds: Optional[float], rr_span_seconds: Optional[float], analysis_track: Optional[str]=None, analysis_duration_seconds: Optional[float]=None, analysis_duration_policy: str='first_24h') -> Dict:
        audit = window_audit.copy()
        n_nominal = int(len(audit))
        rr_eligible = _bool_column(audit, 'layer3_eligible')
        rhythm = audit.get('rhythm_type', pd.Series(index=audit.index, dtype=object))
        classified = rhythm.notna()
        entropy_eligible = _bool_column(audit, 'entropy_eligible')
        if analysis_track is not None and 'analysis_track' in audit.columns:
            entropy_eligible &= audit['analysis_track'].eq(analysis_track)
        n_rr_eligible = int(rr_eligible.sum())
        n_classified = int((rr_eligible & classified).sum())
        n_entropy_eligible = int(entropy_eligible.sum())
        source_duration_hours = _hours(hea_duration_seconds)
        analysis_duration_hours = _hours(analysis_duration_seconds if analysis_duration_seconds is not None else hea_duration_seconds)
        rr_span_hours = _hours(rr_span_seconds)
        classified_rows = audit[rr_eligible & classified]
        rhythm_counts = classified_rows['rhythm_type'].value_counts() if len(classified_rows) else pd.Series(dtype=int)
        ratios = {f'{name.lower()}_ratio_all_classified': float(rhythm_counts.get(name, 0) / n_classified) if n_classified else np.nan for name in ('Sinus', 'Sinus_HV', 'AFib', 'Uncertain')}
        ratios['af_like_ratio_all_classified'] = ratios['afib_ratio_all_classified']
        sinus_ratio = float((rhythm_counts.get('Sinus', 0) + rhythm_counts.get('Sinus_HV', 0)) / n_classified) if n_classified else np.nan
        high_ectopy = audit.get('rhythm_quality', pd.Series(index=audit.index, dtype=object)).eq('rejected_high_ectopy')
        entropy_over_rr = n_entropy_eligible / n_rr_eligible if n_rr_eligible else np.nan
        entropy_over_nominal = n_entropy_eligible / n_nominal if n_nominal else np.nan
        entropy_clock_hours = _nunique_clock_hours(audit, entropy_eligible)
        duration_pass = bool(np.isfinite(analysis_duration_hours) and analysis_duration_hours >= self.thresholds.primary_duration_hours)
        sensitivity_duration_pass = bool(np.isfinite(analysis_duration_hours) and analysis_duration_hours >= self.thresholds.sensitivity_duration_hours)
        windows_pass = n_entropy_eligible >= self.thresholds.min_entropy_windows
        acceptance_pass = bool(np.isfinite(entropy_over_rr) and entropy_over_rr >= self.thresholds.min_acceptance_ratio)
        clock_hours_pass = entropy_clock_hours >= self.thresholds.min_clock_hours
        coverage_pass = bool(windows_pass and acceptance_pass and clock_hours_pass)
        sinus_predominant = bool(np.isfinite(sinus_ratio) and sinus_ratio >= self.thresholds.sinus_predominance_ratio)
        primary_lead_pass = _primary_lead_pass(audit)
        full_day_pass = bool(duration_pass and coverage_pass)
        sensitivity_full_day_pass = bool(sensitivity_duration_pass and coverage_pass)
        sinus_requirements_apply = analysis_track in {'sinus_primary_v2', 'sinus_strict_legacy'}
        failure_reasons = _eligibility_failure_reasons(duration_pass=duration_pass, windows_pass=windows_pass, acceptance_pass=acceptance_pass, clock_hours_pass=clock_hours_pass, sinus_requirements_apply=sinus_requirements_apply, sinus_predominant=sinus_predominant, primary_lead_pass=primary_lead_pass, duration_reason='duration_below_20h')
        sensitivity_failure_reasons = _eligibility_failure_reasons(duration_pass=sensitivity_duration_pass, windows_pass=windows_pass, acceptance_pass=acceptance_pass, clock_hours_pass=clock_hours_pass, sinus_requirements_apply=sinus_requirements_apply, sinus_predominant=sinus_predominant, primary_lead_pass=primary_lead_pass, duration_reason='duration_below_18h')
        return {'eligibility_policy_id': ELIGIBILITY_POLICY_ID, 'hea_duration_hours': source_duration_hours, 'source_duration_hours': source_duration_hours, 'analysis_duration_hours': analysis_duration_hours, 'analysis_duration_policy': analysis_duration_policy, 'rr_span_hours': rr_span_hours, 'n_nominal_windows': n_nominal, 'n_rr_eligible_windows': n_rr_eligible, 'n_classified_windows': n_classified, 'n_entropy_eligible_windows': n_entropy_eligible, 'rr_eligibility_ratio': n_rr_eligible / n_nominal if n_nominal else np.nan, 'entropy_acceptance_ratio_rr_eligible': entropy_over_rr, 'entropy_acceptance_ratio_nominal': entropy_over_nominal, 'n_nominal_clock_hours': _nunique_clock_hours(audit), 'n_rr_eligible_clock_hours': _nunique_clock_hours(audit, rr_eligible), 'n_entropy_eligible_clock_hours': entropy_clock_hours, 'duration_20h_pass': duration_pass, 'duration_18h_pass': sensitivity_duration_pass, 'coverage_windows_pass': bool(windows_pass), 'coverage_acceptance_pass': acceptance_pass, 'coverage_clock_hours_pass': bool(clock_hours_pass), 'coverage_pass': coverage_pass, 'eligible_full_day_entropy': full_day_pass, 'sinus_ratio_all_classified': sinus_ratio, 'sinus_predominant_primary': sinus_predominant, 'fixed_primary_lead_pass': primary_lead_pass, 'eligible_primary_analysis': bool(analysis_track == 'sinus_primary_v2' and full_day_pass and sinus_predominant and primary_lead_pass), 'eligible_primary_18h_sensitivity': bool(analysis_track == 'sinus_primary_v2' and sensitivity_full_day_pass and sinus_predominant and primary_lead_pass), 'eligible_all_rhythm_20h_sensitivity': bool(analysis_track is None and full_day_pass), 'eligible_all_rhythm_sensitivity': bool(analysis_track is None and full_day_pass), 'eligible_af_like_exploratory': bool(analysis_track == 'af_like_exploratory_v3' and full_day_pass), 'legacy_coverage_pass': bool(n_entropy_eligible >= self.thresholds.legacy_min_windows and np.isfinite(entropy_over_rr) and (entropy_over_rr >= self.thresholds.legacy_min_acceptance_ratio)), 'coverage_thresholds_locked': True, 'threshold_primary_duration_hours': self.thresholds.primary_duration_hours, 'threshold_sensitivity_duration_hours': self.thresholds.sensitivity_duration_hours, 'threshold_min_entropy_windows': self.thresholds.min_entropy_windows, 'threshold_min_acceptance_ratio': self.thresholds.min_acceptance_ratio, 'threshold_min_clock_hours': self.thresholds.min_clock_hours, 'threshold_sinus_predominance_ratio': self.thresholds.sinus_predominance_ratio if sinus_requirements_apply else np.nan, 'duration_margin_hours': _margin(analysis_duration_hours, self.thresholds.primary_duration_hours), 'sensitivity_duration_margin_hours': _margin(analysis_duration_hours, self.thresholds.sensitivity_duration_hours), 'entropy_window_margin': float(n_entropy_eligible - self.thresholds.min_entropy_windows), 'acceptance_ratio_margin': _margin(entropy_over_rr, self.thresholds.min_acceptance_ratio), 'clock_hour_margin': float(entropy_clock_hours - self.thresholds.min_clock_hours), 'sinus_ratio_margin': _margin(sinus_ratio, self.thresholds.sinus_predominance_ratio) if sinus_requirements_apply else np.nan, 'eligibility_failure_reasons': ';'.join(failure_reasons), 'n_eligibility_failure_reasons': len(failure_reasons), 'eligibility_18h_failure_reasons': ';'.join(sensitivity_failure_reasons), 'n_eligibility_18h_failure_reasons': len(sensitivity_failure_reasons), 'n_high_ectopy_windows': int(high_ectopy.sum()), **ratios}

def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series(False, index=frame.index)).fillna(False).astype(bool)

def _hours(seconds) -> float:
    return float(seconds) / 3600.0 if seconds is not None and np.isfinite(seconds) else np.nan

def _margin(value, threshold) -> float:
    if value is None or not np.isfinite(value):
        return np.nan
    return float(value - threshold)

def _eligibility_failure_reasons(*, duration_pass: bool, windows_pass: bool, acceptance_pass: bool, clock_hours_pass: bool, sinus_requirements_apply: bool, sinus_predominant: bool, primary_lead_pass: bool, duration_reason: str) -> list[str]:
    pass
    reasons = []
    if not duration_pass:
        reasons.append(duration_reason)
    if not windows_pass:
        reasons.append('entropy_windows_below_20')
    if not acceptance_pass:
        reasons.append('acceptance_ratio_below_0.50')
    if not clock_hours_pass:
        reasons.append('clock_hours_below_12')
    if sinus_requirements_apply and (not sinus_predominant):
        reasons.append('sinus_ratio_below_0.80')
    if sinus_requirements_apply and (not primary_lead_pass):
        reasons.append('primary_lead_mismatch')
    return reasons

def _nunique_clock_hours(frame: pd.DataFrame, mask: Optional[pd.Series]=None) -> int:
    if 'clock_hour' not in frame.columns or frame.empty:
        return 0
    values = frame['clock_hour'] if mask is None else frame.loc[mask, 'clock_hour']
    return int(values.dropna().nunique())

def _primary_lead_pass(frame: pd.DataFrame) -> bool:
    pass
    if frame.empty or not {'primary_lead', 'selected_lead'}.issubset(frame.columns):
        return False
    pairs = frame[['primary_lead', 'selected_lead']].dropna().drop_duplicates()
    return bool(len(pairs) == 1 and pairs.iloc[0]['primary_lead'] == pairs.iloc[0]['selected_lead'])
