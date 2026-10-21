"""Provides subject aggregator functions for the ECG-to-entropy pipeline."""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_info, log_debug, log_warning

class SubjectAggregator:
    pass

    def __init__(self, features_to_aggregate: List[str]=None, aggregation_method: str='median', variability_method: str='iqr', compute_spatial: bool=False):
        self.aggregation_method = aggregation_method
        self.variability_method = variability_method
        self.compute_spatial = compute_spatial
        if features_to_aggregate is None:
            self.features_to_aggregate = ['MSE_long', 'MSE_short', 'CXI', 'CXI_diff', 'CXI_ratio', 'DFA_alpha1', 'DFA_alpha1_intercept', 'DFA_alpha1_r_squared', 'DFA_alpha2', 'DFA_intercept', 'DFA_r_squared', 'DFA_slope_8_64', 'DFA_slope_8_64_intercept', 'DFA_slope_8_64_r_squared', 'rhythm_pRR31', 'rhythm_cv', 'rhythm_cosen']
        else:
            self.features_to_aggregate = features_to_aggregate
        log_info(f'[Layer 7] SubjectAggregator initialized')
        log_debug(f'[Layer 7]   Aggregation: {aggregation_method}')
        log_debug(f'[Layer 7]   Variability: {variability_method}')
        log_debug(f'[Layer 7]   Features: {len(self.features_to_aggregate)}')
        log_debug(f'[Layer 7]   Spatial complexity: {compute_spatial}')

    def aggregate(self, df_windows: pd.DataFrame) -> pd.DataFrame:
        pass
        log_info(f'[Layer 7] Starting aggregation for {len(df_windows)} windows')
        group_columns = ['ecg_id', 'lead']
        if 'analysis_track' in df_windows.columns:
            group_columns.append('analysis_track')
        grouped = df_windows.groupby(group_columns)
        results = []
        n_subjects = df_windows['ecg_id'].nunique()
        n_leads = df_windows['lead'].nunique()
        log_info(f'[Layer 7] Aggregating {n_subjects} subjects × {n_leads} leads')
        for group_key, group_data in grouped:
            if len(group_columns) == 3:
                ecg_id, lead, analysis_track = group_key
            else:
                ecg_id, lead = group_key
                analysis_track = None
            subject_features = self._aggregate_single_subject_lead(ecg_id, lead, group_data, analysis_track=analysis_track)
            results.append(subject_features)
        df_subject = pd.DataFrame(results)
        if self.compute_spatial:
            df_subject = self._add_spatial_complexity(df_subject)
        self._report_aggregation_stats(df_subject)
        log_info(f'[Layer 7] Aggregation completed: {len(df_subject)} subject-lead combinations')
        return df_subject

    def _aggregate_single_subject_lead(self, ecg_id: str, lead: str, group_data: pd.DataFrame, analysis_track: Optional[str]=None) -> Dict[str, Any]:
        pass
        features = {'ecg_id': ecg_id, 'group_code': group_data['group_code'].iloc[0], 'lead': lead, 'n_windows': len(group_data)}
        if analysis_track is not None:
            features['analysis_track'] = analysis_track
        if 'window_sec' in group_data.columns:
            window_sec = pd.to_numeric(group_data['window_sec'], errors='coerce').dropna()
            if len(window_sec) > 0:
                features['window_sec'] = int(window_sec.iloc[0])
        if 'feature_mode' in group_data.columns:
            feature_mode = group_data['feature_mode'].dropna()
            if len(feature_mode) > 0:
                features['feature_mode'] = str(feature_mode.iloc[0])
        provenance_fields = ['rr_detector_mode', 'rr_detector_method', 'rr_selected_polarity', 'rr_chunk_sec', 'rr_transient_prefix_sec', 'rr_primary_lead', 'rr_selected_lead', 'rr_lead_fallback_used', 'rr_n_peaks', 'rr_n_intervals', 'rr_valid_fraction', 'rr_quality_score']
        for field_name in provenance_fields:
            if field_name in group_data.columns:
                values = group_data[field_name].dropna()
                if len(values) > 0:
                    features[field_name] = values.iloc[0]
        candidate_windows = np.nan
        if 'n_candidate_windows' in group_data.columns:
            vals = pd.to_numeric(group_data['n_candidate_windows'], errors='coerce').dropna()
            if len(vals) > 0:
                candidate_windows = int(vals.max())
        features['n_candidate_windows'] = candidate_windows
        if 'n_nominal_windows' in group_data.columns:
            vals = pd.to_numeric(group_data['n_nominal_windows'], errors='coerce').dropna()
            features['n_nominal_windows'] = int(vals.max()) if len(vals) else np.nan
        if pd.notna(candidate_windows) and candidate_windows > 0:
            features['rr_eligible_acceptance_ratio'] = len(group_data) / candidate_windows
            features['acceptance_ratio'] = features['rr_eligible_acceptance_ratio']
            features['coverage'] = features['rr_eligible_acceptance_ratio']
        else:
            expected_windows = 180
            features['acceptance_ratio'] = np.nan
            features['coverage'] = len(group_data) / expected_windows
        features['coverage_legacy_15min_expected180'] = len(group_data) / 180
        if 'rhythm_type' in group_data.columns:
            rhythm_counts = group_data['rhythm_type'].value_counts()
            features['accepted_rhythm_dominant'] = rhythm_counts.index[0]
            features['accepted_rhythm_ratio'] = rhythm_counts.iloc[0] / len(group_data)
        if 'clock_hour' in group_data.columns:
            features['n_accepted_clock_hours'] = int(group_data['clock_hour'].nunique())
        for feature in self.features_to_aggregate:
            if feature in group_data.columns:
                values = pd.to_numeric(group_data[feature], errors='coerce')
                n_valid = int(values.notna().sum())
                features[f'{feature}_n_valid'] = n_valid
                if n_valid == 0:
                    for suffix in ('median', 'iqr', 'mean', 'std', 'min', 'max'):
                        features[f'{feature}_{suffix}'] = np.nan
                    continue
                median_val = values.median()
                features[f'{feature}_median'] = median_val
                q75 = values.quantile(0.75)
                q25 = values.quantile(0.25)
                iqr_val = q75 - q25
                features[f'{feature}_iqr'] = iqr_val
                features[f'{feature}_mean'] = values.mean()
                features[f'{feature}_std'] = values.std()
                features[f'{feature}_min'] = values.min()
                features[f'{feature}_max'] = values.max()
        if 'feature_quality' in group_data.columns:
            quality_counts = group_data['feature_quality'].value_counts()
            features['quality_good_ratio'] = quality_counts.get('good', 0) / len(group_data)
        return features

    def _add_spatial_complexity(self, df_subject: pd.DataFrame) -> pd.DataFrame:
        pass
        from modules.layer6_features import compute_spatial_complexity_simple
        log_info(f'[Layer 7] Computing spatial complexity (12-lead)')
        spatial_results = []
        for ecg_id in df_subject['ecg_id'].unique():
            subject_data = df_subject[df_subject['ecg_id'] == ecg_id]
            lead_features = {}
            for _, row in subject_data.iterrows():
                lead = row['lead']
                mse_long = row.get('MSE_long_median', np.nan)
                if not np.isnan(mse_long):
                    lead_features[lead] = mse_long
            if len(lead_features) >= 9:
                evr_pc1 = compute_spatial_complexity_simple(lead_features, target_feature='MSE_long')
                spatial_results.append({'ecg_id': ecg_id, 'EVR_PC1': evr_pc1, 'n_leads_spatial': len(lead_features)})
            else:
                log_warning(f'[Layer 7] Subject {ecg_id}: insufficient leads ({len(lead_features)}/12)')
                spatial_results.append({'ecg_id': ecg_id, 'EVR_PC1': np.nan, 'n_leads_spatial': len(lead_features)})
        df_spatial = pd.DataFrame(spatial_results)
        df_subject = df_subject.merge(df_spatial, on='ecg_id', how='left')
        n_valid = df_spatial['EVR_PC1'].notna().sum()
        log_info(f'[Layer 7] Spatial complexity computed for {n_valid}/{len(df_spatial)} subjects')
        return df_subject

    def _report_aggregation_stats(self, df_subject: pd.DataFrame):
        pass
        n_subjects = df_subject['ecg_id'].nunique()
        n_leads = df_subject['lead'].nunique()
        n_combinations = len(df_subject)
        log_info(f'[Layer 7] Aggregation statistics:')
        log_info(f'[Layer 7]   Subjects: {n_subjects}')
        log_info(f'[Layer 7]   Leads: {n_leads}')
        log_info(f'[Layer 7]   Total combinations: {n_combinations}')
        if 'n_windows' in df_subject.columns:
            log_info(f'[Layer 7]   Windows per combination:')
            log_info(f"[Layer 7]     Mean: {df_subject['n_windows'].mean():.1f}")
            log_info(f"[Layer 7]     Median: {df_subject['n_windows'].median():.0f}")
            log_info(f"[Layer 7]     Range: [{df_subject['n_windows'].min():.0f}, {df_subject['n_windows'].max():.0f}]")
        if 'coverage' in df_subject.columns:
            log_info(f'[Layer 7]   Coverage:')
            log_info(f"[Layer 7]     Mean: {df_subject['coverage'].mean():.1%}")
            log_info(f"[Layer 7]     Median: {df_subject['coverage'].median():.1%}")
        if 'MSE_long_median' in df_subject.columns:
            mse_valid = df_subject['MSE_long_median'].dropna()
            log_info(f'[Layer 7]   MSE_long_median:')
            log_info(f'[Layer 7]     Mean: {mse_valid.mean():.3f}')
            log_info(f'[Layer 7]     Median: {mse_valid.median():.3f}')
            log_info(f'[Layer 7]     Valid: {len(mse_valid)}/{len(df_subject)}')
        if 'EVR_PC1' in df_subject.columns:
            evr_valid = df_subject['EVR_PC1'].dropna()
            if len(evr_valid) > 0:
                log_info(f'[Layer 7]   EVR_PC1 (spatial complexity):')
                log_info(f'[Layer 7]     Mean: {evr_valid.mean():.3f}')
                log_info(f'[Layer 7]     Valid: {len(evr_valid)}/{n_subjects} subjects')

def aggregate_by_rhythm(df_subject: pd.DataFrame, sinus_ratio: float=0.8, afib_ratio: float=0.8) -> Dict[str, pd.DataFrame]:
    pass
    subject_rhythm = {}
    for ecg_id in df_subject['ecg_id'].unique():
        subject_data = df_subject[df_subject['ecg_id'] == ecg_id]
        if 'rhythm_ratio' in subject_data.columns and 'rhythm_dominant' in subject_data.columns:
            mode_result = subject_data['rhythm_dominant'].mode()
            dominant = mode_result.iloc[0] if len(mode_result) > 0 else 'Unknown'
            ratio = subject_data['rhythm_ratio'].mean()
            if dominant == 'Sinus' and ratio >= sinus_ratio:
                subject_rhythm[ecg_id] = 'Sinus'
            elif dominant == 'AFib' and ratio >= afib_ratio:
                subject_rhythm[ecg_id] = 'AFib'
            else:
                subject_rhythm[ecg_id] = 'Mixed'
        else:
            subject_rhythm[ecg_id] = 'Unknown'
    df_subject['subject_rhythm'] = df_subject['ecg_id'].map(subject_rhythm)
    results = {'Sinus': df_subject[df_subject['subject_rhythm'] == 'Sinus'].copy(), 'AFib': df_subject[df_subject['subject_rhythm'] == 'AFib'].copy(), 'Mixed': df_subject[df_subject['subject_rhythm'] == 'Mixed'].copy()}
    log_info(f'[Layer 7] Rhythm stratification:')
    log_info(f"[Layer 7]   Sinus: {len(results['Sinus']['ecg_id'].unique())} subjects")
    log_info(f"[Layer 7]   AFib: {len(results['AFib']['ecg_id'].unique())} subjects")
    log_info(f"[Layer 7]   Mixed: {len(results['Mixed']['ecg_id'].unique())} subjects")
    return results
