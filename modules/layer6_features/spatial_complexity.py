"""Provides spatial complexity functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_debug, log_warning

class SpatialComplexityCalculator:

    def __init__(self, n_components: int=12, standardize: bool=True):
        self.n_components = n_components
        self.standardize = standardize
        log_debug(f'[Layer 6] SpatialComplexityCalculator initialized')
        log_debug(f'[Layer 6]   n_components={n_components}, standardize={standardize}')

    def compute_for_subject(self, features_12lead: Dict[str, np.ndarray], target_feature: str='MSE_long') -> Dict[str, Any]:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_data = []
        valid_leads = []
        for lead in lead_names:
            if lead in features_12lead:
                value = features_12lead[lead]
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        value = np.mean(value)
                    else:
                        value = np.nan
                if not np.isnan(value) and (not np.isinf(value)):
                    lead_data.append(value)
                    valid_leads.append(lead)
        n_valid = len(lead_data)
        if n_valid < 9:
            log_warning(f'[Layer 6] Insufficient valid leads: {n_valid}/12')
            return self._generate_empty_result(n_valid)
        X = np.array(lead_data).reshape(1, -1)
        if self.standardize:
            if np.std(X) == 0:
                log_warning(f'[Layer 6] Zero variance in lead data')
                return self._generate_empty_result(n_valid)
        try:
            mean_val = np.mean(X)
            std_val = np.std(X)
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                evr_pc1 = 1.0 / (1.0 + cv)
            else:
                cv = np.nan
                evr_pc1 = np.nan
            result = {'EVR_PC1': evr_pc1, 'EVR_PC2': np.nan, 'EVR_PC3': np.nan, 'cumulative_variance': evr_pc1, 'n_leads_used': n_valid, 'spatial_quality': self._assess_quality(evr_pc1, n_valid), 'lead_CV': cv, 'lead_mean': mean_val, 'lead_std': std_val}
            log_debug(f'[Layer 6] Spatial complexity computed: EVR_PC1={evr_pc1:.3f}, n_leads={n_valid}, CV={cv:.3f}')
            return result
        except Exception as e:
            log_warning(f'[Layer 6] PCA failed: {str(e)}')
            return self._generate_empty_result(n_valid)

    def compute_for_multiple_windows(self, windows_12lead: List[Dict[str, Any]], target_feature: str='MSE_long') -> Dict[str, Any]:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_dict = {lead: [] for lead in lead_names}
        for window in windows_12lead:
            lead = window.get('lead')
            value = window.get(target_feature, np.nan)
            if lead in lead_dict:
                lead_dict[lead].append(value)
        n_windows = len(windows_12lead) // 12
        if n_windows < 10:
            log_warning(f'[Layer 6] Insufficient windows for PCA: {n_windows}')
            return self._generate_empty_result(0)
        X_list = []
        for lead in lead_names:
            if lead in lead_dict and len(lead_dict[lead]) >= n_windows:
                X_list.append(lead_dict[lead][:n_windows])
            else:
                X_list.append([np.nan] * n_windows)
        X = np.array(X_list).T
        valid_cols = ~np.all(np.isnan(X), axis=0)
        X_valid = X[:, valid_cols]
        n_valid_leads = np.sum(valid_cols)
        if n_valid_leads < 9:
            log_warning(f'[Layer 6] Insufficient valid leads: {n_valid_leads}/12')
            return self._generate_empty_result(n_valid_leads)
        valid_rows = ~np.any(np.isnan(X_valid), axis=1)
        X_clean = X_valid[valid_rows, :]
        if X_clean.shape[0] < 10:
            log_warning(f'[Layer 6] Insufficient clean windows: {X_clean.shape[0]}')
            return self._generate_empty_result(n_valid_leads)
        if self.standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean
        try:
            pca = PCA(n_components=min(self.n_components, X_scaled.shape[1]))
            pca.fit(X_scaled)
            evr = pca.explained_variance_ratio_
            result = {'EVR_PC1': evr[0] if len(evr) > 0 else np.nan, 'EVR_PC2': evr[1] if len(evr) > 1 else np.nan, 'EVR_PC3': evr[2] if len(evr) > 2 else np.nan, 'cumulative_variance': np.sum(evr[:3]) if len(evr) >= 3 else np.sum(evr), 'n_leads_used': n_valid_leads, 'n_windows_used': X_clean.shape[0], 'spatial_quality': self._assess_quality(evr[0] if len(evr) > 0 else np.nan, n_valid_leads), 'explained_variance_ratio': evr.tolist() if len(evr) > 0 else []}
            log_debug(f"[Layer 6] PCA completed: EVR_PC1={result['EVR_PC1']:.3f}, n_windows={X_clean.shape[0]}, n_leads={n_valid_leads}")
            return result
        except Exception as e:
            log_warning(f'[Layer 6] PCA failed: {str(e)}')
            return self._generate_empty_result(n_valid_leads)

    def _assess_quality(self, evr_pc1: float, n_leads: int) -> str:
        if np.isnan(evr_pc1):
            return 'poor'
        if n_leads >= 10:
            return 'good'
        elif n_leads >= 9:
            return 'warning'
        else:
            return 'poor'

    def _generate_empty_result(self, n_leads: int) -> Dict[str, Any]:
        return {'EVR_PC1': np.nan, 'EVR_PC2': np.nan, 'EVR_PC3': np.nan, 'cumulative_variance': np.nan, 'n_leads_used': n_leads, 'spatial_quality': 'poor'}

def compute_spatial_complexity_simple(features_12lead: Dict[str, float], target_feature: str='MSE_long') -> float:
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    values = []
    for lead in lead_names:
        if lead in features_12lead:
            val = features_12lead[lead]
            if not np.isnan(val) and (not np.isinf(val)):
                values.append(val)
    if len(values) < 9:
        return np.nan
    mean_val = np.mean(values)
    std_val = np.std(values)
    if mean_val == 0:
        return np.nan
    cv = std_val / abs(mean_val)
    evr_pc1 = 1.0 / (1.0 + cv)
    return evr_pc1
