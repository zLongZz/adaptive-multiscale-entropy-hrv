"""Provides rhythm classifier functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from modules.core.logger import log_info, log_debug, log_warning
PRR31_THRESHOLD = 68.79
COSEN_THRESHOLD_BALANCED = 3.0
COSEN_SINUS_HV = 3.2
CV_SINUS_HV = 0.22
CV_THRESHOLD_SENSITIVE = 0.2
CV_THRESHOLD_SPECIFIC = 0.25
SUBWINDOW_60S = True
SUBWINDOW_30BEAT = True
try:
    from config.config_formal import RhythmDetectionConfig
    _default_config = RhythmDetectionConfig()
    PRR31_THRESHOLD = _default_config.prr31_threshold
    CV_THRESHOLD_SENSITIVE = _default_config.cv_threshold_sensitive
    CV_THRESHOLD_SPECIFIC = _default_config.cv_threshold_specific
    SUBWINDOW_60S = _default_config.subwindow_60s
    SUBWINDOW_30BEAT = _default_config.subwindow_30beat
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

class RhythmClassifier:
    VALID_POLICIES = {'non_entropy_v2', 'legacy_entropy_rules'}

    def __init__(self, prr31_threshold: float=None, cosen_threshold_balanced: float=None, cosen_sinus_hv: float=None, cv_sinus_hv: float=None, cv_threshold_sensitive: float=None, cv_threshold_specific: float=None, subwindow_60s: bool=None, subwindow_30beat: bool=None, classification_policy: str='non_entropy_v2'):
        self.prr31_threshold = prr31_threshold if prr31_threshold is not None else PRR31_THRESHOLD
        self.cosen_threshold_balanced = cosen_threshold_balanced if cosen_threshold_balanced is not None else COSEN_THRESHOLD_BALANCED
        self.cosen_sinus_hv = cosen_sinus_hv if cosen_sinus_hv is not None else COSEN_SINUS_HV
        self.cv_sinus_hv = cv_sinus_hv if cv_sinus_hv is not None else CV_SINUS_HV
        self.cv_threshold_sensitive = cv_threshold_sensitive if cv_threshold_sensitive is not None else CV_THRESHOLD_SENSITIVE
        self.cv_threshold_specific = cv_threshold_specific if cv_threshold_specific is not None else CV_THRESHOLD_SPECIFIC
        self.subwindow_60s = subwindow_60s if subwindow_60s is not None else SUBWINDOW_60S
        self.subwindow_30beat = subwindow_30beat if subwindow_30beat is not None else SUBWINDOW_30BEAT
        if classification_policy not in self.VALID_POLICIES:
            raise ValueError(f'Unknown rhythm classification policy: {classification_policy}')
        self.classification_policy = classification_policy
        log_debug(f'[Layer 5] RhythmClassifier initialized')
        log_debug(f'[Layer 5]   pRR31={self.prr31_threshold}% (Buś 2022)')
        log_debug(f'[Layer 5]   CoSEn={self.cosen_threshold_balanced} (data-calibrated)')
        log_debug(f'[Layer 5]   Sinus_HV: CoSEn>{self.cosen_sinus_hv}, CV<{self.cv_sinus_hv}')
        log_debug(f'[Layer 5]   CV={self.cv_threshold_sensitive}/{self.cv_threshold_specific}')
        log_debug(f'[Layer 5]   Policy={self.classification_policy}')
        if _CONFIG_AVAILABLE:
            log_debug(f'[Layer 5]   Parameters loaded from config_formal.py (partial: 5/8 fields)')
        else:
            log_debug(f'[Layer 5]   Using default parameters (config not found)')

    def compute_pRR31(self, rr: np.ndarray, rr_ts: np.ndarray=None) -> float:
        if len(rr) < 2:
            return 0.0
        if self.subwindow_60s and rr_ts is not None and (len(rr) > 300):
            subwindows = self._split_60s_subwindows(rr, rr_ts)
            if len(subwindows) > 0:
                prr31_values = [self._compute_pRR31_single(sw) for sw in subwindows]
                return np.median(prr31_values)
        return self._compute_pRR31_single(rr)

    def _compute_pRR31_single(self, rr: np.ndarray) -> float:
        if len(rr) < 2:
            return 0.0
        diff_rr = np.abs(np.diff(rr)) * 1000
        count = np.sum(diff_rr >= 31)
        pRR31 = count / len(diff_rr) * 100
        return pRR31

    def _split_60s_subwindows(self, rr: np.ndarray, rr_ts: np.ndarray) -> List[np.ndarray]:
        if len(rr) == 0 or len(rr_ts) == 0:
            return []
        subwindows = []
        window_size = 60
        start_time = np.floor(float(rr_ts[0]) / window_size) * window_size
        while start_time < rr_ts[-1]:
            end_time = start_time + window_size
            mask = (rr_ts >= start_time) & (rr_ts < end_time)
            if np.sum(mask) >= 10:
                subwindows.append(rr[mask])
            start_time = end_time
        return subwindows

    def compute_CoSEn(self, rr: np.ndarray, m: int=2, r_factor: float=0.2) -> float:
        if len(rr) < 30:
            return 0.0
        if self.subwindow_30beat and len(rr) > 100:
            subwindows = self._split_30beat_subwindows(rr)
            if len(subwindows) > 0:
                cosen_values = [self._compute_CoSEn_single(sw, m, r_factor) for sw in subwindows]
                cosen_values = [v for v in cosen_values if not np.isnan(v)]
                if len(cosen_values) > 0:
                    return np.median(cosen_values)
        return self._compute_CoSEn_single(rr, m, r_factor)

    def _compute_CoSEn_single(self, rr: np.ndarray, m: int, r_factor: float) -> float:
        if len(rr) < 2 * m + 1:
            return np.nan
        try:
            sampen = self._compute_sample_entropy(rr, m, r_factor)
            std_rr = np.std(rr)
            if std_rr > 0:
                cosen = sampen - np.log(2 * std_rr)
            else:
                cosen = sampen
            return cosen
        except Exception as e:
            log_warning(f'[Layer 5] CoSEn computation failed: {str(e)}')
            return np.nan

    def _compute_sample_entropy(self, rr: np.ndarray, m: int, r_factor: float) -> float:
        N = len(rr)
        if N < 2 * (m + 1):
            return np.nan
        r = r_factor * np.std(rr, ddof=0)
        if r <= 0:
            return 0.0

        def _count_matches(data, dim):
            n = len(data) - dim
            if n <= 1:
                return 0
            templates = np.array([data[i:i + dim] for i in range(n)])
            count = 0
            for i in range(n - 1):
                diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
                count += np.sum(diffs <= r)
            return count
        B = _count_matches(rr, m)
        A = _count_matches(rr, m + 1)
        if B == 0:
            return np.nan
        if A == 0:
            return np.log(B)
        return -np.log(A / B)

    def _split_30beat_subwindows(self, rr: np.ndarray) -> List[np.ndarray]:
        subwindows = []
        window_size = 30
        step = 10
        for i in range(0, len(rr) - window_size + 1, step):
            subwindows.append(rr[i:i + window_size])
        return subwindows

    def compute_CV(self, rr: np.ndarray) -> float:
        if len(rr) < 2:
            return 0.0
        mean_rr = np.mean(rr)
        if mean_rr == 0:
            return 0.0
        std_rr = np.std(rr)
        cv = std_rr / mean_rr
        return cv

    def classify_rhythm(self, rr: np.ndarray, rr_ts: np.ndarray=None) -> Dict[str, Any]:
        prr31 = self.compute_pRR31(rr, rr_ts)
        cv = self.compute_CV(rr)
        cosen = np.nan if self.classification_policy == 'non_entropy_v2' else self.compute_CoSEn(rr)
        features = {'pRR31': prr31, 'cosen': cosen, 'cv': cv}
        if self.classification_policy == 'non_entropy_v2':
            return self._classify_non_entropy_v2(prr31, cv, features)
        return self._classify_legacy(prr31, cosen, cv, features)

    def _classify_non_entropy_v2(self, prr31: float, cv: float, features: Dict[str, float]) -> Dict[str, Any]:
        pass
        common = {'features': features, 'score_type': 'rule_score_not_probability', 'classification_policy': self.classification_policy}
        if prr31 >= self.prr31_threshold and cv >= self.cv_threshold_sensitive:
            return {'rhythm_type': 'AFib', 'confidence': 0.85, 'method': 'v2_prr31_cv_afib', **common}
        if cv >= self.cv_threshold_specific and prr31 >= 40.0:
            return {'rhythm_type': 'AFib', 'confidence': 0.75, 'method': 'v2_cv_prr31_afib', **common}
        if prr31 < 40.0 and cv < 0.15:
            return {'rhythm_type': 'Sinus', 'confidence': 0.85, 'method': 'v2_clear_sinus', **common}
        if prr31 < self.prr31_threshold and cv < self.cv_threshold_sensitive:
            return {'rhythm_type': 'Sinus', 'confidence': 0.75, 'method': 'v2_probable_sinus', **common}
        return {'rhythm_type': 'Uncertain', 'confidence': 0.5, 'method': 'v2_borderline', **common}

    def _classify_legacy(self, prr31: float, cosen: float, cv: float, features: Dict[str, float]) -> Dict[str, Any]:
        common = {'features': features, 'score_type': 'rule_score_not_probability', 'classification_policy': self.classification_policy}
        if prr31 >= self.prr31_threshold:
            if cosen <= self.cosen_threshold_balanced or cv >= self.cv_threshold_specific:
                return {'rhythm_type': 'AFib', 'confidence': 0.9, 'method': 'rule1_pRR31_validated', **common}
            if cosen > self.cosen_sinus_hv and cv < self.cv_sinus_hv:
                return {'rhythm_type': 'Sinus_HV', 'confidence': 0.75, 'method': 'rule1b_sinus_hv', **common}
            return {'rhythm_type': 'Uncertain', 'confidence': 0.6, 'method': 'rule1b_pRR31_unvalidated', **common}
        if cosen <= self.cosen_threshold_balanced and cv >= self.cv_threshold_sensitive:
            return {'rhythm_type': 'AFib', 'confidence': 0.7, 'method': 'rule2_entropy_cv_combination', **common}
        if cv >= self.cv_threshold_specific:
            return {'rhythm_type': 'AFib', 'confidence': 0.65, 'method': 'rule3_cv_primary', **common}
        if cosen > self.cosen_threshold_balanced and cv < self.cv_threshold_specific:
            return {'rhythm_type': 'Sinus', 'confidence': 0.8, 'method': 'rule4a_cosen_sinus', **common}
        if prr31 < 40 and cv < 0.15:
            return {'rhythm_type': 'Sinus', 'confidence': 0.85, 'method': 'rule4b_clear_sinus', **common}
        return {'rhythm_type': 'Uncertain', 'confidence': 0.5, 'method': 'rule5_borderline', **common}

    def classify_batch(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for win in windows:
            rr = win['rr']
            rr_ts = win.get('rr_ts', None)
            result = self.classify_rhythm(rr, rr_ts)
            results.append({**win, **result})
        n_sinus = sum((1 for r in results if r['rhythm_type'] == 'Sinus'))
        n_sinus_hv = sum((1 for r in results if r['rhythm_type'] == 'Sinus_HV'))
        n_afib = sum((1 for r in results if r['rhythm_type'] == 'AFib'))
        n_uncertain = sum((1 for r in results if r['rhythm_type'] == 'Uncertain'))
        log_info(f'[Layer 5] Classified {len(results)} windows: Sinus={n_sinus}, Sinus_HV={n_sinus_hv}, AFib={n_afib}, Uncertain={n_uncertain}')
        return results
