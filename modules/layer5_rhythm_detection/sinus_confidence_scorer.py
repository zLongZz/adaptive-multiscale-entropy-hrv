"""Provides sinus confidence scorer functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import Dict, Any, Tuple
from scipy.stats import kurtosis
from modules.core.logger import log_debug, log_warning

class SinusConfidenceScorer:

    def __init__(self, weight_rhythm_classifier: float=0.4, weight_rr_regularity: float=0.25, weight_ectopy: float=0.15, weight_interpolation: float=0.1, weight_plausibility: float=0.1, cv_delta_excellent: float=0.3, cv_delta_good: float=0.5, cv_delta_fair: float=0.8, pnn50_lower: float=5.0, pnn50_upper: float=70.0, ectopy_excellent: float=0.01, ectopy_good: float=0.05, ectopy_fair: float=0.1, interp_excellent: float=0.05, interp_good: float=0.1, interp_fair: float=0.2, hr_min_normal: float=0.5, hr_max_normal: float=1.5, hr_min_extended: float=0.4, hr_max_extended: float=2.0, cv_rr_min_normal: float=0.05, cv_rr_max_normal: float=0.2, cv_rr_max_extended: float=0.3):
        self.weights = {'rhythm_classifier': weight_rhythm_classifier, 'rr_regularity': weight_rr_regularity, 'ectopy': weight_ectopy, 'interpolation': weight_interpolation, 'plausibility': weight_plausibility}
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            log_warning(f'[SCS] Weight sum = {weight_sum:.3f}, should be 1.0. Auto-normalizing.')
            for key in self.weights:
                self.weights[key] /= weight_sum
        self.cv_delta_excellent = cv_delta_excellent
        self.cv_delta_good = cv_delta_good
        self.cv_delta_fair = cv_delta_fair
        self.pnn50_lower = pnn50_lower
        self.pnn50_upper = pnn50_upper
        self.ectopy_excellent = ectopy_excellent
        self.ectopy_good = ectopy_good
        self.ectopy_fair = ectopy_fair
        self.interp_excellent = interp_excellent
        self.interp_good = interp_good
        self.interp_fair = interp_fair
        self.hr_min_normal = hr_min_normal
        self.hr_max_normal = hr_max_normal
        self.hr_min_extended = hr_min_extended
        self.hr_max_extended = hr_max_extended
        self.cv_rr_min_normal = cv_rr_min_normal
        self.cv_rr_max_normal = cv_rr_max_normal
        self.cv_rr_max_extended = cv_rr_max_extended
        log_debug(f'[SCS] Initialized with weights: {self.weights}')

    def score(self, rhythm_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        rhythm_type = rhythm_result.get('rhythm_type', 'Unknown')
        confidence = rhythm_result.get('confidence', 0.0)
        rr_final = rhythm_result.get('rr_final', np.array([]))
        rr_original = rhythm_result.get('rr', np.array([]))
        ectopy_rate = rhythm_result.get('ectopy_rate', 0.0)
        interpolation_applied = rhythm_result.get('interpolation_applied', False)
        interpolation_success = rhythm_result.get('interpolation_success', False)
        processing_reason = rhythm_result.get('processing_reason', '')
        d1 = self._score_rhythm_classifier(rhythm_type, confidence)
        d2 = self._score_rr_regularity(rr_final)
        d3 = self._score_ectopy(ectopy_rate)
        d4 = self._score_interpolation(rr_final, rr_original, rhythm_type, interpolation_success, interpolation_applied, processing_reason)
        d5 = self._score_plausibility(rr_final)
        scs = self.weights['rhythm_classifier'] * d1 + self.weights['rr_regularity'] * d2 + self.weights['ectopy'] * d3 + self.weights['interpolation'] * d4 + self.weights['plausibility'] * d5
        dimension_scores = {'D1_rhythm_classifier': d1, 'D2_rr_regularity': d2, 'D3_ectopy': d3, 'D4_interpolation': d4, 'D5_plausibility': d5}
        return (scs, dimension_scores)

    def _score_rhythm_classifier(self, rhythm_type: str, confidence: float) -> float:
        if rhythm_type in ('Sinus', 'Sinus_HV'):
            return confidence
        elif rhythm_type in ('Unknown', 'Uncertain'):
            return 0.5
        elif rhythm_type == 'AFib':
            return 1.0 - confidence
        else:
            log_warning(f'[SCS-D1] Unknown rhythm_type: {rhythm_type}, returning 0.5')
            return 0.5

    def _score_rr_regularity(self, rr: np.ndarray) -> float:
        if len(rr) < 10:
            log_debug('[SCS-D2] RR too short for regularity assessment')
            return 0.0
        rr_clean = rr[~np.isnan(rr)]
        if len(rr_clean) < 10:
            return 0.0
        delta_rr = np.diff(rr_clean)
        if len(delta_rr) > 0 and np.mean(np.abs(delta_rr)) > 0:
            cv_delta = np.std(delta_rr) / np.mean(np.abs(delta_rr))
        else:
            cv_delta = 0.0
        if cv_delta < self.cv_delta_excellent:
            cv_score = 1.0
        elif cv_delta < self.cv_delta_good:
            cv_score = 1.0 - 0.3 * (cv_delta - self.cv_delta_excellent) / (self.cv_delta_good - self.cv_delta_excellent)
        elif cv_delta < self.cv_delta_fair:
            cv_score = 0.7 - 0.4 * (cv_delta - self.cv_delta_good) / (self.cv_delta_fair - self.cv_delta_good)
        else:
            cv_score = max(0.0, 0.3 - 0.15 * (cv_delta - self.cv_delta_fair))
        pnn50 = np.mean(np.abs(delta_rr) > 0.05) * 100
        if self.pnn50_lower < pnn50 < self.pnn50_upper:
            pnn50_score = 1.0
        elif pnn50 < self.pnn50_lower:
            pnn50_score = max(0.0, pnn50 / self.pnn50_lower)
        else:
            pnn50_score = max(0.0, 1.0 - (pnn50 - self.pnn50_upper) / 30.0)
        if np.std(rr_clean) < 1e-08:
            kurt_score = 0.5
        else:
            try:
                kurt = kurtosis(rr_clean, fisher=False)
                kurt_deviation = abs(kurt - 3.0)
                kurt_score = max(0.0, 1.0 - kurt_deviation / 10.0)
            except (FloatingPointError, ValueError):
                kurt_score = 0.5
        regularity_score = 0.5 * cv_score + 0.3 * pnn50_score + 0.2 * kurt_score
        return np.clip(regularity_score, 0.0, 1.0)

    def _score_ectopy(self, ectopy_rate: float) -> float:
        if ectopy_rate < self.ectopy_excellent:
            return 1.0
        elif ectopy_rate < self.ectopy_good:
            return 1.0 - 0.2 * (ectopy_rate - self.ectopy_excellent) / (self.ectopy_good - self.ectopy_excellent)
        elif ectopy_rate < self.ectopy_fair:
            return 0.8 - 0.3 * (ectopy_rate - self.ectopy_good) / (self.ectopy_fair - self.ectopy_good)
        else:
            return max(0.0, 0.5 - 2.5 * (ectopy_rate - self.ectopy_fair))

    def _score_interpolation(self, rr_final: np.ndarray, rr_original: np.ndarray, rhythm_type: str, success: bool, applied: bool, reason: str) -> float:
        if rhythm_type not in ('Sinus', 'Sinus_HV'):
            return 0.5
        if not success:
            return 0.0
        if reason == 'no_ectopy':
            return 1.0
        if not applied:
            return 0.5
        if len(rr_final) == 0 or len(rr_original) == 0:
            return 0.5
        if len(rr_final) == len(rr_original):
            n_changed = np.sum(~np.isclose(rr_final, rr_original, rtol=1e-06))
            interp_ratio = n_changed / len(rr_final)
        else:
            interp_ratio = abs(len(rr_final) - len(rr_original)) / max(len(rr_final), len(rr_original))
        if interp_ratio < self.interp_excellent:
            return 1.0
        elif interp_ratio < self.interp_good:
            return 1.0 - 0.2 * (interp_ratio - self.interp_excellent) / (self.interp_good - self.interp_excellent)
        elif interp_ratio < self.interp_fair:
            return 0.8 - 0.3 * (interp_ratio - self.interp_good) / (self.interp_fair - self.interp_good)
        else:
            return max(0.0, 0.5 - 2.5 * (interp_ratio - self.interp_fair))

    def _score_plausibility(self, rr: np.ndarray) -> float:
        if len(rr) < 5:
            return 0.0
        rr_clean = rr[~np.isnan(rr)]
        if len(rr_clean) < 5:
            return 0.0
        mean_rr = np.mean(rr_clean)
        std_rr = np.std(rr_clean)
        cv_rr = std_rr / mean_rr if mean_rr > 0 else 0.0
        if self.hr_min_normal < mean_rr < self.hr_max_normal:
            hr_score = 1.0
        elif self.hr_min_extended < mean_rr < self.hr_max_extended:
            if mean_rr < self.hr_min_normal:
                hr_score = 0.5 + 0.5 * (mean_rr - self.hr_min_extended) / (self.hr_min_normal - self.hr_min_extended)
            else:
                hr_score = 0.5 + 0.5 * (self.hr_max_extended - mean_rr) / (self.hr_max_extended - self.hr_max_normal)
        else:
            hr_score = 0.0
        if self.cv_rr_min_normal < cv_rr < self.cv_rr_max_normal:
            cv_score = 1.0
        elif cv_rr < self.cv_rr_min_normal:
            cv_score = 0.3 + 0.7 * (cv_rr / self.cv_rr_min_normal)
        elif cv_rr < self.cv_rr_max_extended:
            cv_score = 1.0 - 0.4 * (cv_rr - self.cv_rr_max_normal) / (self.cv_rr_max_extended - self.cv_rr_max_normal)
        else:
            cv_score = max(0.0, 0.6 - 2.0 * (cv_rr - self.cv_rr_max_extended))
        plausibility_score = 0.6 * hr_score + 0.4 * cv_score
        return np.clip(plausibility_score, 0.0, 1.0)

    def get_config(self) -> Dict[str, Any]:
        return {'weights': self.weights, 'thresholds': {'cv_delta': {'excellent': self.cv_delta_excellent, 'good': self.cv_delta_good, 'fair': self.cv_delta_fair}, 'pnn50': {'lower': self.pnn50_lower, 'upper': self.pnn50_upper}, 'ectopy': {'excellent': self.ectopy_excellent, 'good': self.ectopy_good, 'fair': self.ectopy_fair}, 'interpolation': {'excellent': self.interp_excellent, 'good': self.interp_good, 'fair': self.interp_fair}, 'heart_rate': {'min_normal': self.hr_min_normal, 'max_normal': self.hr_max_normal, 'min_extended': self.hr_min_extended, 'max_extended': self.hr_max_extended}, 'cv_rr': {'min_normal': self.cv_rr_min_normal, 'max_normal': self.cv_rr_max_normal, 'max_extended': self.cv_rr_max_extended}}}
