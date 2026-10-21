"""Provides rhythm processor functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import Dict, List, Any
from .rhythm_classifier import RhythmClassifier
from .ectopy_detector import EctopyDetector
from .sinus_interpolator import SinusInterpolator
from .afib_filter import AFibFilter
from modules.core.logger import log_info, log_debug, log_warning

class RhythmProcessor:

    def __init__(self, prr31_threshold: float=68.79, cosen_threshold_balanced: float=3.0, cosen_sinus_hv: float=3.2, cv_sinus_hv: float=0.22, cv_threshold_sensitive: float=0.2, cv_threshold_specific: float=0.25, max_ectopy_ratio: float=0.2, classification_policy: str='non_entropy_v2', constant_rr_std_epsilon: float=1e-08, interpolation_variance_ratio_min: float=0.05):
        self.classifier = RhythmClassifier(prr31_threshold=prr31_threshold, cosen_threshold_balanced=cosen_threshold_balanced, cosen_sinus_hv=cosen_sinus_hv, cv_sinus_hv=cv_sinus_hv, cv_threshold_sensitive=cv_threshold_sensitive, cv_threshold_specific=cv_threshold_specific, classification_policy=classification_policy)
        self.ectopy_detector = EctopyDetector()
        self.sinus_interpolator = SinusInterpolator(max_ectopy_ratio=max_ectopy_ratio)
        self.afib_filter = AFibFilter()
        self.constant_rr_std_epsilon = float(constant_rr_std_epsilon)
        self.interpolation_variance_ratio_min = float(interpolation_variance_ratio_min)
        log_debug('[Layer 5] RhythmProcessor initialized')

    def process_window(self, window: Dict[str, Any]) -> Dict[str, Any]:
        rr = np.asarray(window['rr'], dtype=float)
        rr_ts = window.get('rr_ts', None)
        rhythm_result = self.classifier.classify_rhythm(rr, rr_ts)
        rhythm_type = rhythm_result['rhythm_type']
        confidence = rhythm_result['confidence']
        features = rhythm_result['features']
        method = rhythm_result['method']
        ectopic_mask, ectopy_rate = self.ectopy_detector.detect_ectopy(rr)
        if rhythm_type in ('Sinus', 'Sinus_HV'):
            rr_final, success, reason = self.sinus_interpolator.interpolate(rr, ectopic_mask)
            rr_ts_final = rr_ts
            interpolation_success = success
            interpolation_applied = bool(success and reason == 'success')
            if not success:
                quality = 'rejected_high_ectopy'
            else:
                quality = 'accepted'
        elif rhythm_type == 'AFib':
            rr_final, rr_ts_final, acceptable, reason = self.afib_filter.filter_artifacts(rr, rr_ts)
            interpolation_success = True
            interpolation_applied = False
            if not acceptable:
                quality = 'rejected_too_many_artifacts'
            else:
                quality = 'accepted'
        else:
            rr_final = rr
            rr_ts_final = rr_ts
            reason = 'not_applicable_uncertain'
            interpolation_success = False
            interpolation_applied = False
            quality = 'uncertain'
        rr_final = np.asarray(rr_final, dtype=float)
        raw_stats = self._rr_statistics(rr)
        final_stats = self._rr_statistics(rr_final)
        raw_std = raw_stats['std']
        final_std = final_stats['std']
        variance_ratio = final_std / raw_std if np.isfinite(raw_std) and raw_std > self.constant_rr_std_epsilon else np.nan
        near_constant_raw = bool(np.isfinite(raw_std) and raw_std < self.constant_rr_std_epsilon)
        near_constant_final = bool(np.isfinite(final_std) and final_std < self.constant_rr_std_epsilon)
        variance_collapse = bool(interpolation_applied and np.isfinite(variance_ratio) and (variance_ratio < self.interpolation_variance_ratio_min))
        if quality == 'accepted' and variance_collapse:
            quality = 'rejected_interpolation_variance_collapse'
            reason = 'interpolation_variance_collapse'
        elif quality == 'accepted' and near_constant_final:
            quality = 'rejected_near_constant_rr'
            reason = 'near_constant_rr'
        result = {**window, 'rhythm_type': rhythm_type, 'rhythm_type_legacy': rhythm_type, 'rhythm_label_semantic': 'AF_like' if rhythm_type == 'AFib' else rhythm_type, 'confidence': confidence, 'method': method, 'features': features, 'rr_final': rr_final, 'rr_ts_final': rr_ts_final, 'interpolated': interpolation_applied, 'interpolation_applied': interpolation_applied, 'interpolation_success': interpolation_success, 'processing_reason': reason, 'ectopy_rate': ectopy_rate, 'quality': quality, 'rr_mean_raw_window': raw_stats['mean'], 'rr_std_raw_window': raw_std, 'rr_cv_raw_window': raw_stats['cv'], 'rr_mean_final_window': final_stats['mean'], 'rr_std_final_window': final_std, 'rr_cv_final_window': final_stats['cv'], 'rr_variance_ratio': variance_ratio, 'rr_near_constant_raw': near_constant_raw, 'rr_near_constant_final': near_constant_final, 'interpolation_variance_collapse': variance_collapse, 'classification_policy': rhythm_result.get('classification_policy'), 'score_type': rhythm_result.get('score_type')}
        return result

    @staticmethod
    def _rr_statistics(rr: np.ndarray) -> Dict[str, float]:
        finite = np.asarray(rr, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return {'mean': np.nan, 'std': np.nan, 'cv': np.nan}
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        cv = std / mean if mean != 0 else np.nan
        return {'mean': mean, 'std': std, 'cv': cv}

    def process_batch(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for win in windows:
            try:
                result = self.process_window(win)
                results.append(result)
            except Exception as e:
                log_warning(f"[Layer 5] Failed to process window {win.get('window_idx', 'N/A')}: {str(e)}")
                results.append({**win, 'rhythm_type': 'Error', 'confidence': 0.0, 'quality': 'error', 'error': str(e)})
        n_sinus = sum((1 for r in results if r.get('rhythm_type') == 'Sinus'))
        n_sinus_hv = sum((1 for r in results if r.get('rhythm_type') == 'Sinus_HV'))
        n_afib = sum((1 for r in results if r.get('rhythm_type') == 'AFib'))
        n_uncertain = sum((1 for r in results if r.get('rhythm_type') == 'Uncertain'))
        n_accepted = sum((1 for r in results if r.get('quality') == 'accepted'))
        log_info(f'[Layer 5] Processed {len(results)} windows: Sinus={n_sinus}, Sinus_HV={n_sinus_hv}, AFib={n_afib}, Uncertain={n_uncertain}, Accepted={n_accepted}')
        return results

    def get_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(results) == 0:
            return {}
        rhythm_counts = {}
        for r in results:
            rt = r.get('rhythm_type', 'Unknown')
            rhythm_counts[rt] = rhythm_counts.get(rt, 0) + 1
        quality_counts = {}
        for r in results:
            q = r.get('quality', 'unknown')
            quality_counts[q] = quality_counts.get(q, 0) + 1
        confidences = [r.get('confidence', 0) for r in results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        ectopy_rates = [r.get('ectopy_rate', 0) for r in results]
        avg_ectopy = np.mean(ectopy_rates) if ectopy_rates else 0.0
        return {'n_windows': len(results), 'rhythm_counts': rhythm_counts, 'quality_counts': quality_counts, 'avg_confidence': avg_confidence, 'avg_ectopy_rate': avg_ectopy}
