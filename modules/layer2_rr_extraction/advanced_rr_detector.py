"""Provides advanced rr detector functions for the ECG-to-entropy pipeline."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import numpy as np
from scipy import signal
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_debug, log_warning

class AdvancedRRExtractor:
    pass
    VALID_MODES = {'robust', 'legacy'}
    VALID_POLARITY_MODES = {'auto', 'positive', 'negative'}

    def __init__(self, fs: float=200.0, detector_mode: str='robust', chunk_duration_sec: float=3600.0, chunk_overlap_sec: float=10.0, polarity_mode: str='auto', neurokit_method: str='neurokit'):
        if fs <= 0:
            raise ValueError('fs must be positive')
        if detector_mode not in self.VALID_MODES:
            raise ValueError(f'detector_mode must be one of {sorted(self.VALID_MODES)}')
        if polarity_mode not in self.VALID_POLARITY_MODES:
            raise ValueError(f'polarity_mode must be one of {sorted(self.VALID_POLARITY_MODES)}')
        if chunk_duration_sec <= 0:
            raise ValueError('chunk_duration_sec must be positive')
        if chunk_overlap_sec < 0 or chunk_overlap_sec * 2 >= chunk_duration_sec:
            raise ValueError('chunk_overlap_sec must be non-negative and shorter than half a chunk')
        self.fs = float(fs)
        self.detector_mode = detector_mode
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.chunk_overlap_sec = float(chunk_overlap_sec)
        self.polarity_mode = polarity_mode
        self.neurokit_method = neurokit_method
        self.last_diagnostics: Dict[str, Any] = {}

    def preprocess_ecg(self, ecg: np.ndarray) -> np.ndarray:
        pass
        sig = np.asarray(ecg, dtype=float)
        if len(sig) > int(self.fs * 3):
            try:
                sos_hp = signal.butter(2, 0.5, btype='high', fs=self.fs, output='sos')
                sig = signal.sosfiltfilt(sos_hp, sig)
            except Exception:
                sig = signal.detrend(sig)
        if len(sig) > int(self.fs):
            try:
                sos_lp = signal.butter(4, 40.0, btype='low', fs=self.fs, output='sos')
                sig = signal.sosfiltfilt(sos_lp, sig)
            except Exception:
                pass
        if np.std(sig) > 0:
            sig = (sig - np.mean(sig)) / np.std(sig)
        return sig

    def enhance_signal(self, ecg: np.ndarray) -> np.ndarray:
        pass
        diff1 = np.diff(ecg, prepend=ecg[0])
        diff2 = np.diff(diff1, prepend=diff1[0])
        combined_diff = diff1 + 0.5 * diff2
        squared = np.sign(combined_diff) * combined_diff ** 2
        win_size = max(1, int(0.08 * self.fs))
        kernel = np.ones(win_size) / win_size
        return np.convolve(squared, kernel, mode='same')

    def adaptive_threshold(self, enhanced_signal: np.ndarray, learning_rate: float=0.125) -> Tuple[np.ndarray, np.ndarray]:
        pass
        n = len(enhanced_signal)
        threshold1 = np.zeros(n)
        threshold2 = np.zeros(n)
        init_samples = min(int(2 * self.fs), n // 4)
        if init_samples > 10:
            signal_peak = np.percentile(enhanced_signal[:init_samples], 92)
            noise_peak = np.percentile(enhanced_signal[:init_samples], 50)
        else:
            signal_peak = np.max(enhanced_signal) * 0.5
            noise_peak = np.median(enhanced_signal)
        spki = signal_peak
        npki = noise_peak
        for i in range(n):
            threshold1[i] = npki + 0.25 * (spki - npki)
            threshold2[i] = 0.5 * threshold1[i]
            if 0 < i < n - 1:
                is_local_peak = enhanced_signal[i] > enhanced_signal[i - 1] and enhanced_signal[i] > enhanced_signal[i + 1]
                if is_local_peak:
                    if enhanced_signal[i] > threshold1[i]:
                        spki = learning_rate * enhanced_signal[i] + (1 - learning_rate) * spki
                    else:
                        npki = learning_rate * enhanced_signal[i] + (1 - learning_rate) * npki
        return (threshold1, threshold2)

    def find_r_peaks(self, ecg: np.ndarray, min_distance: float=0.3) -> np.ndarray:
        pass
        ecg_array = self._prepare_signal(ecg)
        if len(ecg_array) < int(self.fs * 0.5):
            self.last_diagnostics = self._empty_diagnostics(len(ecg_array), 'signal_too_short')
            return np.array([], dtype=int)
        if min_distance <= 0:
            raise ValueError('min_distance must be positive')
        if self.detector_mode == 'legacy':
            peaks = self._find_r_peaks_legacy(ecg_array, min_distance=min_distance)
            self.last_diagnostics = {'detector_mode': 'legacy', 'detector_method': 'signed_energy_adaptive_threshold', 'polarity_mode': 'positive', 'selected_polarity': 'positive', 'chunk_duration_sec': None, 'chunk_overlap_sec': None, 'n_chunks': 1, 'n_chunks_detected': int(len(peaks) > 0), 'n_chunk_failures': int(len(peaks) == 0), 'n_negative_polarity_chunks': 0, 'n_input_samples': int(len(ecg_array)), 'n_nonfinite_repaired': int(getattr(self, '_n_nonfinite_repaired', 0)), 'transient_prefix_sec': self._estimate_transient_prefix(ecg_array), 'n_peaks_before_distance_filter': int(len(peaks)), 'n_peaks': int(len(peaks))}
            return peaks
        return self._find_r_peaks_robust(ecg_array, min_distance=min_distance)

    def _prepare_signal(self, ecg: np.ndarray) -> np.ndarray:
        values = np.asarray(ecg, dtype=float).reshape(-1)
        finite = np.isfinite(values)
        self._n_nonfinite_repaired = int(np.sum(~finite))
        if np.all(finite):
            return values
        if not np.any(finite):
            return np.zeros_like(values)
        repaired = values.copy()
        indices = np.arange(len(values))
        repaired[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
        return repaired

    def _find_r_peaks_robust(self, ecg: np.ndarray, min_distance: float) -> np.ndarray:
        chunk_samples = max(int(self.chunk_duration_sec * self.fs), int(10 * self.fs))
        overlap_samples = int(self.chunk_overlap_sec * self.fs)
        all_peaks: List[int] = []
        chunk_methods: List[str] = []
        selected_polarities: List[str] = []
        n_chunks = 0
        n_chunk_failures = 0
        for core_start in range(0, len(ecg), chunk_samples):
            core_end = min(len(ecg), core_start + chunk_samples)
            padded_start = max(0, core_start - overlap_samples)
            padded_end = min(len(ecg), core_end + overlap_samples)
            chunk = ecg[padded_start:padded_end]
            n_chunks += 1
            peaks_local, polarity, method = self._detect_robust_chunk(chunk)
            if len(peaks_local) == 0:
                n_chunk_failures += 1
                continue
            peaks_global = peaks_local + padded_start
            keep = (peaks_global >= core_start) & (peaks_global < core_end)
            kept = peaks_global[keep]
            if len(kept) == 0:
                n_chunk_failures += 1
                continue
            all_peaks.extend(kept.astype(int).tolist())
            selected_polarities.append(polarity)
            chunk_methods.append(method)
        peaks_before_filter = np.asarray(sorted(set(all_peaks)), dtype=int)
        peaks = self._enforce_min_distance(peaks_before_filter, ecg, min_distance)
        method_set = sorted(set(chunk_methods))
        if not method_set:
            detector_method = 'none'
        elif len(method_set) == 1:
            detector_method = method_set[0]
        else:
            detector_method = 'hybrid:' + '+'.join(method_set)
        if selected_polarities and all((p == selected_polarities[0] for p in selected_polarities)):
            selected_polarity = selected_polarities[0]
        elif selected_polarities:
            selected_polarity = 'mixed'
        else:
            selected_polarity = 'none'
        self.last_diagnostics = {'detector_mode': 'robust', 'detector_method': detector_method, 'polarity_mode': self.polarity_mode, 'selected_polarity': selected_polarity, 'chunk_duration_sec': self.chunk_duration_sec, 'chunk_overlap_sec': self.chunk_overlap_sec, 'n_chunks': int(n_chunks), 'n_chunks_detected': int(n_chunks - n_chunk_failures), 'n_chunk_failures': int(n_chunk_failures), 'n_negative_polarity_chunks': int(sum((p == 'negative' for p in selected_polarities))), 'n_input_samples': int(len(ecg)), 'n_nonfinite_repaired': int(getattr(self, '_n_nonfinite_repaired', 0)), 'transient_prefix_sec': self._estimate_transient_prefix(ecg), 'n_peaks_before_distance_filter': int(len(peaks_before_filter)), 'n_peaks': int(len(peaks))}
        return peaks

    def _detect_robust_chunk(self, ecg: np.ndarray) -> Tuple[np.ndarray, str, str]:
        polarity_candidates = {'positive': ecg, 'negative': -ecg}
        if self.polarity_mode != 'auto':
            polarity_candidates = {self.polarity_mode: polarity_candidates[self.polarity_mode]}
        candidates: List[Tuple[float, str, np.ndarray]] = []
        for polarity, candidate_signal in polarity_candidates.items():
            peaks = self._detect_chunk_neurokit(candidate_signal)
            score = self._score_peak_train(peaks, len(ecg))
            candidates.append((score, polarity, peaks))
        candidates.sort(key=lambda item: (item[0], item[1] == 'positive'), reverse=True)
        best_score, best_polarity, best_peaks = candidates[0]
        if len(best_peaks) >= 2 and best_score > 0:
            return (best_peaks, best_polarity, f'neurokit2:{self.neurokit_method}')
        fallback_peaks = self._detect_chunk_energy(ecg)
        if len(fallback_peaks) >= 2:
            return (fallback_peaks, 'polarity_invariant', 'energy_fallback')
        return (np.array([], dtype=int), 'none', 'none')

    def _detect_chunk_neurokit(self, ecg: np.ndarray) -> np.ndarray:
        try:
            import neurokit2 as nk
            _, info = nk.ecg_peaks(np.asarray(ecg, dtype=float), sampling_rate=self.fs, method=self.neurokit_method, correct_artifacts=False)
            return np.asarray(info.get('ECG_R_Peaks', []), dtype=int)
        except Exception as exc:
            log_warning(f'[Layer 2] NeuroKit2 chunk detection failed: {exc}')
            return np.array([], dtype=int)

    def _detect_chunk_energy(self, ecg: np.ndarray) -> np.ndarray:
        pass
        clean = self.preprocess_ecg(ecg)
        derivative = np.diff(clean, prepend=clean[0])
        energy = derivative ** 2
        win_size = max(1, int(0.08 * self.fs))
        integrated = np.convolve(energy, np.ones(win_size) / win_size, mode='same')
        median = float(np.median(integrated))
        mad = float(np.median(np.abs(integrated - median)))
        threshold = median + 5.0 * max(mad, np.finfo(float).eps)
        candidates, _ = signal.find_peaks(integrated, height=threshold, distance=max(1, int(0.25 * self.fs)))
        search_window = max(1, int(0.08 * self.fs))
        refined = []
        for peak in candidates:
            start = max(0, peak - search_window)
            end = min(len(ecg), peak + search_window + 1)
            local = ecg[start:end]
            if len(local) == 0:
                continue
            baseline = np.median(local)
            refined.append(start + int(np.argmax(np.abs(local - baseline))))
        return np.asarray(sorted(set(refined)), dtype=int)

    def _score_peak_train(self, peaks: np.ndarray, n_samples: int) -> float:
        if len(peaks) < 2 or n_samples <= 0:
            return 0.0
        rr = np.diff(peaks) / self.fs
        valid = (rr >= 0.3) & (rr <= 2.0)
        valid_fraction = float(np.mean(valid)) if len(rr) else 0.0
        coverage = float((peaks[-1] - peaks[0]) / max(n_samples - 1, 1))
        if np.any(valid):
            median_hr = 60.0 / float(np.median(rr[valid]))
            hr_score = 1.0 if 30.0 <= median_hr <= 180.0 else 0.0
        else:
            hr_score = 0.0
        return 0.55 * valid_fraction + 0.35 * np.clip(coverage, 0.0, 1.0) + 0.1 * hr_score

    def _enforce_min_distance(self, peaks: np.ndarray, ecg: np.ndarray, min_distance: float) -> np.ndarray:
        if len(peaks) < 2:
            return peaks.astype(int)
        min_samples = max(1, int(min_distance * self.fs))
        kept: List[int] = []
        context = max(1, int(0.1 * self.fs))
        for peak_value in peaks:
            peak = int(peak_value)
            if not kept or peak - kept[-1] >= min_samples:
                kept.append(peak)
                continue
            previous = kept[-1]
            start = max(0, min(previous, peak) - context)
            end = min(len(ecg), max(previous, peak) + context + 1)
            baseline = float(np.median(ecg[start:end]))
            previous_amplitude = abs(float(ecg[previous]) - baseline)
            current_amplitude = abs(float(ecg[peak]) - baseline)
            if current_amplitude > previous_amplitude:
                kept[-1] = peak
        return np.asarray(kept, dtype=int)

    def _estimate_transient_prefix(self, ecg: np.ndarray) -> float:
        pass
        block_samples = max(1, int(2 * self.fs))
        probe_samples = min(len(ecg), int(60 * self.fs))
        if probe_samples < block_samples * 5:
            return 0.0
        amplitudes = []
        for start in range(0, probe_samples - block_samples + 1, block_samples):
            block = ecg[start:start + block_samples]
            amplitudes.append(float(np.percentile(block, 99) - np.percentile(block, 1)))
        amplitudes_array = np.asarray(amplitudes)
        baseline = float(np.median(amplitudes_array[min(5, len(amplitudes_array) - 1):]))
        if baseline <= 0:
            return 0.0
        transient_blocks = 0
        for amplitude in amplitudes_array:
            if amplitude > 4.0 * baseline:
                transient_blocks += 1
            else:
                break
        return float(transient_blocks * block_samples / self.fs)

    def _empty_diagnostics(self, n_samples: int, reason: str) -> Dict[str, Any]:
        return {'detector_mode': self.detector_mode, 'detector_method': 'none', 'polarity_mode': self.polarity_mode, 'selected_polarity': 'none', 'chunk_duration_sec': self.chunk_duration_sec, 'chunk_overlap_sec': self.chunk_overlap_sec, 'n_chunks': 0, 'n_chunks_detected': 0, 'n_chunk_failures': 0, 'n_negative_polarity_chunks': 0, 'n_input_samples': int(n_samples), 'n_nonfinite_repaired': int(getattr(self, '_n_nonfinite_repaired', 0)), 'transient_prefix_sec': 0.0, 'n_peaks_before_distance_filter': 0, 'n_peaks': 0, 'failure_reason': reason}

    def _find_r_peaks_legacy(self, ecg: np.ndarray, min_distance: float) -> np.ndarray:
        ecg_clean = self.preprocess_ecg(ecg)
        enhanced = self.enhance_signal(ecg_clean)
        threshold1, threshold2 = self.adaptive_threshold(enhanced)
        min_samples = int(min_distance * self.fs)
        try:
            peaks, _ = signal.find_peaks(enhanced, height=threshold2, distance=min_samples)
        except Exception:
            return np.array([], dtype=int)
        if len(peaks) == 0:
            return np.array([], dtype=int)
        r_peaks = []
        search_window = int(0.05 * self.fs)
        for peak in peaks:
            if enhanced[peak] < threshold1[peak]:
                continue
            start = max(0, peak - search_window)
            end = min(len(ecg), peak + search_window)
            window = ecg[start:end]
            if len(window) > 0:
                r_peaks.append(start + int(np.argmax(window)))
        return self._post_process_r_peaks(np.asarray(r_peaks, dtype=int), ecg)

    def _post_process_r_peaks(self, r_peaks: np.ndarray, ecg: np.ndarray) -> np.ndarray:
        if len(r_peaks) < 3:
            return r_peaks
        rr_intervals = np.diff(r_peaks) / self.fs
        r_amplitudes = ecg[r_peaks]
        keep_mask = np.ones(len(r_peaks), dtype=bool)
        for i, rr_interval in enumerate(rr_intervals):
            if rr_interval < 0.25 and i < len(r_peaks) - 1:
                if r_amplitudes[i] < r_amplitudes[i + 1] * 0.7:
                    keep_mask[i] = False
        n_removed = int(np.sum(~keep_mask))
        if n_removed > 0:
            log_debug(f'[Layer 2] Post-processing removed {n_removed} suspect peaks')
        return r_peaks[keep_mask]

    def assess_quality(self, r_peaks: np.ndarray, ecg: np.ndarray) -> Dict[str, float]:
        pass
        ecg = np.asarray(ecg, dtype=float).reshape(-1)
        finite = np.isfinite(ecg)
        if not np.all(finite):
            if not np.any(finite):
                ecg = np.zeros_like(ecg)
            else:
                repaired = ecg.copy()
                indices = np.arange(len(ecg))
                repaired[~finite] = np.interp(indices[~finite], indices[finite], ecg[finite])
                ecg = repaired
        if len(r_peaks) < 3:
            return {'regularity': 0.0, 'coverage': 0.0, 'beat_density': 0.0, 'valid_rr_fraction': 0.0, 'snr_estimate': 0.0, 'quality_score': 0.0}
        rr_intervals = np.diff(r_peaks) / self.fs
        physiological = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]
        valid_rr_fraction = float(len(physiological) / len(rr_intervals))
        if len(physiological) > 1 and np.mean(physiological) > 0:
            regularity = 1.0 / (1.0 + float(np.std(physiological) / np.mean(physiological)))
        else:
            regularity = 0.0
        total_time = len(ecg) / self.fs
        detected_time = (r_peaks[-1] - r_peaks[0]) / self.fs
        coverage = float(detected_time / total_time) if total_time > 0 else 0.0
        minimum_expected_beats = max(total_time / 2.0, 1.0)
        beat_density = float(np.clip(len(r_peaks) / minimum_expected_beats, 0.0, 1.0))
        r_amplitudes = ecg[r_peaks]
        signal_power = float(np.mean(r_amplitudes ** 2)) if len(r_amplitudes) else 0.0
        stride = max(1, len(ecg) // 200000)
        background = np.asarray(ecg[::stride], dtype=float)
        noise_power = float(np.mean(background ** 2)) if len(background) else 0.0
        snr = 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0
        snr_normalized = float(np.clip(snr / 20.0, 0.0, 1.0))
        quality_score = 0.25 * regularity + 0.25 * np.clip(coverage, 0.0, 1.0) + 0.25 * beat_density + 0.15 * valid_rr_fraction + 0.1 * snr_normalized
        density_gate = float(np.clip(beat_density / 0.5, 0.0, 1.0))
        quality_score *= density_gate * valid_rr_fraction
        return {'regularity': float(regularity), 'coverage': float(coverage), 'beat_density': beat_density, 'valid_rr_fraction': valid_rr_fraction, 'snr_estimate': float(snr), 'quality_score': float(quality_score)}

    def extract_rr(self, ecg: np.ndarray, return_quality: bool=True, min_distance: float=0.3, rr_range: tuple=(0.3, 2.0)) -> Dict[str, Any]:
        r_peaks = self.find_r_peaks(ecg, min_distance=min_distance)
        diagnostics = dict(self.last_diagnostics)
        if len(r_peaks) < 2:
            result: Dict[str, Any] = {'rr_ts': np.array([]), 'rr': np.array([]), 'rr_ts_raw': np.array([]), 'rr_raw': np.array([]), 'rr_physiological_mask': np.array([], dtype=bool), 'r_peaks': r_peaks, 'n_filtered': 0, 'diagnostics': diagnostics}
            if return_quality:
                result['quality'] = self.assess_quality(r_peaks, np.asarray(ecg))
            return result
        r_times = r_peaks / self.fs
        rr_values = np.diff(r_times)
        rr_times = r_times[1:]
        raw_valid_mask = np.isfinite(rr_values) & (rr_values > 0)
        rr_raw = rr_values[raw_valid_mask]
        rr_times_raw = rr_times[raw_valid_mask]
        rr_min, rr_max = rr_range
        physiological_mask = (rr_raw >= rr_min) & (rr_raw <= rr_max)
        rr_filtered = rr_raw[physiological_mask]
        rr_times_filtered = rr_times_raw[physiological_mask]
        n_filtered = int(len(rr_raw) - len(rr_filtered))
        if n_filtered > 0:
            log_debug(f'[Layer 2] Filtered {n_filtered} non-physiological RR intervals')
        diagnostics.update({'n_rr_before_physiological_filter': int(len(rr_values)), 'n_rr_raw': int(len(rr_raw)), 'n_rr': int(len(rr_filtered)), 'n_rr_filtered': n_filtered, 'rr_valid_fraction': float(len(rr_filtered) / len(rr_values)) if len(rr_values) else 0.0})
        result = {'rr_ts': rr_times_filtered, 'rr': rr_filtered, 'rr_ts_raw': rr_times_raw, 'rr_raw': rr_raw, 'rr_physiological_mask': physiological_mask, 'r_peaks': r_peaks, 'n_filtered': n_filtered, 'diagnostics': diagnostics}
        if return_quality:
            result['quality'] = self.assess_quality(r_peaks, np.asarray(ecg))
        return result

def extract_rr_from_ecg(ecg_signal: np.ndarray, fs: float=200.0, detector_mode: str='robust') -> Dict[str, np.ndarray]:
    extractor = AdvancedRRExtractor(fs=fs, detector_mode=detector_mode)
    return extractor.extract_rr(ecg_signal, return_quality=True)
