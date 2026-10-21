"""Provides window rr exporter functions for the ECG-to-entropy pipeline."""
import pickle
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional
import numpy as np

class WindowRRExporter:

    def __init__(self, output_dir: str, filename: str='window_rr_final.pklstream', enabled: bool=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename
        self.enabled = enabled
        self.n_records = 0

    def append_records(self, records: Iterable[Dict]) -> int:
        if not self.enabled:
            return 0
        n_written = 0
        with open(self.path, 'ab') as f:
            for record in records:
                payload = dict(record)
                rr_final = np.asarray(payload.get('rr_final', []), dtype=np.float32)
                payload['rr_final'] = rr_final
                payload['n_rr_final'] = int(len(rr_final))
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                n_written += 1
        self.n_records += n_written
        return n_written

    def retain_ecg_ids(self, completed_ids: set[str]) -> int:
        pass
        if not self.enabled or not self.path.exists():
            return 0
        temp_path = self.path.with_suffix(self.path.suffix + '.resume_tmp')
        n_kept = 0
        try:
            with self.path.open('rb') as source, temp_path.open('wb') as target:
                while True:
                    try:
                        payload = pickle.load(source)
                    except EOFError:
                        break
                    except (pickle.UnpicklingError, AttributeError, ValueError) as exc:
                        raise RuntimeError(f'Corrupt window RR stream after {n_kept} retained records: {self.path}') from exc
                    if str(payload.get('ecg_id')) in completed_ids:
                        pickle.dump(payload, target, protocol=pickle.HIGHEST_PROTOCOL)
                        n_kept += 1
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        os.replace(temp_path, self.path)
        self.n_records = n_kept
        return n_kept

def build_rr_export_record(*, ecg_id: str, lead: str, window: Dict, rr_final: np.ndarray, rhythm_type: str, rhythm_confidence: float, rhythm_method: str, rr_raw: Optional[np.ndarray]=None, rr_ts_raw: Optional[np.ndarray]=None, rr_ts_final: Optional[np.ndarray]=None, rhythm_features: Optional[Dict]=None, sinus_confidence_score: Optional[float]=None, quality_tier: Optional[str]=None, analysis_track: Optional[str]=None, rhythm_quality: Optional[str]=None, processing_reason: Optional[str]=None) -> Dict:
    rhythm_features = rhythm_features or {}
    return {'ecg_id': ecg_id, 'lead': lead, 'window_idx': int(window.get('window_idx', -1)), 'clock_hour': int(window.get('clock_hour', -1)), 'start_time': float(window.get('start_time', np.nan)), 'end_time': float(window.get('end_time', np.nan)), 'timestamp': str(window.get('timestamp', '')), 'rhythm_type': rhythm_type, 'rhythm_type_legacy': rhythm_type, 'rhythm_label_semantic': 'AF_like' if rhythm_type == 'AFib' else rhythm_type, 'rhythm_confidence': float(rhythm_confidence), 'rhythm_method': rhythm_method, 'rhythm_cosen': _safe_float(rhythm_features.get('cosen', np.nan)), 'rhythm_pRR31': _safe_float(rhythm_features.get('pRR31', np.nan)), 'rhythm_cv': _safe_float(rhythm_features.get('cv', np.nan)), 'sinus_confidence_score': _safe_float(sinus_confidence_score), 'quality_tier': quality_tier, 'analysis_track': analysis_track, 'analysis_track_legacy': 'afib_primary_v2' if analysis_track == 'af_like_exploratory_v3' else analysis_track, 'rhythm_quality': rhythm_quality, 'processing_reason': processing_reason, 'rr_raw': np.asarray(rr_raw if rr_raw is not None else window.get('rr', []), dtype=np.float32), 'rr_ts_raw': np.asarray(rr_ts_raw if rr_ts_raw is not None else window.get('rr_ts', []), dtype=np.float32), 'rr_final': np.asarray(rr_final, dtype=np.float32), 'rr_ts_final': np.asarray(rr_ts_final if rr_ts_final is not None else [], dtype=np.float32)}

def iter_records(path: str) -> Iterator[Dict]:
    with open(path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')
