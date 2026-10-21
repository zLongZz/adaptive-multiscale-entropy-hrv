"""Provides record rr exporter functions for the ECG-to-entropy pipeline."""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Dict, Iterator
import numpy as np

class RecordRRExporter:
    pass

    def __init__(self, output_dir: str, filename: str='record_rr_cache.pklstream', enabled: bool=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename
        self.enabled = enabled
        self.n_records = 0

    def append_record(self, record: Dict) -> int:
        if not self.enabled:
            return 0
        payload = _compact_record(record)
        with self.path.open('ab') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.n_records += 1
        return 1

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
                        raise RuntimeError(f'Corrupt record RR stream after {n_kept} retained records: {self.path}') from exc
                    if str(payload.get('ecg_id')) in completed_ids:
                        pickle.dump(payload, target, protocol=pickle.HIGHEST_PROTOCOL)
                        n_kept += 1
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        os.replace(temp_path, self.path)
        self.n_records = n_kept
        return n_kept

def _compact_record(record: Dict) -> Dict:
    payload = dict(record)
    float_arrays = ('rr_raw', 'rr_ts_raw', 'rr_physiological', 'rr_ts_physiological')
    for key in float_arrays:
        payload[key] = np.asarray(payload.get(key, []), dtype=np.float32)
    payload['r_peaks'] = np.asarray(payload.get('r_peaks', []), dtype=np.int32)
    payload['rr_physiological_mask'] = np.asarray(payload.get('rr_physiological_mask', []), dtype=bool)
    payload['n_rr_raw'] = int(len(payload['rr_raw']))
    payload['n_rr_physiological'] = int(len(payload['rr_physiological']))
    payload['n_r_peaks'] = int(len(payload['r_peaks']))
    return payload

def iter_record_rr(path: str) -> Iterator[Dict]:
    with open(path, 'rb') as handle:
        while True:
            try:
                yield pickle.load(handle)
            except EOFError:
                break
