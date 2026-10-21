"""Provides window segmenter functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from modules.core.logger import log_info, log_debug, log_warning

class WindowSegmenter:

    def __init__(self, window_sec: int=900, overlap_ratio: float=0.5, min_beats: int=1000):
        self.window_sec = window_sec
        self.overlap_ratio = overlap_ratio
        self.step_sec = int(window_sec * (1 - overlap_ratio))
        self.min_beats = min_beats
        log_debug(f'[Layer 3] WindowSegmenter initialized: window={window_sec}s, overlap={overlap_ratio}, step={self.step_sec}s, min_beats={min_beats}')

    def segment_windows_with_inventory(self, rr: np.ndarray, rr_ts: np.ndarray, test_datetime: datetime, total_duration_sec: float=None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(rr) != len(rr_ts):
            raise ValueError(f'RR/RR timestamp length mismatch: {len(rr)} != {len(rr_ts)}')
        rr_duration = float(rr_ts[-1]) if len(rr_ts) else 0.0
        total_duration = float(total_duration_sec) if total_duration_sec is not None and total_duration_sec > 0 else rr_duration
        start_times = []
        current_start = 0
        while current_start + self.window_sec <= total_duration:
            start_times.append(current_start)
            current_start += self.step_sec
        log_debug(f'[Layer 3] Total duration: {total_duration / 3600:.2f}h, Generated {len(start_times)} candidate windows')
        windows = []
        inventory = []
        for idx, start_t in enumerate(start_times):
            end_t = start_t + self.window_sec
            start_idx = np.searchsorted(rr_ts, start_t, side='left')
            end_idx = np.searchsorted(rr_ts, end_t, side='right')
            window_rr = rr[start_idx:end_idx]
            window_rr_ts = rr_ts[start_idx:end_idx]
            center_t = start_t + self.window_sec / 2
            window_datetime = test_datetime + timedelta(seconds=center_t)
            clock_hour = window_datetime.hour
            rr_eligible = len(window_rr) >= self.min_beats
            base = {'window_idx': idx, 'start_time': start_t, 'end_time': end_t, 'timestamp': window_datetime, 'clock_hour': clock_hour, 'n_beats': len(window_rr), 'layer3_eligible': bool(rr_eligible), 'layer3_reject_reason': '' if rr_eligible else 'insufficient_beats'}
            inventory.append(base.copy())
            if rr_eligible:
                windows.append({**base, 'rr': window_rr.copy(), 'rr_ts': window_rr_ts.copy()})
        log_info(f'[Layer 3] Segmented {len(windows)} valid windows (from {len(start_times)} candidates, min_beats={self.min_beats})')
        return (windows, inventory)

    def segment_windows(self, rr: np.ndarray, rr_ts: np.ndarray, test_datetime: datetime) -> List[Dict[str, Any]]:
        windows, _ = self.segment_windows_with_inventory(rr, rr_ts, test_datetime)
        return windows

    def segment_multi_lead(self, rr_data: Dict[str, Dict], test_datetime: datetime, ecg_id: str=None, group_code: int=None, reference_lead: str='II') -> List[Dict[str, Any]]:
        if not rr_data:
            log_warning('[Layer 3] rr_data is empty, no windows to segment')
            return []
        if reference_lead not in rr_data:
            log_warning(f"[Layer 3] Reference lead '{reference_lead}' not found, using first available lead")
            reference_lead = list(rr_data.keys())[0]
        ref_rr = rr_data[reference_lead]['rr']
        ref_rr_ts = rr_data[reference_lead]['rr_ts']
        log_info(f"[Layer 3] Using lead '{reference_lead}' as time reference ({len(ref_rr)} RR intervals)")
        time_windows = self.segment_windows(ref_rr, ref_rr_ts, test_datetime)
        log_info(f'[Layer 3] Generated {len(time_windows)} time windows, applying to all {len(rr_data)} leads')
        all_windows = []
        for lead_name, lead_data in rr_data.items():
            lead_rr = lead_data['rr']
            lead_rr_ts = lead_data['rr_ts']
            for win in time_windows:
                start_t = win['start_time']
                end_t = win['end_time']
                start_idx = np.searchsorted(lead_rr_ts, start_t, side='left')
                end_idx = np.searchsorted(lead_rr_ts, end_t, side='right')
                window_rr = lead_rr[start_idx:end_idx]
                window_rr_ts = lead_rr_ts[start_idx:end_idx]
                if len(window_rr) < self.min_beats:
                    log_debug(f"[Layer 3] Lead {lead_name}, window {win['window_idx']}: only {len(window_rr)} beats (< {self.min_beats})")
                window_dict = {'lead': lead_name, 'window_idx': win['window_idx'], 'start_time': start_t, 'end_time': end_t, 'timestamp': win['timestamp'], 'clock_hour': win['clock_hour'], 'rr': window_rr.copy(), 'rr_ts': window_rr_ts.copy(), 'n_beats': len(window_rr)}
                if ecg_id is not None:
                    window_dict['ecg_id'] = ecg_id
                if group_code is not None:
                    window_dict['group_code'] = group_code
                all_windows.append(window_dict)
        log_info(f'[Layer 3] Total windows across all leads: {len(all_windows)}')
        return all_windows
