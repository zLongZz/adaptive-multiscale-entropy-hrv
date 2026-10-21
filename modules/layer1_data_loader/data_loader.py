"""Provides data loader functions for the ECG-to-entropy pipeline."""
from typing import Dict, Optional, Any, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_info, log_debug, log_error
from modules.core.utils import save_pickle
from .locator_strategies import FileLocatorStrategy, create_locator
from .ecg_reader import read_ecg_complete, get_lead_dataframe
from .hea_parser import parse_hea_info

class ECGDataLoader:

    def __init__(self, locator: FileLocatorStrategy):
        self.locator = locator
        log_info(f'[Layer 1] ECGDataLoader initialized with {type(locator).__name__}')

    @classmethod
    def from_config(cls, config):
        data_source = config.paths.data_source
        if data_source == 'formal':
            locator_config = {'data_root': config.paths.formal_data_root, 'metadata_excel': config.paths.formal_metadata_excel, 'sampling_config': {'mode': config.metadata.sampling_mode, 'stratified': config.metadata.stratified_sampling, 'manual': config.metadata.manual_sampling, 'random_seed': config.metadata.sampling_random_seed}}
        else:
            locator_config = {'data_root': config.paths.preliminary_data_root}
        locator = create_locator(data_source, locator_config)
        log_info(f'[Layer 1] Created loader for data source: {data_source}')
        return cls(locator)

    def load_single_sample(self, ecg_id: str, leads: Optional[List[str]]=None) -> Optional[Dict[str, Any]]:
        log_info(f'[Layer 1] Loading sample: {ecg_id}')
        file_info = self.locator.find_by_id(ecg_id)
        if file_info['csv'] is None:
            log_error(f"[Layer 1] CSV not found for {ecg_id}: {file_info.get('error', 'Unknown error')}")
            return None
        try:
            ecg_df = read_ecg_complete(file_info['csv'], lead_names=leads)
            if ecg_df.empty:
                log_error(f'[Layer 1] Empty CSV for {ecg_id}')
                return None
            lead_data = get_lead_dataframe(ecg_df)
            hea_info = parse_hea_info(file_info['hea'])
            sampling_rate = hea_info.get('sampling_rate')
            csv_duration_seconds = len(ecg_df) / sampling_rate if sampling_rate else None
            hea_duration_seconds = hea_info.get('duration_seconds')
            duration_delta_seconds = csv_duration_seconds - hea_duration_seconds if csv_duration_seconds is not None and hea_duration_seconds is not None else None
            metadata = file_info.get('metadata', {})
            log_info(f'[Layer 1] Loaded {ecg_id}: {ecg_df.shape[0]} samples, {len(lead_data)} leads')
            return {'ecg_id': ecg_id, 'group': file_info.get('group'), 'group_code': file_info.get('group_code'), 'nyha_grade': file_info.get('nyha_grade'), 'analysis_population': file_info.get('analysis_population'), 'patient_uid': file_info.get('patient_uid'), 'is_death': file_info.get('is_death', 0), 'center_name': file_info.get('center_name'), 'ecg_data': ecg_df, 'lead_data': lead_data, 'hea_info': hea_info, 'csv_n_samples': int(len(ecg_df)), 'csv_duration_seconds': csv_duration_seconds, 'duration_delta_seconds': duration_delta_seconds, 'metadata': metadata, 'file_paths': {'csv': file_info['csv'], 'hea': file_info['hea']}}
        except Exception as e:
            log_error(f'[Layer 1] Failed to load {ecg_id}: {str(e)}')
            return None

    def load_multiple_samples(self, ecg_ids: List[str]) -> Dict[str, Dict]:
        results = {}
        log_info(f'[Layer 1] Starting batch loading: {len(ecg_ids)} samples')
        for i, ecg_id in enumerate(ecg_ids):
            if (i + 1) % 10 == 0 or i + 1 == len(ecg_ids):
                log_info(f'[Layer 1] Loading progress: {i + 1}/{len(ecg_ids)}')
            data = self.load_single_sample(ecg_id)
            if data is not None:
                results[ecg_id] = data
        log_info(f'[Layer 1] Loaded {len(results)}/{len(ecg_ids)} samples successfully')
        return results

    def load_all_samples(self) -> Dict[str, Dict]:
        all_ids = self.locator.list_all_ids()
        log_info(f'[Layer 1] Loading all available samples: {len(all_ids)}')
        return self.load_multiple_samples(all_ids)

    def get_available_ids(self) -> List[str]:
        return self.locator.list_all_ids()

    def get_statistics(self) -> Dict:
        if hasattr(self.locator, 'get_statistics'):
            return self.locator.get_statistics()
        else:
            all_ids = self.locator.list_all_ids()
            return {'total_samples': len(all_ids), 'data_source': 'preliminary'}
