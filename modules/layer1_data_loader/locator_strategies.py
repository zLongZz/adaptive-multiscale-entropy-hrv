"""Provides locator strategies functions for the ECG-to-entropy pipeline."""
from typing import Optional, Dict, List, Generator
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_info, log_warning, log_error
NYHA_ANALYSIS_GROUPS = {'NYHA_0': ('Healthy', 0), 'NYHA_I': ('NYHA I', 1), 'NYHA_II': ('NYHA II', 2), 'NYHA_III': ('NYHA III', 3), 'NYHA_IV': ('NYHA IV', 4)}

def _analysis_group_from_nyha(nyha_grade) -> tuple:
    pass
    if pd.isna(nyha_grade):
        return ('Unknown', None)
    normalized = str(nyha_grade).strip().upper().replace(' ', '_')
    aliases = {'0': 'NYHA_0', 'HEALTHY': 'NYHA_0', 'NYHA_1': 'NYHA_I', 'NYHA_2': 'NYHA_II', 'NYHA_3': 'NYHA_III', 'NYHA_4': 'NYHA_IV'}
    return NYHA_ANALYSIS_GROUPS.get(aliases.get(normalized, normalized), ('Unknown', None))

def _analysis_population_from_group_code(group_code) -> str:
    pass
    if group_code == 0:
        return 'Healthy'
    if group_code in (1, 2, 3, 4):
        return 'Heart Failure'
    return 'Unknown'

def _coerce_integer_code(value):
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else None

class FileLocatorStrategy(ABC):

    @abstractmethod
    def find_by_id(self, ecg_id: str) -> Dict[str, Optional[str]]:
        pass

    @abstractmethod
    def list_all_ids(self) -> List[str]:
        pass

    @abstractmethod
    def iter_all_files(self) -> Generator[Dict, None, None]:
        pass

class FormalLocator(FileLocatorStrategy):

    def __init__(self, data_root: str, metadata_excel: str, sampling_config: Optional[Dict]=None):
        self.data_root = Path(data_root)
        self.metadata_excel = metadata_excel
        self.sampling_config = sampling_config or {'mode': 'all'}
        self._load_metadata()
        self._apply_sampling()
        log_info(f'[FormalLocator] Initialized')
        log_info(f'  Data root: {self.data_root}')
        log_info(f'  Total metadata records: {len(self.metadata_full)}')
        log_info(f'  Sampled records: {len(self.metadata)}')
        log_info(f"  Sampling mode: {self.sampling_config['mode']}")

    def _load_metadata(self):
        try:
            self.metadata_full = pd.read_excel(self.metadata_excel)
            log_info(f'[FormalLocator] Loaded metadata: {len(self.metadata_full)} records')
            record_id_col = None
            for col in ['record_id', 'Record ID', '记录ID']:
                if col in self.metadata_full.columns:
                    record_id_col = col
                    break
            if record_id_col is None:
                log_error('[FormalLocator] Record ID column not found')
                raise ValueError('Record ID column not found')
            self._nyha_col = None
            for col in ['nyha_std', 'NYHA_std']:
                if col in self.metadata_full.columns:
                    self._nyha_col = col
                    break
            self._is_death_col = 'is_death' if 'is_death' in self.metadata_full.columns else None
            if self._is_death_col is None:
                log_warning("[FormalLocator] 'is_death' column not found in metadata — mortality-related analyses (H4) will be unavailable")
            if self._nyha_col is None:
                raise ValueError("NYHA column not found (tried: 'nyha_std', 'NYHA_std')")
            self._patient_uid_col = 'patient_uid' if 'patient_uid' in self.metadata_full.columns else None
            if self.metadata_full[record_id_col].duplicated().any():
                raise ValueError('Duplicate record_id values found in metadata')
            self.metadata_full.set_index(record_id_col, inplace=True)
            log_info(f"[FormalLocator] Index='{record_id_col}', NYHA='{self._nyha_col}', patient_uid={self._patient_uid_col is not None}, is_death={self._is_death_col is not None}")
            log_info('[FormalLocator] Analysis groups derive only from NYHA; group_label/is_heart_failure/is_healthy are ignored')
        except Exception as e:
            log_error(f'[FormalLocator] Failed to load metadata: {e}')
            raise

    def _apply_sampling(self):
        mode = self.sampling_config.get('mode', 'all')
        if mode == 'all':
            self.metadata = self.metadata_full.copy()
            log_info(f'[FormalLocator] Using all {len(self.metadata)} samples')
        elif mode == 'stratified':
            config = self.sampling_config.get('stratified', {})
            seed = self.sampling_config.get('random_seed', 20260126)
            self.metadata = self._stratified_sampling(config, seed)
            log_info(f'[FormalLocator] Stratified sampling: {len(self.metadata)} samples')
        elif mode == 'manual':
            config = self.sampling_config.get('manual', {})
            seed = self.sampling_config.get('random_seed', 20260126)
            self.metadata = self._manual_sampling(config, seed)
            log_info(f'[FormalLocator] Manual sampling: {len(self.metadata)} samples')
        else:
            log_warning(f'[FormalLocator] Unknown sampling mode: {mode}, using all data')
            self.metadata = self.metadata_full.copy()

    def _stratified_sampling(self, config: Dict[str, Optional[int]], seed: int) -> pd.DataFrame:
        sampled_dfs = []
        for nyha_grade, n_samples in config.items():
            subset = self.metadata_full[self.metadata_full[self._nyha_col] == nyha_grade]
            if len(subset) == 0:
                log_warning(f'[FormalLocator] No samples for {nyha_grade}')
                continue
            if n_samples is None or n_samples >= len(subset):
                sampled_dfs.append(subset)
                log_info(f'  {nyha_grade}: {len(subset)}/{len(subset)} (all)')
            else:
                sampled = subset.sample(n=n_samples, random_state=seed)
                sampled_dfs.append(sampled)
                log_info(f'  {nyha_grade}: {n_samples}/{len(subset)} (sampled)')
        if len(sampled_dfs) == 0:
            log_warning('[FormalLocator] No samples after stratified sampling, using all data')
            return self.metadata_full.copy()
        return pd.concat(sampled_dfs, axis=0)

    def _manual_sampling(self, config: Dict[str, int], seed: int) -> pd.DataFrame:
        return self._stratified_sampling(config, seed)

    def find_by_id(self, ecg_id: str) -> Dict[str, Optional[str]]:
        if ecg_id not in self.metadata.index:
            return {'csv': None, 'hea': None, 'group': None, 'metadata': None, 'error': 'Not in sampled dataset'}
        csv_file = self.data_root / f'{ecg_id}.csv'
        hea_file = self.data_root / f'{ecg_id}.hea'
        if not csv_file.exists():
            log_warning(f'[FormalLocator] CSV not found: {ecg_id}')
            return {'csv': None, 'hea': None, 'group': None, 'metadata': None, 'error': 'CSV file not found'}
        metadata_row = self.metadata.loc[ecg_id].to_dict()
        nyha = metadata_row.get(self._nyha_col, 'Unknown') if self._nyha_col else 'Unknown'
        group, derived_group_code = _analysis_group_from_nyha(nyha)
        source_group_code = metadata_row.get('group_code', None)
        source_group_code_int = _coerce_integer_code(source_group_code)
        if derived_group_code is not None and (not pd.isna(source_group_code)) and (source_group_code_int != derived_group_code):
            log_warning(f'[FormalLocator] {ecg_id}: group_code={source_group_code} conflicts with {nyha}; using {derived_group_code}')
        group_code = derived_group_code
        analysis_population = _analysis_population_from_group_code(group_code)
        is_death_value = metadata_row.get(self._is_death_col, 0) if self._is_death_col else 0
        is_death = int(not pd.isna(is_death_value) and is_death_value == 1)
        patient_uid = metadata_row.get(self._patient_uid_col, None) if self._patient_uid_col else None
        if pd.isna(patient_uid):
            patient_uid = None
        return {'csv': str(csv_file), 'hea': str(hea_file) if hea_file.exists() else None, 'group': group, 'nyha_grade': nyha, 'group_code': group_code, 'analysis_population': analysis_population, 'patient_uid': patient_uid, 'is_death': is_death, 'center_name': metadata_row.get('center_name', None), 'metadata': metadata_row, 'error': None}

    def list_all_ids(self) -> List[str]:
        return self.metadata.index.tolist()

    def iter_all_files(self) -> Generator[Dict, None, None]:
        for ecg_id in self.metadata.index:
            file_info = self.find_by_id(ecg_id)
            if file_info['csv'] is not None:
                yield file_info

    def get_statistics(self) -> Dict:
        stats = {'total_samples': len(self.metadata), 'nyha_distribution': self.metadata[self._nyha_col].value_counts().to_dict() if self._nyha_col else {}, 'sampling_mode': self.sampling_config.get('mode', 'all')}
        if self._patient_uid_col:
            stats['unique_patients'] = int(self.metadata[self._patient_uid_col].nunique())
        existing_files = 0
        for ecg_id in self.metadata.index:
            csv_file = self.data_root / f'{ecg_id}.csv'
            if csv_file.exists():
                existing_files += 1
        stats['files_found'] = existing_files
        stats['files_missing'] = len(self.metadata) - existing_files
        return stats

class PreliminaryLocator(FileLocatorStrategy):
    GROUP_NAME_TO_CODE: Dict[str, int] = {'health': 0, 'healthy': 0, '健康': 0, '健康体检组': 0, 'nyha i': 1, 'nyha i级': 1, 'nyha ii': 2, 'nyha ii级': 2, 'nyha iii': 3, 'nyha iii级': 3, 'nyha iv': 4, 'nyha iv级': 4, 'heart death': 5, '心脏性死亡': 5}

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        log_info(f'[PreliminaryLocator] Initialized: {self.data_root}')

    def find_by_id(self, ecg_id: str) -> Dict[str, Optional[str]]:
        for group_dir in self.data_root.iterdir():
            if not group_dir.is_dir():
                continue
            group_name_lower = group_dir.name.strip().lower()
            if group_name_lower not in self.GROUP_NAME_TO_CODE:
                continue
            group_code = self.GROUP_NAME_TO_CODE[group_name_lower]
            for csv_file in group_dir.glob('*.csv'):
                file_sample_id = self._parse_sample_id(csv_file.name)
                if self._is_match(file_sample_id, csv_file.stem, ecg_id):
                    hea_file = csv_file.with_suffix('.hea')
                    hea_path = str(hea_file) if hea_file.exists() else None
                    return {'csv': str(csv_file), 'hea': hea_path, 'group': group_dir.name, 'group_code': group_code, 'metadata': None, 'error': None}
        return {'csv': None, 'hea': None, 'group': None, 'group_code': None, 'metadata': None, 'error': 'File not found'}

    def list_all_ids(self) -> List[str]:
        ids = []
        for file_info in self.iter_all_files():
            ids.append(file_info['ecg_id'])
        return ids

    def iter_all_files(self) -> Generator[Dict, None, None]:
        for group_dir in sorted([p for p in self.data_root.iterdir() if p.is_dir()]):
            group_name_lower = group_dir.name.strip().lower()
            if group_name_lower not in self.GROUP_NAME_TO_CODE:
                continue
            group_code = self.GROUP_NAME_TO_CODE[group_name_lower]
            group_name = group_dir.name
            for csv_file in sorted(group_dir.glob('*.csv')):
                ecg_id = self._parse_sample_id(csv_file.name)
                hea_file = csv_file.with_suffix('.hea')
                hea_path = str(hea_file) if hea_file.exists() else None
                yield {'ecg_id': ecg_id, 'csv': str(csv_file), 'hea': hea_path, 'group': group_name, 'group_code': group_code, 'metadata': None, 'error': None}

    @staticmethod
    def _parse_sample_id(filename: str) -> str:
        import re
        base_name = Path(filename).stem
        if '-' in base_name:
            parts = base_name.split('-')
            return parts[0].strip()
        m = re.match('([^0-9]+)(\\d{4})', base_name)
        if m:
            return m.group(1).strip()
        return base_name.strip()

    @staticmethod
    def _is_match(file_sample_id: str, file_stem: str, ecg_id: str) -> bool:
        if file_sample_id == ecg_id:
            return True
        if file_stem.startswith(ecg_id + '-'):
            return True
        if file_stem == ecg_id:
            return True
        return False

def create_locator(data_source: str, config: Dict) -> FileLocatorStrategy:
    if data_source == 'formal':
        return FormalLocator(data_root=config['data_root'], metadata_excel=config['metadata_excel'], sampling_config=config.get('sampling_config'))
    elif data_source == 'preliminary':
        return PreliminaryLocator(data_root=config['data_root'])
    else:
        raise ValueError(f'Unknown data source: {data_source}')
