"""Provides ecg file locator functions for the ECG-to-entropy pipeline."""
from typing import Optional, Dict
from pathlib import Path
import re
GROUP_NAME_TO_CODE: Dict[str, int] = {'health': 0, 'healthy': 0, '健康': 0, '健康体检组': 0, 'nyha i': 1, 'nyha i级': 1, 'nyha ii': 2, 'nyha ii级': 2, 'nyha iii': 3, 'nyha iii级': 3, 'nyha iv': 4, 'nyha iv级': 4, 'heart death': 5, '心脏性死亡': 5}

def parse_sample_id_from_filename(filename: str) -> str:
    base_name = Path(filename).stem
    if '-' in base_name:
        parts = base_name.split('-')
        sample_id = parts[0].strip()
        return sample_id
    m = re.match('([^0-9]+)(\\d{4})', base_name)
    if m:
        return m.group(1).strip()
    return base_name.strip()

def find_ecg_files_by_id(ecg_id: str, data_root: str) -> Dict[str, Optional[str]]:
    root = Path(data_root)
    for group_dir in root.iterdir():
        if not group_dir.is_dir():
            continue
        group_name_lower = group_dir.name.strip().lower()
        if group_name_lower not in GROUP_NAME_TO_CODE:
            continue
        group_code = GROUP_NAME_TO_CODE[group_name_lower]
        for csv_file in group_dir.glob('*.csv'):
            file_sample_id = parse_sample_id_from_filename(csv_file.name)
            is_match = False
            if file_sample_id == ecg_id:
                is_match = True
            elif csv_file.stem.startswith(ecg_id + '-'):
                is_match = True
            elif csv_file.stem == ecg_id:
                is_match = True
            if is_match:
                hea_file = csv_file.with_suffix('.hea')
                hea_path = str(hea_file) if hea_file.exists() else None
                return {'csv': str(csv_file), 'hea': hea_path, 'group': group_dir.name, 'group_code': group_code}
    return {'csv': None, 'hea': None, 'group': None, 'group_code': None}

def iter_all_ecg_files(data_root: str):
    root = Path(data_root)
    for group_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        group_name_lower = group_dir.name.strip().lower()
        if group_name_lower not in GROUP_NAME_TO_CODE:
            continue
        group_code = GROUP_NAME_TO_CODE[group_name_lower]
        group_name = group_dir.name
        for csv_file in sorted(group_dir.glob('*.csv')):
            ecg_id = parse_sample_id_from_filename(csv_file.name)
            hea_file = csv_file.with_suffix('.hea')
            hea_path = str(hea_file) if hea_file.exists() else None
            yield {'ecg_id': ecg_id, 'csv_path': str(csv_file), 'hea_path': hea_path, 'group': group_name, 'group_code': group_code}
