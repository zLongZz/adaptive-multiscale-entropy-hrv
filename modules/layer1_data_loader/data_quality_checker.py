"""Provides data quality checker functions for the ECG-to-entropy pipeline."""
import pandas as pd
from typing import Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_info
from .ecg_file_locator import iter_all_ecg_files

def check_file_integrity(csv_path: str) -> Dict:
    checks = {'file_exists': False, 'file_size_mb': 0.0, 'estimated_hours': 0.0}
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return checks
    checks['file_exists'] = True
    checks['file_size_mb'] = csv_file.stat().st_size / (1024 * 1024)
    checks['estimated_hours'] = csv_file.stat().st_size / 150 / 200 / 3600
    return checks

def check_all_files(data_root: str) -> pd.DataFrame:
    log_info('[Quality Check] Starting...')
    results = []
    for file_info in iter_all_ecg_files(data_root):
        checks = check_file_integrity(file_info['csv_path'])
        results.append({'ecg_id': file_info['ecg_id'], 'group': file_info['group'], **checks})
    return pd.DataFrame(results)
