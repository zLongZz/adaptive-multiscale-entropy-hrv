"""Provides utils functions for the ECG-to-entropy pipeline."""
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_pickle(obj: Any, filepath: str):
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    print(f'[INFO] Saved pickle: {Path(filepath).name} ({file_size_mb:.2f} MB)')

def load_pickle(filepath: str) -> Any:
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f'[INFO] Loaded pickle: {Path(filepath).name}')
    return obj

def save_json(obj: Dict, filepath: str):
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
    print(f'[INFO] Saved JSON: {Path(filepath).name}')

def load_json(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    print(f'[INFO] Loaded JSON: {Path(filepath).name}')
    return obj

def save_dataframe_hdf5(df: pd.DataFrame, filepath: str, key: str='data'):
    ensure_dir(Path(filepath).parent)
    df.to_hdf(filepath, key=key, mode='w', complevel=9)
    file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    print(f'[INFO] Saved HDF5: {Path(filepath).name} ({file_size_mb:.2f} MB)')

def load_dataframe_hdf5(filepath: str, key: str='data') -> pd.DataFrame:
    df = pd.read_hdf(filepath, key=key)
    print(f'[INFO] Loaded HDF5: {Path(filepath).name} ({len(df)} rows)')
    return df

def get_file_size_mb(filepath: str) -> float:
    return Path(filepath).stat().st_size / (1024 * 1024)

def parse_datetime_flexible(date_str: str) -> Optional[datetime]:
    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%Y/%m/%d %H:%M', '%d-%b-%Y %H:%M:%S', '%d-%b-%Y %H:%M']
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except:
            continue
    return None

def calculate_time_coverage(hours: np.ndarray, total_hours: int=24) -> float:
    unique_hours = len(np.unique(hours))
    return unique_hours / total_hours

def get_optimal_n_processes(n_processes: Optional[int]=None) -> int:
    import os
    if n_processes is not None:
        return max(1, n_processes)
    cpu_count = os.cpu_count() or 4
    optimal = int(cpu_count * 0.6)
    return max(1, optimal)

def create_experiment_id(prefix: str='formal_exp') -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_config_snapshot(config: Any, output_path: str):
    import sys
    snapshot = {'experiment_id': create_experiment_id(), 'timestamp': datetime.now().isoformat(), 'config': config.to_dict() if hasattr(config, 'to_dict') else str(config), 'software_versions': {'python': sys.version, 'numpy': __import__('numpy').__version__, 'scipy': __import__('scipy').__version__, 'pandas': __import__('pandas').__version__}}
    save_json(snapshot, output_path)
