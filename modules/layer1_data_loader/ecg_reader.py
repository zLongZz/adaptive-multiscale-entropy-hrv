"""Provides ecg reader functions for the ECG-to-entropy pipeline."""
from typing import Iterator, Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import re
import sys
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_info, log_warning
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def _resolve_leads(lead_names: Optional[List[str]]) -> Tuple[List[str], List[int]]:
    requested = LEAD_NAMES if lead_names is None else list(dict.fromkeys(lead_names))
    unknown = [lead for lead in requested if lead not in LEAD_NAMES]
    if unknown:
        raise ValueError(f'Unknown ECG leads: {unknown}')
    ordered = sorted(((LEAD_NAMES.index(lead), lead) for lead in requested))
    return ([lead for _, lead in ordered], [idx for idx, _ in ordered])

def read_ecg_chunks(csv_path: str, chunk_size: int=1000000, lead_names: Optional[List[str]]=None) -> Iterator[pd.DataFrame]:
    selected_names, selected_indices = _resolve_leads(lead_names)

    def _yield_clean(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        if df.shape[1] != len(selected_names):
            log_warning(f'[Layer 1] CSV selected column mismatch: {df.shape[1]} != {len(selected_names)}')
            return
        df = df.copy()
        df.columns = selected_names
        yield df.reset_index(drop=True)
    try:
        skip_n = _detect_numeric_start_row(csv_path)
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, header=None, usecols=selected_indices, dtype=float, encoding='utf-8', na_values=['', 'NaN', 'nan'], keep_default_na=True, low_memory=False, skiprows=skip_n):
            for cleaned in _yield_clean(chunk):
                yield cleaned
    except Exception as e:
        log_warning(f'[Layer 1] C engine failed, fallback to Python engine: {str(e)}')
        skip_n = _detect_numeric_start_row(csv_path)
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, header=None, usecols=selected_indices, dtype=float, engine='python', sep=None, encoding='utf-8', on_bad_lines='skip', skiprows=skip_n):
            for cleaned in _yield_clean(chunk):
                yield cleaned

def read_ecg_complete(csv_path: str, chunk_size: int=1000000, lead_names: Optional[List[str]]=None) -> pd.DataFrame:
    chunks = []
    total_rows = 0
    for chunk in read_ecg_chunks(csv_path, chunk_size, lead_names=lead_names):
        chunks.append(chunk)
        total_rows += len(chunk)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    log_info(f'[Layer 1] Read {total_rows} rows from {Path(csv_path).name}')
    return df

def _detect_numeric_start_row(csv_path: str, max_probe_lines: int=50) -> int:
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                if idx >= max_probe_lines:
                    break
                parts = [p.strip() for p in line.strip().split(',') if p.strip() != '']
                if len(parts) < 12:
                    parts = [p for p in re.split('\\s+', line.strip()) if p]
                if len(parts) >= 12:
                    ok = True
                    for p in parts[:12]:
                        try:
                            float(p)
                        except:
                            ok = False
                            break
                    if ok:
                        return idx
        return 0
    except Exception:
        return 0

def get_lead_dataframe(ecg_df: pd.DataFrame) -> Dict[str, pd.Series]:
    if all((str(col) in LEAD_NAMES for col in ecg_df.columns)):
        return {str(col): ecg_df[col].to_numpy(copy=False) for col in ecg_df.columns}
    if ecg_df.shape[1] == 12:
        return {lead_name: ecg_df.iloc[:, i].to_numpy(copy=False) for i, lead_name in enumerate(LEAD_NAMES)}
    return {}
