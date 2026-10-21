"""Provides hea parser functions for the ECG-to-entropy pipeline."""
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Optional
sys.path.append(str(Path(__file__).parents[2]))
from modules.core.logger import log_debug, log_warning
_DATETIME_PATTERNS = (('(\\d{1,2}-[A-Za-z]{3}-\\d{4})\\s+(\\d{1,2}:\\d{2}:\\d{2})', '%d-%b-%Y %H:%M:%S'), ('(\\d{1,2}-[A-Za-z]{3}-\\d{4})\\s+(\\d{1,2}:\\d{2})', '%d-%b-%Y %H:%M'), ('(\\d{4}[-/]\\d{2}[-/]\\d{2})[ T](\\d{1,2}:\\d{2}:\\d{2})', '%Y-%m-%d %H:%M:%S'), ('(\\d{4}[-/]\\d{2}[-/]\\d{2})[ T](\\d{1,2}:\\d{2})', '%Y-%m-%d %H:%M'), ('(\\d{1,2}:\\d{2}:\\d{2})\\s+(\\d{1,2}-[A-Za-z]{3}-\\d{4})', '%H:%M:%S %d-%b-%Y'))

def _read_first_line(hea_path: Optional[str]) -> Optional[str]:
    if not hea_path:
        return None
    hea_file = Path(hea_path)
    if not hea_file.is_file():
        return None
    try:
        with hea_file.open('r', encoding='utf-8', errors='ignore') as handle:
            return handle.readline().strip()
    except OSError as exc:
        log_warning(f'[Layer 1] HEA read error: {exc}')
        return None

def _read_header_lines(hea_path: Optional[str], n_lines: int) -> list[str]:
    if not hea_path or n_lines <= 0:
        return []
    hea_file = Path(hea_path)
    if not hea_file.is_file():
        return []
    try:
        lines = []
        with hea_file.open('r', encoding='utf-8', errors='ignore') as handle:
            for _ in range(n_lines):
                line = handle.readline()
                if not line:
                    break
                lines.append(line.strip())
        return lines
    except OSError as exc:
        log_warning(f'[Layer 1] HEA read error: {exc}')
        return []

def parse_hea_datetime(hea_path: Optional[str]) -> Optional[datetime]:
    pass
    first_line = _read_first_line(hea_path)
    if not first_line:
        return None
    for pattern, fmt in _DATETIME_PATTERNS:
        match = re.search(pattern, first_line)
        if not match:
            continue
        first, second = match.groups()[:2]
        value = f'{first} {second}'.replace('/', '-')
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        log_debug(f'[Layer 1] Parsed HEA datetime: {parsed}')
        return parsed
    log_warning(f'[Layer 1] HEA datetime not recognized: {first_line}')
    return None

def _parse_sampling_rate(token: str) -> Optional[float]:
    pass
    try:
        sampling_rate = float(token.split('/', 1)[0])
    except (TypeError, ValueError):
        return None
    return sampling_rate if sampling_rate > 0 else None

def parse_hea_info(hea_path: Optional[str]) -> dict:
    pass
    info = {'datetime': None, 'duration': None, 'duration_seconds': None, 'sampling_rate': None, 'n_leads': None, 'total_values': None, 'lead_names': []}
    first_line = _read_first_line(hea_path)
    if not first_line:
        return info
    parts = first_line.split()
    if len(parts) < 4:
        log_warning(f'[Layer 1] HEA first line has fewer than 4 fields: {first_line}')
        return info
    try:
        n_leads = int(parts[1])
        total_values = int(float(parts[3]))
    except (TypeError, ValueError):
        n_leads = None
        total_values = None
    sampling_rate = _parse_sampling_rate(parts[2])
    lead_names = []
    if n_leads:
        header_lines = _read_header_lines(hea_path, n_leads + 1)
        for signal_line in header_lines[1:n_leads + 1]:
            signal_parts = signal_line.split()
            if signal_parts:
                lead_names.append(signal_parts[-1])
    duration_seconds = None
    if n_leads and sampling_rate and (total_values is not None):
        duration_seconds = total_values / n_leads / sampling_rate
    info.update({'datetime': parse_hea_datetime(hea_path), 'duration': duration_seconds, 'duration_seconds': duration_seconds, 'sampling_rate': sampling_rate, 'n_leads': n_leads, 'total_values': total_values, 'lead_names': lead_names})
    return info
