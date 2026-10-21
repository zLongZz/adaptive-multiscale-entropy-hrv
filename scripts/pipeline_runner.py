"""Provides pipeline runner functions for the ECG-to-entropy pipeline."""
from __future__ import annotations
import gc
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config_formal import load_config
from modules.layer1_data_loader.data_loader import ECGDataLoader
from modules.layer2_rr_extraction import AdvancedRRExtractor
from modules.layer3_segmentation import ClockSampler, WindowSegmenter
from modules.layer5_rhythm_detection import RhythmProcessor, SinusConfidenceScorer
from modules.layer6_features import FeatureProcessor
from modules.layer7_aggregation import EligibilityThresholds, RecordRRExporter, RecordEligibilityEvaluator, SubjectAggregator, WindowRRExporter, build_rr_export_record
from modules.layer7_aggregation.record_eligibility import ELIGIBILITY_POLICY_ID
PIPELINE_VERSION = 'entropy_core_sampen_cxi_fix_v2_20260810'
DURATION_TRUNCATION_TOLERANCE_SECONDS = 1.0
RHYTHM_ONLY_NAN_FEATURES = [*(f'MSE_{tau}' for tau in range(1, 21)), 'MSE_short', 'MSE_long', 'CXI', 'CXI_diff', 'CXI_ratio', 'n_valid_scales', 'n_valid_long_scales', 'n_valid_short_scales', 'DFA_alpha1', 'DFA_alpha1_intercept', 'DFA_alpha1_r_squared', 'DFA_alpha2', 'DFA_intercept', 'DFA_r_squared', 'DFA_slope_8_64', 'DFA_slope_8_64_intercept', 'DFA_slope_8_64_r_squared']
WINDOW_AUDIT_COLUMNS = ['ecg_id', 'patient_uid', 'nyha_grade', 'group', 'group_code', 'analysis_population', 'center_name', 'record_datetime', 'timestamp_source', 'source_duration_seconds', 'analysis_duration_seconds', 'analysis_duration_policy', 'duration_truncated_to_analysis_window', 'duration_truncated_strict', 'duration_truncation_seconds', 'meaningful_duration_truncation', 'primary_lead', 'selected_lead', 'window_idx', 'start_time', 'end_time', 'timestamp', 'clock_hour', 'n_beats', 'layer3_eligible', 'layer3_reject_reason', 'rhythm_type', 'rhythm_type_legacy', 'rhythm_label_semantic', 'rhythm_confidence', 'rhythm_method', 'rhythm_classification_policy', 'rhythm_score_type', 'rhythm_quality', 'rhythm_processing_reason', 'ectopy_rate', 'interpolation_success', 'interpolation_applied', 'rr_mean_raw_window', 'rr_std_raw_window', 'rr_cv_raw_window', 'rr_mean_final_window', 'rr_std_final_window', 'rr_cv_final_window', 'rr_variance_ratio', 'rr_near_constant_raw', 'rr_near_constant_final', 'interpolation_variance_collapse', 'sinus_confidence_score', 'analysis_track', 'analysis_track_legacy', 'entropy_eligible', 'legacy_layer5_eligible', 'feature_quality', 'layer6_input_rr_count', 'layer6_min_rr_required', 'layer6_input_eligible', 'rr_n_input', 'rcmse_quality', 'layer6_reject_reason', 'mse_range_flag', 'dfa_range_flag', 'cxi_afib_flag', 'reject_stage', 'reject_reason']

def _build_rhythm_only_features(window: dict, rr_final: np.ndarray) -> dict:
    pass
    features = {'lead': window.get('lead'), 'window_idx': window.get('window_idx'), 'start_time': window.get('start_time'), 'end_time': window.get('end_time'), 'timestamp': window.get('timestamp'), 'clock_hour': window.get('clock_hour'), 'group_code': window.get('group_code'), 'n_beats': int(len(rr_final)), 'DFA_quality': 'not_computed_short_window', 'valid_scales': [], 'feature_quality': 'time_domain_only'}
    for name in RHYTHM_ONLY_NAN_FEATURES:
        features[name] = np.nan
    return features

def _normalize_fallback_leads(lead_name: str, fallback_leads) -> list[str]:
    normalized = []
    for lead in fallback_leads or []:
        value = str(lead).strip()
        if value and value != lead_name and (value not in normalized):
            normalized.append(value)
    return normalized

def _coerce_datetime(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    try:
        parsed = pd.to_datetime(value, errors='raise')
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    parsed = parsed.to_pydatetime() if hasattr(parsed, 'to_pydatetime') else parsed
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed

def _resolve_record_datetime(sample_data: dict) -> tuple:
    pass
    hea_datetime = _coerce_datetime(sample_data.get('hea_info', {}).get('datetime'))
    if hea_datetime is not None:
        return (hea_datetime, 'hea')
    metadata = sample_data.get('metadata', {}) or {}
    for field in ('record_datetime', 'test_datetime'):
        metadata_datetime = _coerce_datetime(metadata.get(field))
        if metadata_datetime is not None:
            return (metadata_datetime, f'metadata:{field}')
    return (None, 'missing')

def _strict_input_issues(sample_data: dict, *, timestamp_source: str, primary_lead: str) -> list[str]:
    pass
    hea_info = sample_data.get('hea_info', {}) or {}
    issues = []
    if timestamp_source != 'hea':
        issues.append('hea_datetime_missing_or_invalid')
    sampling_rate = hea_info.get('sampling_rate')
    if sampling_rate is None or not np.isfinite(sampling_rate) or sampling_rate <= 0:
        issues.append('hea_sampling_rate_missing_or_invalid')
    duration = hea_info.get('duration_seconds')
    if duration is None or not np.isfinite(duration) or duration <= 0:
        issues.append('hea_duration_missing_or_invalid')
    n_leads = hea_info.get('n_leads')
    if not isinstance(n_leads, int) or n_leads <= 0:
        issues.append('hea_lead_count_missing_or_invalid')
    if primary_lead not in hea_info.get('lead_names', []):
        issues.append('primary_lead_absent_from_hea')
    return issues

def _resolve_min_beats(*, window_sec: int, rr_max: float, min_length: int, feature_mode: str) -> int:
    physiological_min = max(3, int(math.ceil(window_sec / rr_max)))
    if feature_mode == 'full' and window_sec >= 600:
        return max(physiological_min, int(min_length * 20))
    return physiological_min

def _resolve_lead_min_distance(min_distance, lead_name: str) -> float:
    pass
    if not isinstance(min_distance, Mapping):
        return float(min_distance)
    if lead_name in min_distance:
        return float(min_distance[lead_name])
    from modules.layer2_rr_extraction.quality_profiles import get_lead_min_distance
    return get_lead_min_distance(lead_name, dict(min_distance))

def _count_eligible_rr_windows(rr_times: np.ndarray, *, window_sec: int, step_sec: int, min_beats: int) -> int:
    if len(rr_times) == 0 or step_sec <= 0 or rr_times[-1] < window_sec:
        return 0
    count = 0
    start = 0
    while start + window_sec <= rr_times[-1]:
        left = np.searchsorted(rr_times, start, side='left')
        right = np.searchsorted(rr_times, start + window_sec, side='right')
        if right - left >= min_beats:
            count += 1
        start += step_sec
    return count

def _extract_rr_with_lead_policy(*, rr_extractor, lead_data: dict, primary_lead: str, fallback_leads: list, min_distance, rr_range: tuple, min_rr_count: int, window_sec: int | None=None, window_step_sec: int | None=None, min_window_beats: int | None=None) -> tuple:
    pass
    attempts = []
    leads_to_try = [primary_lead, *_normalize_fallback_leads(primary_lead, fallback_leads)]
    for attempt_order, lead in enumerate(leads_to_try, start=1):
        signal_data = lead_data.get(lead)
        effective_min_distance = _resolve_lead_min_distance(min_distance, lead)
        attempt = {'attempt_order': attempt_order, 'lead': lead, 'is_primary_lead': lead == primary_lead, 'selected': False, 'min_distance': effective_min_distance}
        if signal_data is None:
            attempt.update({'status': 'missing_lead', 'n_rr': 0, 'n_rr_raw': 0, 'n_r_peaks': 0})
            attempts.append(attempt)
            continue
        try:
            result = rr_extractor.extract_rr(signal_data, return_quality=True, min_distance=effective_min_distance, rr_range=rr_range)
        except Exception as exc:
            attempt.update({'status': 'extraction_error', 'error': f'{type(exc).__name__}: {exc}', 'n_rr': 0, 'n_rr_raw': 0, 'n_r_peaks': 0})
            attempts.append(attempt)
            continue
        rr_raw = np.asarray(result.get('rr_raw', result.get('rr', [])), dtype=float)
        rr_ts_raw = np.asarray(result.get('rr_ts_raw', result.get('rr_ts', [])), dtype=float)
        n_eligible_windows = None
        if all((value is not None for value in (window_sec, window_step_sec, min_window_beats))):
            n_eligible_windows = _count_eligible_rr_windows(rr_ts_raw, window_sec=int(window_sec), step_sec=int(window_step_sec), min_beats=int(min_window_beats))
        status = 'valid' if len(rr_raw) >= min_rr_count else 'insufficient_rr'
        diagnostics = result.get('diagnostics', {})
        quality = result.get('quality', {})
        attempt.update({'status': status, 'n_rr': int(len(result.get('rr', []))), 'n_rr_raw': int(len(rr_raw)), 'n_r_peaks': int(len(result.get('r_peaks', []))), 'n_rr_filtered': int(result.get('n_filtered', 0)), 'n_eligible_windows': n_eligible_windows, **{key if key.startswith('rr_') else f'rr_{key}': value for key, value in diagnostics.items()}, **{f"rr_quality_{key.removeprefix('quality_')}": value for key, value in quality.items()}})
        attempts.append(attempt)
        if status == 'valid':
            attempt['selected'] = True
            return (lead, result, attempts)
    return (None, None, attempts)

def _append_jsonl(path: Path, records: list) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records

def _append_dataframe_csv(path: Path, frame: pd.DataFrame, columns=None) -> None:
    pass
    if frame is None or frame.empty:
        return
    data = frame.copy()
    if columns is not None:
        data = data.reindex(columns=list(columns))
    elif path.exists():
        existing = list(pd.read_csv(path, nrows=0).columns)
        unexpected = [column for column in data.columns if column not in existing]
        if unexpected:
            raise RuntimeError(f'CSV schema drift for {path}: unexpected columns {unexpected}')
        data = data.reindex(columns=existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, mode='a' if path.exists() else 'w', header=not path.exists(), index=False, encoding='utf-8-sig')

def _scalarize_features(features: dict) -> dict:
    pass
    scalar = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (list, tuple, dict, set)):
            scalar[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            scalar[key] = value
    return scalar

def _resolve_formal_data_root(data_root: str) -> str:
    root = Path(data_root).expanduser()
    if any(root.glob('*.csv')):
        return str(root)
    matched_dir = root / '01_matched_clinical_ecg'
    if matched_dir.is_dir() and any(matched_dir.glob('*.csv')):
        print(f'[Layer 0] Resolved ECG data root: {root} -> {matched_dir}')
        return str(matched_dir)
    return str(root)

def _resolve_analysis_duration(source_duration_seconds, *, policy: str='first_24h', max_hours: float=24.0) -> tuple[float | None, bool]:
    pass
    if policy not in {'first_24h', 'full_record'}:
        raise ValueError('analysis_duration_policy must be first_24h or full_record')
    if source_duration_seconds is None or not np.isfinite(source_duration_seconds):
        return (None, False)
    source_duration_seconds = float(source_duration_seconds)
    if policy == 'full_record':
        return (source_duration_seconds, False)
    limit = float(max_hours) * 3600.0
    return (min(source_duration_seconds, limit), bool(source_duration_seconds > limit))

def _duration_truncation_metadata(source_duration_seconds, analysis_duration_seconds, *, tolerance_seconds: float=DURATION_TRUNCATION_TOLERANCE_SECONDS) -> dict:
    pass
    if source_duration_seconds is None or analysis_duration_seconds is None or (not np.isfinite(source_duration_seconds)) or (not np.isfinite(analysis_duration_seconds)):
        truncation_seconds = np.nan
        strict = False
        meaningful = False
    else:
        truncation_seconds = max(0.0, float(source_duration_seconds) - float(analysis_duration_seconds))
        strict = bool(truncation_seconds > 0.0)
        meaningful = bool(truncation_seconds > float(tolerance_seconds))
    return {'duration_truncated_to_analysis_window': strict, 'duration_truncated_strict': strict, 'duration_truncation_seconds': truncation_seconds, 'meaningful_duration_truncation': meaningful}

def _resolve_layer6_min_rr_required(feature_processor) -> int:
    pass
    return int(feature_processor.rcmse_min_length * max(feature_processor.rcmse_scales))

def _assess_layer6_input(rr_count: int, min_required: int) -> dict:
    eligible = int(rr_count) >= int(min_required)
    return {'layer6_input_rr_count': int(rr_count), 'layer6_min_rr_required': int(min_required), 'layer6_input_eligible': bool(eligible)}

def _computation_status_fields(status_name: str) -> dict:
    pass
    return {'status': status_name, 'computation_status': status_name, 'computation_success': status_name == 'success'}

def create_loader(data_root: str | None=None, metadata_excel: str | None=None):
    pass
    config = load_config()
    config.paths.data_source = 'formal'
    if data_root:
        config.paths.formal_data_root = _resolve_formal_data_root(data_root)
    if metadata_excel:
        config.paths.formal_metadata_excel = metadata_excel
    config.metadata.sampling_mode = 'all'
    loader = ECGDataLoader.from_config(config)
    metadata = loader.locator.metadata
    optional_columns = ['is_death', 'death_datetime', 'med_beta_blocker', 'med_acei', 'med_arb', 'med_diuretic']
    missing = [column for column in optional_columns if column not in metadata.columns]
    if missing:
        print(f'[Layer 0] Optional downstream clinical fields missing; entropy extraction remains runnable: {missing}')
    print(f'[Layer 0] Metadata ready (n_records={len(metadata)}, n_columns={len(metadata.columns)})')
    return (loader, config)

def sample_ids(loader, n_per_group: int, random_seed: int=20260328) -> list:
    pass
    metadata = loader.locator.metadata
    nyha_col = loader.locator._nyha_col
    sampled = []
    for value in ['NYHA_0', 'NYHA_I', 'NYHA_II', 'NYHA_III', 'NYHA_IV']:
        group = metadata[metadata[nyha_col] == value]
        n = min(n_per_group, len(group))
        if n:
            sampled.extend(group.sample(n=n, random_state=random_seed).index.tolist())
    return sampled

def _empty_audit_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=WINDOW_AUDIT_COLUMNS)

def _record_base(sample_data: dict, ecg_id: str, record_datetime, timestamp_source: str) -> dict:
    return {'ecg_id': ecg_id, 'patient_uid': sample_data.get('patient_uid'), 'nyha_grade': sample_data.get('nyha_grade', 'Unknown'), 'group': sample_data.get('group', 'Unknown'), 'group_code': sample_data.get('group_code'), 'analysis_population': sample_data.get('analysis_population', 'Unknown'), 'center_name': sample_data.get('center_name'), 'record_datetime': record_datetime, 'timestamp_source': timestamp_source}

def _build_eligibility_rows(evaluator: RecordEligibilityEvaluator, audit_df: pd.DataFrame, *, base: dict, hea_duration_seconds, csv_duration_seconds, duration_delta_seconds, rr_span_seconds, analysis_duration_seconds, analysis_duration_policy, duration_truncated_to_analysis_window, analysis_tracks: list[str]) -> list[dict]:
    rows = []
    for track in [None, *analysis_tracks]:
        summary = evaluator.summarize(audit_df, hea_duration_seconds=hea_duration_seconds, rr_span_seconds=rr_span_seconds, analysis_track=track, analysis_duration_seconds=analysis_duration_seconds, analysis_duration_policy=analysis_duration_policy)
        rows.append({**base, 'analysis_track': track or 'all_entropy_tracks', 'hea_duration_seconds': hea_duration_seconds, 'csv_duration_seconds': csv_duration_seconds, 'duration_delta_seconds': duration_delta_seconds, 'analysis_duration_seconds': analysis_duration_seconds, 'duration_truncated_to_analysis_window': duration_truncated_to_analysis_window, **summary})
    return rows

def _load_completed_ids(status_jsonl: Path) -> set[str]:
    completed = set()
    if not status_jsonl.exists():
        return completed
    with status_jsonl.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                ecg_id = json.loads(line).get('ecg_id')
            except (json.JSONDecodeError, AttributeError):
                continue
            if ecg_id:
                completed.add(str(ecg_id))
    return completed

def _retain_completed_csv_rows(path: Path, completed_ids: set[str]) -> None:
    pass
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if 'ecg_id' not in frame.columns:
        return
    retained = frame[frame['ecg_id'].astype(str).isin(completed_ids)]
    retained.to_csv(path, index=False, encoding='utf-8-sig')

def _csv_boolean_mask(series: pd.Series) -> pd.Series:
    pass
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.astype('string').str.strip().str.lower().isin({'true', '1', 'yes'})

def run_layer07(loader, ecg_ids: list, output_dir: str, lead_name: str='II', window_sec: int=900, batch_description: str='', lambda_param: float=500.0, r_factor: float=0.15, interpolation_threshold: float=0.2, min_length: int=30, min_distance=0.3, calibrated_min_distance_params: dict | None=None, n_processes: int | None=None, skip_detrend: bool=True, quality_tier: str='sinus_strict', min_beats: int | None=None, window_rr_min_override: int | None=None, feature_mode: str='full', write_window_csv: bool=True, export_window_rr: bool=True, export_record_rr_cache: bool=True, rr_detector_mode: str='robust', rr_chunk_sec: float=3600.0, rr_chunk_overlap_sec: float=10.0, rr_polarity_mode: str='auto', fallback_leads: list | None=None, analysis_policy: str='dual_track_v2', rhythm_classification_policy: str='non_entropy_v2', primary_duration_hours: float=20.0, sensitivity_duration_hours: float=18.0, analysis_duration_policy: str='first_24h', analysis_duration_hours: float=24.0, constant_rr_std_epsilon: float=1e-08, interpolation_variance_ratio_min: float=0.05, resume: bool=False, strict_input: bool=False) -> dict:
    pass
    from modules.core.utils import get_optimal_n_processes
    from modules.layer2_rr_extraction.quality_profiles import get_quality_config
    if feature_mode not in {'full', 'rhythm_only'}:
        raise ValueError("feature_mode must be 'full' or 'rhythm_only'")
    if analysis_policy not in {'dual_track_v2', 'legacy_sinus_strict'}:
        raise ValueError('analysis_policy must be dual_track_v2 or legacy_sinus_strict')
    if analysis_duration_policy not in {'first_24h', 'full_record'}:
        raise ValueError('analysis_duration_policy must be first_24h or full_record')
    compute_full_features = feature_mode == 'full'
    qc = get_quality_config(quality_tier)
    if min_beats is None:
        min_beats = _resolve_min_beats(window_sec=window_sec, rr_max=qc.rr_physiological_range[1], min_length=min_length, feature_mode=feature_mode)
    if window_rr_min_override is not None:
        effective_window_rr_min = int(window_rr_min_override)
    elif window_sec < 900:
        effective_window_rr_min = max(3, int(window_sec * 0.5))
    else:
        effective_window_rr_min = qc.window_rr_min
    out = Path(output_dir)
    for subdir in ('windows', 'subjects', 'logs', 'records'):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    windows_csv = out / 'windows' / 'window_features.csv'
    window_audit_csv = out / 'windows' / 'window_audit.csv'
    subjects_csv = out / 'subjects' / 'subject_features.csv'
    eligibility_csv = out / 'subjects' / 'record_eligibility.csv'
    eligibility_jsonl = out / 'logs' / 'record_eligibility.jsonl'
    rr_qc_csv = out / 'logs' / 'rr_extraction_qc.csv'
    rr_qc_jsonl = out / 'logs' / 'rr_extraction_qc.jsonl'
    status_csv = out / 'logs' / 'record_processing_status.csv'
    status_jsonl = out / 'logs' / 'record_processing_status.jsonl'
    rr_exporter = WindowRRExporter(str(out / 'windows'), enabled=export_window_rr)
    record_rr_exporter = RecordRRExporter(str(out / 'records'), enabled=export_record_rr_cache)
    completed_ids = _load_completed_ids(status_jsonl) if resume else set()
    pending_ids = [ecg_id for ecg_id in ecg_ids if str(ecg_id) not in completed_ids]
    if completed_ids:
        for path in (windows_csv, window_audit_csv, subjects_csv):
            _retain_completed_csv_rows(path, completed_ids)
        rr_exporter.retain_ecg_ids(completed_ids)
        record_rr_exporter.retain_ecg_ids(completed_ids)
        print(f'  [Resume] skipped={len(ecg_ids) - len(pending_ids)}, pending={len(pending_ids)}')
    actual_n_processes = get_optimal_n_processes(n_processes)
    if not skip_detrend:
        raise ValueError('Layer 4 detrending is deprecated and cannot be enabled in the frozen entropy pipeline')
    fallback_leads = _normalize_fallback_leads(lead_name, fallback_leads)
    requested_leads = [lead_name, *fallback_leads]
    calibrated_values = calibrated_min_distance_params or min_distance
    segmenter = WindowSegmenter(window_sec=window_sec, overlap_ratio=0.5, min_beats=min_beats)
    clock_sampler = ClockSampler(n_strata=24, keep_all=True)
    detrending = None
    rhythm_processor = RhythmProcessor(max_ectopy_ratio=interpolation_threshold, classification_policy=rhythm_classification_policy, constant_rr_std_epsilon=constant_rr_std_epsilon, interpolation_variance_ratio_min=interpolation_variance_ratio_min)
    sinus_scorer = SinusConfidenceScorer()
    feature_processor = FeatureProcessor(rcmse_r_factor=r_factor, rcmse_min_length=min_length, n_processes=actual_n_processes) if compute_full_features else None
    layer6_min_rr_required = _resolve_layer6_min_rr_required(feature_processor) if compute_full_features else None
    evaluator = RecordEligibilityEvaluator(EligibilityThresholds(primary_duration_hours=primary_duration_hours, sensitivity_duration_hours=sensitivity_duration_hours))
    extractor_cache = {}
    print(f"  [Layer 2] detector={rr_detector_mode}, lead={lead_name}, fallback={fallback_leads or 'disabled'}, calibration={calibrated_values}")
    print(f'  [Layer 3-7] window={window_sec}s, min_beats={min_beats}, policy={analysis_policy}, rhythm={rhythm_classification_policy}, duration={analysis_duration_policy}')
    n_success = 0
    n_fail = 0
    total_windows = 0
    total_candidate_windows = 0
    total_nominal_windows = 0
    n_fallback_used = 0
    selected_lead_counts = Counter()
    failure_reason_counts = Counter()
    rr_qc_rows = []
    status_rows = []
    eligibility_rows_all = []
    for index, ecg_id in enumerate(pending_ids, start=1):
        print(f'  [{index}/{len(pending_ids)}] {ecg_id}...', end='', flush=True)
        selected_lead = None
        rr_result = None
        window_features = []
        rr_export_records = []
        rejection_counts = Counter()
        n_window_errors = 0
        try:
            sample_data = loader.load_single_sample(ecg_id, leads=requested_leads)
            if sample_data is None:
                raise RuntimeError('load_failed')
            record_datetime, timestamp_source = _resolve_record_datetime(sample_data)
            base = _record_base(sample_data, ecg_id, record_datetime, timestamp_source)
            hea_info = sample_data.get('hea_info', {})
            if strict_input:
                input_issues = _strict_input_issues(sample_data, timestamp_source=timestamp_source, primary_lead=lead_name)
                if input_issues:
                    raise RuntimeError('strict_input_failed:' + ','.join(input_issues))
            hea_duration = hea_info.get('duration_seconds')
            csv_duration = sample_data.get('csv_duration_seconds')
            duration_delta = sample_data.get('duration_delta_seconds')
            source_duration = hea_duration if hea_duration is not None and np.isfinite(hea_duration) and (hea_duration > 0) else csv_duration
            analysis_duration, duration_truncated = _resolve_analysis_duration(source_duration, policy=analysis_duration_policy, max_hours=analysis_duration_hours)
            duration_metadata = _duration_truncation_metadata(source_duration, analysis_duration)
            if duration_metadata['duration_truncated_strict'] != duration_truncated:
                raise RuntimeError('duration_truncation_state_mismatch')
            base.update({'source_duration_seconds': source_duration, 'analysis_duration_seconds': analysis_duration, 'analysis_duration_policy': analysis_duration_policy, **duration_metadata})
            if record_datetime is None:
                audit_df = _empty_audit_frame()
                rows = _build_eligibility_rows(evaluator, audit_df, base=base, hea_duration_seconds=hea_duration, csv_duration_seconds=csv_duration, duration_delta_seconds=duration_delta, rr_span_seconds=None, analysis_duration_seconds=analysis_duration, analysis_duration_policy=analysis_duration_policy, duration_truncated_to_analysis_window=duration_truncated, analysis_tracks=['sinus_primary_v2', 'af_like_exploratory_v3'])
                eligibility_rows_all.extend(rows)
                _append_jsonl(eligibility_jsonl, rows)
                n_fail += 1
                failure_reason_counts['missing_record_datetime'] += 1
                status = {**base, **_computation_status_fields('missing_record_datetime'), 'eligibility_policy_id': ELIGIBILITY_POLICY_ID, 'eligible_primary_analysis': False, 'eligible_primary_18h_sensitivity': False, 'eligible_all_rhythm_20h_sensitivity': False, 'eligible_all_rhythm_sensitivity': False, 'eligible_af_like_exploratory': False, 'n_entropy_eligible_windows': 0}
                status_rows.append(status)
                _append_jsonl(status_jsonl, [status])
                print(' FAIL (missing datetime)')
                continue
            fs = hea_info.get('sampling_rate')
            fs_source = 'hea'
            if fs is None or not np.isfinite(fs) or fs <= 0:
                fs = 200.0
                fs_source = 'fallback_200hz'
            fs = float(fs)
            if fs not in extractor_cache:
                extractor_cache[fs] = AdvancedRRExtractor(fs=fs, detector_mode=rr_detector_mode, chunk_duration_sec=rr_chunk_sec, chunk_overlap_sec=rr_chunk_overlap_sec, polarity_mode=rr_polarity_mode)
            selected_lead, rr_result, attempts = _extract_rr_with_lead_policy(rr_extractor=extractor_cache[fs], lead_data=sample_data['lead_data'], primary_lead=lead_name, fallback_leads=fallback_leads, min_distance=calibrated_values, rr_range=qc.rr_physiological_range, min_rr_count=qc.min_rr_count, window_sec=window_sec, window_step_sec=int(window_sec * 0.5), min_window_beats=min_beats)
            subject_qc = [{'ecg_id': ecg_id, 'primary_lead': lead_name, 'hea_sampling_rate': fs, 'sampling_rate_source': fs_source, **attempt} for attempt in attempts]
            rr_qc_rows.extend(subject_qc)
            _append_jsonl(rr_qc_jsonl, subject_qc)
            sample_data.pop('ecg_data', None)
            sample_data.pop('lead_data', None)
            if selected_lead is None or rr_result is None:
                _, inventory = segmenter.segment_windows_with_inventory(np.array([], dtype=float), np.array([], dtype=float), record_datetime, total_duration_sec=analysis_duration)
                audit_rows = []
                for row in inventory:
                    audit_rows.append({**base, 'primary_lead': lead_name, 'selected_lead': None, **row, 'entropy_eligible': False, 'legacy_layer5_eligible': False, 'reject_stage': 'layer2', 'reject_reason': 'rr_extraction_failed'})
                audit_df = pd.DataFrame(audit_rows).reindex(columns=WINDOW_AUDIT_COLUMNS)
                _append_dataframe_csv(window_audit_csv, audit_df, WINDOW_AUDIT_COLUMNS)
                total_nominal_windows += len(audit_df)
                rows = _build_eligibility_rows(evaluator, audit_df, base=base, hea_duration_seconds=hea_duration, csv_duration_seconds=csv_duration, duration_delta_seconds=duration_delta, rr_span_seconds=None, analysis_duration_seconds=analysis_duration, analysis_duration_policy=analysis_duration_policy, duration_truncated_to_analysis_window=duration_truncated, analysis_tracks=['sinus_primary_v2', 'af_like_exploratory_v3'])
                eligibility_rows_all.extend(rows)
                _append_jsonl(eligibility_jsonl, rows)
                n_fail += 1
                failure_reason_counts['rr_extraction_failed'] += 1
                status = {**base, **_computation_status_fields('rr_extraction_failed'), 'eligibility_policy_id': ELIGIBILITY_POLICY_ID, 'eligible_primary_analysis': False, 'eligible_primary_18h_sensitivity': False, 'eligible_all_rhythm_20h_sensitivity': False, 'eligible_all_rhythm_sensitivity': False, 'eligible_af_like_exploratory': False, 'n_nominal_windows': len(audit_df), 'n_entropy_eligible_windows': 0}
                status_rows.append(status)
                _append_jsonl(status_jsonl, [status])
                print(' FAIL (RR)')
                continue
            selected_lead_counts[selected_lead] += 1
            if selected_lead != lead_name:
                n_fallback_used += 1
            rr_raw = np.asarray(rr_result.get('rr_raw', rr_result.get('rr', [])), dtype=float)
            rr_ts_raw = np.asarray(rr_result.get('rr_ts_raw', rr_result.get('rr_ts', [])), dtype=float)
            analysis_rr_mask = rr_ts_raw <= analysis_duration if analysis_duration is not None else np.ones(len(rr_ts_raw), dtype=bool)
            analysis_rr_ts = rr_ts_raw[analysis_rr_mask]
            rr_span_seconds = float(analysis_rr_ts[-1] - analysis_rr_ts[0]) if len(analysis_rr_ts) > 1 else None
            record_rr_exporter.append_record({**base, 'primary_lead': lead_name, 'selected_lead': selected_lead, 'sampling_rate': fs, 'sampling_rate_source': fs_source, 'r_peaks': rr_result.get('r_peaks', []), 'rr_raw': rr_raw, 'rr_ts_raw': rr_ts_raw, 'rr_physiological': rr_result.get('rr', []), 'rr_ts_physiological': rr_result.get('rr_ts', []), 'rr_physiological_mask': rr_result.get('rr_physiological_mask', []), 'rr_diagnostics': rr_result.get('diagnostics', {}), 'rr_quality': rr_result.get('quality', {})})
            windows, inventory = segmenter.segment_windows_with_inventory(rr_raw, rr_ts_raw, record_datetime, total_duration_sec=analysis_duration)
            for window in windows:
                window.update({'lead': selected_lead, 'ecg_id': ecg_id, 'group_code': base['group_code']})
            candidate_windows = clock_sampler.stratified_sample(windows)
            n_nominal = len(inventory)
            n_candidate = len(candidate_windows)
            total_nominal_windows += n_nominal
            total_candidate_windows += n_candidate
            audit_rows = []
            audit_by_index = {}
            for row in inventory:
                audit = {**base, 'primary_lead': lead_name, 'selected_lead': selected_lead, **row, 'entropy_eligible': False, 'legacy_layer5_eligible': False, 'reject_stage': '' if row['layer3_eligible'] else 'layer3', 'reject_reason': row['layer3_reject_reason']}
                audit_rows.append(audit)
                audit_by_index[row['window_idx']] = audit
            for window in candidate_windows:
                audit = audit_by_index[window['window_idx']]
                try:
                    rr_detrended = detrending.detrend(window['rr']) if detrending is not None else window['rr']
                    rhythm_result = rhythm_processor.process_window(window)
                    rhythm_type = rhythm_result.get('rhythm_type', 'Unknown')
                    confidence = float(rhythm_result.get('confidence', 0.0))
                    rhythm_quality = rhythm_result.get('quality', 'unknown')
                    scs, dimensions = sinus_scorer.score(rhythm_result)
                    audit.update({'rhythm_type': rhythm_type, 'rhythm_type_legacy': rhythm_result.get('rhythm_type_legacy', rhythm_type), 'rhythm_label_semantic': rhythm_result.get('rhythm_label_semantic', rhythm_type), 'rhythm_confidence': confidence, 'rhythm_method': rhythm_result.get('method', ''), 'rhythm_classification_policy': rhythm_result.get('classification_policy'), 'rhythm_score_type': rhythm_result.get('score_type'), 'rhythm_quality': rhythm_quality, 'rhythm_processing_reason': rhythm_result.get('processing_reason'), 'ectopy_rate': rhythm_result.get('ectopy_rate'), 'interpolation_success': rhythm_result.get('interpolation_success'), 'interpolation_applied': rhythm_result.get('interpolation_applied'), 'rr_mean_raw_window': rhythm_result.get('rr_mean_raw_window'), 'rr_std_raw_window': rhythm_result.get('rr_std_raw_window'), 'rr_cv_raw_window': rhythm_result.get('rr_cv_raw_window'), 'rr_mean_final_window': rhythm_result.get('rr_mean_final_window'), 'rr_std_final_window': rhythm_result.get('rr_std_final_window'), 'rr_cv_final_window': rhythm_result.get('rr_cv_final_window'), 'rr_variance_ratio': rhythm_result.get('rr_variance_ratio'), 'rr_near_constant_raw': rhythm_result.get('rr_near_constant_raw'), 'rr_near_constant_final': rhythm_result.get('rr_near_constant_final'), 'interpolation_variance_collapse': rhythm_result.get('interpolation_variance_collapse'), 'sinus_confidence_score': scs})
                    legacy_high_af = qc.skip_high_afib and rhythm_type == 'AFib' and (confidence > qc.afib_confidence_high)
                    legacy_scs_pass = qc.scs_threshold <= 0 or scs >= qc.scs_threshold or rhythm_type in ('Sinus', 'Sinus_HV', 'Uncertain')
                    legacy_eligible = bool(not legacy_high_af and legacy_scs_pass)
                    audit['legacy_layer5_eligible'] = legacy_eligible
                    if rhythm_type in ('Sinus', 'Sinus_HV'):
                        analysis_track = 'sinus_primary_v2'
                        analysis_track_legacy = 'sinus_primary_v2'
                    elif rhythm_type == 'AFib':
                        analysis_track = 'af_like_exploratory_v3'
                        analysis_track_legacy = 'afib_primary_v2'
                    else:
                        analysis_track = 'uncertain_unclassified'
                        analysis_track_legacy = 'uncertain_unclassified'
                    audit['analysis_track'] = analysis_track
                    audit['analysis_track_legacy'] = analysis_track_legacy
                    if analysis_policy == 'dual_track_v2':
                        active_eligible = rhythm_quality == 'accepted' and analysis_track in ('sinus_primary_v2', 'af_like_exploratory_v3')
                    else:
                        active_eligible = legacy_eligible and rhythm_quality == 'accepted' and (rhythm_type in ('Sinus', 'Sinus_HV'))
                        analysis_track = 'sinus_strict_legacy'
                        audit['analysis_track'] = analysis_track
                        audit['analysis_track_legacy'] = analysis_track
                    if not active_eligible:
                        reason = rhythm_quality if rhythm_quality not in ('accepted', 'uncertain') else 'uncertain_rhythm' if rhythm_type == 'Uncertain' else 'analysis_policy_rejected'
                        audit.update({'reject_stage': 'layer5', 'reject_reason': reason})
                        rejection_counts[reason] += 1
                        continue
                    rr_final = np.asarray(rhythm_result.get('rr_final', rr_detrended), dtype=float)
                    finite = np.isfinite(rr_final)
                    rr_final = rr_final[finite]
                    rr_ts_final = rhythm_result.get('rr_ts_final')
                    if rr_ts_final is not None:
                        rr_ts_final = np.asarray(rr_ts_final, dtype=float)
                        if len(rr_ts_final) == len(finite):
                            rr_ts_final = rr_ts_final[finite]
                    if len(rr_final) < effective_window_rr_min:
                        audit.update({'reject_stage': 'layer5', 'reject_reason': 'rr_after_clean_below_min'})
                        rejection_counts['rr_after_clean_below_min'] += 1
                        continue
                    if compute_full_features:
                        layer6_input = _assess_layer6_input(len(rr_final), layer6_min_rr_required)
                        audit.update(layer6_input)
                        if not layer6_input['layer6_input_eligible']:
                            audit.update({'rr_n_input': len(rr_final), 'rcmse_quality': 'insufficient_length', 'layer6_reject_reason': 'rr_below_rcmse_min', 'feature_quality': 'rejected', 'reject_stage': 'layer6', 'reject_reason': 'rr_below_rcmse_min'})
                            rejection_counts['rr_below_rcmse_min'] += 1
                            continue
                        feature_input = {**window, 'rr_final': rr_final, 'n_beats': len(rr_final)}
                        features = feature_processor.process_window(feature_input)
                        for key in ('rr_n_input', 'rcmse_quality', 'layer6_reject_reason'):
                            audit[key] = features.get(key)
                        mse_long = features.get('MSE_long', np.nan)
                        mse_short = features.get('MSE_short', np.nan)
                        dfa_alpha2 = features.get('DFA_alpha2', np.nan)
                        audit['mse_range_flag'] = bool(np.isfinite(mse_long) and (not qc.mse_valid_range[0] < mse_long < qc.mse_valid_range[1]))
                        audit['dfa_range_flag'] = bool(np.isfinite(dfa_alpha2) and (not qc.dfa_valid_range[0] < dfa_alpha2 < qc.dfa_valid_range[1]))
                        cxi_ratio = features.get('CXI_ratio', np.nan)
                        audit['cxi_afib_flag'] = bool(np.isfinite(cxi_ratio) and cxi_ratio > qc.cxi_afib_threshold and (rhythm_type not in ('Sinus', 'Sinus_HV')))
                        if not np.isfinite(mse_long) or not np.isfinite(dfa_alpha2):
                            layer6_reason = features.get('layer6_reject_reason') or 'required_feature_missing_unknown'
                            audit.update({'reject_stage': 'layer6', 'reject_reason': layer6_reason})
                            audit['feature_quality'] = features.get('feature_quality')
                            rejection_counts[layer6_reason] += 1
                            continue
                    else:
                        audit.update({'layer6_input_rr_count': len(rr_final), 'layer6_min_rr_required': np.nan, 'layer6_input_eligible': np.nan})
                        features = _build_rhythm_only_features(window, rr_final)
                    rr_diagnostics = rr_result.get('diagnostics', {})
                    rr_quality = rr_result.get('quality', {})
                    features.update({**base, 'lead': selected_lead, 'analysis_track': analysis_track, 'analysis_track_legacy': analysis_track_legacy, 'is_death': sample_data.get('is_death', 0), 'timestamp_valid': True, 'hea_duration_seconds': hea_duration, 'csv_duration_seconds': csv_duration, 'duration_delta_seconds': duration_delta, 'hea_sampling_rate': fs, 'sampling_rate_source': fs_source, 'hea_n_leads': hea_info.get('n_leads'), 'rhythm_type': rhythm_type, 'rhythm_type_legacy': rhythm_result.get('rhythm_type_legacy', rhythm_type), 'rhythm_label_semantic': rhythm_result.get('rhythm_label_semantic', rhythm_type), 'rhythm_confidence': confidence, 'rhythm_method': rhythm_result.get('method', ''), 'rhythm_classification_policy': rhythm_result.get('classification_policy'), 'rhythm_score_type': rhythm_result.get('score_type'), 'rhythm_quality': rhythm_quality, 'rhythm_processing_reason': rhythm_result.get('processing_reason'), 'rhythm_ectopy_rate': rhythm_result.get('ectopy_rate'), 'rhythm_cosen': rhythm_result.get('features', {}).get('cosen', np.nan), 'rhythm_pRR31': rhythm_result.get('features', {}).get('pRR31', np.nan), 'rhythm_cv': rhythm_result.get('features', {}).get('cv', np.nan), 'sinus_confidence_score': scs, 'scs_d1_rhythm': dimensions.get('D1_rhythm_classifier', np.nan), 'scs_d2_regularity': dimensions.get('D2_rr_regularity', np.nan), 'scs_d3_ectopy': dimensions.get('D3_ectopy', np.nan), 'scs_d4_interpolation': dimensions.get('D4_interpolation', np.nan), 'scs_d5_plausibility': dimensions.get('D5_plausibility', np.nan), 'quality_tier': qc.tier_name, 'rr_detector_mode': rr_diagnostics.get('detector_mode', rr_detector_mode), 'rr_detector_method': rr_diagnostics.get('detector_method', ''), 'rr_selected_polarity': rr_diagnostics.get('selected_polarity', ''), 'rr_chunk_sec': rr_diagnostics.get('chunk_duration_sec', rr_chunk_sec), 'rr_transient_prefix_sec': rr_diagnostics.get('transient_prefix_sec', 0.0), 'rr_primary_lead': lead_name, 'rr_selected_lead': selected_lead, 'rr_lead_fallback_used': selected_lead != lead_name, 'rr_n_peaks': int(len(rr_result.get('r_peaks', []))), 'rr_n_intervals': int(len(rr_result.get('rr', []))), 'rr_n_intervals_raw': int(len(rr_raw)), 'rr_valid_fraction': rr_diagnostics.get('rr_valid_fraction', np.nan), 'rr_quality_score': rr_quality.get('quality_score', np.nan), 'window_sec': int(window_sec), 'feature_mode': feature_mode, 'n_nominal_windows': n_nominal, 'n_candidate_windows': n_candidate})
                    scalar_features = _scalarize_features(features)
                    window_features.append(scalar_features)
                    audit.update({'entropy_eligible': True, 'feature_quality': features.get('feature_quality'), 'reject_stage': '', 'reject_reason': ''})
                    if export_window_rr:
                        rr_export_records.append(build_rr_export_record(ecg_id=ecg_id, lead=selected_lead, window=window, rr_raw=window.get('rr'), rr_ts_raw=window.get('rr_ts'), rr_final=rr_final, rr_ts_final=rr_ts_final, rhythm_type=rhythm_type, rhythm_confidence=confidence, rhythm_method=rhythm_result.get('method', ''), rhythm_features=rhythm_result.get('features', {}), sinus_confidence_score=scs, quality_tier=qc.tier_name, analysis_track=analysis_track, rhythm_quality=rhythm_quality, processing_reason=rhythm_result.get('processing_reason')))
                except Exception as exc:
                    n_window_errors += 1
                    rejection_counts['window_error'] += 1
                    audit.update({'reject_stage': 'window_exception', 'reject_reason': f'{type(exc).__name__}: {exc}'})
                    if n_window_errors <= 3:
                        print(f"\n    [WARN] window {window['window_idx']}: {exc}")
            audit_df = pd.DataFrame(audit_rows).reindex(columns=WINDOW_AUDIT_COLUMNS)
            _append_dataframe_csv(window_audit_csv, audit_df, WINDOW_AUDIT_COLUMNS)
            tracks = ['sinus_primary_v2', 'af_like_exploratory_v3'] if analysis_policy == 'dual_track_v2' else ['sinus_strict_legacy']
            eligibility_rows = _build_eligibility_rows(evaluator, audit_df, base=base, hea_duration_seconds=hea_duration, csv_duration_seconds=csv_duration, duration_delta_seconds=duration_delta, rr_span_seconds=rr_span_seconds, analysis_duration_seconds=analysis_duration, analysis_duration_policy=analysis_duration_policy, duration_truncated_to_analysis_window=duration_truncated, analysis_tracks=tracks)
            eligibility_rows_all.extend(eligibility_rows)
            _append_jsonl(eligibility_jsonl, eligibility_rows)
            summary_by_track = {row['analysis_track']: row for row in eligibility_rows[1:]}
            subject_output_created = False
            if window_features:
                window_df = pd.DataFrame(window_features)
                if write_window_csv:
                    _append_dataframe_csv(windows_csv, window_df)
                if export_window_rr:
                    rr_exporter.append_records(rr_export_records)
                total_windows += len(window_df)
                subject_df = SubjectAggregator(aggregation_method='median', variability_method='iqr', compute_spatial=False).aggregate(window_df)
                if not subject_df.empty:
                    for column in ('nyha_grade', 'group', 'analysis_population', 'patient_uid', 'is_death', 'center_name', 'record_datetime', 'timestamp_source'):
                        subject_df[column] = window_df[column].iloc[0]
                    for row_index, subject_row in subject_df.iterrows():
                        track_summary = summary_by_track.get(subject_row.get('analysis_track'), {})
                        for key, value in track_summary.items():
                            if key not in base and key != 'analysis_track':
                                subject_df.at[row_index, key] = value
                    _append_dataframe_csv(subjects_csv, subject_df)
                    subject_output_created = True
            overall = eligibility_rows[0]
            if subject_output_created:
                n_success += 1
                status_name = 'success'
                print(f" OK ({len(window_features)}/{n_candidate} entropy windows, analysis_duration={overall['analysis_duration_hours']:.2f}h, source_duration={overall['source_duration_hours']:.2f}h)")
            else:
                n_fail += 1
                status_name = 'no_entropy_eligible_windows'
                failure_reason_counts[status_name] += 1
                print(f' FAIL (0/{n_candidate} entropy windows)')
            status = {**base, **_computation_status_fields(status_name), 'eligibility_policy_id': ELIGIBILITY_POLICY_ID, 'primary_lead': lead_name, 'selected_lead': selected_lead, 'fallback_used': selected_lead != lead_name, 'n_rr_raw': int(len(rr_raw)), 'n_rr_physiological': int(len(rr_result.get('rr', []))), 'n_nominal_windows': n_nominal, 'n_rr_eligible_windows': n_candidate, 'n_entropy_eligible_windows': len(window_features), 'duration_20h_pass': overall['duration_20h_pass'], 'duration_18h_pass': overall['duration_18h_pass'], 'analysis_duration_hours': overall['analysis_duration_hours'], 'analysis_duration_policy': analysis_duration_policy, 'duration_truncated_to_analysis_window': duration_truncated, 'duration_truncated_strict': duration_metadata['duration_truncated_strict'], 'duration_truncation_seconds': duration_metadata['duration_truncation_seconds'], 'meaningful_duration_truncation': duration_metadata['meaningful_duration_truncation'], 'coverage_pass_all_tracks': overall['coverage_pass'], 'eligible_all_rhythm_sensitivity': overall['eligible_all_rhythm_sensitivity'], 'eligible_all_rhythm_20h_sensitivity': overall['eligible_all_rhythm_20h_sensitivity'], 'eligible_primary_analysis': summary_by_track.get('sinus_primary_v2', {}).get('eligible_primary_analysis', False), 'eligible_primary_18h_sensitivity': summary_by_track.get('sinus_primary_v2', {}).get('eligible_primary_18h_sensitivity', False), 'eligible_af_like_exploratory': summary_by_track.get('af_like_exploratory_v3', {}).get('eligible_af_like_exploratory', False), 'legacy_seven_window_computational_pass': len(window_features) >= qc.min_windows_threshold, 'n_window_errors': n_window_errors, **{f'n_rejected_{reason}': int(count) for reason, count in sorted(rejection_counts.items())}}
            status_rows.append(status)
            _append_jsonl(status_jsonl, [status])
            del sample_data, rr_result, window_features, rr_export_records, audit_df
            gc.collect()
        except Exception as exc:
            n_fail += 1
            reason = 'load_failed' if str(exc) == 'load_failed' else 'record_error'
            failure_reason_counts[reason] += 1
            error_row = {'ecg_id': ecg_id, **_computation_status_fields(reason), 'eligibility_policy_id': ELIGIBILITY_POLICY_ID, 'eligible_primary_analysis': False, 'eligible_primary_18h_sensitivity': False, 'eligible_all_rhythm_20h_sensitivity': False, 'eligible_all_rhythm_sensitivity': False, 'eligible_af_like_exploratory': False, 'primary_lead': lead_name, 'selected_lead': selected_lead, 'error': f'{type(exc).__name__}: {exc}'}
            status_rows.append(error_row)
            _append_jsonl(status_jsonl, [error_row])
            rr_qc_rows.append(error_row)
            _append_jsonl(rr_qc_jsonl, [error_row])
            print(f' ERROR ({exc})')
            gc.collect()
    all_rr_qc_rows = _read_jsonl(rr_qc_jsonl)
    all_status_rows = _read_jsonl(status_jsonl)
    all_eligibility_rows = _read_jsonl(eligibility_jsonl)
    if all_rr_qc_rows:
        rr_qc_frame = pd.DataFrame(all_rr_qc_rows)
        rr_qc_keys = [key for key in ('ecg_id', 'attempt_order', 'lead') if key in rr_qc_frame]
        if rr_qc_keys:
            rr_qc_frame = rr_qc_frame.drop_duplicates(rr_qc_keys, keep='last')
        rr_qc_frame.to_csv(rr_qc_csv, index=False, encoding='utf-8-sig')
    status_frame = pd.DataFrame()
    if all_status_rows:
        status_frame = pd.DataFrame(all_status_rows).drop_duplicates('ecg_id', keep='last')
        status_frame.to_csv(status_csv, index=False, encoding='utf-8-sig')
    if all_eligibility_rows:
        eligibility_frame = pd.DataFrame(all_eligibility_rows)
        eligibility_frame = eligibility_frame.drop_duplicates(['ecg_id', 'analysis_track'], keep='last')
        if all_status_rows:
            final_ids = set(status_frame['ecg_id'].astype(str))
            eligibility_frame = eligibility_frame[eligibility_frame['ecg_id'].astype(str).isin(final_ids)]
        eligibility_frame.to_csv(eligibility_csv, index=False, encoding='utf-8-sig')
    batch_n_success = int(_csv_boolean_mask(status_frame['computation_success']).sum()) if not status_frame.empty and 'computation_success' in status_frame else int(status_frame['status'].eq('success').sum()) if not status_frame.empty else 0
    batch_n_fail = int(len(status_frame) - batch_n_success)
    batch_selected_lead_counts = status_frame['selected_lead'].dropna().value_counts().to_dict() if 'selected_lead' in status_frame else {}
    batch_n_fallback_used = int(_csv_boolean_mask(status_frame['fallback_used']).sum()) if 'fallback_used' in status_frame else 0
    batch_failure_reason_counts = status_frame.loc[~status_frame['status'].eq('success'), 'status'].value_counts().to_dict() if not status_frame.empty else {}
    batch_n_windows = total_windows
    batch_n_candidate_windows = total_candidate_windows
    batch_n_nominal_windows = total_nominal_windows
    if window_audit_csv.exists():
        batch_audit = pd.read_csv(window_audit_csv, usecols=['layer3_eligible', 'entropy_eligible'], low_memory=False)
        batch_n_nominal_windows = int(len(batch_audit))
        batch_n_candidate_windows = int(_csv_boolean_mask(batch_audit['layer3_eligible']).sum())
        batch_n_windows = int(_csv_boolean_mask(batch_audit['entropy_eligible']).sum())
    return {'window_csv': str(windows_csv), 'window_audit_csv': str(window_audit_csv), 'subject_csv': str(subjects_csv), 'record_eligibility_csv': str(eligibility_csv), 'record_eligibility_jsonl': str(eligibility_jsonl), 'n_success': batch_n_success, 'n_fail': batch_n_fail, 'n_computation_success': batch_n_success, 'n_computation_fail': batch_n_fail, 'n_records': int(len(status_frame)), 'n_success_this_run': n_success, 'n_fail_this_run': n_fail, 'n_processed_this_run': int(len(pending_ids)), 'n_windows': batch_n_windows, 'n_candidate_windows': batch_n_candidate_windows, 'n_nominal_windows': batch_n_nominal_windows, 'n_windows_this_run': total_windows, 'n_candidate_windows_this_run': total_candidate_windows, 'n_nominal_windows_this_run': total_nominal_windows, 'entropy_acceptance_ratio_rr_eligible': float(batch_n_windows / batch_n_candidate_windows) if batch_n_candidate_windows else np.nan, 'entropy_acceptance_ratio_nominal': float(batch_n_windows / batch_n_nominal_windows) if batch_n_nominal_windows else np.nan, 'window_csv_written': bool(write_window_csv and windows_csv.exists()), 'feature_mode': feature_mode, 'analysis_policy': analysis_policy, 'analysis_duration_policy': analysis_duration_policy, 'analysis_duration_hours': float(analysis_duration_hours), 'duration_truncation_tolerance_seconds': DURATION_TRUNCATION_TOLERANCE_SECONDS, 'eligibility_policy_id': ELIGIBILITY_POLICY_ID, 'pipeline_version': PIPELINE_VERSION, 'rhythm_classification_policy': rhythm_classification_policy, 'strict_input': bool(strict_input), 'window_rr_pklstream': str(rr_exporter.path) if export_window_rr else None, 'n_rr_exported': int(rr_exporter.n_records), 'record_rr_cache_pklstream': str(record_rr_exporter.path) if export_record_rr_cache else None, 'n_record_rr_cached': int(record_rr_exporter.n_records), 'constant_rr_std_epsilon': float(constant_rr_std_epsilon), 'interpolation_variance_ratio_min': float(interpolation_variance_ratio_min), 'rr_qc_csv': str(rr_qc_csv) if rr_qc_csv.exists() else None, 'rr_qc_jsonl': str(rr_qc_jsonl) if rr_qc_jsonl.exists() else None, 'record_status_csv': str(status_csv) if status_csv.exists() else None, 'record_status_jsonl': str(status_jsonl) if status_jsonl.exists() else None, 'rr_detector_mode': rr_detector_mode, 'rr_chunk_sec': float(rr_chunk_sec), 'rr_chunk_overlap_sec': float(rr_chunk_overlap_sec), 'rr_polarity_mode': rr_polarity_mode, 'fallback_leads': fallback_leads, 'n_fallback_used': batch_n_fallback_used, 'selected_lead_counts': batch_selected_lead_counts, 'failure_reason_counts': batch_failure_reason_counts, 'n_fallback_used_this_run': int(n_fallback_used), 'selected_lead_counts_this_run': dict(selected_lead_counts), 'failure_reason_counts_this_run': dict(failure_reason_counts), 'n_resumed_skipped': len(ecg_ids) - len(pending_ids)}
