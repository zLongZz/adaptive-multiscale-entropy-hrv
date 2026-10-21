"""Provides config formal functions for the ECG-to-entropy pipeline."""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import time

@dataclass
class PathConfig:
    data_source: str = 'formal'
    formal_data_root: str = ''
    formal_metadata_excel: str = ''
    preliminary_data_root: str = ''
    preliminary_metadata_excel: str = ''

    @property
    def data_root(self) -> str:
        return self.formal_data_root if self.data_source == 'formal' else self.preliminary_data_root

    @property
    def metadata_excel(self) -> str:
        return self.formal_metadata_excel if self.data_source == 'formal' else self.preliminary_metadata_excel
    output_root: str = ''
    log_dir: str = ''

@dataclass
class MetadataConfig:
    formal_column_mapping: Dict[str, str] = field(default_factory=lambda: {'ecg_id': '记录ID', 'nyha_grade': 'NYHA_std', 'group_label': 'group_label', 'age': '年龄(岁)', 'sex': '性别', 'bmi': 'BMI(kg/m^2)', 'lvef': '左室射血分数(LVEF)(%)', 'la_diameter': '左房内径(mm)', 'sbp': '收缩压(mmHg)', 'dbp': '舒张压(mmHg)', 'heart_rate': '心率(次/分)', 'nt_probnp': 'NT-proBNP(pg/mL)', 'bnp': 'BNP(pg/mL)', 'outcome_death': '是否死亡', 'death_date': '死亡时间', 'is_cv_death': '心血管死亡', 'death_cause': '死亡原因', 'beta_blocker': 'β阻滞剂', 'acei': 'ACEI', 'arb': 'ARB', 'diuretic': '利尿剂'})
    preliminary_column_mapping: Dict[str, str] = field(default_factory=lambda: {'ecg_id': '待确定', 'nyha_grade': '待确定', 'group_label': '待确定'})
    sampling_mode: str = 'all'
    stratified_sampling: Dict[str, int] = field(default_factory=lambda: {'NYHA_0': None, 'NYHA_I': None, 'NYHA_II': None, 'NYHA_III': None, 'NYHA_IV': None})
    manual_sampling: Dict[str, int] = field(default_factory=lambda: {'NYHA_0': 50, 'NYHA_I': 50, 'NYHA_II': 50, 'NYHA_III': 50, 'NYHA_IV': 30})
    sampling_random_seed: int = 20260126

@dataclass
class WindowConfig:
    primary_window_sec: int = 900
    sensitivity_windows_sec: List[int] = field(default_factory=lambda: [10, 30, 60, 300, 600, 1200, 1800, 3600])
    overlap_ratio: float = 0.5
    min_beats_per_window: int = 1000

@dataclass
class SamplingConfig:
    n_clock_strata: int = 24
    target_sampling_ratio: float = 1.0
    target_windows_per_sample: Optional[int] = None
    windows_per_layer: Optional[int] = None
    lead_alignment: bool = True
    min_recording_hours: float = 20.0
    min_windows_per_subject_lead: int = 7
    min_coverage_ratio: float = 0.5
    filter_level: str = 'subject_by_lead'
    random_seed: int = 20250810

@dataclass
class PreprocessingConfig:
    smoothness_lambda: float = 500.0
    detrend_scope: str = 'per_window'
    sensitivity_lambda: List[float] = field(default_factory=lambda: [100.0, 500.0, 1000.0])

@dataclass
class RhythmDetectionConfig:
    method: str = 'literature_based'
    confidence_threshold: float = 0.7
    prr31_threshold: float = 68.79
    cosen_threshold_balanced: float = 3.0
    cosen_threshold_sensitive: float = 3.2
    cv_threshold_sensitive: float = 0.2
    cv_threshold_specific: float = 0.25
    cv_threshold_sinus_hv: float = 0.22
    subwindow_60s: bool = True
    subwindow_30beat: bool = True
    sinus_ratio_threshold: float = 0.8
    afib_ratio_threshold: float = 0.8
    paroxysmal_min_ratio: float = 0.2
    max_interpolation_ratio: float = 0.2
    ectopy_detection_method: str = 'local_iqr'
    ectopy_window_size: int = 50
    ectopy_iqr_factor: float = 1.5
    extreme_lower: float = 0.2
    extreme_upper: float = 2.5
    min_remaining_ratio_afib: float = 0.7
    output_detailed_features: bool = True
    sensitivity_interpolation: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

@dataclass
class RCMSEConfig:
    scales: range = field(default_factory=lambda: range(1, 21))
    m: int = 2
    r_factor: float = 0.15
    r_adaptive_type: str = 'scale_specific'
    short_scale_range: Tuple[int, int] = (1, 5)
    long_scale_range: Tuple[int, int] = (11, 20)
    min_coarse_grained_length: int = 30
    sensitivity_r_factors: List[float] = field(default_factory=lambda: [0.1, 0.15, 0.2])

@dataclass
class DFAConfig:
    n_min: int = 8
    n_max: int = 64
    fit_method: str = 'leastsq'

@dataclass
class AggregationConfig:
    aggregation_method: str = 'median'
    variability_method: str = 'iqr'
    min_windows: int = 20
    min_coverage: float = 0.5
    expected_windows_per_24h: int = 24
    sinus_ratio_threshold: float = 0.8
    afib_ratio_threshold: float = 0.8
    mixed_label: str = 'Mixed'
    compute_spatial_complexity: bool = True
    spatial_min_leads: int = 9
    output_format: str = 'pkl'
    save_individual_stats: bool = True
    features_to_aggregate: List[str] = field(default_factory=lambda: ['MSE_long', 'MSE_short', 'CXI', 'DFA_alpha2', 'DFA_intercept', 'DFA_r_squared', 'MSE_1', 'MSE_2', 'MSE_3', 'MSE_4', 'MSE_5', 'MSE_11', 'MSE_12', 'MSE_13', 'MSE_14', 'MSE_15', 'MSE_16', 'MSE_17', 'MSE_18', 'MSE_19', 'MSE_20'])

@dataclass
class CircadianConfig:
    day_start: time = time(8, 0)
    night_start: time = time(20, 0)
    cosinor_min_hours: int = 12
    cosinor_min_r2: float = 0.2
    n_bootstrap_amplitude: int = 1000
    amplitude_test_groups: Tuple[int, int] = (0, 1)
    save_complete_results: bool = True

@dataclass
class StatisticalConfig:
    alpha: float = 0.05
    fdr_method: str = 'bh'
    n_bootstrap: int = 1000
    trend_methods: List[str] = field(default_factory=lambda: ['jonckheere_terpstra', 'theil_sen', 'spearman'])
    kendalls_w_threshold: float = 0.7
    min_samples_per_group: int = 3

@dataclass
class PrognosticConfig:
    follow_up_months: int = 36
    follow_up_days: int = 1080
    outcome_type: str = 'cardiac_death'
    quartile_stratification: bool = True
    base_covariates: List[str] = field(default_factory=lambda: ['lvef', 'nyha_grade', 'age', 'sex'])
    medication_vars: List[str] = field(default_factory=lambda: ['beta_blocker', 'acei_arb', 'diuretic'])
    calculate_incremental: bool = True

@dataclass
class SensitivityConfig:
    oa_t_enabled: bool = True
    param_interpolation_ratios: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    param_lambda_values: List[float] = field(default_factory=lambda: [100.0, 500.0, 1000.0])
    param_window_lengths: List[int] = field(default_factory=lambda: [10, 30, 60, 300, 600, 1200, 1800, 3600])
    param_r_factors: List[float] = field(default_factory=lambda: [0.1, 0.15, 0.2])
    total_runs: int = 9
    robustness_criteria: Dict[str, float] = field(default_factory=lambda: {'direction_flip_tolerance': 0.0, 'significance_stability_min': 0.7, 'effect_size_cv_max': 0.3})

@dataclass
class CalibrationConfig:
    enable: bool = True
    use_cache_if_exists: bool = True
    cache_path: str = 'outputs/calibration/calibrated_params.json'
    small_cohort_threshold: int = 100
    stage0_pilot_md: float = 0.5
    stage0_pilot_ratio: float = 0.1
    stage0_pilot_n_min: int = 18
    stage0_pilot_n_max: int = 200
    stage1_sample_ratio: float = 0.03
    stage1_n_min: int = 12
    stage1_n_max: int = 80
    stage1_test_lead: str = 'II'
    stage2_sample_ratio: float = 0.03
    stage2_n_min: int = 12
    stage2_n_max: int = 80
    stage2_use_quality_stratification: bool = True
    stage3_sample_ratio: float = 0.05
    stage3_n_min: int = 12
    stage3_n_max: int = 120
    stage3_max_iterations: int = 8
    stage3_early_stop_threshold: float = 0.85
    metric_weights: Dict[str, float] = field(default_factory=lambda: {'hr_range': 0.3, 'filter_rate': 0.25, 'hr_cv': 0.15, 'quality': 0.15, 'lead_consistency': 0.1, 'group_gradient': 0.05})
    enable_human_validation: bool = False
    random_seed: int = 20250805
    rr_detector_mode: str = 'robust'
    rr_chunk_sec: float = 3600.0
    rr_chunk_overlap_sec: float = 10.0
    rr_polarity_mode: str = 'auto'

@dataclass
class VisualizationConfig:
    dpi: int = 300
    figure_format: str = 'png'
    save_pdf_also: bool = True
    style: str = 'nature'
    font_family: str = 'Arial'
    font_size: int = 10
    figure_sizes: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {1: (12, 6), 2: (10, 8), 3: (14, 10), 4: (12, 8), 5: (12, 10), 6: (8, 8), 7: (12, 8)})

@dataclass
class PerformanceConfig:
    enable_parallel: bool = True
    n_processes: Optional[int] = None
    parallel_level: str = 'sample'
    use_numba_jit: bool = True
    use_gpu: bool = False
    use_streaming: bool = True
    batch_size: int = 200
    checkpoint_interval: int = 100
    large_data_format: str = 'hdf5'
    small_data_format: str = 'pickle'

@dataclass
class LogConfig:
    console_level: str = 'INFO'
    file_level: str = 'DEBUG'
    execution_log: str = 'execution_{time}.log'
    error_log: str = 'error_{time}.log'
    progress_log: str = 'progress_{time}.log'
    progress_every_n_samples: int = 10

@dataclass
class QualityConfig:
    rr_min: float = 0.3
    rr_max: float = 2.0
    min_signal_quality: float = 0.3
    max_missing_ratio: float = 0.1
    enable_precheck: bool = True

@dataclass
class FormalExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    rhythm_detection: RhythmDetectionConfig = field(default_factory=RhythmDetectionConfig)
    rcmse: RCMSEConfig = field(default_factory=RCMSEConfig)
    dfa: DFAConfig = field(default_factory=DFAConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    circadian: CircadianConfig = field(default_factory=CircadianConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    prognostic: PrognosticConfig = field(default_factory=PrognosticConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self):
        pass

    def save_snapshot(self, output_path: str):
        import json
        from datetime import datetime
        import sys
        snapshot = {'experiment_id': f"formal_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 'timestamp': datetime.now().isoformat(), 'config': self.to_dict(), 'software_versions': {'python': sys.version, 'numpy': __import__('numpy').__version__, 'scipy': __import__('scipy').__version__, 'pandas': __import__('pandas').__version__, 'neurokit2': __import__('neurokit2').__version__, 'lifelines': __import__('lifelines').__version__, 'scikit-learn': __import__('sklearn').__version__}}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, default=str, ensure_ascii=False)
        print(f'[INFO] Config snapshot saved: {output_path}')

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)

def load_config() -> FormalExperimentConfig:
    config = FormalExperimentConfig()
    print('[INFO] Configuration loaded')
    print(f'[INFO] Data root: {config.paths.data_root}')
    print(f'[INFO] Output root: {config.paths.output_root}')
    print(f'[INFO] Primary window: {config.windows.primary_window_sec}s')
    print(f'[INFO] Parallel: {config.performance.enable_parallel}')
    print(f'[INFO] Numba JIT: {config.performance.use_numba_jit}')
    return config
