"""Provides quality profiles functions for the ECG-to-entropy pipeline."""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

@dataclass(frozen=True)
class QualityConfig:
    tier_name: str
    min_rr_count: int
    rr_physiological_range: Tuple[float, float]
    window_rr_min: int
    scs_threshold: float
    skip_high_afib: bool
    afib_confidence_high: float
    cxi_afib_filter: bool
    cxi_afib_threshold: float
    mse_valid_range: Tuple[float, float]
    dfa_valid_range: Tuple[float, float]
    min_windows_threshold: int
SINUS_STRICT = QualityConfig(tier_name='sinus_strict', min_rr_count=100, rr_physiological_range=(0.3, 2.0), window_rr_min=100, scs_threshold=0.55, skip_high_afib=True, afib_confidence_high=0.85, cxi_afib_filter=True, cxi_afib_threshold=1.5, mse_valid_range=(0.1, 3.0), dfa_valid_range=(0.3, 2.0), min_windows_threshold=7)
AFIB_ADAPTIVE = QualityConfig(tier_name='afib_adaptive', min_rr_count=100, window_rr_min=100, mse_valid_range=(0.1, 3.0), dfa_valid_range=(0.3, 2.0), afib_confidence_high=0.85, cxi_afib_threshold=1.5, rr_physiological_range=(0.25, 2.0), scs_threshold=0.0, skip_high_afib=False, cxi_afib_filter=False, min_windows_threshold=5)
LEAD_GENERAL = QualityConfig(tier_name='lead_general', min_rr_count=50, rr_physiological_range=(0.25, 2.5), window_rr_min=50, scs_threshold=0.0, skip_high_afib=False, afib_confidence_high=0.0, cxi_afib_filter=False, cxi_afib_threshold=999.0, mse_valid_range=(0.05, 5.0), dfa_valid_range=(0.2, 2.5), min_windows_threshold=3)
PROFILES: Dict[str, QualityConfig] = {'sinus_strict': SINUS_STRICT, 'afib_adaptive': AFIB_ADAPTIVE, 'lead_general': LEAD_GENERAL}
LEAD_QUALITY_MAP: Dict[str, str] = {'II': 'high_quality', 'V5': 'high_quality', 'V6': 'high_quality', 'I': 'medium_quality', 'III': 'medium_quality', 'V1': 'medium_quality', 'V2': 'medium_quality', 'V3': 'medium_quality', 'V4': 'medium_quality', 'aVR': 'low_quality', 'aVL': 'low_quality', 'aVF': 'low_quality'}
DEFAULT_MIN_DISTANCE: Dict[str, float] = {'high_quality': 0.3, 'medium_quality': 0.32, 'low_quality': 0.34}

def get_quality_config(tier: str) -> QualityConfig:
    if tier not in PROFILES:
        raise ValueError(f'Unknown quality tier: {tier}. Available: {list(PROFILES.keys())}')
    return PROFILES[tier]

def get_lead_min_distance(lead_name: str, calibrated_params: Optional[Dict]=None) -> float:
    quality_level = LEAD_QUALITY_MAP.get(lead_name, 'medium_quality')
    if calibrated_params and quality_level in calibrated_params:
        return float(calibrated_params[quality_level])
    return DEFAULT_MIN_DISTANCE[quality_level]

def get_lead_quality_tier(lead_name: str) -> str:
    return LEAD_QUALITY_MAP.get(lead_name, 'medium_quality')
