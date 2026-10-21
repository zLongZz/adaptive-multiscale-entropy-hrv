"""Provides clock sampler functions for the ECG-to-entropy pipeline."""
import numpy as np
from typing import List, Dict, Any
from modules.core.logger import log_info, log_debug

class ClockSampler:

    def __init__(self, n_strata: int=24, target_total: int=None, random_seed: int=20250810, keep_all: bool=True):
        self.n_strata = n_strata
        self.target_total = target_total
        self.random_seed = random_seed
        self.keep_all = keep_all
        self.rng = np.random.RandomState(random_seed)
        if keep_all:
            log_debug(f'[Layer 3] ClockSampler initialized (KEEP_ALL mode): {n_strata} strata, seed={random_seed}')
        else:
            log_debug(f'[Layer 3] ClockSampler initialized: {n_strata} strata, target={target_total}, seed={random_seed}')

    def stratified_sample(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(windows) == 0:
            log_info('[Layer 3] No windows to sample')
            return []
        strata = {}
        for win in windows:
            hour = win['clock_hour']
            if hour not in strata:
                strata[hour] = []
            strata[hour].append(win)
        n_available_strata = len(strata)
        total_candidates = len(windows)
        log_debug(f'[Layer 3] Stratification: {n_available_strata} hours covered, {total_candidates} candidate windows')
        if self.keep_all:
            log_info(f'[Layer 3] KEEP_ALL mode: retaining all {total_candidates} windows')
            return windows
        target_per_stratum = self.target_total / n_available_strata
        sampled_windows = []
        for hour in range(self.n_strata):
            if hour not in strata:
                continue
            candidates = strata[hour]
            n_candidates = len(candidates)
            n_sample = max(1, int(target_per_stratum))
            n_sample = min(n_sample, n_candidates)
            if n_sample < n_candidates:
                indices = self.rng.choice(n_candidates, n_sample, replace=False)
                selected = [candidates[i] for i in indices]
            else:
                selected = candidates
            sampled_windows.extend(selected)
            log_debug(f'[Layer 3]   Hour {hour:02d}: {n_candidates} candidates → {len(selected)} sampled')
        log_info(f'[Layer 3] Clock stratified sampling: {total_candidates} candidates → {len(sampled_windows)} sampled (target={self.target_total})')
        return sampled_windows

    def sample_by_lead(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        leads = {}
        for win in windows:
            lead = win['lead']
            if lead not in leads:
                leads[lead] = []
            leads[lead].append(win)
        log_info(f'[Layer 3] Sampling {len(leads)} leads independently')
        all_sampled = []
        for lead_name, lead_windows in leads.items():
            sampled = self.stratified_sample(lead_windows)
            all_sampled.extend(sampled)
            log_debug(f'[Layer 3]   Lead {lead_name}: {len(lead_windows)} → {len(sampled)} windows')
        log_info(f'[Layer 3] Total sampled windows (all leads): {len(all_sampled)}')
        return all_sampled

    def compute_coverage(self, windows: List[Dict[str, Any]], total_candidates: int) -> float:
        if total_candidates == 0:
            return 0.0
        coverage = len(windows) / total_candidates
        log_debug(f'[Layer 3] Coverage: {len(windows)}/{total_candidates} = {coverage:.2%}')
        return coverage
