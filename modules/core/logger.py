"""Provides logger functions for the ECG-to-entropy pipeline."""
from loguru import logger
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class ExperimentLogger:

    def __init__(self, log_dir: str, experiment_id: Optional[str]=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if experiment_id is None:
            experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = experiment_id
        self.stage_timers = {}
        self.current_stage = None
        logger.remove()
        logger.add(sys.stdout, format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>', level='INFO', colorize=True)
        logger.add(str(self.log_dir / f'execution_{experiment_id}.log'), format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}', level='DEBUG', rotation='500 MB', encoding='utf-8')
        logger.add(str(self.log_dir / f'error_{experiment_id}.log'), format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file}:{line} | {message}', level='ERROR', encoding='utf-8')
        logger.info(f'Logger initialized: experiment_id={experiment_id}')
        logger.info(f'Log directory: {self.log_dir}')

    def log_stage(self, stage_name: str, status: str='Start'):
        if status == 'Start':
            logger.info(f"{'=' * 60}")
            logger.info(f'[TASK] {status}: {stage_name}')
            logger.info(f"{'=' * 60}")
            self.stage_timers[stage_name] = datetime.now()
            self.current_stage = stage_name
        elif stage_name in self.stage_timers:
            elapsed = (datetime.now() - self.stage_timers[stage_name]).total_seconds()
            logger.success(f'[TASK] {status}: {stage_name} (耗时: {elapsed:.1f}s)')
        else:
            logger.success(f'[TASK] {status}: {stage_name}')

    def log_progress(self, current: int, total: int, item_name: str='items'):
        percentage = current / total * 100 if total > 0 else 0
        logger.info(f'[PROGRESS] {item_name}: {current}/{total} ({percentage:.1f}%)')

    def log_quality(self, sample_id: str, metrics: dict):
        logger.debug(f'[QUALITY] {sample_id}: {metrics}')

    def log_error_continue(self, sample_id: str, error: Exception):
        logger.error(f'[ERROR] {sample_id} failed: {str(error)}')
        logger.error(f'[ERROR] Continuing with next sample...')

    def log_summary(self, summary: dict):
        logger.info(f"{'=' * 60}")
        logger.info('[REPORT] Execution Summary:')
        for key, value in summary.items():
            logger.info(f'[REPORT]   {key}: {value}')
        logger.info(f"{'=' * 60}")
_global_logger: Optional[ExperimentLogger] = None

def init_logger(log_dir: str, experiment_id: Optional[str]=None) -> ExperimentLogger:
    global _global_logger
    _global_logger = ExperimentLogger(log_dir, experiment_id)
    return _global_logger

def get_logger() -> ExperimentLogger:
    if _global_logger is None:
        raise RuntimeError('Logger not initialized. Call init_logger() first.')
    return _global_logger

def log_info(msg: str):
    logger.info(msg)

def log_debug(msg: str):
    logger.debug(msg)

def log_warning(msg: str):
    logger.warning(msg)

def log_error(msg: str):
    logger.error(msg)

def log_success(msg: str):
    logger.success(msg)
