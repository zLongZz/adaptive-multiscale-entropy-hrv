"""Runs the V2.0 ECG-to-entropy extraction pipeline."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from scripts.pipeline_runner import create_loader, run_layer07


def main():
    parser = argparse.ArgumentParser(description="Extract V2.0 entropy features from ECG CSV and HEA files")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-excel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-id")
    parser.add_argument("--n-processes", type=int, default=1)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    metadata_excel = args.metadata_excel.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"Data directory does not exist: {data_root}")
    if not metadata_excel.is_file():
        parser.error(f"Metadata workbook does not exist: {metadata_excel}")
    if args.n_processes < 1:
        parser.error("--n-processes must be positive")
    package_root = Path(__file__).resolve().parent
    if output_root == data_root or output_root == package_root or data_root in output_root.parents or package_root in output_root.parents:
        parser.error("Output directory must be outside the data and package directories")

    loader, _ = create_loader(str(data_root), str(metadata_excel))
    ids = [args.record_id] if args.record_id else [str(item) for item in loader.locator.metadata.index]
    if args.record_id and args.record_id not in loader.locator.metadata.index.astype(str):
        parser.error(f"Record ID is absent from metadata: {args.record_id}")
    if not ids:
        parser.error("No records found in metadata")

    batch = output_root / f"entropy_v2_0_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    batch.mkdir(parents=True, exist_ok=False)
    result = run_layer07(
        loader,
        ids,
        str(batch),
        lead_name="II",
        window_sec=900,
        batch_description="entropy_v2_0_public",
        lambda_param=500.0,
        r_factor=0.15,
        interpolation_threshold=0.20,
        min_length=30,
        min_distance=0.30,
        n_processes=args.n_processes,
        skip_detrend=True,
        quality_tier="sinus_strict",
        feature_mode="full",
        write_window_csv=True,
        export_window_rr=True,
        export_record_rr_cache=True,
        rr_detector_mode="robust",
        rr_chunk_sec=3600.0,
        rr_chunk_overlap_sec=10.0,
        rr_polarity_mode="auto",
        fallback_leads=[],
        analysis_policy="dual_track_v2",
        rhythm_classification_policy="non_entropy_v2",
        primary_duration_hours=20.0,
        sensitivity_duration_hours=18.0,
        analysis_duration_policy="first_24h",
        analysis_duration_hours=24.0,
        constant_rr_std_epsilon=1e-8,
        interpolation_variance_ratio_min=0.05,
        strict_input=True,
    )
    source_hash = hashlib.sha256(metadata_excel.read_bytes()).hexdigest()
    manifest = {
        "pipeline": "entropy_v2_0_public",
        "created_at": datetime.now().isoformat(),
        "metadata_sha256": source_hash,
        "selected_records": len(ids),
        "result": result,
    }
    (batch / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(batch), "n_success": result.get("n_success"), "n_fail": result.get("n_fail"), "n_windows": result.get("n_windows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
