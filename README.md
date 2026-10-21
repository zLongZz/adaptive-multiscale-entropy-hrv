# Adaptive Multiscale Entropy for Heart Rate Variability

This repository provides an independently developed **adaptive multiscale entropy workflow** for extracting nonlinear heart rate variability (HRV) features from long-term ECG recordings. It converts ECG signals to RR intervals, applies rhythm and quality checks, and calculates entropy across multiple RR scales. The outputs are features for subsequent HRV analysis, physiological research, or model development.

The calculation builds on established sample entropy, multiscale entropy, and refined composite multiscale entropy methods [1–4]. In this implementation, “adaptive” means that the similarity tolerance is recalculated from the coarse-grained RR series **at each scale**. The algorithm does not learn from clinical labels or outcomes.

## Input

- For each recording, a numerical 12-lead `<record_id>.csv` and a matching `<record_id>.hea` header. CSV columns must be ordered `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`; this release analyses lead II.
- The HEA header supplies the sampling rate, recording start time, lead information, and sample count.
- An XLSX metadata file with unique `record_id` (`Record ID` or `记录ID` is also accepted) and `NYHA_std` (`nyha_std` is also accepted). The group field labels outputs; it does not train the entropy calculation. Supported labels are `NYHA_0`, `NYHA_I`, `NYHA_II`, `NYHA_III`, and `NYHA_IV`.

Input data are supplied by the user and are not included in this package.

## License and independent verification

Copyright (c) 2026 Kai Zheng. This repository's code and documentation are available under the [MIT License](LICENSE). You may use, modify, and redistribute them, including for independent verification, provided the copyright and license notice are retained. The license does not cover third-party dependencies or study data.

For a reproducible code review, record the Git commit identifier, use the supplied `environment.yml`, and retain the input format and processing settings described here. `SOURCE_MANIFEST.json` identifies the V2.0 source lineage and hashes of the released code files. Clinical data and original ECG recordings are not in this repository; reproducing study-level numerical results requires access to the corresponding study inputs and analysis materials.

## Environment and use

The included `environment.yml` lists the direct dependencies and versions used for the local package check. Create a separate Conda environment, then run the pipeline from the package directory:

```bash
conda env create -f environment.yml
conda activate entropy_v2_public
python run_entropy.py \
  --data-root /path/to/ecg_csv_hea \
  --metadata-excel /path/to/metadata.xlsx \
  --output-root /path/to/output \
  --n-processes 1
```

Add `--record-id RECORD_ID` to process one recording. The output location must be outside the input and package directories. Other operating systems and dependency combinations have not yet been checked.

If a quality-controlled RR sequence in **seconds** is already available, the entropy calculation can also be called directly:

```python
from modules.layer6_features.rcmse_numba import compute_rcmse_fast

features = compute_rcmse_fast(rr_seconds)
```

This direct call does not perform ECG-to-RR extraction, rhythm assessment, or full-pipeline quality checks.

## Output

The pipeline creates a timestamped directory. `windows/window_features.csv` contains entropy at scales 1–20 (`MSE_1`–`MSE_20`) for eligible 15-minute RR windows, along with short-scale (`MSE_short`, scales 1–5) and long-scale (`MSE_long`, scales 11–20) summaries, related features, and quality indicators. `subjects/subject_features.csv` contains recording-level summaries; status and quality-control files document processed and rejected records or windows.

Invalid scales or windows may yield missing values. Scales count **RR intervals**, not fixed durations in seconds. The outputs are research features; this package does not provide a clinical diagnosis or a trained risk-prediction model. The public package excludes the internal formal-run profile and study data. `SOURCE_MANIFEST.json` records source provenance and file hashes.

## Methodological references

References 1–4 underpin the entropy calculation. Reference 5 documents software used for ECG peak detection.

1. Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy. *Am J Physiol Heart Circ Physiol*. 2000;278(6):H2039–H2049. [doi:10.1152/ajpheart.2000.278.6.H2039](https://doi.org/10.1152/ajpheart.2000.278.6.H2039).
2. Costa M, Goldberger AL, Peng CK. Multiscale entropy analysis of complex physiologic time series. *Phys Rev Lett*. 2002;89(6):068102. [doi:10.1103/PhysRevLett.89.068102](https://doi.org/10.1103/PhysRevLett.89.068102).
3. Costa M, Goldberger AL, Peng CK. Multiscale entropy analysis of biological signals. *Phys Rev E*. 2005;71(2 Pt 1):021906. [doi:10.1103/PhysRevE.71.021906](https://doi.org/10.1103/PhysRevE.71.021906).
4. Wu SD, Wu CW, Lin SG, Lee KY, Peng CK. Analysis of complex time series using refined composite multiscale entropy. *Physics Letters A*. 2014;378(20):1369–1374. [doi:10.1016/j.physleta.2014.03.034](https://doi.org/10.1016/j.physleta.2014.03.034).
5. Makowski D, Pham T, Lau ZJ, Brammer JC, Lespinasse F, Pham H, et al. NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behav Res Methods*. 2021;53(4):1689–1696. [doi:10.3758/s13428-020-01516-y](https://doi.org/10.3758/s13428-020-01516-y).
