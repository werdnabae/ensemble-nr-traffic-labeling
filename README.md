# Traffic State Based Labeling of Nonrecurrent Traffic Disturbances

Reference implementation for:

> **Traffic State Based Labeling of Nonrecurrent Disturbances from Speed Data
> Using Interpretable Ensemble Detection**  
> Andrew J. Bae — Carnegie Mellon University

---

## Overview

This repository provides the code for an interpretable ensemble framework that
constructs link-level nonrecurrent traffic disturbance labels directly from
probe-based speed observations. The framework combines three speed-based
detectors — robust deviation, temporal gradient, and upstream slowdown contrast
— with persistence filtering, free-flow speed confirmation, and traffic-driven
recovery logic to produce coherent disturbance episodes with well-defined onset
and termination times.

Incident reports serve as optional supporting evidence only. The primary
evaluation metric is **abnormal delay capture** — the fraction of excess travel
time (speed below the estimated recurrent lower bound) that is covered by each
method's labels.

---

## Repository structure

```
src/
  ensemble_labeling.py        Core detectors, thresholds, episode construction,
                              recovery logic, and the main run_ensemble_labeler()
  duan_baseline.py            Duan et al. (2024) slowdown-based baseline
  delay_metrics.py            Excess and abnormal delay computation
  timing_metrics.py           Episode extraction, detection latency, boundary analysis
  run_revised_experiments.py  Main pipeline — runs all experiments and saves results

data/
  cranberry/                  ← place Cranberry data files here (see below)
  tsmo/                       ← place TSMO data files here (see below)

final_outputs/
  tables/                     Pre-computed result tables (CSV and Parquet)
  figures/                    Paper figures (PNG)
  ml_baseline_xgb.pkl         Trained XGBoost baseline model (TSMO calibration)
  ml_simple_xgb.pkl           Trained XGBoost model (simple threshold variant)

README.md
requirements.txt
.gitignore
```

---

## Data

**Data cannot be included due to a data-use agreement with the provider.**

### Expected file layout

```
data/
  cranberry/
    cranberry_speed_data.parquet
    cranberry_incident_reports.parquet
    cranberry_network.geojson
    cranberry_upstream_mapping.json
  tsmo/
    tsmo_speed_data.parquet
    tsmo_incident_reports.parquet
    tsmo_network.geojson
    tsmo_upstream_mapping.json
```

### Data format

#### Speed data (`*_speed_data.parquet`)

| Property | Specification |
|---|---|
| Index | `pandas.DatetimeIndex`, 5-minute resolution |
| Columns | TMC segment ID strings (e.g. `"104-04540"`) |
| Values | Float, speed in mph (NaN for missing observations) |
| Active hours | 05:30–20:55 on weekdays only — no overnight or weekend rows |

#### Incident reports (`*_incident_reports.parquet`)

Same shape and index as the speed file. Values are binary 0/1 indicating
whether a crowd-sourced incident report overlaps the link at that time interval.

#### Network geometry (`*_network.geojson`)

Standard GeoJSON FeatureCollection with required properties per feature:

| Property | Description |
|---|---|
| `tmc` | TMC segment ID — must match speed/incident column names |
| `miles` | Link length in miles |
| `roadnumber` | Route designation (e.g. `"I-76"`, `"I-79"`) |
| `geometry` | LineString or MultiLineString (WGS84) |

#### Upstream adjacency mapping (`*_upstream_mapping.json`)

JSON dict mapping each TMC ID to a list of its immediate upstream TMC IDs:
`{"104-04540": ["104-04539"], ...}`

### Data source

Speed data are derived from INRIX probe-speed feeds for TMC segments, available
through RITIS (Regional Integrated Transportation Information System). Incident
reports are from a crowd-sourced incident feed spatially matched to TMC segments.
Both are available to agencies and researchers through data-sharing agreements.
The pipeline is compatible with any provider that matches the schema above.

### Network summary

| Network | Location | Links | Sessions | Calib / Eval | Period |
|---|---|---|---|---|---|
| TSMO (calibration) | Howard County, MD | 228 | 260 | 65 / 195 | Feb 2022–Feb 2023 |
| Cranberry (transfer) | Pittsburgh, PA | 78 | 522 | 130 / 392 | Feb 2022–Jan 2024 |

---

## Installation

```bash
git clone https://github.com/anomalyco/ensemble-nr-traffic-labeling
cd ensemble-nr-traffic-labeling
pip install -r requirements.txt
```

Or with uv:

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Reproducing results

Place data files in the locations described above, then run:

```bash
python src/run_revised_experiments.py
```

Runtime: approximately 5–15 minutes on a modern laptop.

This script:
1. Loads speed and incident data for both networks
2. Applies the 25/75 temporal split (calibration → held-out evaluation)
3. Computes all calibration-period statistics with no leakage into the eval period
4. Runs all methods: incident reports only, Duan (2024) baseline, XGBoost
   baseline, and full ensemble
5. Calibrates parameters via grid search on TSMO, transfers unchanged to Cranberry
6. Runs the detector ablation study and staged component ablation
7. Runs ±20% parameter sensitivity analysis
8. Computes detection latency, episode quality, and boundary error rates
9. Saves all result tables to `final_outputs/tables/`

Pre-computed results are already in `final_outputs/tables/` if you want to
inspect outputs without re-running.

### Using the pre-trained XGBoost model

The trained XGBoost model is provided in `final_outputs/ml_baseline_xgb.pkl`.
`run_revised_experiments.py` will load it automatically if found and skip
retraining. To force retraining, delete the `.pkl` file before running.

---

## Calibration–transfer protocol

Parameters are calibrated on **TSMO** (the denser, higher-frequency network)
and transferred to **Cranberry** without any re-tuning.

| | TSMO | Cranberry |
|---|---|---|
| Hyperparameter selection | Grid search on first 25% | Not used |
| Speed statistics | First 25% | First 25% (local baseline only) |
| Evaluation | Last 75% | Last 75% |

The IQR-based normalisation in each detector adapts automatically to each
network's historical speed distribution; only the multiplier values are shared.

### Final calibrated parameters

| Parameter | Value | Description |
|---|---|---|
| `snd_c` — deviation IQR multiplier | 2.0 | Deviation detector threshold |
| `grad_c` — gradient IQR multiplier | 1.0 | Gradient detector threshold |
| `conf_f` — confirmation factor | 0.75 | Speed must drop below 75% of FFS |
| `min_dur` — minimum persistence | 15 min (3 steps) | Noise pre-filter |

---

## Baselines

### Duan et al. (2024)

Implemented from:

> Duan, H., Wu, H., and Qian, S. (2024). *Know unreported roadway incidents in
> real-time: Early traffic anomaly detection.* arXiv:2412.10892v2.

Core denoising functions reproduced from the reference; session-based day
segmentation replaces fixed-length day splits. See `src/duan_baseline.py`.

### XGBoost baseline (incident-supervised)

An XGBoost classifier trained to predict crowd-sourced incident report indicators
from raw probe-speed features: current speed, four lags, and instantaneous speed
change — no ensemble-derived features. Trained on TSMO calibration data and
applied to both networks at a threshold selected to match the ensemble's 5.3%
anomaly rate on TSMO. See `src/run_revised_experiments.py`.

---

## Key results

### Abnormal delay capture — held-out evaluation

| Method | TSMO (calibration) | Cranberry (transfer) |
|---|---|---|
| Incident reports | 6.5% | 14.9% |
| Duan (2024) baseline | 20.2% | 26.1% |
| XGBoost (speed-only) | 46.9% | 19.4% |
| **Full ensemble** | **55.6%** | **38.6%** |

The XGBoost baseline captures 46.9% on TSMO but only 19.4% on Cranberry —
incident-supervised prediction fails to transfer across networks. The ensemble
transfers cleanly (55.6% → 38.6%) because its IQR-based normalisation adapts
to each network's speed distribution.

### Episode quality

| Method | Network | Near-zero episodes | Quality (%) | Median dur. |
|---|---|---|---|---|
| Incident reports | TSMO | 37.0% | 72.9% | 30 min |
| XGBoost | TSMO | 37.5% | 70.7% | 25 min |
| **Full ensemble** | TSMO | **4.9%** | **97.3%** | **60 min** |
| **Full ensemble** | Cranberry | **4.6%** | **100%** | **50 min** |

### Detection timing (TSMO)

The ensemble achieves a median onset latency of −5 min relative to the first
below-v_rec step (gradient detector fires before speed crosses the recurrent
lower bound), compared to 0 min for incident reports and +5 min for Duan
and XGBoost.

---

## Output files

| File | Description |
|---|---|
| `tables/heldout_delay_results.csv` | Abnormal delay capture, all methods × both networks |
| `tables/ablation_results.csv` | Detector-subset and component ablation |
| `tables/sensitivity_results.csv` | ±20% parameter sensitivity |
| `tables/filtering_table.csv` | Per-episode near-zero fraction and quality |
| `tables/episode_duration_statistics.csv` | Duration distribution statistics |
| `tables/timing_results.csv` | Detection latency and episode quality |
| `tables/boundary_error_rates.csv` | Recovery-end boundary analysis |
| `tables/temporal_precision.csv` | Median detection offset from delay peak |
| `tables/persistence_staged_ablation.csv` | Staged pipeline component contributions |
| `tables/grid_search_tsmo.csv` | Full TSMO hyperparameter grid results |
| `tables/tsmo_ml_labels.parquet` | XGBoost labels for TSMO eval period |
| `tables/cran_ml_labels.parquet` | XGBoost labels for Cranberry eval period |
| `figures/ensemble_diagram.png` | Conceptual overview of the framework |
| `figures/detector_signal_comparison.png` | Detector activation decomposition |
| `figures/fig_network_layout.png` | Spatial layout of both networks |
| `figures/fig_recovery_end.png` | Recovery-end comparison (4 methods) |
| `figures/case_study_comparison.png` | Case studies (4 methods, TSMO) |
| `figures/fig_appendix_casestudies_1.png` | Appendix case studies (panels a–c) |
| `figures/fig_appendix_casestudies_2.png` | Appendix case studies (panels d–f) |

---

---

## License

MIT License. See `LICENSE` for details.
