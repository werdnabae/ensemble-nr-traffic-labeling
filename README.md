# Real-Time Ensemble Labeling of Nonrecurrent Traffic Disturbances

Reference implementation for the paper:

> **Real-Time Ensemble Labeling of Nonrecurrent Traffic Disturbances from Speed Data**  
> Andrew J. Bae — Carnegie Mellon University

---

## Overview

This repository provides the code for an ensemble anomaly labeling framework that
generates link-level nonrecurrent traffic disturbance labels directly from
probe-based speed measurements.  The framework combines three complementary
detectors (robust deviation, speed gradient, upstream slowdown) with
persistence logic, speed confirmation, and traffic-driven recovery rules.

Incident reports (e.g. Waze) are used only as optional supporting evidence.
The primary evaluation metric is the share of *abnormal delay* — delay
occurring when speed falls below the estimated recurrent variability band —
captured on held-out data.

---

## Repository structure

```
src/
  ensemble_labeling.py   Detectors, thresholds, graph neighbors, labeler
  duan_baseline.py       Duan et al. (2024) slowdown-based baseline
  delay_metrics.py       Excess and abnormal delay computation
  timing_metrics.py      Detection latency and termination overhang
  evaluation_pipeline.py Full evaluation entry point (single command)

data/
  cranberry/             ← place your Cranberry data files here (see below)
  tsmo/                  ← place your TSMO data files here (see below)

final_outputs/
  tables/                Pre-computed result tables (CSV)
  figures/               Pre-computed figures (PNG)

FINAL_RESULTS.md         Complete results in markdown
requirements.txt
```

---

## Data

**The data cannot be included in this repository due to a data-use agreement
with the data provider.**

To reproduce the results, you will need to obtain 5-minute probe speed data
and incident-report data for the two networks independently, then place them
in the expected locations described below.

### Expected file locations

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

### Data format specification

#### Speed data  (`*_speed_data.parquet`)

A Parquet file that loads as a `pandas.DataFrame` with the following structure:

| Property | Specification |
|---|---|
| Index | `pandas.DatetimeIndex`, 5-minute resolution |
| Columns | TMC segment ID strings (e.g. `"104-04540"`) |
| Values | Float, speed in mph |
| Active hours | 05:30–20:55 on weekdays only (no overnight or weekend rows) |
| Coverage | Cranberry: Feb 2022 – Jan 2024 · TSMO: Feb 2022 – Feb 2023 |

```python
import pandas as pd
df = pd.read_parquet("data/cranberry/cranberry_speed_data.parquet")
# df.shape → (97092, 78) for Cranberry full dataset
# df.index → DatetimeIndex(['2022-02-01 05:30:00', '2022-02-01 05:35:00', ...])
# df.columns → Index(['104-04439', '104-04440', ...], dtype='object')
# df.dtypes → float64 (may contain NaN for missing observations)
```

#### Incident reports  (`*_incident_reports.parquet`)

Same shape and index as the speed file.  Values are binary 0/1 indicating
whether a crowd-sourced incident report is mapped to that link at that time.

```python
inc = pd.read_parquet("data/cranberry/cranberry_incident_reports.parquet")
# inc.shape → same as speed_df
# inc.dtypes → int64 or float64, values in {0, 1}
```

#### Network geometry  (`*_network.geojson`)

Standard GeoJSON `FeatureCollection`.  Each feature represents one TMC
segment.  Required properties:

| Property | Type | Description |
|---|---|---|
| `tmc` | string | TMC segment ID — must match speed/incident column names |
| `miles` | float | Link length in miles |
| `roadnumber` | string | Route designation (e.g. `"I-76"`, `"I-79"`, `"I-695"`) |
| `geometry` | LineString or MultiLineString | WGS84 / CRS84 |

Optional properties used for context (not required to run): `direction`,
`roadname`, `county`, `state`.

```python
import geopandas as gpd
gdf = gpd.read_file("data/cranberry/cranberry_network.geojson")
# gdf.columns includes: tmc, miles, roadnumber, geometry, ...
```

#### Upstream adjacency mapping  (`*_upstream_mapping.json`)

A JSON object mapping each TMC segment ID to a list of its immediate upstream
(predecessor) TMC segment IDs in the directed road graph.

```json
{
  "104-04540": ["104-04539"],
  "104-04539": ["104-04538", "104N04538"],
  ...
}
```

```python
import json
with open("data/cranberry/cranberry_upstream_mapping.json") as f:
    upstream = json.load(f)
# upstream["104-04540"] → ["104-04539"]
```

### Data source

The speed data used in this work were derived from INRIX probe-speed feeds
for TMC segments, available through RITIS (Regional Integrated Transportation
Information System).  Incident reports were sourced from Waze for Cities
crowd-sourced incident feeds, spatially matched to TMC segments.  Both feeds
are available to transportation agencies and researchers through data-sharing
agreements.

If you have access to compatible 5-minute TMC speed data from another provider
(e.g. HERE, TomTom), the pipeline will work as long as the files match the
schema above.

---

## Installation

```bash
git clone https://github.com/<your-username>/ensemble-traffic-labeling
cd ensemble-traffic-labeling
pip install -r requirements.txt
```

Or with uv:

```bash
uv venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

---

## Reproducing results

Place data files in the locations described above, then run:

```bash
python src/evaluation_pipeline.py
```

This single command (~4–8 minutes on a modern laptop):

1. Loads and filters speed and incident data for both networks
2. Applies the 25 / 75 temporal split (calibration → held-out evaluation)
3. Computes all calibration-period frozen statistics (no leakage into eval)
4. Runs three methods: incident-reports-only, Duan (2024) baseline, full ensemble
5. Runs the ablation study (four detector subsets)
6. Computes timing analysis (detection latency and termination overhang)
7. Analyses unlabeled abnormal delay (heatmaps)
8. Saves all tables to `final_outputs/tables/`
9. Saves all figures to `final_outputs/figures/`
10. Writes `FINAL_RESULTS.md`

Pre-computed results (produced with the original data) are already in
`final_outputs/` and `FINAL_RESULTS.md` for reference.

---

## Parameter configuration

The final configuration was calibrated on the Cranberry network calibration
period and applied without modification to TSMO:

| Parameter | Value | Description |
|---|---|---|
| SND IQR multiplier | 2.5 | Robust deviation detector sensitivity |
| Gradient IQR multiplier | 1.2 | Speed gradient detector sensitivity |
| Confirmation threshold | 0.70 × FFS | Speed must drop below 70% of free-flow to confirm |
| Minimum run duration | 20 min | Shortest labeled episode |

Temporal protocol: first 25% of sessions = calibration, remaining 75% = evaluation.

---

## Key results (from pre-computed outputs)

### Abnormal delay capture — held-out evaluation

| Method | Cranberry | TSMO |
|---|---|---|
| Incident reports only | 14.9% | 6.5% |
| Duan (2024) baseline | 26.1% | 20.2% |
| **Full ensemble** | **33.5%** | **43.4%** |

Detector-only abnormal delay (no incident report present):
Cranberry 21.9%, TSMO 37.2% of total abnormal delay.

Incident-report periods excluded by the method: 61–83% have near-zero
abnormal delay, supporting the claim that the confirmation filter correctly
rejects low-impact reports.

Unlabeled abnormal delay: 61% (Cranberry) and 67% (TSMO) occurs during
predictable AM/PM peak hours, indicating it reflects recurrent congestion
correctly excluded by the method.

---

## Baseline

The Duan baseline is implemented from:

> Duan, H., Wu, H., and Qian, S. (2024). *Know unreported roadway incidents
> in real-time: Early traffic anomaly detection.* arXiv:2412.10892v2.

Core functions (`label_all_incident_contain_significant_sd` and
`label_long_last_abnormal_sd_as_incident`) are reproduced verbatim.
The only adaptation is session-based day segmentation (replacing the original
fixed-length day split) to handle data gaps correctly.  See
`src/duan_baseline.py` for full documentation.

---

## License

MIT License.  See `LICENSE` for details.
