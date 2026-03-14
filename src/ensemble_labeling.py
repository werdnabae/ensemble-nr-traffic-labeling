"""Ensemble nonrecurrent traffic disturbance labeling.

Consolidated module covering:
  - Data loading and TMC selection
  - Preprocessing (FFS, slowdown speed, speed gradients)
  - IQR-based threshold computation
  - Graph neighbor construction
  - Core labeling functions (per-link)
  - Calibration-period frozen threshold helpers (for temporal evaluation)
  - Main orchestrator: run_ensemble_labeler()

Terminology: "incident reports" refers to crowd-sourced incident feeds
(e.g. Waze) that are used as optional supporting evidence, not as ground truth.

Reference configuration (C02, Cranberry-calibrated):
  snd_c=2.5, grad_c=1.2, conf_f=0.70, min_dur=20 min
"""

from __future__ import annotations

import heapq, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shapely_box
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
INTERVAL_MIN = 5
FFS_QUANTILE = 0.85
CALIB_FRAC = 0.25  # fraction of sessions for calibration
ALPHA_VREC = 0.7  # recurrent lower-bound IQR multiplier

# Final configuration (Cranberry-calibrated, frozen)
FINAL_SND_C = 2.5
FINAL_GRAD_C = 1.2
FINAL_CONF_F = 0.70
FINAL_MIN_DUR = 20

# Network-specific parameters (paper Table 1)
NETWORK_PARAMS = {
    "cranberry": dict(
        freeway_road_numbers=["I-76", "I-79"],
        bbox=dict(min_lon=-80.3, max_lon=-79.9, min_lat=40.55, max_lat=41.0),
        snd_c_report=2.3,
        slowdown_c=3.5,
        confirmation_factor_report=0.75,
        snd_window_min=15,
    ),
    "tsmo": dict(
        freeway_road_numbers=[
            "I-695",
            "I-95",
            "I-70",
            "I-195",
            "I-895",
            "US-29",
            "MD-100",
            "MD-32",
        ],
        bbox=None,
        snd_c_report=1.5,
        slowdown_c=3.5,
        confirmation_factor_report=0.85,
        snd_window_min=15,
    ),
}
RECOVERY_SOFT_C_SHORT = 0.7
RECOVERY_SOFT_C_LONG = 1.2
HARD_RECOVERY_SHORT = 0.80
HARD_RECOVERY_LONG = 0.70
HARD_SPEED_FACTOR = 0.70
MAX_GAP_MIN = 10

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_network_data(
    data_dir: str | Path, network: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load speed, incident reports, and upstream mapping for a network.

    Parameters
    ----------
    data_dir : path to data/{network}/
    network  : "cranberry" or "tsmo"

    Returns
    -------
    speed_df, incident_df (both with DatetimeIndex), upstream_mapping (dict)
    """
    d = Path(data_dir)
    pfx = network.lower()
    speed_df = pd.read_parquet(d / f"{pfx}_speed_data.parquet")
    incident_df = pd.read_parquet(d / f"{pfx}_incident_reports.parquet")
    speed_df.index = pd.to_datetime(speed_df.index)
    incident_df.index = pd.to_datetime(incident_df.index)
    with open(d / f"{pfx}_upstream_mapping.json") as f:
        upstream = json.load(f)
    return speed_df.sort_index(), incident_df.sort_index(), upstream


def get_geojson_path(data_dir: str | Path, network: str) -> Path:
    d = Path(data_dir)
    return d / f"{network.lower()}_network.geojson"


# ─────────────────────────────────────────────────────────────────────────────
# TMC selection
# ─────────────────────────────────────────────────────────────────────────────


def select_tmcs(
    speed_df: pd.DataFrame, geojson_path: str | Path, network: str
) -> List[str]:
    """Return selected freeway TMC IDs for a network.

    Applies road-number whitelist and optional bounding-box filter.
    Only returns TMCs present in speed_df.
    """
    params = NETWORK_PARAMS[network.lower()]
    gdf = gpd.read_file(str(geojson_path))
    id_col = "tmc" if "tmc" in gdf.columns else "id_tmc"
    gdf[id_col] = gdf[id_col].astype(str)
    sel = gdf[gdf["roadnumber"].isin(params["freeway_road_numbers"])]
    if params["bbox"]:
        bb = params["bbox"]
        geom = shapely_box(bb["min_lon"], bb["min_lat"], bb["max_lon"], bb["max_lat"])
        sel = sel[sel.geometry.within(geom)]
    speed_set = set(speed_df.columns.astype(str))
    return sorted(str(t) for t in sel[id_col] if str(t) in speed_set)


def get_link_lengths(geojson_path: str | Path, links: List[str]) -> pd.Series:
    gdf = gpd.read_file(str(geojson_path))
    id_col = "tmc" if "tmc" in gdf.columns else "id_tmc"
    gdf[id_col] = gdf[id_col].astype(str)
    lengths = gdf.set_index(id_col)["miles"].astype(float)
    return lengths.reindex(links).fillna(lengths.median())


# ─────────────────────────────────────────────────────────────────────────────
# Session / temporal split
# ─────────────────────────────────────────────────────────────────────────────


def compute_session_ids(
    index: pd.DatetimeIndex, interval_minutes: int = INTERVAL_MIN
) -> np.ndarray:
    """Assign integer session IDs based on timestamp gaps.

    A new session begins when the gap to the previous timestamp exceeds
    2 × interval_minutes (handles overnight and weekend gaps).
    """
    threshold = pd.Timedelta(minutes=2 * interval_minutes)
    diffs = pd.Series(index).diff()
    new_sess = (diffs > threshold).to_numpy().copy()
    new_sess[0] = True
    return new_sess.cumsum().astype(np.int64)


def temporal_split(
    df: pd.DataFrame, calib_frac: float = CALIB_FRAC
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Split by session fraction → (calib_mask, eval_mask, n_calib, n_sess)."""
    sids = compute_session_ids(df.index, INTERVAL_MIN)
    n_sess = int(sids[-1])
    n_calib = max(1, int(n_sess * calib_frac))
    return sids <= n_calib, sids > n_calib, n_calib, n_sess


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def compute_free_flow_speed(
    speed_df: pd.DataFrame, q: float = FFS_QUANTILE
) -> pd.Series:
    """85th-percentile speed per link (using all provided data as training)."""
    return speed_df.quantile(q=q, axis=0).astype("float32")


def compute_slowdown_speed(
    speed_df: pd.DataFrame, upstream_kmile_dict: Dict[str, List[str]]
) -> pd.DataFrame:
    """Slowdown speed = max(0, mean_upstream_speed − current_speed) per link.

    Defined in paper §4.3:  SS_i(t) = max( v̄_Ui(t) − v_i(t), 0 )
    """
    available = set(map(str, speed_df.columns))
    result = pd.DataFrame(
        np.float32(0.0),
        index=speed_df.index,
        columns=speed_df.columns,
        dtype=np.float32,
    )
    for link in speed_df.columns:
        nbrs = [
            n
            for n in upstream_kmile_dict.get(str(link), [])
            if n in available and n != str(link)
        ]
        if nbrs:
            up_mean = speed_df[nbrs].mean(axis=1)
            result[link] = (up_mean - speed_df[link]).clip(lower=0.0).astype(np.float32)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Graph neighbor construction
# ─────────────────────────────────────────────────────────────────────────────


def _reverse_adjacency(adj: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, Set[str]] = {}
    for u, nbrs in adj.items():
        for v in nbrs:
            out.setdefault(v, set()).add(u)
    return {k: sorted(v) for k, v in out.items()}


def _build_length_dict(geojson_path: str | Path) -> Dict[str, float]:
    gdf = gpd.read_file(str(geojson_path))
    id_col = "tmc" if "tmc" in gdf.columns else "id_tmc"
    gdf[id_col] = gdf[id_col].astype(str)
    df = gdf[[id_col, "miles"]].dropna()
    df["miles"] = pd.to_numeric(df["miles"], errors="coerce")
    return {str(t): float(m) for t, m in zip(df[id_col], df["miles"]) if pd.notna(m)}


def _neighbors_within_k_miles(start, neighbor_dict, length_dict, k_miles, allowed=None):
    start = str(start)
    inf = 1e30
    dist: Dict[str, float] = {}
    heap = []
    for nb in neighbor_dict.get(start, []):
        nb = str(nb)
        if nb not in length_dict:
            continue
        d = float(length_dict[nb])
        if d < dist.get(nb, inf):
            dist[nb] = d
            heapq.heappush(heap, (d, nb))
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist.get(u):
            continue
        if d >= k_miles:
            continue
        for nb in neighbor_dict.get(u, []):
            nb = str(nb)
            if nb not in length_dict:
                continue
            nd = d + float(length_dict[nb])
            if nd < dist.get(nb, inf):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    out = sorted(set(dist.keys()))
    if allowed is not None:
        out = [x for x in out if x in allowed]
    return out


def build_upstream_neighbors(
    speed_df: pd.DataFrame,
    geojson_path: str | Path,
    raw_upstream: Dict[str, List[str]],
    k_miles: float = 0.3,
) -> Dict[str, List[str]]:
    """Build k-mile upstream neighbor dict for computing slowdown speed."""
    speed_set = set(map(str, speed_df.columns))
    length_dict = _build_length_dict(geojson_path)
    # Filter adjacency to links with known lengths
    adj = {
        str(k): [str(v) for v in vals if str(v) in length_dict]
        for k, vals in raw_upstream.items()
        if str(k) in length_dict
    }

    targets = sorted(speed_set & set(adj.keys()))
    up03 = {
        t: _neighbors_within_k_miles(t, adj, length_dict, k_miles, speed_set)
        for t in targets
    }
    for lnk in speed_df.columns:
        up03.setdefault(str(lnk), [])
    return up03


# ─────────────────────────────────────────────────────────────────────────────
# Threshold computation
# ─────────────────────────────────────────────────────────────────────────────


def frozen_thresholds(
    calib_df: pd.DataFrame,
    full_df: pd.DataFrame,
    group_kind: str,
    c: float,
    mode: str,
    window_minutes: int = 15,
) -> pd.DataFrame:
    """IQR thresholds from calib_df, broadcast to full_df.index.

    Prevents temporal leakage: evaluation period never enters the statistics.
    """
    if group_kind == "dow_timebin":
        c_k0 = calib_df.index.dayofweek
        c_k1 = (calib_df.index.hour * 60 + calib_df.index.minute) // window_minutes
        f_k0 = full_df.index.dayofweek
        f_k1 = (full_df.index.hour * 60 + full_df.index.minute) // window_minutes
        knames = ["dow", "bin"]
    elif group_kind == "dow_time":
        c_k0 = calib_df.index.dayofweek
        c_k1 = calib_df.index.time
        f_k0 = full_df.index.dayofweek
        f_k1 = full_df.index.time
        knames = ["dow", "tod"]
    else:
        raise ValueError(f"Unknown group_kind: {group_kind}")

    seg = list(calib_df.columns)
    calc = calib_df.copy()
    calc["_k0"] = c_k0
    calc["_k1"] = c_k1
    g = calc.groupby(["_k0", "_k1"])[seg]
    med = g.median()
    iqr = (g.quantile(0.75) - g.quantile(0.25)).clip(lower=1e-3)
    thr = med - c * iqr if mode == "minus" else med + c * iqr
    thr.index.names = knames
    broad = thr.reindex(pd.MultiIndex.from_arrays([f_k0, f_k1], names=knames))
    broad.index = full_df.index
    return broad.astype("float32")


def frozen_confirmation_thresholds(
    calib_speed: pd.DataFrame, factors, q: float = 0.85
) -> dict:
    """Per-link scalar confirmation thresholds from calibration period."""
    filtered = pd.concat(
        [
            calib_speed.between_time("09:00", "16:00"),
            calib_speed.between_time("19:00", "22:00"),
        ]
    ).sort_index()
    base = filtered.quantile(q=q, axis=0).astype("float32")
    return {float(f): (base * float(f)).astype("float32") for f in factors}


def frozen_percentile(calib_df: pd.DataFrame, q: float) -> pd.Series:
    return calib_df.quantile(q=q, axis=0).astype("float32")


def four_point_slopes(speed_df: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    """4-point rolling OLS speed gradient per link."""
    y = speed_df.to_numpy(dtype=np.float32, copy=False)
    t, n = y.shape
    slopes = np.full((t, n), np.nan, dtype=np.float32)
    if weighted:
        w = np.array([1, 2, 3, 4], dtype=np.float32)
        x = np.arange(4, dtype=np.float32)
        x_m = np.average(x, weights=w)
        x_d = x - x_m
        x2 = np.sum(w * x_d * x_d)
        for i in range(t - 3):
            win = y[i : i + 4]
            y_m = np.average(win, axis=0, weights=w)
            cov = np.sum((w[:, None] * x_d[:, None]) * (win - y_m[None, :]), axis=0)
            slopes[i] = cov / x2
    else:
        x = np.arange(4, dtype=np.float32)
        x_m = np.mean(x)
        x_d = x - x_m
        x2 = np.sum(x_d * x_d)
        for i in range(t - 3):
            win = y[i : i + 4]
            y_m = np.mean(win, axis=0)
            cov = np.sum(x_d[:, None] * (win - y_m[None, :]), axis=0)
            slopes[i] = cov / x2
    slopes[np.abs(slopes) < 1e-6] = 0.0
    return pd.DataFrame(
        slopes, index=speed_df.index, columns=speed_df.columns, dtype=np.float32
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-link labeling helpers (paper Algorithm 1, batch implementation)
# ─────────────────────────────────────────────────────────────────────────────


def _find_runs(
    arr: np.ndarray, session_ids: Optional[np.ndarray] = None
) -> List[Tuple[int, int]]:
    """Return [(start, end), ...] for each run of 1s, respecting sessions."""
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 1:
            j = i
            if session_ids is not None:
                while j < n and arr[j] == 1 and session_ids[j] == session_ids[i]:
                    j += 1
            else:
                while j < n and arr[j] == 1:
                    j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def early_labeling_with_incident_reports(
    incident_reports: pd.Series,
    detector_indicator: pd.Series,
    interval_minutes: int = INTERVAL_MIN,
) -> pd.Series:
    """Confirm Waze-gated onset: require 2 consecutive co-fires of incident report
    and detector indicator within each incident-report run.

    Labels the entire incident-report run if confirmed, discards it otherwise.
    Session boundaries are respected.
    """
    result = pd.Series(0.0, index=incident_reports.index)
    inc_arr = np.nan_to_num(incident_reports.values, nan=0.0)
    sd_arr = np.nan_to_num(detector_indicator.values, nan=0.0)
    session_ids = compute_session_ids(incident_reports.index, interval_minutes)
    n = len(inc_arr)
    i = 0
    while i < n:
        if inc_arr[i] == 1:
            j = i
            while j < n and inc_arr[j] == 1 and session_ids[j] == session_ids[i]:
                j += 1
            run_end = j - 1
            for k in range(i, run_end):
                if sd_arr[k] == 1 and sd_arr[k + 1] == 1:
                    result.iloc[i : run_end + 1] = 1.0
                    break
            i = run_end + 1
        else:
            i += 1
    return result


def filter_by_confirmation_thresholds(
    labels: pd.Series,
    speed: pd.Series,
    gradient: pd.Series,
    confirmation_threshold: float,
    min_len: int = 3,
    window_size: int = 3,
    interval_minutes: int = INTERVAL_MIN,
) -> pd.Series:
    """Confirmation filter (non-report branch).

    For each labeled run of length >= min_len:
    1. Gradient at run[2] must be < 0 (confirmed deceleration).
    2. Slide window through run: any speed < confirmation_threshold → keep from
       that position onward.
    """
    result = pd.Series(0.0, index=labels.index)
    lbl_arr = np.nan_to_num(labels.values, nan=0.0)
    spd_arr = speed.values.astype(float)
    grd_arr = gradient.values.astype(float)
    session_ids = compute_session_ids(labels.index, interval_minutes)

    for rs, re in _find_runs(lbl_arr, session_ids):
        if re - rs + 1 < min_len:
            continue
        grd_pos = rs + 2
        if grd_pos > re:
            continue
        if np.isnan(grd_arr[grd_pos]) or grd_arr[grd_pos] >= 0:
            continue
        for shift_start in range(re - rs):
            gs = rs + shift_start
            win = spd_arr[gs : min(gs + window_size, re + 1)]
            v = win[~np.isnan(win)]
            if v.size > 0 and np.any(v < confirmation_threshold):
                result.iloc[gs : re + 1] = 1.0
                break
    return result


def filter_by_confirmation_thresholds_reports(
    labels: pd.Series,
    speed: pd.Series,
    gradient: pd.Series,
    confirmation_threshold: float,
    min_len: int = 1,
    backtrack: int = 2,
    interval_minutes: int = INTERVAL_MIN,
) -> pd.Series:
    """Confirmation filter (incident-report branch).

    Within each labeled run find first position where gradient < 0 AND
    speed < threshold; keep from (position − backtrack) to run end.
    """
    result = pd.Series(0.0, index=labels.index)
    lbl_arr = np.nan_to_num(labels.values, nan=0.0)
    spd_arr = speed.values.astype(float)
    grd_arr = gradient.values.astype(float)
    session_ids = compute_session_ids(labels.index, interval_minutes)

    for rs, re in _find_runs(lbl_arr, session_ids):
        if re - rs + 1 < min_len:
            continue
        for offset in range(1, re - rs + 1):
            gi = rs + offset
            if np.isnan(grd_arr[gi]) or np.isnan(spd_arr[gi]):
                continue
            if grd_arr[gi] < 0 and spd_arr[gi] < confirmation_threshold:
                result.iloc[max(rs, gi - backtrack) : re + 1] = 1.0
    return result


def filter_short_ones_with_time(
    labels: pd.Series,
    min_minutes: int = FINAL_MIN_DUR,
    interval_minutes: int = INTERVAL_MIN,
) -> pd.Series:
    """Remove labeled runs shorter than min_minutes."""
    min_steps = max(1, min_minutes // interval_minutes)
    result = labels.copy().astype(float)
    arr = np.nan_to_num(labels.values, nan=0.0)
    session_ids = compute_session_ids(labels.index, interval_minutes)
    for i, j in _find_runs(arr, session_ids):
        if j - i + 1 < min_steps:
            result.iloc[i : j + 1] = 0.0
    return result


def fill_short_gaps(
    labels: pd.Series,
    max_gap_minutes: int = MAX_GAP_MIN,
    interval_minutes: int = INTERVAL_MIN,
) -> pd.Series:
    """Fill gaps ≤ max_gap_minutes between labeled runs within the same session."""
    max_gap = max(1, max_gap_minutes // interval_minutes)
    result = labels.copy().astype(float)
    arr = np.nan_to_num(labels.values, nan=0.0)
    sids = compute_session_ids(labels.index, interval_minutes)
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 1:
            j = i
            while j < n and arr[j] == 1:
                j += 1
            if j >= n:
                break
            k = j
            while k < n and arr[k] == 0:
                k += 1
            if k < n and k - j <= max_gap and sids[j - 1] == sids[k]:
                result.iloc[j:k] = 1.0
            i = k
        else:
            i += 1
    return result


def extend_recovery(
    labels: pd.Series,
    speed: pd.Series,
    soft_short: pd.Series,
    soft_long: pd.Series,
    hard_short: float,
    hard_long: float,
    interval_minutes: int = INTERVAL_MIN,
) -> pd.Series:
    """Extend labeled intervals forward until traffic recovers.

    Recovery criteria (from paper §4.5):
    - Short: 2 of next 3 steps > max(hard_short, soft_short[t])
    - Long:  3 of next 5 steps > max(hard_long,  soft_long[t])
    Extension is capped at the end of the same session.
    """
    result = labels.copy().astype(float)
    spd_arr = speed.values.astype(float)
    ss_arr = soft_short.values.astype(float)
    sl_arr = soft_long.values.astype(float)
    sids = compute_session_ids(labels.index, interval_minutes)
    n = len(spd_arr)
    i = 0
    while i < n:
        if result.iloc[i] == 1:
            j = i
            while j < n and result.iloc[j] == 1:
                j += 1
            run_end = j - 1
            end_session = sids[run_end]
            t = run_end + 1
            while t < n and sids[t] == end_session:
                es = max(hard_short, ss_arr[t])
                el = max(hard_long, sl_arr[t])
                w3 = spd_arr[t : min(t + 3, n)]
                v3 = w3[~np.isnan(w3)]
                if len(v3) >= 2 and np.sum(v3 > es) >= 2:
                    break
                w5 = spd_arr[t : min(t + 5, n)]
                v5 = w5[~np.isnan(w5)]
                if len(v5) >= 3 and np.sum(v5 > el) >= 3:
                    break
                result.iloc[t] = 1.0
                t += 1
            i = t
        else:
            i += 1
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def run_ensemble_labeler(
    speed_df: pd.DataFrame,
    incident_df: pd.DataFrame,
    slowdown_speed_df: pd.DataFrame,
    snd_threshold_nonreport: pd.DataFrame,
    snd_threshold_report: pd.DataFrame,
    slowdown_threshold: pd.DataFrame,
    nw_gradient_df: pd.DataFrame,
    w_gradient_df: pd.DataFrame,
    nw_gradient_threshold: pd.DataFrame,
    w_gradient_threshold: pd.DataFrame,
    confirmation_threshold_nonreport: pd.Series,
    confirmation_threshold_report: pd.Series,
    recovery_soft_short: pd.DataFrame,
    recovery_soft_long: pd.DataFrame,
    ffs: pd.Series,
    hard_speed_factor: float = HARD_SPEED_FACTOR,
    hard_rec_short: float = HARD_RECOVERY_SHORT,
    hard_rec_long: float = HARD_RECOVERY_LONG,
    min_nonreport_min: int = FINAL_MIN_DUR,
    max_gap_min: int = MAX_GAP_MIN,
    interval_min: int = INTERVAL_MIN,
    conf_min_len: int = 3,
    conf_window: int = 3,
    links: Optional[List[str]] = None,
    verbose: bool = True,
    use_snd: bool = True,
    use_gradient: bool = True,
    use_slowdown: bool = True,
) -> pd.DataFrame:
    """Run the full per-link ensemble labeling pipeline.

    Ablation flags (all True by default):
      use_snd      – include SND detector in non-report OR
      use_gradient – include NW and weighted gradient detectors
      use_slowdown – include slowdown speed detector
    """
    if links is None:
        links = list(speed_df.columns)
    out = pd.DataFrame(
        np.float32(0.0),
        index=speed_df.index,
        columns=speed_df.columns,
        dtype=np.float32,
    )
    it = tqdm(links, desc="Labeling") if verbose else links

    for link in it:
        lk = str(link)
        if lk not in speed_df.columns:
            continue
        spd = speed_df[lk]
        inc = (
            incident_df[lk]
            if lk in incident_df.columns
            else pd.Series(0.0, index=spd.index)
        )
        slow = slowdown_speed_df[lk]
        nw_g = nw_gradient_df[lk]
        w_g = w_gradient_df[lk]

        conf_nr = float(confirmation_threshold_nonreport.get(lk, np.nan))
        conf_r = float(confirmation_threshold_report.get(lk, np.nan))
        lk_ffs = float(ffs.get(lk, np.nan))
        h_spd = lk_ffs * hard_speed_factor
        h_rs = lk_ffs * hard_rec_short
        h_rl = lk_ffs * hard_rec_long

        snd_nr = (spd < snd_threshold_nonreport[lk]).fillna(False).astype(float)
        snd_r = (spd < snd_threshold_report[lk]).fillna(False).astype(float)
        sl_ind = (slow > slowdown_threshold[lk]).fillna(False).astype(float)
        nwi = (
            ((nw_g < nw_gradient_threshold[lk]) & (nw_g < 0))
            .fillna(False)
            .astype(float)
        )
        wi = ((w_g < w_gradient_threshold[lk]) & (w_g < 0)).fillna(False).astype(float)
        h_ind = (
            (spd < h_spd).fillna(False).astype(float)
            if np.isfinite(h_spd)
            else pd.Series(0.0, index=spd.index)
        )

        # apply ablation flags
        _s = snd_nr if use_snd else pd.Series(0.0, index=spd.index)
        _sl = sl_ind if use_slowdown else pd.Series(0.0, index=spd.index)
        _n = nwi if use_gradient else pd.Series(0.0, index=spd.index)
        _w = wi if use_gradient else pd.Series(0.0, index=spd.index)

        # NON-REPORT BRANCH
        nr_raw = ((_s == 1) | (_sl == 1) | (_n == 1) | (_w == 1)).astype(float)
        nr_flt = filter_short_ones_with_time(nr_raw, min_nonreport_min, interval_min)
        nr_lbl = (
            filter_by_confirmation_thresholds(
                nr_flt, spd, nw_g, conf_nr, conf_min_len, conf_window, interval_min
            )
            if np.isfinite(conf_nr)
            else nr_flt
        )

        # INCIDENT-REPORT BRANCH
        rep_raw = ((snd_r == 1) | (h_ind == 1)).astype(float)
        rep_lbl = early_labeling_with_incident_reports(inc, rep_raw, interval_min)
        if np.isfinite(conf_r):
            rep_lbl = filter_by_confirmation_thresholds_reports(
                rep_lbl, spd, nw_g, conf_r, 1, 2, interval_min
            )

        # COMBINE
        combined = ((rep_lbl == 1) | (nr_lbl == 1)).astype(float)
        combined = fill_short_gaps(combined, max_gap_min, interval_min)
        if not (np.isnan(h_rs) or np.isnan(h_rl)):
            combined = extend_recovery(
                combined,
                spd,
                recovery_soft_short[lk],
                recovery_soft_long[lk],
                h_rs,
                h_rl,
                interval_min,
            )
            combined = fill_short_gaps(combined, max_gap_min, interval_min)
        out[lk] = combined.astype(np.float32)

    return out
