"""Timing analysis: detection latency and termination overhang.

Definitions
-----------
true_start : first timestep where speed < v_rec, searching backward from the
             detected episode start within the same session.
true_end   : first timestep at or after the detected episode end where
             speed >= v_rec within the same session.

detection_latency    = detected_start - true_start  (minutes)
                       positive = late,  negative = early
termination_overhang = detected_end   - true_end    (minutes)
                       positive = runs past true end,  negative = terminates early

Only episodes with at least one below-v_rec step are included in latency /
overhang distributions; all others are counted as "no speed anomaly".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.ensemble_labeling import INTERVAL_MIN


def extract_episodes(
    labels_df: pd.DataFrame, interval_min: int = INTERVAL_MIN
) -> pd.DataFrame:
    rows = []
    for link in labels_df.columns:
        col = labels_df[link].values.astype(float)
        i = 0
        while i < len(col):
            if col[i] == 1:
                j = i
                while j < len(col) and col[j] == 1:
                    j += 1
                rows.append(
                    dict(
                        link=link,
                        start_idx=i,
                        end_idx=j - 1,
                        start=labels_df.index[i],
                        end=labels_df.index[j - 1],
                        duration_min=(j - i) * interval_min,
                    )
                )
                i = j
            else:
                i += 1
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=["link", "start_idx", "end_idx", "start", "end", "duration_min"]
        )
    )


def _find_true_boundaries(
    spd: np.ndarray, vr: np.ndarray, sids: np.ndarray, det_s: int, det_e: int
) -> tuple[int | None, int | None]:
    """Core timing computation for one episode."""
    sid = sids[det_s]
    n = len(spd)

    # true_start: walk backward from det_s for the earliest below-v_rec step
    if spd[det_s] < vr[det_s]:
        ts = det_s
        i = det_s - 1
        while i >= 0 and sids[i] == sid:
            if spd[i] < vr[i]:
                ts = i
                i -= 1
            else:
                break
    else:
        ts = None
        for i in range(det_s, min(det_e + 1, n)):
            if sids[i] != sid:
                break
            if spd[i] < vr[i]:
                ts = i
                break
        if ts is None:
            return None, None  # no speed anomaly in window

    # true_end: first step at or after det_e where speed >= v_rec
    te = det_e
    for i in range(det_e + 1, n):
        if sids[i] != sid:
            break
        if spd[i] >= vr[i]:
            te = i
            break

    return ts, te


def compute_timing(
    episodes: pd.DataFrame,
    speed_df: pd.DataFrame,
    vrec_df: pd.DataFrame,
    sids: np.ndarray,
) -> pd.DataFrame:
    """Compute true start/end and timing metrics for all episodes."""
    rows = []
    for _, ep in episodes.iterrows():
        lnk = ep.link
        if lnk not in speed_df.columns:
            continue
        ts, te = _find_true_boundaries(
            speed_df[lnk].values.astype(float),
            vrec_df[lnk].values.astype(float),
            sids,
            int(ep.start_idx),
            int(ep.end_idx),
        )

        if ts is None:
            rows.append(
                dict(
                    link=lnk,
                    start=ep.start,
                    end=ep.end,
                    duration_min=ep.duration_min,
                    true_start=None,
                    true_end=None,
                    true_duration_min=None,
                    detection_latency_min=None,
                    termination_overhang_min=None,
                    has_speed_anomaly=False,
                )
            )
        else:
            ts_t = speed_df.index[ts]
            te_t = speed_df.index[te]
            rows.append(
                dict(
                    link=lnk,
                    start=ep.start,
                    end=ep.end,
                    duration_min=ep.duration_min,
                    true_start=ts_t,
                    true_end=te_t,
                    true_duration_min=(te_t - ts_t).total_seconds() / 60.0,
                    detection_latency_min=(ep.start - ts_t).total_seconds() / 60.0,
                    termination_overhang_min=(ep.end - te_t).total_seconds() / 60.0,
                    has_speed_anomaly=True,
                )
            )
    return pd.DataFrame(rows)


def summarise_timing(tdf: pd.DataFrame, method: str, network: str) -> dict:
    valid = tdf[tdf.has_speed_anomaly == True]
    lat = valid.detection_latency_min.dropna()
    over = valid.termination_overhang_min.dropna()
    dur = tdf.duration_min.dropna()
    return dict(
        method=method,
        network=network,
        n_episodes=len(tdf),
        n_with_speed_anomaly=len(valid),
        pct_with_speed_anomaly=round(100 * len(valid) / max(1, len(tdf)), 1),
        latency_median=round(float(lat.median()), 1) if len(lat) else None,
        latency_mean=round(float(lat.mean()), 1) if len(lat) else None,
        latency_p25=round(float(lat.quantile(0.25)), 1) if len(lat) else None,
        latency_p75=round(float(lat.quantile(0.75)), 1) if len(lat) else None,
        overhang_median=round(float(over.median()), 1) if len(over) else None,
        overhang_mean=round(float(over.mean()), 1) if len(over) else None,
        duration_median=round(float(dur.median()), 1) if len(dur) else None,
        duration_mean=round(float(dur.mean()), 1) if len(dur) else None,
    )
