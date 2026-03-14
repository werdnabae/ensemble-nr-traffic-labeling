"""Delay-based evaluation metrics.

Defines excess travel time and abnormal delay relative to a time-of-week
reference speed calibrated from the first 25% of each network's data.

Key definitions (paper §Delay measures):
  TT       = link_length / observed_speed  (hours per traversal)
  TT_ref   = link_length / reference_speed
  excess   = max(0, TT - TT_ref)
  v_rec    = median - alpha * IQR  (recurrent lower bound)
  abnormal = excess where observed_speed < v_rec

All quantities are in link-hours of excess travel time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_delay_matrices(
    speed_df: pd.DataFrame,
    ref_speed: pd.DataFrame,
    v_rec: pd.DataFrame,
    link_lengths: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-(link, timestep) delay matrices.

    Returns
    -------
    excess_delay   : link-hours, max(0, TT - TT_ref);  NaN where speed unavailable
    abnormal_flag  : bool, True where speed < v_rec AND speed > 0
    abnormal_delay : excess_delay masked to abnormal timesteps
    """
    spd = speed_df.values.astype("float32")
    ref = ref_speed.values.astype("float32")
    vrec = v_rec.values.astype("float32")
    L = link_lengths.reindex(speed_df.columns).values.astype("float32")

    with np.errstate(divide="ignore", invalid="ignore"):
        TT = np.where((spd > 0) & np.isfinite(spd), L[None, :] / spd, np.nan)
        TT_ref = np.where((ref > 0) & np.isfinite(ref), L[None, :] / ref, np.nan)

    excess = np.maximum(0.0, TT - TT_ref)
    excess = np.where(np.isnan(TT_ref), np.nan, excess)
    abnormal = (spd < vrec) & (spd > 0) & np.isfinite(spd)
    ab_delay = np.where(abnormal, excess, 0.0)

    idx = speed_df.index
    cols = speed_df.columns
    return (
        pd.DataFrame(excess, index=idx, columns=cols, dtype="float32"),
        pd.DataFrame(abnormal, index=idx, columns=cols, dtype="bool"),
        pd.DataFrame(ab_delay, index=idx, columns=cols, dtype="float32"),
    )


def _sum(df: pd.DataFrame, mask=None) -> float:
    if mask is None:
        return float(np.nansum(df.values))
    m = mask.values if hasattr(mask, "values") else mask
    return float(np.nansum(df.values[m]))


def _near_zero_pct(delay_df: pd.DataFrame, mask, threshold: float = 0.001) -> float:
    m = mask.values if hasattr(mask, "values") else mask
    vals = delay_df.values[m]
    if vals.size == 0:
        return 0.0
    return round(100 * np.sum((vals < threshold) | np.isnan(vals)) / vals.size, 1)


def delay_decomposition(
    labels: pd.DataFrame,
    incident_df: pd.DataFrame,
    excess: pd.DataFrame,
    ab_delay: pd.DataFrame,
    method_name: str = "method",
) -> dict:
    """Full delay decomposition for one labeling method.

    Returns a dict with total, method-captured, incident-excluded, and
    detector-only delay statistics for both excess and abnormal delay.
    """
    nr = labels == 1
    inc = incident_df == 1
    nr_only = nr & ~inc
    inc_only = ~nr & inc

    tot_exc = _sum(excess)
    tot_abn = _sum(ab_delay)

    def pct(n, d):
        return round(100 * n / d, 2) if d > 0 else 0.0

    nr_exc = _sum(excess, nr)
    nr_abn = _sum(ab_delay, nr)
    in_exc = _sum(excess, inc)
    in_abn = _sum(ab_delay, inc)
    ix_exc = _sum(excess, inc_only)
    ix_abn = _sum(ab_delay, inc_only)
    do_exc = _sum(excess, nr_only)
    do_abn = _sum(ab_delay, nr_only)
    ov_exc = _sum(excess, nr & inc)
    ov_abn = _sum(ab_delay, nr & inc)

    return dict(
        method=method_name,
        total_excess_lh=round(tot_exc, 4),
        total_abnormal_lh=round(tot_abn, 4),
        method_excess_lh=round(nr_exc, 4),
        method_abnormal_lh=round(nr_abn, 4),
        method_pct_total_excess=pct(nr_exc, tot_exc),
        method_pct_total_abnormal=pct(nr_abn, tot_abn),
        incident_reports_pct_excess=pct(in_exc, tot_exc),
        incident_reports_pct_abnormal=pct(in_abn, tot_abn),
        incident_excluded_excess_lh=round(ix_exc, 4),
        incident_excluded_abnormal_lh=round(ix_abn, 4),
        incident_excluded_pct_excess=pct(ix_exc, tot_exc),
        incident_excluded_pct_abnormal=pct(ix_abn, tot_abn),
        incident_excluded_near_zero_exc=_near_zero_pct(excess, inc_only),
        incident_excluded_near_zero_abn=_near_zero_pct(ab_delay, inc_only),
        detector_only_excess_lh=round(do_exc, 4),
        detector_only_abnormal_lh=round(do_abn, 4),
        detector_only_pct_excess=pct(do_exc, tot_exc),
        detector_only_pct_abnormal=pct(do_abn, tot_abn),
        overlap_excess_lh=round(ov_exc, 4),
        overlap_abnormal_lh=round(ov_abn, 4),
        labeled_steps=int(nr.values.sum()),
        anomaly_rate_pct=round(100 * nr.values.sum() / max(1, labels.size), 3),
    )


def unlabeled_abnormal_summary(
    speed_eval: pd.DataFrame,
    labels_eval: pd.DataFrame,
    v_rec_eval: pd.DataFrame,
    excess_eval: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Analyse timesteps that are abnormal (speed < v_rec) but not labeled.

    Returns (stats_dict, pivot_df) where pivot_df is a
    (day_of_week × hour) heatmap of unlabeled abnormal delay.
    """
    unlabeled_abn = ((speed_eval < v_rec_eval) & (labels_eval != 1)).values
    ex_vals = excess_eval.values.copy()
    ex_vals[np.isnan(ex_vals)] = 0.0
    contrib = np.where(unlabeled_abn, ex_vals, 0.0)
    per_step = contrib.sum(axis=1)

    ts = speed_eval.index
    hour = ts.hour
    dow = ts.dayofweek
    total = per_step.sum()
    peak = ((hour >= 6) & (hour <= 9)) | ((hour >= 15) & (hour <= 19))
    peak_share = round(100 * per_step[peak].sum() / max(1, total), 1)

    df_map = pd.DataFrame({"delay": per_step, "dow": dow, "hour": hour})
    pivot = df_map.groupby(["dow", "hour"])["delay"].sum().unstack(fill_value=0.0)

    return (
        dict(total_unlabeled_abn_lh=round(float(total), 3), peak_share_pct=peak_share),
        pivot,
    )
