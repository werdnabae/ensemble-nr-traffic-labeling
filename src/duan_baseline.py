"""Duan et al. (2024) slowdown-based label denoising baseline.

Implements the logic from Duan et al. (2024) arXiv:2412.10892 as faithfully
as possible.  Original functions are preserved verbatim; adaptations are
documented inline.

Reference:
  Duan, H., Wu, H., and Qian, S. (2024).  Know unreported roadway incidents
  in real-time: Early traffic anomaly detection.  arXiv:2412.10892v2.

Verbatim from original code (TSMO_2 notebook, Cell 7):
  - label_all_incident_contain_significant_sd
  - label_long_last_abnormal_sd_as_incident

Adaptations:
  - Day segmentation uses compute_session_ids() instead of fixed-length
    day_list_length splits, to handle variable-length sessions, weekends,
    and data gaps correctly.
  - Percentile threshold computed from calibration period only (original
    uses full dataset) to prevent temporal leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.ensemble_labeling import compute_session_ids, INTERVAL_MIN

# ── Duan (2024) verbatim functions ──────────────────────────────────────────


def _label_all_incident_contain_significant_sd(
    raw_inc_arr: np.ndarray,
    sign_sd_arr: np.ndarray,
) -> np.ndarray:
    """Verbatim: label_all_incident_contain_significant_sd (per session).

    Marks incident-report runs that overlap with ≥1 significant-slowdown step.
    Back-fills and forward-fills within the Waze run to its full extent.
    """
    day_len = len(raw_inc_arr)
    sum_list = raw_inc_arr + sign_sd_arr
    overlap_list = (sum_list == 2).astype(int)
    overlap_idx = np.where(sum_list == 2)[0]
    for index in overlap_idx:
        i = 1
        while (index - i) >= 0:
            if raw_inc_arr[index - i] == 1:
                overlap_list[index - i] = 1
                i += 1
            else:
                break
        j = 1
        while (index + j) < day_len:
            if raw_inc_arr[index + j] == 1:
                overlap_list[index + j] = 1
                j += 1
            else:
                break
    return overlap_list


def _label_long_last_abnormal_sd(
    sign_sd_arr: np.ndarray,
    minimum_length: int,
) -> np.ndarray:
    """Verbatim: label_long_last_abnormal_sd_as_incident (per session).

    Labels runs of consecutive significant-slowdown steps of length
    >= minimum_length.
    """
    day_len = len(sign_sd_arr)
    day_labels = np.zeros(day_len, dtype=int)
    count = 0
    for t in range(day_len):
        if sign_sd_arr[t] == 1:
            count += 1
            if count >= minimum_length:
                for idx in range(t - count + 1, t + 1):
                    day_labels[idx] = 1
        else:
            count = 0
    return day_labels


# ── Public entry point ──────────────────────────────────────────────────────


def duan_baseline(
    incident_df: pd.DataFrame,
    slowdown_df: pd.DataFrame,
    p95_slowdown: pd.Series,
    session_ids: np.ndarray,
    minimum_length: int = 3,
) -> pd.DataFrame:
    """Apply the Duan et al. (2024) denoising to both networks.

    Parameters
    ----------
    incident_df   : binary incident-report labels (0/1), evaluation period
    slowdown_df   : slowdown speed values, evaluation period
    p95_slowdown  : per-link 95th-percentile slowdown speed from calibration
    session_ids   : session ID array aligned to incident_df.index
    minimum_length: minimum consecutive slowdown steps to label standalone
                    anomaly (default 3 = 15 min, per original code)

    Returns
    -------
    pd.DataFrame  : binary labels (0/1), same shape as incident_df
    """
    result = pd.DataFrame(
        0, index=incident_df.index, columns=incident_df.columns, dtype="float32"
    )

    for link in incident_df.columns:
        p95 = float(p95_slowdown.get(str(link), np.nan))
        if np.isnan(p95):
            continue

        sign_sd = (slowdown_df[link] > p95).astype(int)
        raw_inc = incident_df[link].astype(int)

        inc_contain = np.zeros(len(raw_inc), dtype=int)
        long_sd = np.zeros(len(raw_inc), dtype=int)

        for sid in np.unique(session_ids):
            pos = np.where(session_ids == sid)[0]
            if len(pos) == 0:
                continue
            inc_contain[pos] = _label_all_incident_contain_significant_sd(
                raw_inc.values[pos], sign_sd.values[pos]
            )
            long_sd[pos] = _label_long_last_abnormal_sd(
                sign_sd.values[pos], minimum_length
            )

        result[link] = np.maximum(inc_contain, long_sd).astype("float32")

    return result
