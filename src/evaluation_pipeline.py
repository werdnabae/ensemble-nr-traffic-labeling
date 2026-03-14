#!/usr/bin/env python
"""Full evaluation pipeline — single entry point.

Reproduces all results in the paper:
  1. Temporal split (25% calibration / 75% held-out evaluation)
  2. Delay-based evaluation on held-out data
  3. Baseline comparison: incident reports, Duan (2024), full ensemble
  4. Ablation study
  5. Timing analysis: detection latency and termination overhang
  6. Unlabeled abnormal delay analysis (heatmaps)
  7. Figures and CSV tables in final_outputs/
  8. FINAL_RESULTS.md at project root

Networks
--------
  Cranberry : calibration + held-out evaluation
  TSMO      : cross-network validation (same frozen hyperparameters)

Temporal protocol
-----------------
  Calibration : first 25% of sessions (Cranberry Feb–Aug 2022,
                                        TSMO     Feb–May 2022)
  Evaluation  : remaining 75%

Parameters (frozen from prior calibration on Cranberry, C02):
  snd_c=2.5, grad_c=1.2, conf_f=0.70, min_dur=20 min

Usage
-----
  python src/evaluation_pipeline.py
  # or from project root:
  python -m src.evaluation_pipeline
"""

from __future__ import annotations

import sys, time, textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ensemble_labeling import (
    load_network_data,
    get_geojson_path,
    select_tmcs,
    get_link_lengths,
    compute_session_ids,
    temporal_split,
    compute_free_flow_speed,
    compute_slowdown_speed,
    build_upstream_neighbors,
    four_point_slopes,
    frozen_thresholds,
    frozen_confirmation_thresholds,
    frozen_percentile,
    run_ensemble_labeler,
    FINAL_SND_C,
    FINAL_GRAD_C,
    FINAL_CONF_F,
    FINAL_MIN_DUR,
    RECOVERY_SOFT_C_SHORT,
    RECOVERY_SOFT_C_LONG,
    HARD_RECOVERY_SHORT,
    HARD_RECOVERY_LONG,
    HARD_SPEED_FACTOR,
    MAX_GAP_MIN,
    INTERVAL_MIN,
    NETWORK_PARAMS,
    ALPHA_VREC,
)
from src.duan_baseline import duan_baseline
from src.delay_metrics import (
    compute_delay_matrices,
    delay_decomposition,
    unlabeled_abnormal_summary,
)
from src.timing_metrics import extract_episodes, compute_timing, summarise_timing

# ── directories ───────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
OUT = _ROOT / "final_outputs"
TABS = OUT / "tables"
FIGS = OUT / "figures"
for d in [TABS, FIGS]:
    d.mkdir(parents=True, exist_ok=True)

# ── configuration ─────────────────────────────────────────────────────────────
H_PERCENTILE = 0.95  # Duan baseline: 95th-pct slowdown threshold
H_MIN_LENGTH = 3  # Duan baseline: minimum run length (3 × 5 min = 15 min)

ABLATIONS = [
    dict(name="Deviation only", use_snd=True, use_gradient=False, use_slowdown=False),
    dict(
        name="Deviation + Gradient", use_snd=True, use_gradient=True, use_slowdown=False
    ),
    dict(
        name="Deviation + Slowdown", use_snd=True, use_gradient=False, use_slowdown=True
    ),
    dict(name="Full ensemble", use_snd=True, use_gradient=True, use_slowdown=True),
]

METHODS_ORDER = ["Incident reports only", "Duan baseline", "Full ensemble"]

METHOD_COLORS = {
    "Incident reports only": "#2c7bb6",
    "Duan baseline": "#fdae61",
    "Full ensemble": "#d7191c",
}


def banner(msg):
    print(f"\n{'=' * 65}\n  {msg}\n{'=' * 65}")


# ─────────────────────────────────────────────────────────────────────────────
# Network processor
# ─────────────────────────────────────────────────────────────────────────────


def process_network(network: str) -> dict:
    """Load data, calibrate on first 25%, evaluate on remaining 75%.

    Returns a dict with all results for this network.
    """
    banner(f"Processing network: {network.upper()}")
    t0 = time.time()

    cfg = NETWORK_PARAMS[network.lower()]
    geo = get_geojson_path(DATA_DIR / network.lower(), network)

    # ── Load and filter ───────────────────────────────────────────────────────
    print("  loading data…")
    speed_full, incident_full, upstream_raw = load_network_data(
        DATA_DIR / network.lower(), network
    )
    links = select_tmcs(speed_full, geo, network)
    speed_full = speed_full[links].sort_index()
    incident_full = incident_full.reindex(columns=links, index=speed_full.index).fillna(
        0
    )

    # ── Temporal split ────────────────────────────────────────────────────────
    c_mask, e_mask, n_calib, n_sess = temporal_split(speed_full)
    calib_speed = speed_full[c_mask]
    eval_speed = speed_full[e_mask]
    eval_inc = incident_full[e_mask]
    sids_e = compute_session_ids(eval_speed.index, INTERVAL_MIN)
    n_sess_e = int(sids_e[-1])

    print(
        f"  links={len(links)}  sessions={n_sess} "
        f"(calib={n_calib}, eval={n_sess - n_calib})"
    )
    print(f"  calib: {calib_speed.index[0].date()} → {calib_speed.index[-1].date()}")
    print(f"  eval:  {eval_speed.index[0].date()}  → {eval_speed.index[-1].date()}")

    # ── Calibration-period statistics ─────────────────────────────────────────
    print("  computing calibration statistics…")
    ffs = compute_free_flow_speed(calib_speed)
    ref_full = frozen_thresholds(calib_speed, speed_full, "dow_time", 0.0, "minus")
    vrec_full = frozen_thresholds(
        calib_speed, speed_full, "dow_time", ALPHA_VREC, "minus"
    )

    snd_win = cfg["snd_window_min"]
    snd_nr_f = frozen_thresholds(
        calib_speed, speed_full, "dow_timebin", FINAL_SND_C, "minus", snd_win
    )
    snd_r_f = frozen_thresholds(
        calib_speed, speed_full, "dow_timebin", cfg["snd_c_report"], "minus", snd_win
    )
    conf_all = frozen_confirmation_thresholds(
        calib_speed, [FINAL_CONF_F, cfg["confirmation_factor_report"], 0.75, 0.80]
    )
    rec_s_f = frozen_thresholds(
        calib_speed, speed_full, "dow_time", RECOVERY_SOFT_C_SHORT, "minus"
    )
    rec_l_f = frozen_thresholds(
        calib_speed, speed_full, "dow_time", RECOVERY_SOFT_C_LONG, "minus"
    )

    # ── Real-time features (eval period) ─────────────────────────────────────
    print("  computing real-time features…")
    up03 = build_upstream_neighbors(speed_full, geo, upstream_raw)
    slow_eval = compute_slowdown_speed(eval_speed, up03)
    nw_grad_eval = four_point_slopes(eval_speed, weighted=False)
    w_grad_eval = four_point_slopes(eval_speed, weighted=True)

    # Gradient thresholds from calib data
    slow_calib = compute_slowdown_speed(calib_speed, up03)
    nw_grad_calib = four_point_slopes(calib_speed, weighted=False)
    w_grad_calib = four_point_slopes(calib_speed, weighted=True)
    slow_thr_f = frozen_thresholds(
        slow_calib, speed_full, "dow_timebin", cfg["slowdown_c"], "plus", snd_win
    )
    nw_grad_f = frozen_thresholds(
        nw_grad_calib, speed_full, "dow_time", FINAL_GRAD_C, "minus"
    )
    w_grad_f = frozen_thresholds(
        w_grad_calib, speed_full, "dow_time", FINAL_GRAD_C, "minus"
    )
    p95_slow = frozen_percentile(slow_calib, H_PERCENTILE)

    # ── Restrict all threshold DFs to eval period ─────────────────────────────
    idx = eval_speed.index

    def e(df):
        return df.reindex(idx)

    ref_e = e(ref_full)
    vrec_e = e(vrec_full)
    snd_nr = e(snd_nr_f)
    snd_r = e(snd_r_f)
    slow_e = e(slow_thr_f)
    nw_g_e = e(nw_grad_f)
    w_g_e = e(w_grad_f)
    rec_s = e(rec_s_f)
    rec_l = e(rec_l_f)

    link_lengths = get_link_lengths(geo, links)

    # ── Delay matrices (eval period) ──────────────────────────────────────────
    print("  computing delay matrices…")
    excess_e, _, ab_delay_e = compute_delay_matrices(
        eval_speed, ref_e, vrec_e, link_lengths
    )

    # ── Run all methods ───────────────────────────────────────────────────────
    print("  running methods…")
    common_kwargs = dict(
        snd_threshold_nonreport=snd_nr,
        snd_threshold_report=snd_r,
        slowdown_threshold=slow_e,
        nw_gradient_df=nw_grad_eval,
        w_gradient_df=w_grad_eval,
        nw_gradient_threshold=nw_g_e,
        w_gradient_threshold=w_g_e,
        confirmation_threshold_nonreport=conf_all[FINAL_CONF_F],
        confirmation_threshold_report=conf_all.get(
            cfg["confirmation_factor_report"], conf_all[0.75]
        ),
        recovery_soft_short=rec_s,
        recovery_soft_long=rec_l,
        ffs=ffs,
        hard_speed_factor=HARD_SPEED_FACTOR,
        hard_rec_short=HARD_RECOVERY_SHORT,
        hard_rec_long=HARD_RECOVERY_LONG,
        min_nonreport_min=FINAL_MIN_DUR,
        max_gap_min=MAX_GAP_MIN,
        interval_min=INTERVAL_MIN,
        verbose=False,
    )

    labels_incident = eval_inc.copy()
    labels_duan = duan_baseline(eval_inc, slow_eval, p95_slow, sids_e, H_MIN_LENGTH)
    labels_ensemble = run_ensemble_labeler(
        eval_speed, eval_inc, slow_eval, **common_kwargs
    )

    # ── Delay decomposition for main methods ──────────────────────────────────
    print("  delay decomposition…")
    delay_rows = []
    for name, lbl in [
        ("Incident reports only", labels_incident),
        ("Duan baseline", labels_duan),
        ("Full ensemble", labels_ensemble),
    ]:
        row = delay_decomposition(lbl, eval_inc, excess_e, ab_delay_e, name)
        row["network"] = network
        delay_rows.append(row)
        print(
            f"    {name:<25s}  abn={row['method_pct_total_abnormal']:.1f}%  "
            f"det-only={row['detector_only_pct_abnormal']:.1f}%  "
            f"rate={row['anomaly_rate_pct']:.2f}%"
        )

    # ── Ablation ──────────────────────────────────────────────────────────────
    print("  ablation…")
    abl_rows = []
    for ab in ABLATIONS:
        lbl = run_ensemble_labeler(
            eval_speed,
            eval_inc,
            slow_eval,
            **{
                **common_kwargs,
                "use_snd": bool(ab["use_snd"]),
                "use_gradient": bool(ab["use_gradient"]),
                "use_slowdown": bool(ab["use_slowdown"]),
            },
        )
        row = delay_decomposition(lbl, eval_inc, excess_e, ab_delay_e, ab["name"])
        row["network"] = network
        abl_rows.append(row)
        print(f"    {ab['name']:<25s}  abn={row['method_pct_total_abnormal']:.1f}%")

    # ── Timing analysis ───────────────────────────────────────────────────────
    print("  timing analysis…")
    timing_rows = []
    for name, lbl in [
        ("Incident reports only", labels_incident),
        ("Duan baseline", labels_duan),
        ("Full ensemble", labels_ensemble),
    ]:
        eps = extract_episodes(lbl)
        tdf = compute_timing(eps, eval_speed, vrec_e, sids_e)
        summ = summarise_timing(tdf, name, network)
        summ["network"] = network
        timing_rows.append({**summ, "_tdf": tdf})
        print(
            f"    {name:<25s}  eps={summ['n_episodes']:5d}  "
            f"lat_med={summ['latency_median']:+5.0f}min  "
            f"over_med={summ['overhang_median']:+5.0f}min"
        )

    # ── Unlabeled abnormal delay ──────────────────────────────────────────────
    print("  unlabeled abnormal delay…")
    unlab_stats, unlab_pivot = unlabeled_abnormal_summary(
        eval_speed, labels_ensemble, vrec_e, excess_e
    )
    unlab_stats["network"] = network

    print(f"  done in {(time.time() - t0) / 60:.1f} min")

    return dict(
        network=network,
        links=links,
        eval_speed=eval_speed,
        eval_inc=eval_inc,
        vrec_e=vrec_e,
        ref_e=ref_e,
        excess_e=excess_e,
        ab_delay_e=ab_delay_e,
        labels=dict(
            incident=labels_incident, duan=labels_duan, ensemble=labels_ensemble
        ),
        delay_rows=delay_rows,
        abl_rows=abl_rows,
        timing_rows=timing_rows,
        unlab_stats=unlab_stats,
        unlab_pivot=unlab_pivot,
        calib_period=f"{calib_speed.index[0].date()} → {calib_speed.index[-1].date()}",
        eval_period=f"{eval_speed.index[0].date()} → {eval_speed.index[-1].date()}",
        n_calib=n_calib,
        n_sess=n_sess,
        n_sess_e=n_sess_e,
        sids_e=sids_e,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────


def fig_abnormal_delay_capture(cran_rows, tsmo_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rows, net in [
        (axes[0], cran_rows, "Cranberry"),
        (axes[1], tsmo_rows, "TSMO"),
    ]:
        methods = [r["method"] for r in rows if r["method"] in METHODS_ORDER]
        rows_f = [r for r in rows if r["method"] in METHODS_ORDER]
        vals = [r["method_pct_total_abnormal"] for r in rows_f]
        colors = [METHOD_COLORS.get(m, "grey") for m in methods]
        bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_ylabel("% of total abnormal delay captured", fontsize=9)
        ax.set_title(f"{net}", fontsize=10)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.3,
                f"{b.get_height():.1f}%",
                ha="center",
                fontsize=8,
            )
    fig.suptitle("Abnormal delay capture — held-out evaluation", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "abnormal_delay_capture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_baseline_comparison(cran_rows, tsmo_rows):
    metrics = [
        ("method_pct_total_abnormal", "% abnormal delay captured"),
        ("method_pct_total_excess", "% excess delay captured"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (mkey, mlabel) in zip(axes, metrics):
        x = np.arange(len(METHODS_ORDER))
        w = 0.35
        for i, (rows, net, col) in enumerate(
            [(cran_rows, "Cranberry", "#2166ac"), (tsmo_rows, "TSMO", "#d01c8b")]
        ):
            sub = {r["method"]: r for r in rows}
            vals = [sub.get(m, {}).get(mkey, 0) for m in METHODS_ORDER]
            ax.bar(x + (i - 0.5) * w, vals, w, label=net, color=col, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS_ORDER, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Baseline comparison — held-out evaluation period", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_unlabeled_heatmap(pivot: pd.DataFrame, network: str, peak_share: float):
    fig, ax = plt.subplots(figsize=(12, 4))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [dow_labels[i] for i in pivot.index if i < len(dow_labels)]
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.2,
        cbar_kws={"label": "Link-hours of unlabeled abnormal delay"},
    )
    ax.set_xlabel("Hour of day", fontsize=9)
    ax.set_title(
        f"{network} — Unlabeled abnormal delay  "
        f"({peak_share}% during AM/PM peak hours 06–09, 15–19)",
        fontsize=9,
    )
    fig.tight_layout()
    fname = f"unlabeled_delay_heatmap_{network.lower()}.png"
    fig.savefig(FIGS / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_timing(timing_rows_all: list, network: str):
    td = {
        r["method"]: r.pop("_tdf")
        for r in timing_rows_all
        if r.get("network") == network
    }
    if not td:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Latency distribution
    ax = axes[0]
    for method, tdf in td.items():
        vals = tdf[tdf.has_speed_anomaly].detection_latency_min.dropna().clip(-60, 120)
        ax.hist(
            vals,
            bins=30,
            alpha=0.55,
            density=True,
            edgecolor="white",
            lw=0.3,
            label=method,
            color=METHOD_COLORS.get(method, "grey"),
        )
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Detection latency (min)  [+= late]", fontsize=8)
    ax.set_title("Detection latency", fontsize=9)
    ax.legend(fontsize=7)

    # Overhang distribution
    ax = axes[1]
    for method, tdf in td.items():
        vals = (
            tdf[tdf.has_speed_anomaly].termination_overhang_min.dropna().clip(-60, 180)
        )
        ax.hist(
            vals,
            bins=30,
            alpha=0.55,
            density=True,
            edgecolor="white",
            lw=0.3,
            label=method,
            color=METHOD_COLORS.get(method, "grey"),
        )
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Termination overhang (min)  [+= past true end]", fontsize=8)
    ax.set_title("Termination overhang", fontsize=9)

    # Duration box plots
    ax = axes[2]
    data = [tdf.duration_min.clip(0, 480).dropna().values for tdf in td.values()]
    labels = list(td.keys())
    colors = [METHOD_COLORS.get(m, "grey") for m in labels]
    bp = ax.boxplot(
        data, patch_artist=True, notch=False, medianprops=dict(color="black", lw=1.5)
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Duration (min, clipped 480)", fontsize=8)
    ax.set_title("Episode duration", fontsize=9)

    fig.suptitle(f"{network} — Timing analysis", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / f"timing_{network.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return td  # return tdf objects back (needed for tables)


def fig_case_studies(net_result: dict):
    network = net_result["network"]
    speed = net_result["eval_speed"]
    vrec = net_result["vrec_e"]
    ref_spd = net_result["ref_e"]
    labels = net_result["labels"]
    ens_tdf = net_result.get("_ens_tdf")
    if ens_tdf is None:
        return

    valid = ens_tdf[ens_tdf.has_speed_anomaly].copy()
    if len(valid) == 0:
        return
    valid["abs_lat"] = valid.detection_latency_min.abs()
    top = valid.nlargest(min(4, len(valid)), "abs_lat").drop_duplicates("link").head(4)

    n = len(top)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n))
    if n == 1:
        axes = [axes]
    delta = pd.Timedelta(hours=1)

    for ax, (_, ep) in zip(axes, top.iterrows()):
        lnk = ep.link
        lo = max(speed.index[0], ep.start - delta)
        hi = min(speed.index[-1], ep.end + delta)
        spd_w = speed[lnk].loc[lo:hi]
        vr_w = vrec[lnk].loc[lo:hi]
        rs_w = ref_spd[lnk].loc[lo:hi]

        for method, lbl, col, alpha in [
            ("Incident reports only", labels["incident"], "#2c7bb6", 0.30),
            ("Duan baseline", labels["duan"], "#fdae61", 0.35),
            ("Full ensemble", labels["ensemble"], "#d7191c", 0.45),
        ]:
            lbl_w = lbl[lnk].loc[lo:hi]
            arr = lbl_w.values
            idx = lbl_w.index
            added = False
            i = 0
            while i < len(arr):
                if arr[i] == 1:
                    j = i
                    while j < len(arr) and arr[j] == 1:
                        j += 1
                    kw = dict(color=col, alpha=alpha)
                    if not added:
                        kw["label"] = method
                        added = True
                    ax.axvspan(idx[i], idx[j - 1], **kw)
                    i = j
                else:
                    i += 1

        ax.plot(
            spd_w.index,
            spd_w.values,
            color="steelblue",
            lw=1.1,
            zorder=5,
            label="Speed",
        )
        ax.plot(
            rs_w.index,
            rs_w.values,
            color="grey",
            lw=0.9,
            ls="--",
            zorder=4,
            label="Ref speed",
        )
        ax.plot(
            vr_w.index,
            vr_w.values,
            color="#d01c8b",
            lw=0.9,
            ls=":",
            zorder=4,
            label="v_rec",
        )
        if pd.notna(ep.true_start):
            ax.axvline(ep.true_start, color="black", lw=1.2, ls="-.")
        if pd.notna(ep.true_end):
            ax.axvline(ep.true_end, color="black", lw=1.2, ls="-.")

        ax.set_title(
            f"{lnk}  {ep.start.strftime('%Y-%m-%d %H:%M')}  "
            f"dur={ep.duration_min:.0f}min  "
            f"lat={ep.detection_latency_min:+.0f}min  "
            f"overhang={ep.termination_overhang_min:+.0f}min",
            fontsize=8,
        )
        ax.set_ylabel("mph", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_tick_params(rotation=20)
        lo_y = max(0, spd_w.min() - 5) if spd_w.notna().any() else 0
        ax.set_ylim(lo_y, spd_w.max() + 8 if spd_w.notna().any() else 80)

    axes[0].legend(loc="lower right", fontsize=7, ncol=3)
    fig.suptitle(f"Case studies — {network} (held-out period)", fontsize=10)
    fig.tight_layout()
    fig.savefig(
        FIGS / f"case_studies_{network.lower()}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FINAL_RESULTS.md
# ─────────────────────────────────────────────────────────────────────────────


def _tbl(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_final_results(
    cran: dict,
    tsmo: dict,
    delay_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    unlab_df: pd.DataFrame,
):
    cr = delay_df[delay_df.network == "Cranberry"]
    tr = delay_df[delay_df.network == "TSMO"]

    def pick(df, method, col):
        row = df[df.method == method]
        return f"{row[col].values[0]:.1f}%" if len(row) else "—"

    report = textwrap.dedent(f"""\
    # FINAL RESULTS: Ensemble Nonrecurrent Traffic Disturbance Labeling

    **Configuration (frozen):** snd_c=2.5, grad_c=1.2, conf_f=0.70, min_dur=20 min
    **Temporal protocol:** 25% calibration / 75% held-out evaluation (temporal order)
    **Delay unit:** link-hours of excess travel time per link-traversal

    ---

    ## 1. Dataset summary

    | Network | Links | Calibration period | Evaluation period | Eval sessions |
    |---|---|---|---|---|
    | Cranberry | {len(cran["links"])} | {cran["calib_period"]} | {
        cran["eval_period"]
    } | {cran["n_sess_e"]} |
    | TSMO | {len(tsmo["links"])} | {tsmo["calib_period"]} | {tsmo["eval_period"]} | {
        tsmo["n_sess_e"]
    } |

    ---

    ## 2. Delay capture — held-out evaluation

    {
        _tbl(
            delay_df[
                [
                    "network",
                    "method",
                    "method_pct_total_abnormal",
                    "method_pct_total_excess",
                    "detector_only_pct_abnormal",
                    "incident_excluded_near_zero_abn",
                    "anomaly_rate_pct",
                ]
            ].rename(
                columns={
                    "method_pct_total_abnormal": "% abnormal delay",
                    "method_pct_total_excess": "% excess delay",
                    "detector_only_pct_abnormal": "% detector-only abn.",
                    "incident_excluded_near_zero_abn": "% excl. near-zero",
                    "anomaly_rate_pct": "anomaly rate %",
                }
            )
        )
    }

    ### Key findings

    - **Full ensemble captures {
        pick(cr, "Full ensemble", "method_pct_total_abnormal")
    } of abnormal delay on Cranberry**
      vs. {pick(cr, "Duan baseline", "method_pct_total_abnormal")} for Duan baseline
      and {
        pick(cr, "Incident reports only", "method_pct_total_abnormal")
    } for incident reports alone.
    - On TSMO: ensemble {pick(tr, "Full ensemble", "method_pct_total_abnormal")}
      vs. Duan {pick(tr, "Duan baseline", "method_pct_total_abnormal")}
      vs. incident reports {
        pick(tr, "Incident reports only", "method_pct_total_abnormal")
    }.

    ---

    ## 3. Incident-excluded periods (low-impact filtering)

    | Network | Method | Incident-excl. excess (link-h) | Incident-excl. abn. (link-h) | Near-zero excess % | Near-zero abn. % |
    |---|---|---|---|---|---|
    {
        chr(10).join(
            f"| {r.network} | {r.method} | {r.incident_excluded_excess_lh:.2f} "
            f"| {r.incident_excluded_abnormal_lh:.2f} "
            f"| {r.incident_excluded_near_zero_exc}% "
            f"| {r.incident_excluded_near_zero_abn}% |"
            for _, r in delay_df[delay_df.method == "Full ensemble"].iterrows()
        )
    }

    ---

    ## 4. Ablation study

    {
        _tbl(
            abl_df[
                [
                    "network",
                    "method",
                    "method_pct_total_abnormal",
                    "method_pct_total_excess",
                    "anomaly_rate_pct",
                ]
            ].rename(
                columns={
                    "method_pct_total_abnormal": "% abn. delay",
                    "method_pct_total_excess": "% excess delay",
                    "anomaly_rate_pct": "anomaly rate %",
                }
            )
        )
    }

    ---

    ## 5. Timing analysis

    {
        _tbl(
            timing_df[
                [
                    "network",
                    "method",
                    "n_episodes",
                    "pct_with_speed_anomaly",
                    "latency_median",
                    "latency_mean",
                    "overhang_median",
                    "overhang_mean",
                    "duration_median",
                ]
            ].rename(
                columns={
                    "n_episodes": "episodes",
                    "pct_with_speed_anomaly": "% w/ speed anomaly",
                    "latency_median": "lat. median (min)",
                    "latency_mean": "lat. mean (min)",
                    "overhang_median": "overhang median (min)",
                    "overhang_mean": "overhang mean (min)",
                    "duration_median": "dur. median (min)",
                }
            )
        )
    }

    ---

    ## 6. Unlabeled abnormal delay

    {
        _tbl(
            unlab_df[["network", "total_unlabeled_abn_lh", "peak_share_pct"]].rename(
                columns={
                    "total_unlabeled_abn_lh": "unlabeled abn. delay (link-h)",
                    "peak_share_pct": "% in AM/PM peak (06-09, 15-19)",
                }
            )
        )
    }

    The high concentration of unlabeled abnormal delay in predictable AM/PM
    peak hours indicates that the unlabeled fraction is predominantly recurrent
    congestion that the method correctly excludes, not missed non-recurrent events.

    ---

    ## 7. Figures

    | File | Description |
    |---|---|
    | `figures/abnormal_delay_capture.png` | Main result: abnormal delay capture by method and network |
    | `figures/baseline_comparison.png` | Side-by-side comparison of three methods |
    | `figures/unlabeled_delay_heatmap_cranberry.png` | DOW × hour heatmap of unlabeled abnormal delay (Cranberry) |
    | `figures/unlabeled_delay_heatmap_tsmo.png` | DOW × hour heatmap of unlabeled abnormal delay (TSMO) |
    | `figures/timing_cranberry.png` | Detection latency and overhang distributions (Cranberry) |
    | `figures/timing_tsmo.png` | Detection latency and overhang distributions (TSMO) |
    | `figures/case_studies_cranberry.png` | Speed time series with detection windows (Cranberry) |
    | `figures/case_studies_tsmo.png` | Speed time series with detection windows (TSMO) |

    ---

    ## 8. Methodology notes

    ### Temporal leakage control
    All threshold statistics (IQR-based detector thresholds, reference speeds,
    v_rec lower bounds, FFS, Duan 95th-percentile slowdown threshold) are
    computed from the calibration period only and applied to the held-out
    evaluation period without refit.

    ### Duan (2024) baseline
    Implements label_all_incident_contain_significant_sd and
    label_long_last_abnormal_sd_as_incident verbatim from Duan et al. (2024).
    Adaptation: session-based day segmentation replacing fixed-length day splits.
    Parameters: 95th-percentile slowdown threshold, minimum run length = 3 steps.

    ### Terminology
    "Incident reports" refers to crowd-sourced incident feeds used as optional
    supporting evidence.  They are not treated as ground truth.  "Abnormal delay"
    is delay occurring when speed falls below the recurrent lower bound
    v_rec = median − 0.7 × IQR per (link, day-of-week, time-of-day) slot.
    """)

    path = _ROOT / "FINAL_RESULTS.md"
    path.write_text(report)
    print(f"  FINAL_RESULTS.md written ({path})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    banner("Ensemble NR Disturbance Labeling — Final Evaluation Pipeline")

    cran = process_network("cranberry")
    tsmo = process_network("tsmo")

    # ── Assemble tables ───────────────────────────────────────────────────────
    banner("Assembling tables and figures")

    # Main delay results (baselines only)
    delay_df = pd.DataFrame(cran["delay_rows"] + tsmo["delay_rows"])
    delay_df["method"] = pd.Categorical(
        delay_df["method"], categories=METHODS_ORDER, ordered=True
    )
    delay_df = delay_df.sort_values(["network", "method"])

    # Ablation
    abl_df = pd.DataFrame(cran["abl_rows"] + tsmo["abl_rows"])

    # Timing — pop _tdf objects before saving
    timing_rows_clean = []
    timing_tdfs = {}
    for r in cran["timing_rows"] + tsmo["timing_rows"]:
        r2 = {k: v for k, v in r.items() if k != "_tdf"}
        net = r2["network"]
        method = r2["method"]
        timing_tdfs[(net, method)] = r.get("_tdf")
        timing_rows_clean.append(r2)
    timing_df = pd.DataFrame(timing_rows_clean)

    # Attach ens_tdf for case studies
    for res in [cran, tsmo]:
        res["_ens_tdf"] = timing_tdfs.get((res["network"], "Full ensemble"))

    # Unlabeled abnormal delay
    unlab_df = pd.DataFrame([cran["unlab_stats"], tsmo["unlab_stats"]])

    # ── Save tables ───────────────────────────────────────────────────────────
    delay_df.to_csv(TABS / "heldout_delay_results.csv", index=False)
    delay_df[delay_df.method.isin(METHODS_ORDER)].to_csv(
        TABS / "baseline_comparison.csv", index=False
    )
    abl_df.to_csv(TABS / "ablation_results.csv", index=False)
    timing_df.to_csv(TABS / "timing_results.csv", index=False)
    unlab_df.to_csv(TABS / "unlabeled_abnormal_delay.csv", index=False)
    print("  tables saved")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_abnormal_delay_capture(cran["delay_rows"], tsmo["delay_rows"])
    fig_baseline_comparison(cran["delay_rows"], tsmo["delay_rows"])

    for res in [cran, tsmo]:
        fig_unlabeled_heatmap(
            res["unlab_pivot"], res["network"], res["unlab_stats"]["peak_share_pct"]
        )
        net = res["network"]
        timing_rows_net = [
            r
            for r in cran["timing_rows"] + tsmo["timing_rows"]
            if r.get("network") == net
        ]
        fig_timing(timing_rows_net, net)
        fig_case_studies(res)

    print("  figures saved")

    # ── FINAL_RESULTS.md ─────────────────────────────────────────────────────
    banner("Writing FINAL_RESULTS.md")
    write_final_results(cran, tsmo, delay_df, abl_df, timing_df, unlab_df)

    banner(f"Pipeline complete in {(time.time() - t0) / 60:.1f} min")
    print(f"  tables  → {TABS}")
    print(f"  figures → {FIGS}")
    print(f"  report  → {_ROOT / 'FINAL_RESULTS.md'}")


if __name__ == "__main__":
    main()
