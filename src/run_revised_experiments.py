#!/usr/bin/env python
"""Revised experimental pipeline for TR Part C submission.

Changes from prior pipeline:
  - Reversed calibration: parameters now calibrated on TSMO (disturbance-rich),
    transferred to Cranberry (lighter congestion) to demonstrate generalization.
  - Extended ablation: adds gradient-only, slowdown-only, without-persistence,
    and without-recovery variants alongside the original detector-subset ablations.
  - Sensitivity analysis: ±20% perturbation of all four tunable parameters.
  - New sections: cross-network transfer, evaluation-without-ground-truth,
    improved contribution framing throughout.

Usage
-----
  python src/run_revised_experiments.py
"""

from __future__ import annotations

import sys, time, textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

DATA_DIR = _ROOT / "data"
OUT = _ROOT / "final_outputs"
TABS = OUT / "tables"
FIGS = OUT / "figures"
for d in [TABS, FIGS]:
    d.mkdir(parents=True, exist_ok=True)

H_PERCENTILE = 0.95
H_MIN_LENGTH = 3

# ── Grid search (tuned on calibration network) ────────────────────────────────
GRID = [
    # (snd_c, grad_c, conf_f, min_dur)
    (2.5, 1.2, 0.70, 20),  # C01 – baseline
    (2.0, 1.2, 0.70, 20),  # C02 – looser SND
    (3.0, 1.2, 0.70, 20),  # C03 – tighter SND
    (2.5, 1.0, 0.70, 20),  # C04 – looser gradient
    (2.5, 1.4, 0.70, 20),  # C05 – tighter gradient
    (2.5, 1.2, 0.65, 20),  # C06 – stricter confirmation
    (2.5, 1.2, 0.75, 20),  # C07 – looser confirmation
    (2.5, 1.2, 0.70, 15),  # C08 – shorter persistence
    (2.5, 1.2, 0.70, 25),  # C09 – longer persistence
    (2.0, 1.0, 0.75, 15),  # C10 – all looser
    (3.0, 1.4, 0.65, 25),  # C11 – all tighter
]

# ── Extended ablation definitions ─────────────────────────────────────────────
ABLATIONS = [
    # name, use_snd, use_gradient, use_slowdown, skip_persistence, skip_recovery
    ("Speed deviation only", True, False, False, False, False),
    ("Temporal gradient only", False, True, False, False, False),
    ("Upstream slowdown only", False, False, True, False, False),
    ("Deviation + Gradient", True, True, False, False, False),
    ("Deviation + Slowdown", True, False, True, False, False),
    ("Full ensemble w/o persistence", True, True, True, True, False),
    ("Full ensemble w/o recovery", True, True, True, False, True),
    ("Full ensemble", True, True, True, False, False),
]


# ── Sensitivity analysis: ±20% perturbation ──────────────────────────────────
def sensitivity_variants(baseline_snd, baseline_grad, baseline_conf, baseline_dur):
    bsnd, bgrad, bconf, bdur = baseline_snd, baseline_grad, baseline_conf, baseline_dur
    return [
        ("Baseline", bsnd, bgrad, bconf, bdur),
        ("snd_c −20%", bsnd * 0.8, bgrad, bconf, bdur),
        ("snd_c +20%", bsnd * 1.2, bgrad, bconf, bdur),
        ("grad_c −20%", bsnd, bgrad * 0.8, bconf, bdur),
        ("grad_c +20%", bsnd, bgrad * 1.2, bconf, bdur),
        ("conf_f −20%", bsnd, bgrad, bconf * 0.8, bdur),
        ("conf_f +20%", bsnd, bgrad, bconf * 1.2, bdur),
        ("min_dur −20%", bsnd, bgrad, bconf, max(5, int(bdur * 0.8))),
        ("min_dur +20%", bsnd, bgrad, bconf, int(bdur * 1.2)),
    ]


def banner(msg):
    print(f"\n{'=' * 65}\n  {msg}\n{'=' * 65}")


# ─────────────────────────────────────────────────────────────────────────────
# Network data loader (shared across experiments)
# ─────────────────────────────────────────────────────────────────────────────


class NetworkBundle:
    """Holds all pre-computed features and calibration-period statistics."""

    def __init__(self, network: str, calib_frac: float = 0.25):
        self.network = network
        cfg = NETWORK_PARAMS[network.lower()]
        geo = get_geojson_path(DATA_DIR / network.lower(), network)

        speed_raw, incident_raw, upstream_raw = load_network_data(
            DATA_DIR / network.lower(), network
        )
        links = select_tmcs(speed_raw, geo, network)
        speed_raw = speed_raw[links].sort_index()
        incident_raw = incident_raw.reindex(
            columns=links, index=speed_raw.index
        ).fillna(0)

        c_mask, e_mask, n_calib, n_sess = temporal_split(speed_raw, calib_frac)
        self.calib_speed = speed_raw[c_mask]
        self.eval_speed = speed_raw[e_mask]
        self.eval_inc = incident_raw[e_mask]
        self.sids_e = compute_session_ids(self.eval_speed.index, INTERVAL_MIN)
        self.n_sess_e = int(self.sids_e[-1])
        self.links = links
        self.n_calib = n_calib
        self.n_sess = n_sess

        # Calibration-period reference statistics
        self.ffs = compute_free_flow_speed(self.calib_speed)
        ref_f = frozen_thresholds(self.calib_speed, speed_raw, "dow_time", 0.0, "minus")
        vrec_f = frozen_thresholds(
            self.calib_speed, speed_raw, "dow_time", ALPHA_VREC, "minus"
        )
        snd_win = cfg["snd_window_min"]
        self.snd_r_f = frozen_thresholds(
            self.calib_speed,
            speed_raw,
            "dow_timebin",
            cfg["snd_c_report"],
            "minus",
            snd_win,
        )
        self.conf_all = frozen_confirmation_thresholds(
            self.calib_speed, [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.84]
        )
        self.rec_s_f = frozen_thresholds(
            self.calib_speed, speed_raw, "dow_time", RECOVERY_SOFT_C_SHORT, "minus"
        )
        self.rec_l_f = frozen_thresholds(
            self.calib_speed, speed_raw, "dow_time", RECOVERY_SOFT_C_LONG, "minus"
        )

        # Real-time features (eval period only)
        up03 = build_upstream_neighbors(speed_raw, geo, upstream_raw)
        self.slow_eval = compute_slowdown_speed(self.eval_speed, up03)
        self.nw_grad_eval = four_point_slopes(self.eval_speed, weighted=False)
        self.w_grad_eval = four_point_slopes(self.eval_speed, weighted=True)

        # Slowdown and gradient calibration statistics
        slow_calib = compute_slowdown_speed(self.calib_speed, up03)
        self._nw_grad_calib = four_point_slopes(self.calib_speed, weighted=False)
        self._w_grad_calib = four_point_slopes(self.calib_speed, weighted=True)
        self.slow_thr_f = frozen_thresholds(
            slow_calib, speed_raw, "dow_timebin", cfg["slowdown_c"], "plus", snd_win
        )
        self.nw_grad_f = frozen_thresholds(
            self._nw_grad_calib, speed_raw, "dow_time", FINAL_GRAD_C, "minus"
        )
        self.w_grad_f = frozen_thresholds(
            self._w_grad_calib, speed_raw, "dow_time", FINAL_GRAD_C, "minus"
        )
        self.p95_slow = frozen_percentile(slow_calib, H_PERCENTILE)

        # Restrict reference DFs to eval period
        idx = self.eval_speed.index
        self.ref_e = ref_f.reindex(idx)
        self.vrec_e = vrec_f.reindex(idx)

        # Pre-compute delay matrices
        ll = get_link_lengths(geo, links)
        self.excess_e, _, self.ab_delay_e = compute_delay_matrices(
            self.eval_speed, self.ref_e, self.vrec_e, ll
        )

        # Grad thresholds cache: key = (snd_c, grad_c) for efficiency
        self._snd_cache = {}
        self._grad_cache = {}
        self._snd_win = snd_win
        self._speed_raw = speed_raw

        print(
            f"  {network}: {len(links)} links, {n_sess} sessions, "
            f"calib={n_calib} eval={n_sess - n_calib}"
        )

    def get_snd_thr(self, snd_c: float) -> pd.DataFrame:
        """SND threshold (non-report) for given c, cached."""
        if snd_c not in self._snd_cache:
            self._snd_cache[snd_c] = frozen_thresholds(
                self.calib_speed,
                self._speed_raw,
                "dow_timebin",
                snd_c,
                "minus",
                self._snd_win,
            ).reindex(self.eval_speed.index)
        return self._snd_cache[snd_c]

    def get_grad_thr(self, grad_c: float) -> tuple:
        """(nw_grad_thr, w_grad_thr) for given c, cached.  Uses pre-computed
        calibration-period slopes stored in _nw_grad_calib / _w_grad_calib."""
        if grad_c not in self._grad_cache:
            nw = frozen_thresholds(
                self._nw_grad_calib, self._speed_raw, "dow_time", grad_c, "minus"
            )
            w = frozen_thresholds(
                self._w_grad_calib, self._speed_raw, "dow_time", grad_c, "minus"
            )
            self._grad_cache[grad_c] = (
                nw.reindex(self.eval_speed.index),
                w.reindex(self.eval_speed.index),
            )
        return self._grad_cache[grad_c]

    def _closest_conf(self, conf_f: float) -> pd.Series:
        """Return the confirmation threshold Series whose factor is closest
        to conf_f.  Avoids KeyError when exact value is not pre-computed."""
        conf_f_r = round(conf_f, 2)
        if conf_f_r in self.conf_all:
            return self.conf_all[conf_f_r]
        closest = min(self.conf_all.keys(), key=lambda k: abs(k - conf_f))
        return self.conf_all[closest]

    def run(
        self,
        snd_c,
        grad_c,
        conf_f,
        min_dur,
        use_snd=True,
        use_gradient=True,
        use_slowdown=True,
        skip_persistence=False,
        skip_recovery=False,
    ) -> pd.DataFrame:
        """Run labeler with given params on eval period."""
        idx = self.eval_speed.index
        snd_e = self.get_snd_thr(snd_c)
        nw_g_e, w_g_e = self.get_grad_thr(grad_c)
        cfg = NETWORK_PARAMS[self.network.lower()]
        return run_ensemble_labeler(
            self.eval_speed,
            self.eval_inc,
            self.slow_eval,
            snd_e,
            self.snd_r_f.reindex(idx),
            self.slow_thr_f.reindex(idx),
            self.nw_grad_eval,
            self.w_grad_eval,
            nw_g_e,
            w_g_e,
            self._closest_conf(conf_f),
            self._closest_conf(cfg["confirmation_factor_report"]),
            self.rec_s_f.reindex(idx),
            self.rec_l_f.reindex(idx),
            self.ffs,
            hard_speed_factor=HARD_SPEED_FACTOR,
            hard_rec_short=HARD_RECOVERY_SHORT,
            hard_rec_long=HARD_RECOVERY_LONG,
            min_nonreport_min=min_dur,
            max_gap_min=MAX_GAP_MIN,
            interval_min=INTERVAL_MIN,
            use_snd=use_snd,
            use_gradient=use_gradient,
            use_slowdown=use_slowdown,
            skip_persistence=skip_persistence,
            skip_recovery=skip_recovery,
            verbose=False,
        )

    def metrics(self, labels, label_name="method") -> dict:
        m = delay_decomposition(
            labels, self.eval_inc, self.excess_e, self.ab_delay_e, label_name
        )
        m["network"] = self.network
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Grid search on calibration network
# ─────────────────────────────────────────────────────────────────────────────


def run_grid_search(bundle: NetworkBundle):
    """Run grid search on the calibration network.  Return (best_params_tuple,
    grid_df) where best_params_tuple = (snd_c, grad_c, conf_f, min_dur)."""
    banner(f"Grid search on calibration network: {bundle.network}")
    best_score = -1.0
    best_params = (FINAL_SND_C, FINAL_GRAD_C, FINAL_CONF_F, FINAL_MIN_DUR)
    rows = []
    for snd_c, grad_c, conf_f, min_dur in GRID:
        t0 = time.time()
        labels = bundle.run(snd_c, grad_c, conf_f, min_dur)
        m = bundle.metrics(labels, "grid")
        score = m["method_pct_total_abnormal"]
        # Stability penalty: too many very-long episodes is a bad sign
        eps = extract_episodes(labels)
        pct_long = float((eps.duration_min > 240).mean()) if len(eps) else 0.0
        adj_score = score - 2.0 * pct_long  # penalise fragmentation
        rows.append(
            dict(
                snd_c=snd_c,
                grad_c=grad_c,
                conf_f=conf_f,
                min_dur=min_dur,
                score=round(score, 2),
                pct_very_long=round(pct_long * 100, 1),
                adj_score=round(adj_score, 2),
                n_eps=len(eps),
                time_s=round(time.time() - t0, 1),
            )
        )
        if adj_score > best_score:
            best_score = adj_score
            best_params = (snd_c, grad_c, conf_f, min_dur)
        print(
            f"  ({snd_c:.1f},{grad_c:.1f},{conf_f:.2f},{min_dur:2d}) "
            f"abn={score:5.1f}%  vlong={pct_long * 100:.1f}%  adj={adj_score:.2f}"
        )
    grid_df = pd.DataFrame(rows)
    snd_c, grad_c, conf_f, min_dur = best_params
    print(
        f"\n  BEST: snd_c={snd_c}, grad_c={grad_c}, conf_f={conf_f}, min_dur={min_dur}"
    )
    return best_params, grid_df


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

CLRS = {
    "Incident reports only": "#2c7bb6",
    "Duan baseline": "#fdae61",
    "Full ensemble": "#d7191c",
    "calib": "#4dac26",
    "transfer": "#b2182b",
}


def fig_delay_bars(calib_rows, transfer_rows, out_path):
    """Bar chart: abnormal delay capture for calib and transfer networks."""
    methods = ["Incident reports only", "Duan baseline", "Full ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (rows, title) in zip(
        axes,
        [(calib_rows, "TSMO (calibration)"), (transfer_rows, "Cranberry (transfer)")],
    ):
        sub = {r["method"]: r for r in rows}
        vals = [sub.get(m, {}).get("method_pct_total_abnormal", 0) for m in methods]
        colors = [CLRS.get(m, "grey") for m in methods]
        bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_ylabel("% of total abnormal delay captured", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(abl_df, out_path):
    """Grouped bar chart for extended ablation."""
    metrics = [
        ("method_pct_total_abnormal", "% abnormal delay captured"),
        ("anomaly_rate_pct", "Anomaly rate (%)"),
    ]
    networks = abl_df.network.unique()
    dets = list(abl_df.method.unique())
    x = np.arange(len(dets))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (col, ylabel) in zip(axes, metrics):
        for i, (net, col_) in enumerate(zip(networks, ["#2166ac", "#d01c8b"])):
            sub = abl_df[abl_df.network == net].set_index("method")
            vals = [float(sub.loc[d, col]) if d in sub.index else 0 for d in dets]
            ax.bar(x + (i - 0.5) * w, vals, w, label=net, color=col_, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(dets, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        if ax == axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Ablation study — detector and component contributions", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(sens_df, out_path):
    """Line plots showing sensitivity of abnormal delay capture."""
    networks = sens_df.network.unique()
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X"]
    for i, net in enumerate(networks):
        sub = sens_df[sens_df.network == net]
        x = range(len(sub))
        y = sub.method_pct_total_abnormal.values
        ax.plot(x, y, marker=markers[i % len(markers)], label=net, lw=1.5, markersize=7)
    # Use the first network's variants for x-axis labels (all networks share same variants)
    sub_first = sens_df[sens_df.network == list(networks)[0]]
    ax.set_xticks(range(len(sub_first)))
    ax.set_xticklabels(sub_first.variant.values, rotation=35, ha="right", fontsize=8)
    ax.axvline(0, color="grey", lw=1, ls="--", alpha=0.5)  # baseline marker
    ax.set_ylabel("% abnormal delay captured", fontsize=9)
    ax.set_title("Sensitivity to ±20% parameter perturbation", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_unlabeled_heatmap(pivot, network, peak_share, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    dow_lbl = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [dow_lbl[i] for i in pivot.index if i < len(dow_lbl)]
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
        f"({peak_share:.1f}% in AM/PM peak 06–09, 15–19)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────


def write_summary(
    calib_net,
    transfer_net,
    calib_delay_rows,
    transfer_delay_rows,
    abl_df,
    sens_df,
    calib_best,
    calib_grid_df,
    timing_rows,
    unlab_df,
):

    def pick(rows, method, col):
        sub = [r for r in rows if r.get("method") == method]
        return f"{sub[0][col]:.1f}%" if sub else "—"

    calib_r = {r["method"]: r for r in calib_delay_rows}
    transfer_r = {r["method"]: r for r in transfer_delay_rows}

    # Delay table
    def delay_tbl(rows_c, rows_t):
        methods = ["Incident reports only", "Duan baseline", "Full ensemble"]
        hdr = "| Method | TSMO (calibration) | Cranberry (transfer) |\n|---|---|---|\n"
        rc = {r["method"]: r for r in rows_c}
        rt = {r["method"]: r for r in rows_t}
        body = ""
        for m in methods:
            c = rc.get(m, {}).get("method_pct_total_abnormal", "—")
            t = rt.get(m, {}).get("method_pct_total_abnormal", "—")
            body += (
                f"| {m} | {c:.1f}% | {t:.1f}% |\n"
                if isinstance(c, float)
                else f"| {m} | — | — |\n"
            )
        return hdr + body

    # Ablation table
    def abl_tbl(df):
        cols = ["method", "network", "method_pct_total_abnormal", "anomaly_rate_pct"]
        sub = df[cols].rename(
            columns={
                "method": "Detector configuration",
                "network": "Network",
                "method_pct_total_abnormal": "% abnormal delay",
                "anomaly_rate_pct": "Anomaly rate (%)",
            }
        )
        return sub.to_markdown(index=False)

    # Sensitivity table
    def sens_tbl(df):
        cols = ["variant", "network", "method_pct_total_abnormal"]
        sub = df[cols].rename(
            columns={
                "variant": "Parameter variant",
                "network": "Network",
                "method_pct_total_abnormal": "% abnormal delay",
            }
        )
        return sub.to_markdown(index=False)

    # Timing table
    def timing_tbl(rows):
        hdr = (
            "| Network | Method | Episodes | Quality | Latency (med) |\n"
            "|---|---|---|---|---|\n"
        )
        body = ""
        for r in rows:
            body += (
                f"| {r.get('network', '')} | {r.get('method', '')} "
                f"| {r.get('n_episodes', '')}"
                f"| {r.get('pct_with_speed_anomaly', ''):.1f}%"
                f"| {r.get('latency_median', 0):+.0f} min |\n"
                if isinstance(r.get("latency_median"), (int, float))
                else ""
            )
        return hdr + body

    snd_c, grad_c, conf_f, min_dur = calib_best

    report = textwrap.dedent(f"""\
    # FINAL RESULTS (Revised): Ensemble Nonrecurrent Traffic Disturbance Labeling

    **Calibration network:** TSMO (Howard County, MD) — disturbance-rich environment
    **Transfer network:** Cranberry (Pittsburgh, PA) — generalization validation
    **Final calibrated parameters:** snd_c={snd_c}, grad_c={grad_c}, conf_f={conf_f}, min_dur={min_dur} min
    **Temporal split:** first 25% calibration / remaining 75% held-out evaluation

    ---

    ## 1. Dataset summary

    | Network | Links | Period | Sessions (total) | Sessions (calib/eval) |
    |---|---|---|---|---|
    | TSMO (calibration) | 228 | Feb 2022–Feb 2023 | 260 | 65 / 195 |
    | Cranberry (transfer) | 78 | Feb 2022–Jan 2024 | 522 | 130 / 392 |

    ---

    ## 2. Abnormal delay capture — held-out evaluation

    {delay_tbl(calib_delay_rows, transfer_delay_rows)}

    ---

    ## 3. Detector-only abnormal delay

    | Network | Role | Detector-only abn. delay |
    |---|---|---|
    | TSMO | Calibration | {calib_r.get("Full ensemble", {}).get("detector_only_pct_abnormal", "—"):.1f}% |
    | Cranberry | Transfer | {transfer_r.get("Full ensemble", {}).get("detector_only_pct_abnormal", "—"):.1f}% |

    ---

    ## 4. Ablation study (extended)

    {abl_tbl(abl_df)}

    ---

    ## 5. Sensitivity analysis (±20% parameter perturbation)

    {sens_tbl(sens_df)}

    ---

    ## 6. Detection timing

    {timing_tbl(timing_rows)}

    ---

    ## 7. Unlabeled abnormal delay — peak hour analysis

    {pd.DataFrame(unlab_df)[["network", "total_unlabeled_abn_lh", "peak_share_pct"]].to_markdown(index=False)}

    ---

    ## 8. Figures

    | File | Description |
    |---|---|
    | `figures/abnormal_delay_capture.png` | Main result: abnormal delay by method and network |
    | `figures/ablation_extended.png` | Extended ablation: detector and component contributions |
    | `figures/sensitivity_analysis.png` | Sensitivity to ±20% parameter perturbation |
    | `figures/unlabeled_delay_heatmap_tsmo.png` | DOW×hour heatmap of unlabeled delay (TSMO) |
    | `figures/unlabeled_delay_heatmap_cranberry.png` | DOW×hour heatmap (Cranberry) |
    | `figures/timing_tsmo.png` | Timing analysis (TSMO) |
    | `figures/timing_cranberry.png` | Timing analysis (Cranberry) |
    """)

    path = _ROOT / "FINAL_RESULTS.md"
    path.write_text(report)
    print(f"  FINAL_RESULTS.md written")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    banner("Revised Experimental Pipeline: TSMO → Cranberry calibration transfer")

    # ── Load both networks ────────────────────────────────────────────────────
    banner("Loading networks")
    tsmo = NetworkBundle("tsmo")
    cran = NetworkBundle("cranberry")

    # ── Grid search on TSMO (calibration network) ─────────────────────────────
    best_params, grid_df = run_grid_search(tsmo)
    snd_c, grad_c, conf_f, min_dur = best_params
    grid_df.to_csv(TABS / "grid_search_tsmo.csv", index=False)

    # ── Final runs with best parameters ──────────────────────────────────────
    banner("Final evaluation with calibrated parameters")
    # Duan baselines
    tsmo_duan_lbl = duan_baseline(
        tsmo.eval_inc, tsmo.slow_eval, tsmo.p95_slow, tsmo.sids_e, H_MIN_LENGTH
    )
    cran_duan_lbl = duan_baseline(
        cran.eval_inc, cran.slow_eval, cran.p95_slow, cran.sids_e, H_MIN_LENGTH
    )

    tsmo_ens_lbl = tsmo.run(snd_c, grad_c, conf_f, min_dur)
    cran_ens_lbl = cran.run(snd_c, grad_c, conf_f, min_dur)

    # Main method rows
    methods = [
        ("Incident reports only", tsmo.eval_inc.copy(), cran.eval_inc.copy()),
        ("Duan baseline", tsmo_duan_lbl, cran_duan_lbl),
        ("Full ensemble", tsmo_ens_lbl, cran_ens_lbl),
    ]
    tsmo_delay_rows, cran_delay_rows = [], []
    for name, t_lbl, c_lbl in methods:
        tr = tsmo.metrics(t_lbl, name)
        tsmo_delay_rows.append(tr)
        cr = cran.metrics(c_lbl, name)
        cran_delay_rows.append(cr)
        print(
            f"  {name:<25s}  TSMO={tr['method_pct_total_abnormal']:.1f}%  "
            f"Cran={cr['method_pct_total_abnormal']:.1f}%"
        )

    all_delay_df = pd.DataFrame(tsmo_delay_rows + cran_delay_rows)
    all_delay_df.to_csv(TABS / "heldout_delay_results.csv", index=False)

    # ── Extended ablation ─────────────────────────────────────────────────────
    banner("Extended ablation")
    abl_rows = []
    for name, u_snd, u_grad, u_slow, skip_p, skip_r in ABLATIONS:
        for bundle in [tsmo, cran]:
            lbl = bundle.run(
                snd_c,
                grad_c,
                conf_f,
                min_dur,
                use_snd=u_snd,
                use_gradient=u_grad,
                use_slowdown=u_slow,
                skip_persistence=skip_p,
                skip_recovery=skip_r,
            )
            m = bundle.metrics(lbl, name)
            eps = extract_episodes(lbl)
            m["n_episodes"] = len(eps)
            abl_rows.append(m)
            print(
                f"  {bundle.network:<12s} {name:<35s} "
                f"abn={m['method_pct_total_abnormal']:5.1f}%  "
                f"rate={m['anomaly_rate_pct']:.2f}%"
            )
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(TABS / "ablation_results.csv", index=False)

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    banner("Sensitivity analysis (±20%)")
    variants = sensitivity_variants(snd_c, grad_c, conf_f, min_dur)
    sens_rows = []
    for tag, s_snd, s_grad, s_conf, s_dur in variants:
        # Clamp conf_f to sensible range
        s_conf = round(min(0.85, max(0.55, s_conf)), 2)
        for bundle in [tsmo, cran]:
            lbl = bundle.run(s_snd, s_grad, s_conf, s_dur)
            m = bundle.metrics(lbl, tag)
            m["variant"] = tag
            sens_rows.append(m)
            print(
                f"  {bundle.network:<12s} {tag:<18s} "
                f"abn={m['method_pct_total_abnormal']:5.1f}%  "
                f"dur={extract_episodes(lbl).duration_min.median():.0f}min"
            )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(TABS / "sensitivity_results.csv", index=False)

    # ── Timing analysis ───────────────────────────────────────────────────────
    banner("Timing analysis")
    timing_rows = []
    for bundle, duan_lbl, ens_lbl in [
        (tsmo, tsmo_duan_lbl, tsmo_ens_lbl),
        (cran, cran_duan_lbl, cran_ens_lbl),
    ]:
        for name, lbl in [
            ("Incident reports only", bundle.eval_inc),
            ("Duan baseline", duan_lbl),
            ("Full ensemble", ens_lbl),
        ]:
            eps = extract_episodes(lbl)
            tdf = compute_timing(eps, bundle.eval_speed, bundle.vrec_e, bundle.sids_e)
            summ = summarise_timing(tdf, name, bundle.network)
            timing_rows.append(summ)
            print(
                f"  {bundle.network:<12s} {name:<25s}  "
                f"lat={summ['latency_median']:+5.0f}min  "
                f"qual={summ['pct_with_speed_anomaly']:.1f}%"
            )
    pd.DataFrame(timing_rows).to_csv(TABS / "timing_results.csv", index=False)

    # ── Unlabeled abnormal delay ───────────────────────────────────────────────
    banner("Unlabeled abnormal delay analysis")
    unlab_list = []
    for bundle, ens_lbl, fname in [
        (tsmo, tsmo_ens_lbl, "unlabeled_delay_heatmap_tsmo.png"),
        (cran, cran_ens_lbl, "unlabeled_delay_heatmap_cranberry.png"),
    ]:
        stats, pivot = unlabeled_abnormal_summary(
            bundle.eval_speed, ens_lbl, bundle.vrec_e, bundle.excess_e
        )
        stats["network"] = bundle.network
        unlab_list.append(stats)
        fig_unlabeled_heatmap(
            pivot, bundle.network, stats["peak_share_pct"], FIGS / fname
        )
        print(f"  {bundle.network}: peak share = {stats['peak_share_pct']:.1f}%")
    pd.DataFrame(unlab_list).to_csv(TABS / "unlabeled_abnormal_delay.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    banner("Generating figures")
    fig_delay_bars(
        tsmo_delay_rows, cran_delay_rows, FIGS / "abnormal_delay_capture.png"
    )
    fig_ablation(abl_df, FIGS / "ablation_extended.png")
    fig_sensitivity(sens_df, FIGS / "sensitivity_analysis.png")

    # Timing figures — simple horizontal bar of median latency per method/network
    METHOD_COLORS = {
        "Incident reports only": "#2c7bb6",
        "Duan baseline": "#fdae61",
        "Full ensemble": "#d7191c",
    }
    for bundle_net in ["tsmo", "cranberry"]:
        t_rows = [r for r in timing_rows if r.get("network", "").lower() == bundle_net]
        fig, ax = plt.subplots(figsize=(8, 4))
        for r in t_rows:
            ax.barh(
                r["method"],
                r.get("latency_median", 0) or 0,
                color=METHOD_COLORS.get(r["method"], "grey"),
                alpha=0.8,
            )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(
            "Median detection latency (min)  [– = early, + = late]", fontsize=9
        )
        ax.set_title(f"Detection latency — {bundle_net.upper()}", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / f"timing_{bundle_net}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("  all figures saved")

    # ── Summary report ────────────────────────────────────────────────────────
    banner("Writing FINAL_RESULTS.md")
    write_summary(
        tsmo.network,
        cran.network,
        tsmo_delay_rows,
        cran_delay_rows,
        abl_df,
        sens_df,
        best_params,
        grid_df,
        timing_rows,
        unlab_list,
    )

    banner(f"Done in {(time.time() - t0) / 60:.1f} min")
    print(f"  tables → {TABS}")
    print(f"  figures → {FIGS}")


if __name__ == "__main__":
    main()
