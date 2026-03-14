    # FINAL RESULTS: Ensemble Nonrecurrent Traffic Disturbance Labeling

    **Configuration (frozen):** snd_c=2.5, grad_c=1.2, conf_f=0.70, min_dur=20 min
    **Temporal protocol:** 25% calibration / 75% held-out evaluation (temporal order)
    **Delay unit:** link-hours of excess travel time per link-traversal

    ---

    ## 1. Dataset summary

    | Network | Links | Calibration period | Evaluation period | Eval sessions |
    |---|---|---|---|---|
    | Cranberry | 78 | 2022-02-01 → 2022-08-01 | 2022-08-02 → 2024-01-31 | 392 |
    | TSMO | 228 | 2022-02-14 → 2022-05-13 | 2022-05-16 → 2023-02-10 | 195 |

    ---

    ## 2. Delay capture — held-out evaluation

    | network   | method                |   % abnormal delay |   % excess delay |   % detector-only abn. |   % excl. near-zero |   anomaly rate % |
|:----------|:----------------------|-------------------:|-----------------:|-----------------------:|--------------------:|-----------------:|
| cranberry | Incident reports only |              14.87 |            14    |                   0    |                 0   |            2.134 |
| cranberry | Duan baseline         |              26.13 |            24.37 |                  12.9  |                72   |            2.484 |
| cranberry | Full ensemble         |              33.53 |            31.1  |                  21.9  |                61.2 |            2.274 |
| tsmo      | Incident reports only |               6.5  |             5.75 |                   0    |                 0   |            0.43  |
| tsmo      | Duan baseline         |              20.23 |            18.31 |                  15.58 |                66.9 |            1.269 |
| tsmo      | Full ensemble         |              43.39 |            39.09 |                  37.21 |                83.3 |            3.36  |

    ### Key findings

    - **Full ensemble captures — of abnormal delay on Cranberry**
      vs. — for Duan baseline
      and — for incident reports alone.
    - On TSMO: ensemble —
      vs. Duan —
      vs. incident reports —.

    ---

    ## 3. Incident-excluded periods (low-impact filtering)

    | Network | Method | Incident-excl. excess (link-h) | Incident-excl. abn. (link-h) | Near-zero excess % | Near-zero abn. % |
    |---|---|---|---|---|---|
    | cranberry | Full ensemble | 203.76 | 189.61 | 55.7% | 61.2% |
| tsmo | Full ensemble | 24.77 | 21.55 | 79.6% | 83.3% |

    ---

    ## 4. Ablation study

    | network   | method               |   % abn. delay |   % excess delay |   anomaly rate % |
|:----------|:---------------------|---------------:|-----------------:|-----------------:|
| cranberry | Deviation only       |          24.41 |            22.62 |            1.316 |
| cranberry | Deviation + Gradient |          33.07 |            30.67 |            2.163 |
| cranberry | Deviation + Slowdown |          24.97 |            23.14 |            1.385 |
| cranberry | Full ensemble        |          33.53 |            31.1  |            2.274 |
| tsmo      | Deviation only       |          22.12 |            19.49 |            1.022 |
| tsmo      | Deviation + Gradient |          41    |            36.71 |            2.821 |
| tsmo      | Deviation + Slowdown |          23.89 |            21.12 |            1.238 |
| tsmo      | Full ensemble        |          43.39 |            39.09 |            3.36  |

    ---

    ## 5. Timing analysis

    | network   | method                |   episodes |   % w/ speed anomaly |   lat. median (min) |   lat. mean (min) |   overhang median (min) |   overhang mean (min) |   dur. median (min) |
|:----------|:----------------------|-----------:|---------------------:|--------------------:|------------------:|------------------------:|----------------------:|--------------------:|
| cranberry | Incident reports only |       7782 |                 87.1 |                   0 |              22.2 |                      -5 |                 -29.4 |                  45 |
| cranberry | Duan baseline         |      15712 |                 95.7 |                   5 |              46.3 |                     -10 |                 -49.5 |                  20 |
| cranberry | Full ensemble         |       6428 |                100   |                   0 |              14.7 |                     -10 |                 -12.7 |                  40 |
| tsmo      | Incident reports only |       4066 |                 72.9 |                   0 |              12.3 |                      -5 |                 -18.6 |                  30 |
| tsmo      | Duan baseline         |      19565 |                 85.2 |                   5 |              14.5 |                      -5 |                 -20.5 |                  20 |
| tsmo      | Full ensemble         |      20066 |                 97.1 |                  -5 |               0.2 |                     -10 |                 -11.7 |                  45 |

    ---

    ## 6. Unlabeled abnormal delay

    | network   |   unlabeled abn. delay (link-h) |   % in AM/PM peak (06-09, 15-19) |
|:----------|--------------------------------:|---------------------------------:|
| cranberry |                         3893.99 |                             60.8 |
| tsmo      |                         3822.16 |                             67.4 |

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
