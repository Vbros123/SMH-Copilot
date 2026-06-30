"""
diagnostics.py
==============
Quantitative validation and diagnostic utilities for the Round 20 SEIRS
calibration pipeline.

Design constraints
------------------
- No VE equations live here.  All VE math is in scenario_adjustments.py.
- No epiweek arithmetic.  All date/epiweek handling delegates to simulate.py
  via get_calibration_fit(), so diagnostics always match the simulator.
- No simulation logic.  This module only analyses and visualises output.

Public API
----------
compute_validation_metrics(params, obs_hosp, cal_dates)
    Returns dict: rmse_log, mae, r2, continuity_error, mean_relative_error.

verify_properties(scenario_df, obs_df=None)
    Prints PASS/WARN for scientific checks.  Never raises exceptions.

plot_calibration(params, cal_dates, obs_hosp, forecast_df=None, output_path=None)
    Saves or shows a single diagnostic figure: observed + fit + forecast.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper: calibration fit via simulate (no epiweek duplication)
# ---------------------------------------------------------------------------

def _get_fit_array(params, cal_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Return the deterministic SEIRS calibration fit as a float array.

    Delegates entirely to simulate.get_calibration_fit() so that all
    epiweek conversion is done in exactly one place (simulate.py), matching
    the simulator.  No ISO-week arithmetic is duplicated here.
    """
    from simulate import get_calibration_fit  # noqa: PLC0415
    return get_calibration_fit(params, cal_dates).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Quantitative calibration metrics
# ---------------------------------------------------------------------------

def compute_validation_metrics(
    params,
    obs_hosp: np.ndarray,
    cal_dates: pd.DatetimeIndex,
) -> Dict[str, float]:
    """
    Compute calibration fit quality metrics against observed hospitalisations.

    Parameters
    ----------
    params    : Calibrated SEIRSParams from simulate.get_calibrated_params().
    obs_hosp  : Observed weekly inc hosp array, aligned to cal_dates.
    cal_dates : DatetimeIndex of calibration epi-weeks.

    Returns
    -------
    dict
        rmse_log             – RMSE of log(sim) − log(obs) over calibration window
        mae                  – Mean absolute error (count space)
        r2                   – Coefficient of determination (count space)
        last_week_residual   – |sim_last − obs_last| / obs_last
                               Measures how well the calibration fit matches the
                               last observed week (calibration quality at origin).
        mean_relative_error  – mean |sim − obs| / obs
    """
    sim = _get_fit_array(params, cal_dates)
    valid = ~np.isnan(obs_hosp)
    obs_v = obs_hosp[valid].astype(float)
    sim_v = sim[valid].astype(float)

    # RMSE in log-space (proportional errors treated uniformly)
    rmse_log = float(np.sqrt(np.mean(
        (np.log(np.maximum(sim_v, 1.0)) - np.log(np.maximum(obs_v, 1.0))) ** 2
    )))

    # MAE in count space
    mae = float(np.mean(np.abs(sim_v - obs_v)))

    # R² in count space
    ss_res = float(np.sum((obs_v - sim_v) ** 2))
    ss_tot = float(np.sum((obs_v - np.mean(obs_v)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Last-week residual: |sim_last - obs_last| / obs_last
    # This measures the calibration fit quality at the last calibration week,
    # NOT the calibration→forecast transition (which is always ~1–2% because
    # the forecast continues deterministically from the warm-start state).
    last_valid = valid.nonzero()[0][-1] if valid.any() else None
    if last_valid is not None and float(obs_hosp[last_valid]) > 0:
        last_week_residual = (
            abs(float(sim[last_valid]) - float(obs_hosp[last_valid]))
            / float(obs_hosp[last_valid])
        )
    else:
        last_week_residual = float("nan")

    # Mean relative error across calibration window
    mean_rel_err = float(
        np.mean(np.abs(sim_v - obs_v) / np.maximum(obs_v, 1.0))
    )

    metrics = {
        "rmse_log": rmse_log,
        "mae": mae,
        "r2": r2,
        "last_week_residual": last_week_residual,
        "mean_relative_error": mean_rel_err,
    }
    logger.info(
        "compute_validation_metrics: RMSE_log=%.4f  MAE=%.1f  R²=%.4f  "
        "last_week_residual=%.4f  mean_rel_err=%.4f",
        rmse_log, mae, r2, last_week_residual, mean_rel_err,
    )
    return metrics


# ---------------------------------------------------------------------------
# Scenario property verification
# ---------------------------------------------------------------------------

def verify_properties(
    scenario_df: pd.DataFrame,
    obs_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Check scientific properties of scenario trajectories.

    Prints measured values with PASS / WARN labels.
    Never raises exceptions — designed for informational use only.

    Parameters
    ----------
    scenario_df : Output of scenario_adjustments.apply_all_scenarios().
                  Required columns: scenario_id, date, forecast.
    obs_df      : Optional observed data (columns: date, observation) for
                  Scenario A vs. observed comparison.

    Returns
    -------
    dict of measured values, one key per check.
    """
    results: Dict[str, object] = {}

    try:
        from scipy.signal import find_peaks as _sp_peaks  # noqa: PLC0415
        def _count_peaks(arr: np.ndarray) -> int:
            return int(len(_sp_peaks(arr, distance=8)[0]))
    except ImportError:
        def _count_peaks(arr: np.ndarray) -> int:  # type: ignore[misc]
            return int(np.sum(
                (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])
            ))

    medians = (
        scenario_df
        .groupby(["scenario_id", "date"])["forecast"]
        .median()
        .reset_index()
    )
    sorted_dates = sorted(scenario_df["date"].unique())
    sc_ids = sorted(scenario_df["scenario_id"].unique())
    sc_a = next((s for s in sc_ids if s.startswith("A-")), None)
    sc_b = next((s for s in sc_ids if s.startswith("B-")), None)
    sc_d = next((s for s in sc_ids if s.startswith("D-")), None)

    # ── Scenario ordering at week 26 ─────────────────────────────────────
    if len(sorted_dates) >= 26:
        wk26 = sorted_dates[25]
        m26 = (
            medians[medians["date"] == wk26]
            .set_index("scenario_id")["forecast"]
        )
        vals = {s: float(m26.get(s, float("nan"))) for s in sc_ids}
        results["medians_week_26"] = vals
        a_ge_b = vals.get(sc_a, float("nan")) >= vals.get(sc_b, float("nan"))
        b_ge_d = vals.get(sc_b, float("nan")) >= vals.get(sc_d, float("nan"))
        ok = a_ge_b and b_ge_d
        results["ordering_A_ge_B_ge_D_week26"] = ok
        print(
            f"[{'PASS' if ok else 'WARN'}] Scenario ordering (week 26): "
            + "  ".join(f"{k}={v:.0f}" for k, v in vals.items())
        )

    # ── Seasonal peaks in Scenario A ─────────────────────────────────────
    if sc_a:
        a_ts = (
            medians[medians["scenario_id"] == sc_a]
            .sort_values("date")["forecast"]
            .to_numpy(float)
        )
        n_peaks = _count_peaks(a_ts)
        results["n_peaks_scenario_A"] = n_peaks
        print(
            f"[{'PASS' if n_peaks >= 2 else 'WARN'}] "
            f"Peaks in Scenario A: {n_peaks} (≥2 expected)"
        )

    # ── State differentiation ─────────────────────────────────────────────
    if "location" in scenario_df.columns and sc_a:
        loc_med = (
            scenario_df[scenario_df["scenario_id"] == sc_a]
            .groupby("location")["forecast"]
            .median()
        )
        if len(loc_med) > 1:
            cv = (
                float(loc_med.std() / loc_med.mean())
                if loc_med.mean() > 0 else 0.0
            )
            results["state_cv"] = cv
            print(
                f"[{'PASS' if cv > 0.10 else 'WARN'}] "
                f"State differentiation CV: {cv:.3f}"
            )

    # ── Scenario A vs. observed at weeks 1–4 ─────────────────────────────
    if obs_df is not None and sc_a and len(sorted_dates) >= 4:
        obs_idx = obs_df.set_index("date")["observation"]
        for d in sorted_dates[:4]:
            if d not in obs_idx.index:
                continue
            obs_val = float(obs_idx[d])
            row = medians[
                (medians["scenario_id"] == sc_a) & (medians["date"] == d)
            ]
            a_val = float(row["forecast"].iloc[0]) if not row.empty else float("nan")
            ratio = a_val / obs_val if obs_val > 0 else float("nan")
            key = f"A_vs_obs_{pd.Timestamp(d).date()}"
            results[key] = ratio
            print(
                f"[INFO] Scenario A / observed at {pd.Timestamp(d).date()}: "
                f"{ratio:.3f}"
            )

    print("[INFO] verify_properties complete.")
    return results


# ---------------------------------------------------------------------------
# Calibration diagnostic plot
# ---------------------------------------------------------------------------

def plot_calibration(
    params,
    cal_dates: pd.DatetimeIndex,
    obs_hosp: np.ndarray,
    forecast_df: Optional[pd.DataFrame] = None,
    output_path: Optional["pathlib.Path | str"] = None,
) -> None:
    """
    Single diagnostic figure: observed data + calibration fit + forecast band.

    All epiweek alignment is delegated to _get_fit_array() → simulate.py.
    No ISO-week arithmetic in this function.

    Parameters
    ----------
    params      : Calibrated SEIRSParams.
    cal_dates   : DatetimeIndex of calibration epi-weeks.
    obs_hosp    : Observed inc hosp aligned to cal_dates.
    forecast_df : Optional trajectory DataFrame
                  (columns: trajectory_id, date, forecast).
    output_path : Save path (.png).  None → show interactively.
    """
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("plot_calibration: matplotlib unavailable; skipping.")
        return

    from load_data import ORIGIN_DATE  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(13, 5))

    # Observed data
    ax.scatter(
        cal_dates, obs_hosp, s=16, c="black", zorder=5,
        label="Observed (inc hosp)",
    )

    # Calibration fit — delegated entirely to simulate.py
    sim = _get_fit_array(params, cal_dates)
    ax.plot(cal_dates, sim, c="steelblue", lw=1.8, label="Calibration fit")

    # Forecast trajectories (Scenario A baseline, if provided)
    if forecast_df is not None and not forecast_df.empty:
        grp = forecast_df.groupby("date")["forecast"]
        ax.fill_between(
            grp.quantile(0.10).index,
            grp.quantile(0.10).values,
            grp.quantile(0.90).values,
            alpha=0.22, color="tomato",
            label="Forecast 10–90th pct",
        )
        ax.plot(
            grp.median().index, grp.median().values,
            c="tomato", lw=1.4, label="Forecast median",
        )

    ax.axvline(
        ORIGIN_DATE, c="#555", ls="--", lw=1.0,
        label=f"Forecast origin ({ORIGIN_DATE.date()})",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly incident hospitalisations")
    ax.set_title("SEIRS Calibration Diagnostic — Observed vs. Fit vs. Forecast")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    if output_path is not None:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        logger.info("plot_calibration: saved to %s", out)
    else:
        plt.show()
    plt.close(fig)
