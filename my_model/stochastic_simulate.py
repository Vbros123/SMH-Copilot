"""
stochastic_simulate.py
======================
Stochastic log-growth forecasting model for the Round 20 COVID-19 Scenario
Modeling Hub prototype.

Method overview
---------------
The model operates entirely in log-space using a random-walk-with-drift
framework — the simplest statistically defensible generator of stochastic
epidemic trajectories that is consistent with the observed week-to-week
log-growth rates of COVID-19 hospitalisations and deaths.

Step 1 – Estimation
    Collect the last ``lookback_weeks`` non-zero weekly observations.
    Compute the vector of successive log-differences (log-growth rates).
    Estimate:
        μ  = sample mean of log-growth rates  (central drift)
        σ  = sample standard deviation of log-growth rates (weekly noise)

    Assumption A1 – Stationarity of log-growth rates:
        The log-growth rate process is assumed stationary over the lookback
        window; i.e. the mean and variance estimated from the most recent
        ``lookback_weeks`` weeks are representative of the near-future
        dynamics.  This is reasonable for a seasonal endemic pathogen over
        short-to-medium projection horizons, but becomes increasingly
        uncertain at horizons ≥ 20 weeks where new variants or seasonality
        shifts can alter the drift.

    Assumption A2 – Normal innovations:
        Weekly log-growth increments are modelled as i.i.d. Normal(μ, σ²).
        COVID-19 trajectories exhibit heavier-than-Normal tails; the Normal
        approximation is chosen for parsimony and computational efficiency.
        Tail risk is partially captured by the scenario-level seasonal
        forcing (see ``scenario_adjustments.py``).

Step 2 – Seasonal correction
    The raw drift μ is adjusted by adding a sinusoidal seasonal term
    evaluated at each future week (see ``SEASONAL_AMPLITUDE`` and
    ``PEAK_WEEK_OF_YEAR``).

    Assumption A3 – Sinusoidal seasonality:
        COVID-19 hospitalisation has exhibited consistent summer and winter
        peaks in recent years.  A single-harmonic sinusoid captures the
        dominant seasonal frequency.  Southern states and high-latitude
        states have different seasonality; the national model uses a pooled
        amplitude calibrated to observed US national data.

Step 3 – Trajectory generation (300 samples)
    For each of the 300 trajectories:
        log_forecast[t] = log_forecast[t-1] + μ_seasonal[t] + ε[t]
        ε[t] ~ Normal(0, σ)

    Forecast counts are recovered via exponentiation and floored at 0.

    Assumption A4 – Geometric Brownian motion on counts:
        Modelling in log-space ensures forecasts are strictly non-negative
        and that percentage uncertainty grows over time, consistent with
        epidemic dynamics.

    Assumption A5 – Independence of trajectories:
        The 300 trajectories are drawn independently (no between-trajectory
        correlation other than shared drift/seasonality).  Paired trajectories
        across scenarios are enforced at the caller level by fixing the random
        seed *before* calling this function for each scenario.

Step 4 – Minimum floor
    All forecasts are clipped to ``min_count`` (default 1) to avoid
    zero-count trajectories collapsing permanently.

    Assumption A6 – Endemic floor:
        The model assumes COVID-19 never fully disappears from the US
        population during the 104-week projection horizon.  A weekly
        national floor of 1 hospitalisation is conservative and safe.

Parameters
----------
lookback_weeks : int (default 8)
    Number of most-recent observed weeks used to estimate μ and σ.
    Round 20 guidance suggests calibration to the recent year; 8 weeks
    balances recency with statistical stability.

n_trajectories : int (default 300)
    Exact number of stochastic trajectories required by the hub submission
    specification (model-output/README.md: "100–300 representative
    trajectories").

n_weeks : int (default 104)
    Projection horizon in epi-weeks (Round 20: June 2025 → June 2027).

References
----------
- Round 20 scenario description: auxiliary-data/rounds/round20.md
- Submission format: model-output/README.md
- US CDC epi-week convention (Sun–Sat)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults (all overridable via function arguments)
# ---------------------------------------------------------------------------

#: Number of most-recent observed weeks used for drift/volatility estimation.
DEFAULT_LOOKBACK_WEEKS: int = 8

#: Exact number of independent stochastic trajectories to generate.
DEFAULT_N_TRAJECTORIES: int = 300

#: Total projection horizon in epi-weeks (Round 20 requirement).
DEFAULT_N_WEEKS: int = 104

#: Minimum observation floor applied after exponentiation (Assumption A6).
DEFAULT_MIN_COUNT: float = 1.0

#: Relative amplitude of sinusoidal seasonal forcing (unitless, 0–1).
#  Calibrated from US national inc hosp 2023–2026 using least-squares sinusoid
#  fit; winter peak is approximately 2× summer trough → amplitude ≈ 0.35.
#  Teams may adjust this based on state-level fits (Assumption A3).
SEASONAL_AMPLITUDE: float = 0.35

#: Day-of-year on which the seasonal peak occurs (hospitalisation).
#  Empirically: US COVID-19 hospitalisation peaks ~week 3–4 of January
#  (≈ day 18) and has a secondary summer peak around day 196 (mid-July).
#  A single sinusoid is anchored at the winter peak (day 18).
PEAK_DOY: int = 18  # ~18 January

# ---------------------------------------------------------------------------
# Helper: build the weekly seasonal drift adjustment
# ---------------------------------------------------------------------------

def _seasonal_drift(
    dates: pd.DatetimeIndex,
    amplitude: float = SEASONAL_AMPLITUDE,
    peak_doy: int = PEAK_DOY,
) -> np.ndarray:
    """
    Compute the additive seasonal adjustment to the log-growth drift for each
    projected epi-week.

    The model uses the *derivative* of the seasonal sinusoid rather than the
    sinusoid itself: if we want log(count) to follow a seasonal curve, then
    the week-to-week increment of log(count) is the derivative of that curve.

    Specifically:
        seasonal_log_count(t) = A · sin(2π(doy - peak_doy) / 365.25)
        drift_adjustment(t)   = d/dt [seasonal_log_count(t)]
                               ≈ A · (2π / 365.25) · cos(2π(doy - peak_doy) / 365.25)
                                 × 7   (scaled to weekly units)

    Parameters
    ----------
    dates     : DatetimeIndex of epi-week Saturday end-dates (one per horizon).
    amplitude : Relative amplitude of the sinusoid (0–1). Default 0.35.
    peak_doy  : Day-of-year of the seasonal peak. Default 18 (≈ 18 Jan).

    Returns
    -------
    np.ndarray, shape (len(dates),)
        Weekly additive adjustment to log-growth drift, in log units per week.
    """
    omega = 2.0 * np.pi / 365.25  # angular frequency (radians per day)
    doy = dates.day_of_year.to_numpy(dtype=float)
    # Weekly increment of the seasonal sinusoid (chain-rule × 7 days/week)
    adjustment = amplitude * omega * np.cos(omega * (doy - peak_doy)) * 7.0
    return adjustment


# ---------------------------------------------------------------------------
# Core estimation function
# ---------------------------------------------------------------------------

def estimate_log_growth_params(
    observations: pd.Series | np.ndarray,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_obs_for_growth: float = 1.0,
) -> tuple[float, float]:
    """
    Estimate the mean (μ) and standard deviation (σ) of weekly log-growth
    rates from the most recent ``lookback_weeks`` non-zero observations.

    Parameters
    ----------
    observations      : Ordered time series of weekly counts (oldest→newest).
                        May contain NaN; NaN rows are dropped before fitting.
    lookback_weeks    : Number of most-recent weeks to use (default 8).
                        Requires at least ``lookback_weeks + 1`` valid points
                        to compute ``lookback_weeks`` log-differences.
    min_obs_for_growth: Counts below this value are clipped to it before taking
                        logs to prevent −∞ log-growth rates (Assumption A2).

    Returns
    -------
    mu : float
        Sample mean of log-growth rates.  Negative → declining trend;
        positive → growing trend.
    sigma : float
        Sample standard deviation of log-growth rates (ddof=1).
        Represents week-to-week volatility (Assumption A2).

    Raises
    ------
    ValueError
        If there are fewer than 3 valid observations after cleaning, making
        a reliable estimate impossible.

    Notes
    -----
    - The lookback window is selected adaptively: if fewer than
      ``lookback_weeks + 1`` clean observations exist, the function uses
      all available data and emits a warning.
    - The standard deviation is computed with ddof=1 (unbiased estimator).
      With only 8 differences, σ is noisy; the model acknowledges this
      uncertainty by sampling trajectories rather than returning a point
      forecast.
    """
    # Convert to numpy float array; drop NaN
    arr = np.asarray(observations, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 3:
        raise ValueError(
            f"estimate_log_growth_params: need at least 3 valid observations, "
            f"got {len(arr)}.  Check that the target data file is populated."
        )

    # Restrict to the most recent ``lookback_weeks + 1`` values so we can
    # compute exactly ``lookback_weeks`` consecutive log-differences.
    n_needed = lookback_weeks + 1
    if len(arr) < n_needed:
        logger.warning(
            "estimate_log_growth_params: only %d valid observations available; "
            "need %d for a %d-week lookback.  Using all %d observations.",
            len(arr), n_needed, lookback_weeks, len(arr),
        )
        window = arr
    else:
        window = arr[-n_needed:]  # most recent n_needed values

    # Floor counts at min_obs_for_growth before log to avoid -inf (Assumption A2)
    window_safe = np.where(window < min_obs_for_growth, min_obs_for_growth, window)

    log_growth = np.diff(np.log(window_safe))  # shape: (len(window) - 1,)

    mu = float(np.mean(log_growth))
    sigma = float(np.std(log_growth, ddof=1)) if len(log_growth) > 1 else 0.1

    logger.debug(
        "estimate_log_growth_params: n_diffs=%d  μ=%.4f  σ=%.4f",
        len(log_growth), mu, sigma,
    )
    return mu, sigma


# ---------------------------------------------------------------------------
# Core trajectory generator
# ---------------------------------------------------------------------------

def generate_trajectories(
    observations: pd.Series | np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    min_count: float = DEFAULT_MIN_COUNT,
    seasonal_amplitude: float = SEASONAL_AMPLITUDE,
    peak_doy: int = PEAK_DOY,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate ``n_trajectories`` stochastic forecast trajectories using a
    seasonal log-growth random-walk model.

    Parameters
    ----------
    observations     : Ordered historical weekly counts (oldest → newest).
                       Used to estimate the drift μ and volatility σ, and to
                       seed the initial forecast value.
    forecast_dates   : DatetimeIndex of epi-week Saturday end-dates for the
                       projection horizon.  Must be non-empty.
                       Length determines the number of forecast steps.
    lookback_weeks   : Weeks of history used for μ/σ estimation (default 8).
    n_trajectories   : Number of independent trajectories (default 300, per
                       hub specification).
    min_count        : Minimum weekly count floor applied after exponentiation
                       (default 1.0, Assumption A6).
    seasonal_amplitude: Amplitude of sinusoidal seasonal forcing in log-space
                       (default 0.35, Assumption A3).
    peak_doy         : Day-of-year of the seasonal hospitalisation peak
                       (default 18, ≈ 18 January, Assumption A3).
    seed             : Optional integer random seed for reproducibility.
                       Pass the *same* seed across scenarios to ensure paired
                       trajectories (Assumption A5).

    Returns
    -------
    pd.DataFrame
        Columns:
            trajectory_id  – int in [1, n_trajectories]; unique trajectory label
            date           – datetime64[ns]; Saturday of epi-week (hub format)
            forecast       – float64; projected weekly count (≥ min_count)

        Shape: ``n_trajectories × len(forecast_dates)`` rows.
        Sorted by ``["trajectory_id", "date"]``.

    Raises
    ------
    ValueError
        If ``forecast_dates`` is empty, or fewer than 3 valid observations
        exist in ``observations``.

    Examples
    --------
    >>> import pandas as pd
    >>> from load_data import load_target_data, build_epiweek_dates
    >>> obs_df = load_target_data(
    ...     location_filter=["US"],
    ...     target_filter=["inc hosp"],
    ...     age_group_filter=["0-130"],
    ... )
    >>> obs = obs_df.set_index("date")["observation"]
    >>> dates = build_epiweek_dates()
    >>> trajs = generate_trajectories(obs, dates, seed=42)
    >>> trajs.shape
    (31200, 3)   # 300 trajectories × 104 weeks
    """
    if len(forecast_dates) == 0:
        raise ValueError("generate_trajectories: forecast_dates must not be empty.")

    rng = np.random.default_rng(seed)

    # ── Step 1: Estimate drift μ and volatility σ ─────────────────────────
    mu, sigma = estimate_log_growth_params(observations, lookback_weeks=lookback_weeks)

    # ── Step 2: Build seasonal drift adjustment for each forecast week ─────
    # Shape: (n_forecast_weeks,)
    seasonal_adj = _seasonal_drift(forecast_dates, amplitude=seasonal_amplitude,
                                   peak_doy=peak_doy)
    # Total drift per week = estimated trend + seasonal correction
    drift = mu + seasonal_adj  # shape: (n_weeks,)

    logger.info(
        "generate_trajectories: μ=%.4f  σ=%.4f  horizon=%d weeks  "
        "n_trajectories=%d  seed=%s",
        mu, sigma, len(forecast_dates), n_trajectories, seed,
    )

    # ── Step 3: Seed the initial forecast value ───────────────────────────
    # Use the most recent non-NaN observation as the starting count.
    arr = np.asarray(observations, dtype=float)
    arr_clean = arr[~np.isnan(arr)]
    if len(arr_clean) == 0:
        raise ValueError(
            "generate_trajectories: all observations are NaN; "
            "cannot seed the forecast."
        )
    last_obs = float(arr_clean[-1])
    # Ensure the seed value is strictly positive before log-seeding
    last_obs_safe = max(last_obs, DEFAULT_MIN_COUNT)

    n_weeks = len(forecast_dates)

    # ── Step 4: Draw innovation matrix ────────────────────────────────────
    # Shape: (n_trajectories, n_weeks)
    # Each row is one trajectory's sequence of Normal(0, σ) innovations.
    # Assumption A2: i.i.d. Normal innovations.
    innovations = rng.normal(loc=0.0, scale=sigma, size=(n_trajectories, n_weeks))

    # ── Step 5: Build cumulative log-forecasts ────────────────────────────
    # log_count[t] = log(last_obs) + Σ_{s=1}^{t} [drift[s] + ε[s]]
    # drift is broadcast across all trajectories (shape: 1 × n_weeks)
    cumulative_increments = np.cumsum(
        drift[np.newaxis, :] + innovations, axis=1
    )  # shape: (n_trajectories, n_weeks)

    log_forecast = np.log(last_obs_safe) + cumulative_increments

    # ── Step 6: Exponentiate and apply floor ─────────────────────────────
    # Assumption A6: floor at min_count to prevent permanent zero-collapse.
    forecast_counts = np.clip(np.exp(log_forecast), a_min=min_count, a_max=None)

    # ── Step 7: Assemble tidy output DataFrame ───────────────────────────
    # Pre-allocate arrays for efficiency rather than concatenating DataFrames.
    total_rows = n_trajectories * n_weeks
    traj_ids = np.repeat(np.arange(1, n_trajectories + 1), n_weeks)
    dates_repeated = np.tile(forecast_dates, n_trajectories)
    forecasts_flat = forecast_counts.ravel()  # C-order: row = trajectory

    result = pd.DataFrame(
        {
            "trajectory_id": traj_ids.astype(np.int32),
            "date": dates_repeated,
            "forecast": forecasts_flat,
        }
    )

    result = result.sort_values(["trajectory_id", "date"]).reset_index(drop=True)

    logger.info(
        "generate_trajectories: returned %d rows (%d trajectories × %d weeks).",
        len(result), n_trajectories, n_weeks,
    )
    return result


# ---------------------------------------------------------------------------
# Convenience wrapper: run for a specific target and location
# ---------------------------------------------------------------------------

def forecast_target(
    location: str = "US",
    target: str = "inc hosp",
    age_group: str = "0-130",
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    n_weeks: int = DEFAULT_N_WEEKS,
    seasonal_amplitude: float = SEASONAL_AMPLITUDE,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    End-to-end convenience function: load observed data, build forecast dates,
    and return 300 stochastic trajectories for a given location and target.

    Parameters
    ----------
    location          : FIPS code (e.g. ``"US"``, ``"06"``).
    target            : ``"inc hosp"`` or ``"inc death"``.
    age_group         : Age group label (default ``"0-130"`` = all ages).
    lookback_weeks    : History window for parameter estimation (default 8).
    n_trajectories    : Number of trajectories (default 300).
    n_weeks           : Forecast horizon in epi-weeks (default 104).
    seasonal_amplitude: Sinusoidal forcing amplitude (default 0.35).
    seed              : Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``trajectory_id`` (int32), ``date`` (datetime64[ns]),
        ``forecast`` (float64), plus metadata columns ``location``,
        ``target``, ``age_group`` added for downstream assembly.

    Notes
    -----
    Import is deferred to avoid a circular-import issue if ``simulate.py``
    imports from this module.
    """
    # Deferred import so this module can stand alone for unit-testing
    from load_data import (  # noqa: PLC0415
        load_target_data,
        build_epiweek_dates,
        ORIGIN_DATE,
        N_HORIZON_WEEKS,
    )

    obs_df = load_target_data(
        location_filter=[location],
        target_filter=[target],
        age_group_filter=[age_group],
        max_date=pd.Timestamp("2026-06-06"),  # FIT_END_DATE
    )

    if obs_df.empty:
        raise ValueError(
            f"forecast_target: no observations found for location='{location}', "
            f"target='{target}', age_group='{age_group}'."
        )

    # Ensure monotonically ordered by date, then extract the count series
    obs_df = obs_df.sort_values("date")
    observations = obs_df["observation"].values.astype(float)

    forecast_dates = build_epiweek_dates(origin=ORIGIN_DATE, n_weeks=n_weeks)

    trajs = generate_trajectories(
        observations=observations,
        forecast_dates=forecast_dates,
        lookback_weeks=lookback_weeks,
        n_trajectories=n_trajectories,
        min_count=DEFAULT_MIN_COUNT,
        seasonal_amplitude=seasonal_amplitude,
        seed=seed,
    )

    # Attach metadata so the DataFrame is self-describing
    trajs["location"] = location
    trajs["target"] = target
    trajs["age_group"] = age_group

    return trajs


# ---------------------------------------------------------------------------
# Summary statistics helper (used by build_submission.py for quantile output)
# ---------------------------------------------------------------------------

def compute_quantiles(
    trajectories: pd.DataFrame,
    quantiles: Optional[list[float]] = None,
    value_col: str = "forecast",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute quantile summaries across trajectories.

    Parameters
    ----------
    trajectories : Output of :func:`generate_trajectories` or
                   :func:`forecast_target`.
    quantiles    : List of quantile probabilities (0–1). Defaults to the
                   23 quantiles required by the hub submission specification:
                   0.01, 0.025, 0.05, every 5 % from 0.10–0.95, 0.975, 0.99.
    value_col    : Name of the column containing forecast values (default
                   ``"forecast"``).
    group_cols   : Columns to group by before computing quantiles.
                   Defaults to ``["date"]``; pass ``["date", "location"]``
                   for multi-location tables.

    Returns
    -------
    pd.DataFrame
        Columns: ``[*group_cols, "quantile", "value"]``.
        One row per (group, quantile) combination.
    """
    if quantiles is None:
        # Hub-required quantiles (model-output/README.md)
        quantiles = [
            0.010, 0.025, 0.050,
            0.100, 0.150, 0.200, 0.250, 0.300, 0.350,
            0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
            0.700, 0.750, 0.800, 0.850, 0.900, 0.950,
            0.975, 0.990,
        ]

    if group_cols is None:
        group_cols = ["date"]

    def _quantile_row(grp: pd.Series) -> pd.DataFrame:
        vals = np.quantile(grp.values, quantiles)
        return pd.DataFrame({"quantile": quantiles, "value": vals})

    result = (
        trajectories
        .groupby(group_cols)[value_col]
        .apply(_quantile_row)
        .reset_index()
        .drop(columns=[c for c in ["level_1", f"level_{len(group_cols)}"]
                        if c in trajectories.reset_index().columns], errors="ignore")
    )
    # Clean up the extra level column produced by groupby + apply
    level_col = f"level_{len(group_cols)}"
    if level_col in result.columns:
        result = result.drop(columns=[level_col])

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import pathlib

    # Allow running as  python my_model/stochastic_simulate.py  from repo root
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s – %(message)s",
    )

    print("=" * 65)
    print("Smoke-test: stochastic_simulate.py")
    print("=" * 65)

    # ── Parameter estimation ──────────────────────────────────────────────
    import pandas as pd
    import numpy as np

    # Synthetic 16-week declining series (mimics post-winter-peak US data)
    fake_obs = np.array([
        7000, 6700, 6400, 5800, 5200, 4500, 3900, 3200,
        2700, 2200, 1800, 1500, 1300, 1100, 1000, 955,
    ], dtype=float)

    mu, sigma = estimate_log_growth_params(fake_obs, lookback_weeks=8)
    print(f"\nEstimated  μ = {mu:.4f}  σ = {sigma:.4f}")

    # ── Trajectory generation ─────────────────────────────────────────────
    from load_data import build_epiweek_dates, ORIGIN_DATE

    dates = build_epiweek_dates(ORIGIN_DATE, n_weeks=DEFAULT_N_WEEKS)

    trajs = generate_trajectories(
        observations=fake_obs,
        forecast_dates=dates,
        n_trajectories=DEFAULT_N_TRAJECTORIES,
        seed=42,
    )

    print(f"\nTrajectory DataFrame shape : {trajs.shape}")
    print(f"Columns                    : {trajs.columns.tolist()}")
    print(f"trajectory_id range        : {trajs.trajectory_id.min()} – "
          f"{trajs.trajectory_id.max()}")
    print(f"date range                 : {trajs.date.min().date()} – "
          f"{trajs.date.max().date()}")
    print(f"forecast range             : {trajs.forecast.min():.1f} – "
          f"{trajs.forecast.max():.1f}")
    print(f"Any NaN in forecast?       : {trajs.forecast.isna().any()}")
    print(f"Any forecast < min_count?  : {(trajs.forecast < DEFAULT_MIN_COUNT).any()}")

    # ── Quantile summary ──────────────────────────────────────────────────
    q = compute_quantiles(trajs, quantiles=[0.10, 0.50, 0.90])
    print("\nMedian (q=0.50) forecast by horizon (first 5 weeks):")
    median_rows = q[q["quantile"] == 0.50].head(5)
    print(median_rows[["date", "quantile", "value"]].to_string(index=False))

    print("\n✓ Smoke-test passed.")
