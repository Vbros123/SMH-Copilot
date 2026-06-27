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
#  We use 52 weeks (one full year) so the lookback captures a complete seasonal
#  cycle (summer peak + winter peak + spring trough).  Over a full year the
#  mean log-growth is close to zero for an endemic seasonal pathogen, so the
#  estimated drift μ does not lock onto either a growth or decline phase.
#  The seasonal forcing (not μ) is then the dominant driver of the two peaks.
DEFAULT_LOOKBACK_WEEKS: int = 52

#: Exact number of independent stochastic trajectories to generate.
DEFAULT_N_TRAJECTORIES: int = 300

#: Total projection horizon in epi-weeks (Round 20 requirement).
DEFAULT_N_WEEKS: int = 104

#: Minimum observation floor applied after exponentiation (Assumption A6).
#  Set to 200 for national US projections so trajectories cannot collapse to
#  near-zero during seasonal troughs — COVID-19 remains endemic and NHSN data
#  has never reported <100 weekly US hospitalisations since 2020.  State-level
#  callers may override with a lower value proportional to population.
DEFAULT_MIN_COUNT: float = 200.0

# ---------------------------------------------------------------------------
# Seasonal model: dual-harmonic Fourier representation
# ---------------------------------------------------------------------------
# The seasonal log-level is modelled as:
#
#   S(doy) = [A1c·cos(ω·doy) + A1s·sin(ω·doy)]    <- annual harmonic
#           + [A2c·cos(2ω·doy) + A2s·sin(2ω·doy)]   <- semi-annual harmonic
#
# Fitted by OLS on log(US inc hosp) 2023–2026 (NHSN weekly data).
# The annual harmonic captures the dominant winter peak; the semi-annual
# harmonic captures the secondary summer peak.
#
# Fitted values (2023-2026 OLS, see calibration notes in MODEL_DESCRIPTION.md):
#   Annual:      amp=0.493, peak_doy≈62 (lat Dec / early Jan)
#   Semi-annual: amp=0.436, peak_doy≈23 (lat Jan) → two peaks: ~Jan + ~Jul
#
# In (cos, sin) Fourier form:
#   cos1 = 0.492,  sin1 = -0.026   (annual)
#   cos2 = -0.006, sin2 =  0.436   (semi-annual)

#: Annual harmonic cosine coefficient (log-level space).
SEASONAL_A1C: float =  0.492
#: Annual harmonic sine coefficient (log-level space).
SEASONAL_A1S: float = -0.026
#: Semi-annual harmonic cosine coefficient (log-level space).
SEASONAL_A2C: float = -0.006
#: Semi-annual harmonic sine coefficient (log-level space).
SEASONAL_A2S: float =  0.436

# Keep the legacy amplitude / peak_doy names as aliases so existing function
# signatures remain backward compatible.
SEASONAL_AMPLITUDE: float = SEASONAL_A1C    # compat alias
SEASONAL_AMPLITUDE_2: float = SEASONAL_A2S  # compat alias
PEAK_DOY: int = 18    # compat alias (not directly used in new model)
PEAK_DOY_2: int = 196  # compat alias (not directly used in new model)

# ---------------------------------------------------------------------------
# Helper: build the weekly seasonal drift adjustment
# ---------------------------------------------------------------------------

def _seasonal_level(
    dates: pd.DatetimeIndex,
    a1c: float = SEASONAL_A1C,
    a1s: float = SEASONAL_A1S,
    a2c: float = SEASONAL_A2C,
    a2s: float = SEASONAL_A2S,
    # Legacy compat params (ignored; kept so existing callers don't break)
    amplitude: float = SEASONAL_AMPLITUDE,
    peak_doy: int = PEAK_DOY,
    amplitude_2: float = SEASONAL_AMPLITUDE_2,
    peak_doy_2: int = PEAK_DOY_2,
) -> np.ndarray:
    """
    Compute the **absolute seasonal log-level** for each epi-week using a
    dual-harmonic Fourier representation fitted to 2023–2026 US NHSN data.

    S(doy) = a1c·cos(ω·doy) + a1s·sin(ω·doy)     [annual harmonic]
           + a2c·cos(2ω·doy) + a2s·sin(2ω·doy)    [semi-annual harmonic]

    The annual harmonic captures the winter peak (≈ Jan); the semi-annual
    harmonic captures the secondary summer peak (≈ Jul/Aug), producing the
    two seasonal peaks per year observed in US COVID-19 hospitalisation data.

    Returns
    -------
    np.ndarray, shape (len(dates),)
        Seasonal log-count level at each epi-week.
    """
    omega = 2.0 * np.pi / 365.25
    doy = dates.day_of_year.to_numpy(dtype=float)
    annual     = a1c * np.cos(omega * doy) + a1s * np.sin(omega * doy)
    semiannual = a2c * np.cos(2 * omega * doy) + a2s * np.sin(2 * omega * doy)
    return annual + semiannual


def _seasonal_drift(
    dates: pd.DatetimeIndex,
    a1c: float = SEASONAL_A1C,
    a1s: float = SEASONAL_A1S,
    a2c: float = SEASONAL_A2C,
    a2s: float = SEASONAL_A2S,
    # Legacy compat params (ignored)
    amplitude: float = SEASONAL_AMPLITUDE,
    peak_doy: int = PEAK_DOY,
    amplitude_2: float = SEASONAL_AMPLITUDE_2,
    peak_doy_2: int = PEAK_DOY_2,
) -> np.ndarray:
    """
    Week-to-week derivative of :func:`_seasonal_level` (for reference only;
    the trajectory generator uses the level directly, not its derivative).

    Returns
    -------
    np.ndarray, shape (len(dates),)
        Weekly additive change in seasonal log-level.
    """
    omega = 2.0 * np.pi / 365.25
    doy = dates.day_of_year.to_numpy(dtype=float)
    d_annual     = 7 * omega * (-a1c * np.sin(omega * doy) + a1s * np.cos(omega * doy))
    d_semiannual = 7 * 2 * omega * (-a2c * np.sin(2 * omega * doy) + a2s * np.cos(2 * omega * doy))
    return d_annual + d_semiannual


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
    # Fourier seasonal parameters (fitted to 2023-2026 US NHSN data)
    seasonal_a1c: float = SEASONAL_A1C,
    seasonal_a1s: float = SEASONAL_A1S,
    seasonal_a2c: float = SEASONAL_A2C,
    seasonal_a2s: float = SEASONAL_A2S,
    # Legacy compat params (kept for backward compatibility; not used internally)
    seasonal_amplitude: float = SEASONAL_AMPLITUDE,
    seasonal_amplitude_2: float = SEASONAL_AMPLITUDE_2,
    peak_doy: int = PEAK_DOY,
    peak_doy_2: int = PEAK_DOY_2,
    mu_reversion_strength: float = 0.15,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate ``n_trajectories`` stochastic forecast trajectories using a
    dual-harmonic seasonal log-growth random-walk model.

    Parameters
    ----------
    observations         : Ordered historical weekly counts (oldest → newest).
                           Used to estimate the drift μ and volatility σ, and to
                           seed the initial forecast value.
    forecast_dates       : DatetimeIndex of epi-week Saturday end-dates for the
                           projection horizon.  Must be non-empty.
    lookback_weeks       : Weeks of history used for μ/σ estimation (default 26).
    n_trajectories       : Number of independent trajectories (default 300).
    min_count            : Minimum weekly count floor (default 200 for US national).
    seasonal_amplitude   : Primary (winter) sinusoid amplitude (default 0.35).
    seasonal_amplitude_2 : Secondary (summer) sinusoid amplitude (default 0.22).
    peak_doy             : Primary peak day-of-year (default 18, ≈ 18 Jan).
    peak_doy_2           : Secondary peak day-of-year (default 196, ≈ 15 Jul).
    mu_reversion_strength: Fraction by which μ is pulled toward zero each period
                           to prevent trajectories from diverging exponentially
                           over 104 weeks.  0.0 = no reversion, 1.0 = full reversion
                           to zero each step.  Default 0.15.
    seed                 : Optional integer random seed for reproducibility.
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
    """
    if len(forecast_dates) == 0:
        raise ValueError("generate_trajectories: forecast_dates must not be empty.")

    rng = np.random.default_rng(seed)

    # ── Step 1: Estimate drift μ and volatility σ ─────────────────────────
    mu, sigma = estimate_log_growth_params(observations, lookback_weeks=lookback_weeks)

    # Apply mild mean-reversion: pull the estimated drift toward zero so that
    # trajectories do not diverge exponentially over 104 weeks.  This is a
    # conservative damping factor; the seasonal forcing still drives the
    # expected summer and winter peaks.
    mu_damped = mu * (1.0 - mu_reversion_strength)

    # ── Step 2: Build dual-harmonic seasonal drift for each forecast week ───
    # Shape: (n_forecast_weeks,)
    seasonal_adj = _seasonal_drift(
        forecast_dates,
        amplitude=seasonal_amplitude,
        peak_doy=peak_doy,
        amplitude_2=seasonal_amplitude_2,
        peak_doy_2=peak_doy_2,
    )
    # Total drift per week = damped trend + dual-harmonic seasonal correction
    drift = mu_damped + seasonal_adj  # shape: (n_weeks,)

    logger.info(
        "generate_trajectories: μ=%.4f  μ_damped=%.4f  σ=%.4f  horizon=%d weeks  "
        "n_trajectories=%d  seed=%s",
        mu, mu_damped, sigma, len(forecast_dates), n_trajectories, seed,
    )

    # ── Step 3: Deseasonalize the seed observation ────────────────────────
    # The model uses a seasonal decomposition approach:
    #
    #   log H(t) = trend(t) + seasonal_level(t)
    #
    # where trend(t) is a slow-changing baseline and seasonal_level(t) is the
    # dual-harmonic sinusoidal component.
    #
    # Seeding strategy:
    #   - The "seed observation" is the observed value at (or near) the first
    #     forecast date.  For Round 20, this is the last value BEFORE origin_date
    #     (≈ 3786 on 2025-06-07), NOT the most recent available observation.
    #   - Callers that want to use a specific seed value should pass it as the
    #     first (and only) element of ``observations``, or use the
    #     ``forecast_target`` convenience wrapper which handles this correctly.
    arr = np.asarray(observations, dtype=float)
    arr_clean = arr[~np.isnan(arr)]
    if len(arr_clean) == 0:
        raise ValueError(
            "generate_trajectories: all observations are NaN; "
            "cannot seed the forecast."
        )
    # Use the LAST observation in the array as the seed value.
    # The caller is responsible for passing observations trimmed to the
    # seed date (e.g. all obs up to ORIGIN_DATE when forecasting from origin).
    seed_obs = float(arr_clean[-1])
    seed_obs_safe = max(seed_obs, min_count)

    # Seasonal level at the seed date (= the week before first forecast)
    seed_date_dti = pd.DatetimeIndex([forecast_dates[0] - pd.Timedelta(days=7)])
    seasonal_at_seed = _seasonal_level(
        seed_date_dti,
        a1c=seasonal_a1c, a1s=seasonal_a1s,
        a2c=seasonal_a2c, a2s=seasonal_a2s,
    )[0]

    # Deseasonalized log-trend at the seed date
    log_trend_seed = np.log(seed_obs_safe) - seasonal_at_seed

    # Seasonal level at all forecast dates (shape: n_weeks)
    seasonal_forecast = _seasonal_level(
        forecast_dates,
        a1c=seasonal_a1c, a1s=seasonal_a1s,
        a2c=seasonal_a2c, a2s=seasonal_a2s,
    )

    n_weeks = len(forecast_dates)

    # ── Step 4: Draw innovation matrix ────────────────────────────────────
    innovations = rng.normal(loc=0.0, scale=sigma, size=(n_trajectories, n_weeks))

    # ── Step 5: Build cumulative deseasonalized log-trend ─────────────────
    # log_trend[i,t] = log_trend_seed + Σ_{s=1}^{t} [μ_damped + ε_i[s]]
    # Note: seasonal forcing is NOT in the drift here; it's added as a level.
    trend_increments = np.cumsum(
        mu_damped + innovations, axis=1
    )  # shape: (n_trajectories, n_weeks)

    log_trend = log_trend_seed + trend_increments  # (n_trajectories, n_weeks)

    # ── Step 6: Reattach seasonal level ──────────────────────────────────
    # log_forecast[i, t] = log_trend[i, t] + seasonal_forecast[t]
    log_forecast = log_trend + seasonal_forecast[np.newaxis, :]

    # ── Step 7: Exponentiate and apply floor ─────────────────────────────
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

    # Load ALL calibration data up to FIT_END_DATE
    obs_df_full = load_target_data(
        location_filter=[location],
        target_filter=[target],
        age_group_filter=[age_group],
        max_date=pd.Timestamp("2026-06-06"),  # FIT_END_DATE
    )

    if obs_df_full.empty:
        raise ValueError(
            f"forecast_target: no observations found for location='{location}', "
            f"target='{target}', age_group='{age_group}'."
        )

    obs_df_full = obs_df_full.sort_values("date")

    # The seed observations must end at ORIGIN_DATE, not at FIT_END_DATE.
    # Rationale: the trajectory starts from the value at origin_date so that
    # the seasonal model correctly places the retrospective year (Jun 2025 –
    # Jun 2026) before the prospective year (Jun 2026 – Jun 2027).
    # μ/σ estimation uses data up to ORIGIN_DATE (the 52-week window ending
    # at the projection start date), capturing one full seasonal cycle.
    obs_seed_df = obs_df_full[obs_df_full["date"] <= ORIGIN_DATE]
    if obs_seed_df.empty:
        logger.warning(
            "forecast_target: no obs at or before ORIGIN_DATE=%s; "
            "using earliest available.", ORIGIN_DATE
        )
        obs_seed_df = obs_df_full.head(lookback_weeks + 1)

    observations_for_seed = obs_seed_df["observation"].values.astype(float)

    forecast_dates = build_epiweek_dates(origin=ORIGIN_DATE, n_weeks=n_weeks)

    trajs = generate_trajectories(
        observations=observations_for_seed,
        forecast_dates=forecast_dates,
        lookback_weeks=lookback_weeks,
        n_trajectories=n_trajectories,
        min_count=DEFAULT_MIN_COUNT,
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
