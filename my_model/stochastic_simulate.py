"""
stochastic_simulate.py
======================
Public interface for trajectory generation used by the test pipeline,
build_submission.py, and scenario_adjustments.py.

This module is the STABLE API LAYER — it exposes the same symbols and
function signatures that the rest of the pipeline depends on, while
delegating the actual epidemic simulation to simulate.py (SEIRS model).

Architecture:
    stochastic_simulate.py  <- public API (this file)
           |
           v
    simulate.py             <- SEIRS compartmental model + calibration
           |
           v
    load_data.py            <- data loading (observations, populations)

All previously exported names are preserved for backward compatibility:
  - generate_trajectories       now delegates to SEIRS
  - estimate_log_growth_params  kept for test assertions
  - DEFAULT_N_TRAJECTORIES      300
  - DEFAULT_N_WEEKS             104
  - DEFAULT_MIN_COUNT           1.0
  - SEASONAL_AMPLITUDE          informational alias
  - PEAK_DOY                    informational alias
  - forecast_target             convenience wrapper
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants - preserved for backward compatibility with test pipeline
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK_WEEKS: int = 52
DEFAULT_N_TRAJECTORIES: int = 300
DEFAULT_N_WEEKS: int = 104
DEFAULT_MIN_COUNT: float = 1.0

# Legacy seasonal constants - kept for backward compatibility
SEASONAL_AMPLITUDE: float = 0.30
SEASONAL_AMPLITUDE_2: float = 0.15
SEASONAL_A1C: float = 0.492
SEASONAL_A1S: float = -0.026
SEASONAL_A2C: float = -0.006
SEASONAL_A2S: float = 0.436
PEAK_DOY: int = 18
PEAK_DOY_2: int = 249


def estimate_log_growth_params(
    observations,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_obs_for_growth: float = 1.0,
) -> tuple:
    """
    Estimate mean (mu) and std (sigma) of weekly log-growth rates.

    Retained for backward compatibility with tests/test_pipeline.py.
    The SEIRS model does NOT use this function internally.

    Returns
    -------
    (mu, sigma) both finite floats with sigma > 0.
    """
    arr = np.asarray(observations, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 3:
        raise ValueError(
            f"estimate_log_growth_params: need >= 3 valid observations, got {len(arr)}."
        )

    n_needed = lookback_weeks + 1
    if len(arr) < n_needed:
        logger.warning(
            "estimate_log_growth_params: only %d observations (need %d for %d-week lookback); using all.",
            len(arr), n_needed, lookback_weeks,
        )
        window = arr
    else:
        window = arr[-n_needed:]

    window_safe = np.where(window < min_obs_for_growth, min_obs_for_growth, window)
    log_growth = np.diff(np.log(window_safe))

    mu = float(np.mean(log_growth))
    sigma = float(np.std(log_growth, ddof=1)) if len(log_growth) > 1 else 0.1
    return mu, sigma


def generate_trajectories(
    observations,
    forecast_dates: pd.DatetimeIndex,
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    min_count: float = DEFAULT_MIN_COUNT,
    seasonal_amplitude: float = SEASONAL_AMPLITUDE,
    seasonal_amplitude_2: float = SEASONAL_AMPLITUDE_2,
    seasonal_a1c: float = SEASONAL_A1C,
    seasonal_a1s: float = SEASONAL_A1S,
    seasonal_a2c: float = SEASONAL_A2C,
    seasonal_a2s: float = SEASONAL_A2S,
    peak_doy: int = PEAK_DOY,
    peak_doy_2: int = PEAK_DOY_2,
    mu_reversion_strength: float = 0.15,
    location: str = "US",
    seed: Optional[int] = None,
    scenario_id: str = "A-2026-05-11",
) -> pd.DataFrame:
    """
    Generate n_trajectories stochastic SEIRS forecast trajectories.

    This is the primary public interface. It:
    1. Calibrates a SEIRS model against recent observed hospitalisations.
    2. Warm-starts the SEIRS compartments to the forecast origin date.
    3. Generates n_trajectories stochastic realisations over forecast_dates.

    Parameters
    ----------
    observations  : Historical weekly inc hosp (oldest -> newest).
                    Used to pass legacy API; actual calibration uses all
                    available data loaded internally.
    forecast_dates: DatetimeIndex of 104 epi-week Saturday end-dates.
    n_trajectories: Number of stochastic trajectories (default 300).
    min_count     : Minimum weekly hospitalisation floor (default 1.0).
    location      : FIPS code for state-specific calibration.
    seed          : RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: trajectory_id (int32), date, forecast (float64)
    Shape: n_trajectories x len(forecast_dates) rows.
    """
    from simulate import (
        get_calibrated_params,
        generate_seirs_trajectories,
        run_seirs,
        SEIRSParams,
        MIN_HOSP_FLOOR,
        CAL_START,
    )
    from load_data import (
        ORIGIN_DATE,
        get_population_map,
        load_target_data,
        build_historical_vax_multipliers,
        build_continuous_vax_multipliers,
    )

    if len(forecast_dates) == 0:
        raise ValueError("generate_trajectories: forecast_dates must not be empty.")

    pop_map = get_population_map()
    population = pop_map.get(location, 330_000_000)

    logger.info(
        "generate_trajectories [%s]: calibrating SEIRS  n_trajectories=%d  n_weeks=%d",
        location, n_trajectories, len(forecast_dates),
    )

    params = get_calibrated_params(location=location, population=population)

    # Warm-start: run deterministic SEIRS from CAL_START to ORIGIN_DATE
    cal_obs_df = load_target_data(
        location_filter=[location],
        target_filter=["inc hosp"],
        age_group_filter=["0-130"],
        min_date=CAL_START,
        max_date=ORIGIN_DATE,
    ).sort_values("date")

    n_warmup = len(cal_obs_df)
    start_epiweek_cal = int(cal_obs_df["date"].iloc[0].isocalendar()[1]) if n_warmup > 0 else 1

    if n_warmup > 0:
        # Apply historical vaccination during the warm-start (CAL_START →
        # ORIGIN_DATE) so the reconstructed SEIRS compartments at ORIGIN_DATE
        # reflect real accumulated immunity from the 2024 and 2025-26 campaigns.
        # The warm-start dates are entirely historical; no future scenario
        # vaccination is applied here.
        warmup_vax_mult = build_historical_vax_multipliers(
            pd.DatetimeIndex(cal_obs_df["date"])
        )
        warmup = run_seirs(
            params,
            n_weeks=n_warmup,
            start_epiweek=start_epiweek_cal,
            stochastic=False,
            vax_multipliers=warmup_vax_mult,
        )
        S_init = float(warmup["S"][-1])
        E_init = float(warmup["E"][-1])
        I_init = float(warmup["I"][-1])
        R_init = float(warmup["R"][-1])
    else:
        N = float(population)
        I_init = 10.0 ** params.log_i0 * N
        R_init = params.r0_frac * N
        E_init = I_init * params.de / params.di
        S_init = max(N - E_init - I_init - R_init, 1.0)

    logger.info(
        "generate_trajectories [%s]: state at origin — S=%.0f  E=%.0f  I=%.0f  R=%.0f",
        location, S_init, E_init, I_init, R_init,
    )

    params_at_origin = SEIRSParams(
        beta0=params.beta0,
        seasonal_amp=params.seasonal_amp,
        seasonal_phase=params.seasonal_phase,
        p_hosp=params.p_hosp,
        log_i0=float(np.log10(max(I_init / population, 1e-10))),
        de=params.de,
        di=params.di,
        tau_r=params.tau_r,
        eps_nb=params.eps_nb,
        r0_frac=R_init / population,
        population=population,
        location=location,
    )

    # ── Continuous vaccination timeline for forecast ───────────────────
    # Build one merged coverage curve (historical + future scenario) and
    # perform the VE convolution once.  The forecast slice is passed into
    # run_seirs so immunity carries forward continuously from calibration.
    # scenario_id="A-2026-05-11" (default) means no additional future
    # campaign — the Scenario A baseline.  Callers pass a different
    # scenario_id to incorporate future campaign coverage inside SEIRS;
    # post-hoc scenario_adjustments multipliers still apply on top for
    # the incremental B–E differences.
    _, forecast_vax_mult = build_continuous_vax_multipliers(
        cal_dates=pd.DatetimeIndex(cal_obs_df["date"]),
        forecast_dates=forecast_dates,
        scenario_id=scenario_id,
    )

    result_df = generate_seirs_trajectories(
        params=params_at_origin,
        forecast_dates=forecast_dates,
        n_trajectories=n_trajectories,
        seed=seed,
        vax_multipliers=forecast_vax_mult,
    )

    floor = max(float(min_count), float(MIN_HOSP_FLOOR))
    result_df["forecast"] = result_df["forecast"].clip(lower=floor)

    logger.info(
        "generate_trajectories [%s]: done  rows=%d  range [%.1f, %.1f]",
        location, len(result_df),
        result_df["forecast"].min(), result_df["forecast"].max(),
    )
    return result_df


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
    Convenience wrapper: load data, build dates, generate SEIRS trajectories.

    Returns pd.DataFrame with trajectory_id, date, forecast, location, target, age_group.
    """
    from load_data import load_target_data, build_epiweek_dates, ORIGIN_DATE, FIT_END_DATE

    obs_df = load_target_data(
        location_filter=[location],
        target_filter=[target],
        age_group_filter=[age_group],
        max_date=FIT_END_DATE,
    ).sort_values("date")

    if obs_df.empty:
        raise ValueError(
            f"forecast_target: no data for location='{location}' target='{target}'."
        )

    observations = obs_df["observation"].values.astype(float)
    forecast_dates = build_epiweek_dates(origin=ORIGIN_DATE, n_weeks=n_weeks)

    trajs = generate_trajectories(
        observations=observations,
        forecast_dates=forecast_dates,
        n_trajectories=n_trajectories,
        min_count=DEFAULT_MIN_COUNT,
        location=location,
        seed=seed,
    )

    trajs["location"] = location
    trajs["target"] = target
    trajs["age_group"] = age_group
    return trajs
