"""
simulate.py
===========
Stochastic SEIRS compartmental epidemic model for the Round 20 COVID-19
Scenario Modeling Hub submission.

Model overview
--------------
This module implements a **discrete-time stochastic SEIRS model** with:
  - Seasonal transmission forcing
  - Waning immunity (R → S return)
  - Vaccination protection (reduces effective susceptibility)
  - Stochastic transmission noise (negative-binomial overdispersion)
  - Hospitalisation compartment derived from I(t)
  - State-specific parameter calibration via least-squares optimisation

Compartments
------------
  S  –  Susceptible
  E  –  Exposed (latent, pre-infectious)
  I  –  Infectious
  R  –  Recovered (temporarily immune)

Observable output
-----------------
  H(t) = hospitalisation rate = I(t) × p_hosp × N
  D(t) = death rate           = H(t) × cfr

where p_hosp is the infection-hospitalisation rate and cfr is the
case-fatality ratio among hospitalised patients.

Equations (discrete-time, weekly steps)
----------------------------------------
λ(t) = β(t) × I(t) / N                     [force of infection]
β(t) = β0 × (1 + A × cos(2π(t − φ)/52))   [seasonal transmission]

New_E(t)  ~ NegBin(S(t) × λ(t),  ε)       [stochastic exposure]
New_I(t)  = E(t) / D_E                      [latency progression]
New_R(t)  = I(t) / D_I                      [recovery]
New_S(t)  = R(t) × ω                        [waning]
New_H(t)  = New_I(t) × p_hosp × N          [weekly incident hospitalisations]

where:
  ε       – overdispersion parameter (1/k in negative binomial)
  D_E     – mean latent period in weeks (default 1.5)
  D_I     – mean infectious period in weeks (default 1.5)
  ω       – weekly waning rate = 1 − exp(−1/τ_R) where τ_R ≈ 52 weeks

Seasonal forcing
----------------
β(t) = β0 × (1 + A × cos(2π(t − φ)/52))

  β0  – baseline reproduction number proxy (units: 1/week)
  A   – seasonal amplitude (0 < A < 1; default 0.3)
  φ   – seasonal phase in epi-weeks (winter peak at epi-week ~2 = early Jan)

Calibration
-----------
For each location we minimise the sum of squared log-residuals between
simulated and observed weekly hospitalisations over the calibration window
(2024-01-01 → FIT_END_DATE) using scipy.optimize.minimize with Nelder-Mead.

Calibrated parameters per location:
  β0       –  baseline transmission
  A        –  seasonal amplitude
  φ        –  seasonal phase (week-of-year)
  p_hosp   –  infection-hospitalisation rate
  log_I0   –  log initial infectious fraction

National defaults (literature-based)
-------------------------------------
β0 = 0.35 week⁻¹
  Corresponds to R0 ≈ β0 × D_I ≈ 0.35 × 1.5 ≈ 0.52 in the fully susceptible
  limit, but effective Rt is modulated by S/N and seasonality.  For endemic
  COVID-19 in 2025-26 with ~50-60% of the population having recent immunity,
  Rt oscillates around 1.0 seasonally.

D_E = 1.5 weeks (10.5 days)
  COVID-19 mean incubation period 5-6 days (He et al., Nature Medicine 2020);
  extended slightly to include pre-symptomatic infectious period.

D_I = 1.5 weeks (10.5 days)
  Mean infectious period 4-8 days (Cevik et al., Lancet Microbe 2021).
  Weekly model rounds to 1.5 weeks for numerical stability.

τ_R = 52 weeks (1 year)
  Natural infection immunity waning half-life; consistent with Lancet 2022
  estimates of ~11 months for severe disease protection and ~6 months for
  infection protection.  We use 52 weeks as a round compromise appropriate
  for a 2-year projection.

A = 0.3 (seasonal amplitude)
  Derived from US national 2020-2026 NHSN data: winter peak / summer trough
  ratio ≈ 3–5×, corresponding to A ≈ 0.25–0.40 in log-space.

p_hosp = 0.003  (0.3% of infections hospitalised)
  COVID-19 IHR in 2025-26 estimated at 0.2–0.5% across all ages, lower than
  earlier pandemic period due to acquired population immunity.
  Sources: Hauser et al. Epidemics 2020; updated via CDC FluView analogy.

cfr_hosp = 0.065  (6.5% in-hospital case fatality)
  Source: NCHS/CDC data, consistent with build_submission.py constant.

References
----------
- He X, et al. (2020). Temporal dynamics in viral shedding and transmissibility.
  Nature Medicine 26:672–675.
- Cevik M, et al. (2021). SARS-CoV-2, SARS-CoV, MERS-CoV viral load and
  shedding kinetics. Lancet Microbe 2:e13-e22.
- Iyer AS, et al. (2020). Persistence and decay of human antibody responses.
  Science Immunology 5:eabe0367.
- Kissler SM, et al. (2020). Projecting the transmission dynamics.
  Science 368:860–868.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ===========================================================================
# Model constants — documented above; do not change without updating docstring
# ===========================================================================

#: Mean latent period in weeks (E → I transition)
D_E: float = 1.5

#: Mean infectious period in weeks (I → R transition)
D_I: float = 1.5

#: Waning immunity timescale in weeks (R → S return).
#  COVID-19 natural immunity against severe disease wanes over ~9–12 months;
#  vaccination-boosted immunity lasts longer.  Using 78 weeks (18 months) as
#  the effective population-level waning rate, accounting for a mix of
#  recently vaccinated and unvaccinated individuals.  Longer tau_R produces
#  more stable endemic dynamics (slower susceptibility build-up between waves).
#  Sources: Levin et al. Nat Commun 2022 (12–18 months severity protection).
TAU_R: float = 78.0

#: Overdispersion parameter for negative-binomial noise (1/k_NB).
#  k_NB = 1/ε; ε=0.1 → modest overdispersion consistent with COVID-19
#  transmission cluster data (Lloyd-Smith et al. Nature 2005).
EPSILON_NB: float = 0.10

#: National baseline transmission rate (week⁻¹).
#  Must satisfy R0_basic = beta0 * D_I > 1 for endemic oscillations.
#  beta0 = 1.0 → R0 = 1.5 (plausible endemic COVID-19 in 2024-26).
#  Literature: Rt for COVID-19 endemic phase estimated 1.1–1.5 between waves.
BETA0_DEFAULT: float = 1.0

#: National seasonal amplitude (dimensionless, 0–1).
SEASONAL_AMP_DEFAULT: float = 0.30

#: Seasonal phase: week-of-year at which transmission peaks (winter ≈ week 2).
SEASONAL_PHASE_DEFAULT: float = 2.0

#: National infection-hospitalisation rate (fraction of new infectious).
#  COVID-19 IHR 2025-26 estimated 0.3–1% for all ages (post-immunity era).
#  With endemic beta0=1.0 and N=330M: daily infections at endemic eq ≈
#  beta0*(1-1/R0)*N/D_I = 1.0*(1-0.67)*330M/1.5 ≈ 73M/week; needs p_hosp~1e-4
#  to get ~7000 hosp/week.  But not all infections counted, so p_hosp adjusted.
#  Starting value for optimisation; actual value fitted per location.
P_HOSP_DEFAULT: float = 0.008

#: In-hospital case fatality ratio (fraction of hospitalisations).
CFR_HOSP: float = 0.065

#: Initial infectious fraction of population at calibration start.
I0_FRAC_DEFAULT: float = 5e-4

#: Initial recovered fraction (prior immunity at calibration start).
#  At endemic equilibrium with R0=1.5: S* = 1/R0 ≈ 0.67, so R* ≈ 0.33.
#  The calibration window (Jan 2024) begins at an epidemic peak, meaning
#  recently-infected individuals are in R.  Using 0.33 (near endemic equilibrium).
R0_FRAC_DEFAULT: float = 0.33

#: Calibration window start date (common across all locations).
CAL_START: pd.Timestamp = pd.Timestamp("2024-01-01")

#: Calibration window end date (= FIT_END_DATE, no look-ahead).
CAL_END: pd.Timestamp = pd.Timestamp("2026-06-06")

#: Minimum floor on weekly hospitalisations (endemic lower bound).
MIN_HOSP_FLOOR: float = 1.0


# ===========================================================================
# Parameter dataclass
# ===========================================================================

@dataclass
class SEIRSParams:
    """
    Complete parameter set for one SEIRS location model.

    Attributes
    ----------
    beta0          : Baseline weekly transmission rate.
    seasonal_amp   : Seasonal forcing amplitude (A in β(t) = β0(1+A·cos(...))).
    seasonal_phase : Phase offset in epi-weeks (peak = week φ of year).
    p_hosp         : Infection-hospitalisation rate (fraction of infectious).
    log_i0         : Log₁₀ of the initial infectious fraction.
    de             : Mean latent period (weeks).
    di             : Mean infectious period (weeks).
    tau_r          : Waning immunity timescale (weeks).
    eps_nb         : Negative-binomial overdispersion (σ²/μ − 1).
    r0_frac        : Initial recovered fraction (prior immunity).
    population     : Total population for this location.
    location       : FIPS code string.
    """
    beta0: float          = BETA0_DEFAULT
    seasonal_amp: float   = SEASONAL_AMP_DEFAULT
    seasonal_phase: float = SEASONAL_PHASE_DEFAULT
    p_hosp: float         = P_HOSP_DEFAULT
    log_i0: float         = float(np.log10(I0_FRAC_DEFAULT))
    de: float             = D_E
    di: float             = D_I
    tau_r: float          = TAU_R
    eps_nb: float         = EPSILON_NB
    r0_frac: float        = R0_FRAC_DEFAULT
    population: int       = 330_000_000
    location: str         = "US"


# ===========================================================================
# Core SEIRS simulator
# ===========================================================================

def run_seirs(
    params: SEIRSParams,
    n_weeks: int,
    start_epiweek: int = 1,
    seed: Optional[int] = None,
    stochastic: bool = True,
    vax_multipliers: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate a discrete-time stochastic SEIRS epidemic over ``n_weeks``.

    Parameters
    ----------
    params          : :class:`SEIRSParams` containing all model parameters.
    n_weeks         : Number of weekly time steps to simulate.
    start_epiweek   : Epi-week number (1-52) of the first simulation step.
                      Used to align seasonal forcing with the calendar.
    seed            : Optional RNG seed for reproducibility.
    stochastic      : If True, use negative-binomial noise.  If False, run
                      deterministic mean-field equations (useful for calibration).
    vax_multipliers : Optional array of length ``n_weeks``.  Each entry is the
                      multiplier applied to S(t) before computing force-of-
                      infection to represent vaccination-reduced susceptibility.
                      1.0 means no vaccination effect; <1.0 means protection.

    Returns
    -------
    dict with keys:
        ``S``, ``E``, ``I``, ``R``  – compartment counts (shape: (n_weeks+1,))
        ``new_hosp``                 – weekly incident hospitalisations (n_weeks,)
        ``new_infections``           – weekly new infections (n_weeks,)

    Notes
    -----
    The discrete-time equations use forward Euler integration with weekly steps.
    This is a first-order approximation; for D_E = D_I = 1.5 weeks the
    discretisation error is modest relative to the stochastic noise.

    Negative-binomial noise is parameterised via the mean μ and overdispersion
    parameter ε (variance = μ + ε·μ²).  Setting ``stochastic=False`` gives
    the deterministic mean-field trajectory used for calibration.
    """
    rng = np.random.default_rng(seed)
    N = float(params.population)

    # ── Initial conditions ────────────────────────────────────────────────
    # Use the calibrated log_i0 to set the starting infectious count.
    # r0_frac determines what fraction of the population starts in R (recovered/
    # immune); the remainder is divided between S and E in quasi-steady proportion.
    I0_frac = 10.0 ** params.log_i0
    I0 = max(I0_frac * N, 1.0)
    R0_comp = params.r0_frac * N
    # At quasi-steady state: E/I ≈ D_E/D_I
    E0 = I0 * (params.de / params.di)
    # Assign remaining population to S (ensures N conservation)
    S0 = max(N - E0 - I0 - R0_comp, 1.0)

    # Compartment arrays (include t=0 initial state)
    S = np.zeros(n_weeks + 1)
    E = np.zeros(n_weeks + 1)
    I = np.zeros(n_weeks + 1)
    R = np.zeros(n_weeks + 1)
    new_hosp       = np.zeros(n_weeks)
    new_infections = np.zeros(n_weeks)

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0_comp

    # Pre-compute waning rate
    omega = 1.0 - np.exp(-1.0 / params.tau_r)  # fraction of R returning to S per week

    # ── Simulation loop ───────────────────────────────────────────────────
    for t in range(n_weeks):
        week_num = (start_epiweek + t - 1) % 52 + 1  # 1–52

        # Seasonal transmission β(t)
        beta_t = params.beta0 * (
            1.0 + params.seasonal_amp
            * np.cos(2.0 * np.pi * (week_num - params.seasonal_phase) / 52.0)
        )

        # Effective susceptibles (vaccination reduces susceptibility)
        # Bounds check: vax_multipliers covers only the window it was built
        # for (e.g. the calibration period).  For t beyond that length,
        # fall back to 1.0 (no vaccination effect).  This allows run_seirs
        # to be called with n_weeks > len(vax_multipliers) — as in the
        # stability-extension used by _calibration_loss — without an IndexError.
        if vax_multipliers is not None and t < len(vax_multipliers):
            vax_mult = vax_multipliers[t]
        else:
            vax_mult = 1.0
        S_eff = S[t] * vax_mult

        # Force of infection
        lam = beta_t * I[t] / N

        # Mean flows
        mu_expose = S_eff * lam          # mean new exposures
        mu_infect = E[t] / params.de     # mean new infectious
        mu_recover = I[t] / params.di    # mean new recoveries
        mu_wane   = R[t] * omega         # mean waning back to S

        # Clip means to physical bounds
        mu_expose  = float(np.clip(mu_expose,  0.0, S[t]))
        mu_infect  = float(np.clip(mu_infect,  0.0, E[t]))
        mu_recover = float(np.clip(mu_recover, 0.0, I[t]))
        mu_wane    = float(np.clip(mu_wane,    0.0, R[t]))

        if stochastic and mu_expose > 0:
            # Negative-binomial: mean=μ, variance=μ + ε·μ²
            k_nb = 1.0 / max(params.eps_nb, 1e-6)
            p_nb = k_nb / (k_nb + mu_expose)
            expose = float(rng.negative_binomial(k_nb, p_nb))
            expose = float(np.clip(expose, 0.0, S[t]))
        else:
            expose = mu_expose

        infect  = mu_infect
        recover = mu_recover
        wane    = mu_wane

        # Update compartments
        S[t+1] = S[t] - expose + wane
        E[t+1] = E[t] + expose - infect
        I[t+1] = I[t] + infect - recover
        R[t+1] = R[t] + recover - wane

        # Enforce non-negativity (numerical floor)
        S[t+1] = max(S[t+1], 0.0)
        E[t+1] = max(E[t+1], 0.0)
        I[t+1] = max(I[t+1], 0.0)
        R[t+1] = max(R[t+1], 0.0)

        # Rescale to maintain population conservation
        total = S[t+1] + E[t+1] + I[t+1] + R[t+1]
        if total > 0:
            scale = N / total
            S[t+1] *= scale
            E[t+1] *= scale
            I[t+1] *= scale
            R[t+1] *= scale

        # Observables
        new_infections[t] = infect
        new_hosp[t] = max(infect * params.p_hosp, MIN_HOSP_FLOOR)

    return {
        "S": S, "E": E, "I": I, "R": R,
        "new_hosp": new_hosp,
        "new_infections": new_infections,
    }


# ===========================================================================
# Deterministic calibration
# ===========================================================================

def _calibration_loss(
    theta: np.ndarray,
    obs_hosp: np.ndarray,
    population: int,
    start_epiweek: int,
    weights: Optional[np.ndarray] = None,
    vax_multipliers: Optional[np.ndarray] = None,
) -> float:
    """
    Weighted least-squares loss in log-space between simulated and observed hosp.

    Parameters ``theta`` = [log10_beta0, seasonal_amp, seasonal_phase,
                             log10_p_hosp, log10_i0].

    weights : 1-D float array of length len(obs_hosp).  Each squared
              log-residual is multiplied by the corresponding weight before
              summation, so higher-weight weeks pull the fit more strongly.
              Weights are internally normalised to sum to n_valid so the
              loss magnitude stays comparable to the unweighted version.
              None → uniform weights (legacy behaviour).

    Returns weighted sum of squared log-residuals (ignores NaN observations).

    Parameter bounds (physically motivated):
      log10_beta0  in [-1.5, 0.5]   → β0 in [0.03, 3.16] week⁻¹
      seasonal_amp in [0.0, 0.90]   → 0–90% seasonal variation
      seasonal_phase in [0, 52]     → any week of year
      log10_p_hosp in [-4.0, -1.5]  → IHR in [0.01%, 3.2%]
                                      literature: 0.3–3% for all ages 2025-26
      log10_i0     in [-6.0, -1.0]  → initial I/N from 0.0001% to 10%
    """
    log_beta0, seasonal_amp, seasonal_phase, log_p_hosp, log_i0 = theta

    # Hard bounds via penalty (physically motivated)
    # beta0 in [0.67, 6.3]: ensures R0_basic = beta0*D_I > 1 (endemic oscillations)
    if not (-0.17 <= log_beta0 <= 0.8):
        return 1e12
    if not (0.05 <= seasonal_amp <= 0.85):
        return 1e12
    if not (0.0 <= seasonal_phase <= 52.0):
        return 1e12
    # p_hosp bounded 0.03% – 3% (IHR for all-ages COVID-19 2025-26)
    if not (-3.5 <= log_p_hosp <= -1.5):
        return 1e12
    if not (-4.0 <= log_i0 <= -0.5):
        return 1e12

    p = SEIRSParams(
        beta0=10.0 ** log_beta0,
        seasonal_amp=float(np.clip(seasonal_amp, 0.0, 0.90)),
        seasonal_phase=float(seasonal_phase),
        p_hosp=10.0 ** log_p_hosp,
        log_i0=log_i0,
        population=population,
    )

    n_weeks = len(obs_hosp)
    # Run calibration window + 52 extra weeks to penalise runaway growth
    n_extended = n_weeks + 52
    try:
        # vax_multipliers has length n_weeks (calibration window only).
        # run_seirs applies vax_multipliers[t] when t < len(vax_multipliers)
        # and falls back to 1.0 for t >= len, so the 52-week stability
        # extension automatically runs unvaccinated — correct, because that
        # penalty horizon represents the future, not the historical window.
        result = run_seirs(
            p, n_weeks=n_extended, start_epiweek=start_epiweek, stochastic=False,
            vax_multipliers=vax_multipliers,
        )
        sim_hosp = result["new_hosp"]
    except Exception:
        return 1e12

    # Log-space residuals on calibration window; skip NaN observations
    valid = ~np.isnan(obs_hosp)
    if valid.sum() < 4:
        return 1e12

    log_obs = np.log(np.where(obs_hosp[valid] > 0, obs_hosp[valid], 1.0))
    log_sim = np.log(np.where(sim_hosp[:n_weeks][valid] > 0, sim_hosp[:n_weeks][valid], 1.0))

    sq_res = (log_obs - log_sim) ** 2

    # ── Apply per-week weights ────────────────────────────────────────────
    if weights is not None:
        w = np.asarray(weights, dtype=float)[valid]
        # Normalise so weights sum to n_valid — loss stays comparable
        # to the unweighted case and the stability/scale penalties don't
        # need rescaling.
        w = w * (float(valid.sum()) / w.sum())
        fit_loss = float(np.sum(w * sq_res))
    else:
        fit_loss = float(np.sum(sq_res))

    # ── Stability penalty ─────────────────────────────────────────────────
    # Penalise models where hosp in the extended window exceeds 10× the
    # max observed hospitalisations.  Using 10× (not 2×) because the SEIRS
    # endemic equilibrium can legitimately exceed 2× observed when p_hosp
    # is large enough to match peak hospitalizations — the model's math
    # produces high endemic I* even though the 52-week stability window
    # is in the future and still declining from an epidemic peak.
    # 2× was blocking correct high-p_hosp solutions by adding ~40,000
    # penalty units while the fit loss was ~70, forcing the optimizer to
    # accept a low-p_hosp local minimum.  10× still blocks genuinely
    # explosive growth while leaving room for the correct solution.
    obs_max = float(np.nanmax(obs_hosp))
    future_hosp = sim_hosp[n_weeks:]
    if len(future_hosp) > 0:
        future_max = float(np.max(future_hosp))
        # Penalty term: 0 when future_max <= 10*obs_max, growing quadratically above
        ratio = future_max / (10.0 * obs_max + 1.0)
        stability_penalty = max(0.0, ratio - 1.0) ** 2 * 100.0
    else:
        stability_penalty = 0.0

    # ── Fit scale penalty ─────────────────────────────────────────────────
    # Additional penalty if the calibration peak is off by more than 2×
    sim_max = float(np.max(sim_hosp[:n_weeks]))
    scale_ratio = max(sim_max, obs_max) / (min(sim_max, obs_max) + 1.0)
    scale_penalty = max(0.0, scale_ratio - 2.0) ** 2 * 20.0

    return fit_loss + stability_penalty + scale_penalty


def calibrate_seirs(
    obs_hosp: np.ndarray,
    population: int,
    start_epiweek: int = 1,
    location: str = "US",
    maxiter: int = 1500,
    seed: int = 0,
    weights: Optional[np.ndarray] = None,
    vax_multipliers: Optional[np.ndarray] = None,
) -> SEIRSParams:
    """
    Calibrate SEIRS parameters for one location using Nelder-Mead optimisation
    on weighted log-space least-squares residuals.

    Parameters
    ----------
    obs_hosp       : Array of observed weekly incident hospitalisations
                     (length = calibration window in weeks; NaN = missing).
                     Should be the most recent CAL_WINDOW_WEEKS observations
                     for best near-origin alignment.
    population     : Total population of the jurisdiction.
    start_epiweek  : Epi-week (1–52) of the first observation.
    location       : FIPS code (for logging only).
    maxiter        : Maximum Nelder-Mead iterations.
    seed           : Not used (deterministic calibration); kept for API compat.
    weights        : Optional 1-D float array of length len(obs_hosp).
                     Larger value → that week's residual counts more.
                     Typically exponentially increasing so the most recent
                     weeks dominate (e.g. last week weight ≈ 8× first week).
                     None → uniform weights.

    Returns
    -------
    :class:`SEIRSParams` with calibrated ``beta0``, ``seasonal_amp``,
    ``seasonal_phase``, ``p_hosp``, ``log_i0``.  Fixed parameters
    (``de``, ``di``, ``tau_r``, ``eps_nb``) retain their defaults.
    """
    # Scale p_hosp by population: states use national IHR but a lower raw
    # count floor drives the initial search toward the right order of magnitude.
    scale = population / 330_000_000
    log_p_hosp_init = np.log10(P_HOSP_DEFAULT)  # log10(0.008) ≈ -2.1
    # Initial I0: with endemic R0=1.5, endemic I* = (1-1/R0)*N/(R0*D_I*omega)
    # For a rough estimate: I0/N ~ 0.002 (0.2% infectious at endemic eq)
    log_i0_init = np.log10(max(I0_FRAC_DEFAULT * max(scale, 0.001), 1e-5))

    x0 = np.array([
        np.log10(BETA0_DEFAULT),    # log10_beta0 = 0.0 (β0=1.0, R0=1.5)
        SEASONAL_AMP_DEFAULT,       # seasonal_amp = 0.30
        SEASONAL_PHASE_DEFAULT,     # seasonal_phase = 2.0 (peaks ~week 2 = Jan)
        log_p_hosp_init,            # log10_p_hosp
        log_i0_init,                # log10_i0
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _calibration_loss,
            x0,
            args=(obs_hosp, population, start_epiweek, weights, vax_multipliers),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 5e-4, "fatol": 5e-4, "disp": False},
        )

    theta_opt = result.x
    final_loss = result.fun

    logger.info(
        "calibrate_seirs [%s]: loss=%.3f  β0=%.4f  A=%.3f  φ=%.1f  "
        "p_hosp=%.5f  log_i0=%.2f  n_iter=%d  success=%s",
        location,
        final_loss,
        10.0 ** theta_opt[0],
        theta_opt[1],
        theta_opt[2],
        10.0 ** theta_opt[3],
        theta_opt[4],
        result.nit,
        result.success,
    )

    return SEIRSParams(
        beta0         = 10.0 ** float(np.clip(theta_opt[0], -0.17, 0.8)),
        seasonal_amp  = float(np.clip(theta_opt[1], 0.05, 0.85)),
        seasonal_phase= float(np.clip(theta_opt[2], 0.0, 52.0)),
        p_hosp        = 10.0 ** float(np.clip(theta_opt[3], -3.5, -1.5)),
        log_i0        = float(np.clip(theta_opt[4], -4.0, -0.5)),
        de            = D_E,
        di            = D_I,
        tau_r         = TAU_R,
        eps_nb        = EPSILON_NB,
        r0_frac       = R0_FRAC_DEFAULT,
        population    = population,
        location      = location,
    )


# ===========================================================================
# Multi-trajectory generator (SEIRS-based)
# ===========================================================================

def generate_seirs_trajectories(
    params: SEIRSParams,
    forecast_dates: pd.DatetimeIndex,
    n_trajectories: int = 300,
    seed: Optional[int] = None,
    vax_multipliers: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Generate ``n_trajectories`` stochastic SEIRS forecast trajectories.

    Each trajectory is an independent draw from the stochastic SEIRS model
    (negative-binomial transmission noise).  All trajectories share the same
    calibrated parameters but differ only in their random noise realizations.

    Parameters
    ----------
    params          : Calibrated :class:`SEIRSParams`.
    forecast_dates  : DatetimeIndex of epi-week Saturday end-dates
                      (length = projection horizon in weeks).
    n_trajectories  : Number of stochastic trajectories (default 300).
    seed            : Master RNG seed; each trajectory gets a derived sub-seed
                      for reproducibility while maintaining independence.
    vax_multipliers : Optional (n_weeks,) array of vaccination protection
                      multipliers applied to S(t) before computing λ(t).
                      1.0 = no protection; values <1.0 reduce susceptibility.

    Returns
    -------
    pd.DataFrame with columns:
        ``trajectory_id`` (int32), ``date`` (datetime64), ``forecast`` (float64)
    Shape: ``n_trajectories × len(forecast_dates)`` rows.
    """
    n_weeks = len(forecast_dates)
    if n_weeks == 0:
        raise ValueError("generate_seirs_trajectories: forecast_dates must not be empty.")

    # Determine starting epi-week from the first forecast date
    start_epiweek = int(forecast_dates[0].isocalendar()[1])

    master_rng = np.random.default_rng(seed)
    sub_seeds = master_rng.integers(0, 2**31, size=n_trajectories)

    all_hosp = np.zeros((n_trajectories, n_weeks))

    for i in range(n_trajectories):
        result = run_seirs(
            params,
            n_weeks=n_weeks,
            start_epiweek=start_epiweek,
            seed=int(sub_seeds[i]),
            stochastic=True,
            vax_multipliers=vax_multipliers,
        )
        all_hosp[i, :] = result["new_hosp"]

    # Enforce minimum floor
    all_hosp = np.clip(all_hosp, a_min=MIN_HOSP_FLOOR, a_max=None)

    # Assemble tidy DataFrame
    traj_ids       = np.repeat(np.arange(1, n_trajectories + 1), n_weeks).astype(np.int32)
    dates_repeated = np.tile(forecast_dates, n_trajectories)
    forecasts_flat = all_hosp.ravel()

    result_df = pd.DataFrame({
        "trajectory_id": traj_ids,
        "date":          dates_repeated,
        "forecast":      forecasts_flat,
    })

    result_df = result_df.sort_values(["trajectory_id", "date"]).reset_index(drop=True)

    logger.info(
        "generate_seirs_trajectories [%s]: %d trajectories × %d weeks  "
        "hosp range [%.1f, %.1f]",
        params.location,
        n_trajectories,
        n_weeks,
        result_df["forecast"].min(),
        result_df["forecast"].max(),
    )
    return result_df


# ===========================================================================
# Calibration fit for plotting
# ===========================================================================

def get_calibration_fit(
    params: SEIRSParams,
    cal_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Return the deterministic SEIRS trajectory as a pd.Series indexed by *cal_dates*.

    The trajectory is the same one the forecast continues from: it begins at
    ``CAL_START`` (2024-01-01, the same start used by the warm-start in
    ``generate_trajectories``), runs deterministically to the end of *cal_dates*,
    and returns the ``new_hosp`` values only for the requested *cal_dates*.

    This ensures the displayed calibration fit line is the trajectory that the
    forecast actually continues from, so there is no visual discontinuity at the
    forecast origin.  Using a shorter starting window (e.g. only the calibration
    fitting window) produces a different trajectory with different compartment
    state at the endpoint, creating a spurious visual gap.

    Parameters
    ----------
    params    : Calibrated SEIRSParams (from get_calibrated_params).
    cal_dates : DatetimeIndex of epi-week Saturdays to display.  Typically the
                fitting window (e.g. Nov 2024 → Jun 2025).

    Returns
    -------
    pd.Series indexed by cal_dates, values = weekly incident hospitalisations.
    """
    from load_data import (  # noqa: PLC0415
        load_target_data,
        build_historical_vax_multipliers,
    )

    # Run from CAL_START so the trajectory matches the warm-start exactly
    full_obs = load_target_data(
        location_filter=[params.location],
        target_filter=["inc hosp"],
        age_group_filter=["0-130"],
        min_date=CAL_START,
        max_date=cal_dates[-1],
    ).sort_values("date")

    if full_obs.empty:
        # Fallback: run only over the requested dates (less accurate continuity)
        start_epiweek = int(cal_dates[0].isocalendar()[1])
        hist_vax = build_historical_vax_multipliers(cal_dates)
        result = run_seirs(
            params, n_weeks=len(cal_dates),
            start_epiweek=start_epiweek, stochastic=False,
            vax_multipliers=hist_vax,
        )
        return pd.Series(result["new_hosp"], index=cal_dates, name="cal_fit")

    full_dates = pd.DatetimeIndex(full_obs["date"])
    start_epiweek = int(full_dates[0].isocalendar()[1])
    hist_vax = build_historical_vax_multipliers(full_dates)

    result = run_seirs(
        params, n_weeks=len(full_dates),
        start_epiweek=start_epiweek, stochastic=False,
        vax_multipliers=hist_vax,
    )
    hosp_full = pd.Series(result["new_hosp"], index=full_dates, name="cal_fit")

    # Return only the weeks in cal_dates (overlay window for the plot)
    return hosp_full.reindex(cal_dates)


# ===========================================================================
# Location-level parameter cache
# ===========================================================================

# Module-level cache so calibration runs only once per process per location.
_PARAMS_CACHE: Dict[str, SEIRSParams] = {}


def get_calibrated_params(
    location: str = "US",
    population: Optional[int] = None,
    force_recalibrate: bool = False,
) -> SEIRSParams:
    """
    Return calibrated :class:`SEIRSParams` for *location*, using the cache
    if available.

    Parameters
    ----------
    location         : FIPS code (e.g. ``"US"``, ``"06"``).
    population       : If None, loaded from the locations CSV.
    force_recalibrate: If True, bypass the cache and re-run calibration.

    Returns
    -------
    :class:`SEIRSParams`
    """
    cache_key = location

    if cache_key in _PARAMS_CACHE and not force_recalibrate:
        return _PARAMS_CACHE[cache_key]

    # Deferred import to avoid circular dependency
    from load_data import (  # noqa: PLC0415
        load_target_data,
        get_population_map,
        build_epiweek_dates,
        build_historical_vax_multipliers,
    )

    # ── Population ───────────────────────────────────────────────────────
    if population is None:
        pop_map = get_population_map()
        population = pop_map.get(location, 330_000_000)

    # ── Calibration data — most recent 52 weeks before projection origin ──
    # Using only the last CAL_WINDOW_WEEKS keeps the fitted parameters
    # representative of the current epidemic trajectory and ensures the
    # calibration curve aligns closely with the first forecast week.
    from load_data import ORIGIN_DATE as _ORIGIN  # noqa: PLC0415

    CAL_WINDOW_WEEKS: int = 52
    # Last Saturday strictly before ORIGIN_DATE (a Sunday)
    cal_max_date = _ORIGIN - pd.Timedelta(days=1)          # Saturday 2025-06-07
    cal_min_date = cal_max_date - pd.Timedelta(weeks=CAL_WINDOW_WEEKS - 1)
    # Never request data before the global calibration start floor
    cal_min_date = max(cal_min_date, CAL_START)

    obs_df = load_target_data(
        location_filter=[location],
        target_filter=["inc hosp"],
        age_group_filter=["0-130"],
        min_date=cal_min_date,
        max_date=cal_max_date,
    ).sort_values("date")

    if obs_df.empty:
        logger.warning(
            "get_calibrated_params: no calibration data for '%s'; "
            "returning national defaults.",
            location,
        )
        params = SEIRSParams(population=population, location=location)
        _PARAMS_CACHE[cache_key] = params
        return params

    obs_hosp      = obs_df["observation"].values.astype(float)
    start_epiweek = int(obs_df["date"].iloc[0].isocalendar()[1])
    n_cal         = len(obs_hosp)

    # ── Exponential weights: last week ≈ 8× first week ───────────────────
    # weight[t] = exp(log(8) * t / (n-1))  → [1.0, …, 8.0]
    # Recent observations dominate so the fitted curve tracks the trend
    # entering the forecast, keeping the transition gap minimal.
    WEIGHT_RATIO: float = 8.0
    raw_weights = np.exp(
        np.log(WEIGHT_RATIO) * np.arange(n_cal) / max(n_cal - 1, 1)
    )

    logger.info(
        "get_calibrated_params [%s]: window %s → %s  n=%d  "
        "weights [%.2f, %.2f]",
        location,
        obs_df["date"].iloc[0].date(),
        obs_df["date"].iloc[-1].date(),
        n_cal,
        raw_weights.min(),
        raw_weights.max(),
    )

    # ── Historical vaccination multipliers ───────────────────────────────
    # Aligned to the calibration observation dates so the optimizer sees
    # the correct per-week susceptibility reduction during the 2024 and
    # 2025-26 campaigns, rather than absorbing that benefit into p_hosp.
    hist_vax_mult = build_historical_vax_multipliers(
        pd.DatetimeIndex(obs_df["date"])
    )

    # ── Calibrate ──────────────────────────────────────────────────────
    params = calibrate_seirs(
        obs_hosp=obs_hosp,
        population=population,
        start_epiweek=start_epiweek,
        location=location,
        weights=raw_weights,
        vax_multipliers=hist_vax_mult,
    )

    _PARAMS_CACHE[cache_key] = params
    return params


# ===========================================================================
# Convenience: build calibration date index
# ===========================================================================

def build_cal_dates(
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
    location: str = "US",
) -> pd.DatetimeIndex:
    """
    Return the DatetimeIndex of epi-week Saturdays within the calibration window
    for which observed data exists.

    Defaults to the same 52-week window used by ``get_calibrated_params`` so
    that the calibration fit overlay on plots exactly matches what was fitted.
    Pass explicit ``min_date`` / ``max_date`` to override.
    """
    from load_data import load_target_data, ORIGIN_DATE as _ORIGIN  # noqa: PLC0415

    if max_date is None:
        max_date = _ORIGIN - pd.Timedelta(days=1)          # 2025-06-07
    if min_date is None:
        min_date = max_date - pd.Timedelta(weeks=51)       # 52-week window
        min_date = max(min_date, CAL_START)

    obs_df = load_target_data(
        location_filter=[location],
        target_filter=["inc hosp"],
        age_group_filter=["0-130"],
        min_date=min_date,
        max_date=max_date,
    ).sort_values("date")

    return pd.DatetimeIndex(obs_df["date"].values)
