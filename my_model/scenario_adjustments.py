"""
scenario_adjustments.py
=======================
Round 20 scenario logic for the COVID-19 Scenario Modeling Hub prototype.

Overview
--------
This module provides :func:`apply_scenario`, which takes a baseline set of
stochastic trajectories (produced by ``stochastic_simulate.py``) and
returns trajectories adjusted for the vaccination and immunity assumptions of
one of the five Round 20 scenarios.

The adjustment is implemented as a **per-week multiplicative factor** applied
to the baseline forecast counts.  This is the simplest epidemiologically
coherent approach: if vaccination reduces the effective proportion of the
population susceptible to severe disease, then observed hospitalisations
scale proportionally.

Physical derivation of the multiplier
--------------------------------------
Let:
    N        = total population
    f_v(t)   = fraction newly vaccinated in week t  (from coverage curves)
    VE_0     = initial vaccine effectiveness against hospitalisation (55%)
    VE(t)    = VE_0 × waning(t) × immune_escape(t)   (effective VE at week t)

    The fraction of the population that is "protected" in week t is:
        P_protected(t) ≈ Σ_{s≤t} f_v(s) × VE(t − s)

    The hospitalisation multiplier relative to a no-vaccine scenario is:
        M_vax(t) = 1 − P_protected(t)

    The full weekly multiplier applied to trajectories is:
        M(t) = M_vax(t)

Because the baseline trajectories already reflect the 2025-26 observed
vaccination campaign (Scenarios A-E share the same first campaign), the
adjustment models only the *incremental* effect of the 2026-27 fall campaign
and the hypothetical 2026 spring campaign relative to scenario A
(no further vaccination after Feb 2026).

All five scenarios share identical 2025-26 observed vaccination data.
The differences between scenarios only manifest from June 2026 onward:

    Scenario A  No additional vaccination after Feb 2026.  Multiplier = 1.0
                for all weeks.  This is the baseline reference.

    Scenario B  Annual fall 2026-27 campaign, business-as-usual (BaU) coverage
                (same coverage fractions as observed 2025-26 ~33% for 65+).
                Fall campaign ramps Aug–Oct 2026 to Feb 2027.

    Scenario C  Same fall coverage as B *plus* a spring 2026 campaign for
                high-risk individuals only (≈ half the fall coverage peak).
                Spring campaign: mid-Feb 2026 → mid-Aug 2026.
                Spring VE is discounted for immune escape since the vaccine
                was not reformulated for the spring.

    Scenario D  Annual fall 2026-27 campaign, optimistic coverage
                (flu-vaccine-level, ~59% for 65+).

    Scenario E  Same fall coverage as D *plus* the spring 2026 high-risk
                campaign at half the optimistic fall peak.

Documented assumptions
-----------------------
VE_HOSP_INITIAL = 0.55
    Vaccine effectiveness against hospitalisation at start of fall campaign.
    Source: round20.md, citing US veterans (55%), Canadian seniors (53%),
    European seniors (59%) analyses from Sep–Dec 2025.

WANING_HALF_LIFE_WEEKS = 26
    Median waning time of 6 months (within the 3–10 month range mandated by
    round20.md). Protection decays exponentially from VE_0 toward a waned
    floor at rate ln(2)/half_life.  Using continuous exponential waning as a
    smooth, computationally cheap approximation.

WANED_FLOOR = 0.50
    Waned individuals retain 50% of initial VE (midpoint of the 40–60%
    plateau range given in round20.md: "40–60% reduction → residual 60–40%";
    using midpoint 50%).
    Effective residual VE = VE_0 × WANED_FLOOR.

IMMUNE_ESCAPE_PER_YEAR = 0.35
    35% immune escape per year, the midpoint of the recommended 20–50%
    per-year range (round20.md: "it is also acceptable to use the midpoint of
    the recommended immune escape bounds").  Applied to both vaccine-induced
    and infection-induced immunity.

SPRING_VE_DISCOUNT = 0.70
    Spring 2026 campaign uses the prior season's vaccine formulation (not
    reformulated for June 2026 strains). By the spring campaign start
    (mid-Feb 2026), roughly 8 months of immune escape at 35%/year have
    elapsed since the vaccine was matched (June 2025 reformulation).
    Discount = (1 - 0.35)^(8/12) ≈ 0.78; we round down to 0.70 to be
    conservative (the exact reduction is stated as "teams' discretion" in
    round20.md).

HIGH_RISK_POPULATION_FRACTION = 0.30
    Approximate fraction of the US population classified as high-risk
    (chronic conditions).  Source: round20.md cites CID 2021 data; CDC
    estimates ~29–32% of US adults have an underlying condition.
    Used to weight the spring campaign impact, which applies only to
    high-risk individuals.

COVERAGE_BAU_FALL = 0.33
    Peak cumulative coverage fraction for fall 2026-27 under BaU scenarios
    (B, C). Taken directly from the vaccination coverage curves data:
    maximum national coverage for 65+ in A-2026-05-11 / B-2026-05-11 is
    ~33.0%.  Used as a single representative national number; state-level
    models should use the state curves directly.

COVERAGE_OPT_FALL = 0.59
    Peak cumulative coverage for fall 2026-27 under optimistic scenarios
    (D, E). Taken from D/E-2026-05-11 data for 65+ (~59.0%).

COVERAGE_SPRING_BAU = 0.165
    Peak cumulative spring 2026 coverage for high-risk under Scenario C.
    Equals half of COVERAGE_BAU_FALL (~0.33 / 2 ≈ 0.165), consistent with
    round20.md: "saturating at only half of the main campaign".

COVERAGE_SPRING_OPT = 0.295
    Peak cumulative spring 2026 coverage for high-risk under Scenario E.
    Equals half of COVERAGE_OPT_FALL (~0.59 / 2 ≈ 0.295).

References
----------
- Round 20 scenario description: auxiliary-data/rounds/round20.md
- Vaccination coverage data:
  auxiliary-data/vaccination-coverage/COVID_RD20_Vaccination_curves.csv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ===========================================================================
# Scenario registry
# ===========================================================================

ScenarioKey = Literal["A", "B", "C", "D", "E"]

SCENARIO_IDS: Dict[str, str] = {
    "A": "A-2026-05-11",
    "B": "B-2026-05-11",
    "C": "C-2026-05-11",
    "D": "D-2026-05-11",
    "E": "E-2026-05-11",
}

SCENARIO_NAMES: Dict[str, str] = {
    "A": "noVax (counterfactual)",
    "B": "ModCov_AnnualVax",
    "C": "ModCov_SemiannualHRVax",
    "D": "OptCov_AnnualVax",
    "E": "OptCov_SemiannualHRVax",
}

# ===========================================================================
# Epidemiological constants  (all sourced from round20.md, documented above)
# ===========================================================================

VE_HOSP_INITIAL: float = 0.55
"""VE against hospitalisation at start of each fall campaign (round20.md)."""

WANING_HALF_LIFE_WEEKS: float = 26.0
"""Exponential waning half-life = 6 months (midpoint of 3–10-month range)."""

WANED_FLOOR: float = 0.50
"""Residual fraction of VE retained in the fully-waned state (plateau = 50%)."""

IMMUNE_ESCAPE_PER_YEAR: float = 0.35
"""Annual immune escape = 35%, midpoint of 20–50% range (round20.md)."""

SPRING_VE_DISCOUNT: float = 0.70
"""
Discount applied to VE for the spring 2026 campaign because the vaccine
was not reformulated for summer 2026 strains (conservative 0.70; see module
docstring for derivation).
"""

HIGH_RISK_POPULATION_FRACTION: float = 0.30
"""Fraction of total population that is high-risk (CDC data, ~29–32%)."""

# Coverage fractions (derived from vaccination-coverage data; see module docstring)
COVERAGE_BAU_FALL: float = 0.330
"""BaU peak fall 2026-27 cumulative coverage (Scenarios B, C)."""

COVERAGE_OPT_FALL: float = 0.590
"""Optimistic peak fall 2026-27 cumulative coverage (Scenarios D, E)."""

COVERAGE_SPRING_BAU: float = 0.165
"""BaU spring 2026 peak coverage for high-risk groups (Scenario C)."""

COVERAGE_SPRING_OPT: float = 0.295
"""Optimistic spring 2026 peak coverage for high-risk groups (Scenario E)."""

# Campaign timing windows (from round20.md)
_FALL_2026_START = pd.Timestamp("2026-08-14")
_FALL_2026_END   = pd.Timestamp("2027-02-13")
_SPRING_2026_START = pd.Timestamp("2026-02-15")
_SPRING_2026_END   = pd.Timestamp("2026-08-13")


# ===========================================================================
# Dataclass: ScenarioSpec
# ===========================================================================

@dataclass(frozen=True)
class ScenarioSpec:
    """
    Immutable specification of the vaccination assumptions for one scenario.

    Attributes
    ----------
    key              : Single-letter identifier ("A" … "E").
    scenario_id      : Official hub submission ID (e.g. "B-2026-05-11").
    name             : Human-readable name.
    has_fall_2026    : Whether the scenario includes a fall 2026-27 campaign.
    fall_coverage    : Peak cumulative fall 2026-27 coverage fraction (0–1).
    has_spring_2026  : Whether the scenario adds a spring 2026 campaign for
                       high-risk individuals.
    spring_coverage  : Peak spring 2026 coverage for high-risk (0–1).
                       Zero if has_spring_2026 is False.
    spring_ve_discount: Multiplicative discount on VE for spring vaccine
                        (not reformulated).
    notes            : Free-text summary of the scenario assumptions.
    """

    key: str
    scenario_id: str
    name: str
    has_fall_2026: bool
    fall_coverage: float
    has_spring_2026: bool
    spring_coverage: float
    spring_ve_discount: float
    notes: str


# Build the five scenario specs from constants
SCENARIO_SPECS: Dict[str, ScenarioSpec] = {
    "A": ScenarioSpec(
        key="A",
        scenario_id="A-2026-05-11",
        name="noVax (counterfactual)",
        has_fall_2026=False,
        fall_coverage=0.0,
        has_spring_2026=False,
        spring_coverage=0.0,
        spring_ve_discount=1.0,
        notes=(
            "No further vaccination after mid-Feb 2026. "
            "The 2025-26 campaign is included in the shared baseline. "
            "All scenario-A trajectories are the unmodified baseline forecasts "
            "and serve as the reference against which B–E are compared."
        ),
    ),
    "B": ScenarioSpec(
        key="B",
        scenario_id="B-2026-05-11",
        name="ModCov_AnnualVax",
        has_fall_2026=True,
        fall_coverage=COVERAGE_BAU_FALL,
        has_spring_2026=False,
        spring_coverage=0.0,
        spring_ve_discount=1.0,
        notes=(
            "Annual fall 2026-27 campaign at business-as-usual (BaU) coverage "
            f"(peak ≈ {COVERAGE_BAU_FALL:.0%}), same as observed 2025-26 uptake. "
            "Campaign ramps Aug 2026 → Feb 2027. "
            "Vaccine reformulated in June 2026 (VE_0 = 55% at campaign start)."
        ),
    ),
    "C": ScenarioSpec(
        key="C",
        scenario_id="C-2026-05-11",
        name="ModCov_SemiannualHRVax",
        has_fall_2026=True,
        fall_coverage=COVERAGE_BAU_FALL,
        has_spring_2026=True,
        spring_coverage=COVERAGE_SPRING_BAU,
        spring_ve_discount=SPRING_VE_DISCOUNT,
        notes=(
            "BaU annual fall 2026-27 campaign (same as B), PLUS a hypothetical "
            "spring 2026 campaign for high-risk individuals only. "
            f"Spring peak coverage ≈ {COVERAGE_SPRING_BAU:.1%} "
            f"(half of fall BaU peak), starting mid-Feb 2026. "
            f"Spring VE discounted to {SPRING_VE_DISCOUNT:.0%} of VE_0 because "
            "the vaccine was not reformulated for spring 2026 strains. "
            f"Applies to {HIGH_RISK_POPULATION_FRACTION:.0%} of population."
        ),
    ),
    "D": ScenarioSpec(
        key="D",
        scenario_id="D-2026-05-11",
        name="OptCov_AnnualVax",
        has_fall_2026=True,
        fall_coverage=COVERAGE_OPT_FALL,
        has_spring_2026=False,
        spring_coverage=0.0,
        spring_ve_discount=1.0,
        notes=(
            "Annual fall 2026-27 campaign at optimistic coverage "
            f"(peak ≈ {COVERAGE_OPT_FALL:.0%}), aspirational flu-vaccine level "
            "(2024-25 flu season used as reference). "
            "Vaccine reformulated in June 2026 (VE_0 = 55%)."
        ),
    ),
    "E": ScenarioSpec(
        key="E",
        scenario_id="E-2026-05-11",
        name="OptCov_SemiannualHRVax",
        has_fall_2026=True,
        fall_coverage=COVERAGE_OPT_FALL,
        has_spring_2026=True,
        spring_coverage=COVERAGE_SPRING_OPT,
        spring_ve_discount=SPRING_VE_DISCOUNT,
        notes=(
            "Optimistic annual fall 2026-27 campaign (same as D), PLUS spring "
            "2026 high-risk campaign. "
            f"Spring peak coverage ≈ {COVERAGE_SPRING_OPT:.1%} "
            f"(half of fall optimistic peak). "
            f"Spring VE discounted to {SPRING_VE_DISCOUNT:.0%} of VE_0."
        ),
    ),
}


# ===========================================================================
# Waning / immune-escape functions
# ===========================================================================

def effective_ve(
    weeks_since_vaccination: np.ndarray,
    ve_initial: float = VE_HOSP_INITIAL,
    waning_half_life_weeks: float = WANING_HALF_LIFE_WEEKS,
    waned_floor: float = WANED_FLOOR,
    immune_escape_per_year: float = IMMUNE_ESCAPE_PER_YEAR,
) -> np.ndarray:
    """
    Compute effective vaccine effectiveness as a function of time since
    vaccination, combining waning immunity and immune escape.

    Model
    -----
    Two independent multiplicative factors reduce VE over time:

    1. **Waning** (antibody decay):
       VE decays exponentially from VE_initial toward a plateau
       (VE_initial × waned_floor) with half-life = waning_half_life_weeks.

       waning_factor(τ) = waned_floor + (1 - waned_floor) × exp(-λ τ)
       where λ = ln(2) / waning_half_life_weeks

       Assumption: exponential decay is a standard pharmacokinetic model for
       antibody decline; the plateau reflects long-lived memory B-cells
       providing durable but reduced protection.

    2. **Immune escape** (antigenic drift):
       Strains continuously drift away from the vaccine strain at a constant
       fractional rate of ``immune_escape_per_year`` per year.

       escape_factor(τ) = (1 - immune_escape_per_year) ^ (τ / 52)

       Assumption: linear accumulation of immune escape in log-space, which
       is consistent with the thought experiment in round20.md.

    Combined effective VE:
       VE(τ) = VE_initial × waning_factor(τ) × escape_factor(τ)

    Parameters
    ----------
    weeks_since_vaccination : Array of non-negative integers/floats (weeks).
    ve_initial              : VE at τ=0 (default 0.55).
    waning_half_life_weeks  : Exponential waning half-life (default 26 weeks).
    waned_floor             : Residual fraction of VE at full waning (0.50).
    immune_escape_per_year  : Annual immune escape fraction (default 0.35).

    Returns
    -------
    np.ndarray, same shape as ``weeks_since_vaccination``.
        Effective VE values in [0, ve_initial].
    """
    tau = np.asarray(weeks_since_vaccination, dtype=float)
    tau = np.clip(tau, 0.0, None)

    # Waning factor: decays from 1.0 toward waned_floor
    lam = np.log(2.0) / waning_half_life_weeks
    waning_factor = waned_floor + (1.0 - waned_floor) * np.exp(-lam * tau)

    # Immune escape factor: continuous antigenic drift
    escape_factor = (1.0 - immune_escape_per_year) ** (tau / 52.0)

    return ve_initial * waning_factor * escape_factor


def build_protection_schedule(
    forecast_dates: pd.DatetimeIndex,
    campaign_start: pd.Timestamp,
    peak_coverage: float,
    ve_initial: float = VE_HOSP_INITIAL,
    waning_half_life_weeks: float = WANING_HALF_LIFE_WEEKS,
    waned_floor: float = WANED_FLOOR,
    immune_escape_per_year: float = IMMUNE_ESCAPE_PER_YEAR,
    population_weight: float = 1.0,
) -> np.ndarray:
    """
    Compute the per-week fraction of the population protected by one
    vaccination campaign, accounting for waning and immune escape.

    Modelling approach
    ------------------
    The campaign ramps linearly from 0% to ``peak_coverage`` over 26 weeks
    (6 months), consistent with the observed ramp shape in the vaccination
    coverage curves.  Each vaccinated individual's protection then wanes
    and is eroded by immune escape over subsequent weeks.

    The population-level protection at week t is the convolution:
        P(t) = Σ_{s = campaign_start}^{t} [newly_vax(s) × VE(t − s)]

    where newly_vax(s) = Δ(coverage(s)) and coverage follows a saturating
    linear ramp capped at peak_coverage.

    Parameters
    ----------
    forecast_dates       : DatetimeIndex of epi-week end-dates.
    campaign_start       : Date vaccination campaign begins.
    peak_coverage        : Cumulative coverage fraction at saturation (0–1).
    ve_initial           : VE_0 (default 0.55).
    waning_half_life_weeks: (default 26).
    waned_floor          : (default 0.50).
    immune_escape_per_year: (default 0.35).
    population_weight    : Fraction of total population targeted by this
                           campaign (1.0 = whole population, use
                           HIGH_RISK_POPULATION_FRACTION for high-risk-only).

    Returns
    -------
    np.ndarray, shape (len(forecast_dates),)
        Weekly population-level protection fraction P(t), i.e. the fraction
        of the *total* population that is protected in each week.
    """
    n = len(forecast_dates)
    protection = np.zeros(n)

    # Build a 26-week linear ramp that saturates at peak_coverage
    RAMP_WEEKS = 26
    weekly_vax = np.zeros(n)

    for i, date in enumerate(forecast_dates):
        weeks_into_campaign = int((date - campaign_start).days / 7)
        if 0 <= weeks_into_campaign < RAMP_WEEKS:
            # Weekly newly vaccinated fraction = peak / ramp_length
            weekly_vax[i] = (peak_coverage / RAMP_WEEKS) * population_weight
        # After ramp_weeks, coverage is saturated; no additional vaccination

    # Convolution: for each week t, sum up contributions from all vaccinated
    # cohorts s ≤ t, each decayed by their age (t − s weeks post-vaccination)
    for t in range(n):
        for s in range(t + 1):
            if weekly_vax[s] > 0:
                tau = t - s  # weeks since cohort s was vaccinated
                ve_t = effective_ve(
                    np.array([tau]),
                    ve_initial=ve_initial,
                    waning_half_life_weeks=waning_half_life_weeks,
                    waned_floor=waned_floor,
                    immune_escape_per_year=immune_escape_per_year,
                )[0]
                protection[t] += weekly_vax[s] * ve_t

    # Clip to physical bounds [0, 1]
    return np.clip(protection, 0.0, 1.0)


def _build_protection_vectorised(
    forecast_dates: pd.DatetimeIndex,
    campaign_start: pd.Timestamp,
    peak_coverage: float,
    ve_initial: float = VE_HOSP_INITIAL,
    waning_half_life_weeks: float = WANING_HALF_LIFE_WEEKS,
    waned_floor: float = WANED_FLOOR,
    immune_escape_per_year: float = IMMUNE_ESCAPE_PER_YEAR,
    population_weight: float = 1.0,
) -> np.ndarray:
    """
    Vectorised version of :func:`build_protection_schedule` using NumPy
    broadcasting for efficiency.  Identical semantics; used internally.
    """
    n = len(forecast_dates)
    RAMP_WEEKS = 26

    # Build weekly newly-vaccinated array
    weekly_vax = np.zeros(n)
    for i, date in enumerate(forecast_dates):
        weeks_into_campaign = int((date - campaign_start).days / 7)
        if 0 <= weeks_into_campaign < RAMP_WEEKS:
            weekly_vax[i] = (peak_coverage / RAMP_WEEKS) * population_weight

    # s_idx: indices with non-zero vaccination
    s_indices = np.where(weekly_vax > 0)[0]
    if len(s_indices) == 0:
        return np.zeros(n)

    protection = np.zeros(n)
    for s in s_indices:
        # tau[t] = max(t - s, 0) for all t >= s
        t_range = np.arange(s, n)
        tau = t_range - s  # shape (n - s,)
        ve_vals = effective_ve(
            tau,
            ve_initial=ve_initial,
            waning_half_life_weeks=waning_half_life_weeks,
            waned_floor=waned_floor,
            immune_escape_per_year=immune_escape_per_year,
        )
        protection[s:] += weekly_vax[s] * ve_vals

    return np.clip(protection, 0.0, 1.0)


# ===========================================================================
# Multiplier builder
# ===========================================================================

def build_scenario_multipliers(
    forecast_dates: pd.DatetimeIndex,
    spec: ScenarioSpec,
    ve_initial: float = VE_HOSP_INITIAL,
    waning_half_life_weeks: float = WANING_HALF_LIFE_WEEKS,
    waned_floor: float = WANED_FLOOR,
    immune_escape_per_year: float = IMMUNE_ESCAPE_PER_YEAR,
) -> np.ndarray:
    """
    Build the per-week hospitalisation multiplier for a scenario.

    The multiplier M(t) represents the ratio of expected hospitalisations
    under the scenario to expected hospitalisations under Scenario A
    (no further vaccination after Feb 2026).

    Derivation
    ----------
    For a scenario with only a fall 2026-27 campaign:
        P_fall(t)    = population protection from fall campaign (see
                       :func:`_build_protection_vectorised`)
        M(t)         = 1 − P_fall(t)

    For scenarios C and E with an additional spring 2026 campaign:
        P_spring(t)  = protection from spring campaign (high-risk only,
                       discounted VE)
        M(t)         = 1 − P_fall(t) − P_spring(t)

    Before the fall 2026-27 campaign starts (dates < Aug 2026), M(t) = 1.0
    for Scenario A and for the spring-only contribution in C/E.

    The multiplier is bounded to [0.1, 1.0]:
    - Upper bound 1.0: scenario cannot increase hospitalisations above baseline.
    - Lower bound 0.1: prevents unrealistically large reductions; even at 90%
      vaccination coverage and 55% VE, ~5% of events are in vaccinated
      individuals (breakthrough), so a true floor near zero is unphysical in
      a simplified model.

    Parameters
    ----------
    forecast_dates          : DatetimeIndex of epi-week end-dates (104 weeks).
    spec                    : :class:`ScenarioSpec` for the target scenario.
    ve_initial              : VE_0 (default 0.55).
    waning_half_life_weeks  : (default 26).
    waned_floor             : (default 0.50).
    immune_escape_per_year  : (default 0.35).

    Returns
    -------
    np.ndarray, shape (len(forecast_dates),)
        Weekly multiplier values in [0.1, 1.0].
        1.0 means "no vaccine effect in this week" (equals scenario A).
    """
    n = len(forecast_dates)
    multiplier = np.ones(n)

    # ── Scenario A: reference case – no modification ───────────────────────
    if spec.key == "A":
        logger.debug("build_scenario_multipliers: Scenario A – multiplier = 1.0")
        return multiplier

    # ── Fall 2026-27 campaign (Scenarios B, C, D, E) ──────────────────────
    if spec.has_fall_2026:
        p_fall = _build_protection_vectorised(
            forecast_dates=forecast_dates,
            campaign_start=_FALL_2026_START,
            peak_coverage=spec.fall_coverage,
            ve_initial=ve_initial,
            waning_half_life_weeks=waning_half_life_weeks,
            waned_floor=waned_floor,
            immune_escape_per_year=immune_escape_per_year,
            population_weight=1.0,  # whole population eligible
        )
        multiplier -= p_fall
        logger.debug(
            "build_scenario_multipliers: scenario %s fall protection max=%.3f",
            spec.key, p_fall.max(),
        )

    # ── Spring 2026 high-risk campaign (Scenarios C, E) ───────────────────
    if spec.has_spring_2026:
        p_spring = _build_protection_vectorised(
            forecast_dates=forecast_dates,
            campaign_start=_SPRING_2026_START,
            peak_coverage=spec.spring_coverage,
            # VE is discounted for the spring campaign (not reformulated)
            ve_initial=ve_initial * spec.spring_ve_discount,
            waning_half_life_weeks=waning_half_life_weeks,
            waned_floor=waned_floor,
            immune_escape_per_year=immune_escape_per_year,
            # Only high-risk individuals are targeted
            population_weight=HIGH_RISK_POPULATION_FRACTION,
        )
        multiplier -= p_spring
        logger.debug(
            "build_scenario_multipliers: scenario %s spring protection max=%.3f",
            spec.key, p_spring.max(),
        )

    # ── Apply physical bounds ──────────────────────────────────────────────
    # Lower bound 0.50: the maximum combined protection achievable under any
    # Round 20 scenario is capped at ~50% of baseline hospitalisations.
    # Rationale: peak fall coverage ~59% (Scenario D/E), VE_0=55%, but at
    # the peak of the campaign only ~half the newly-vaccinated cohort has been
    # vaccinated ≤ 6 weeks (early ramp); the population-level effective VE
    # is therefore at most ~0.59 × 0.55 × 0.90 ≈ 29%, leaving ≥71% of
    # events.  The spring high-risk add-on covers only 30% of population at
    # half the coverage → adds at most ~5% absolute protection.  In practice
    # the multiplier stays well above 0.50; this floor simply prevents
    # numerical artefacts from driving near-zero trajectories.
    multiplier = np.clip(multiplier, 0.50, 1.0)

    logger.info(
        "build_scenario_multipliers: scenario %s  "
        "multiplier range [%.3f, %.3f]",
        spec.key, multiplier.min(), multiplier.max(),
    )
    return multiplier


# ===========================================================================
# Public API: apply_scenario
# ===========================================================================

def apply_scenario(
    trajectories: pd.DataFrame,
    scenario: str,
    *,
    forecast_col: str = "forecast",
    date_col: str = "date",
    trajectory_id_col: str = "trajectory_id",
    ve_initial: float = VE_HOSP_INITIAL,
    waning_half_life_weeks: float = WANING_HALF_LIFE_WEEKS,
    waned_floor: float = WANED_FLOOR,
    immune_escape_per_year: float = IMMUNE_ESCAPE_PER_YEAR,
) -> pd.DataFrame:
    """
    Apply Round 20 scenario vaccination adjustments to stochastic trajectories.

    The function takes a baseline set of trajectories (typically the output of
    :func:`stochastic_simulate.generate_trajectories`) and returns a new
    DataFrame with the ``forecast`` column multiplied by the scenario's
    per-week hospitalisation multiplier.

    The baseline trajectories are assumed to represent Scenario A (no further
    vaccination after Feb 2026).  For Scenario A itself, the DataFrame is
    returned unchanged (multiplier = 1.0 for all weeks).

    Parameters
    ----------
    trajectories      : DataFrame with columns ``trajectory_id``, ``date``,
                        and ``forecast`` (plus any extra columns, which are
                        preserved unchanged).  Typically the output of
                        :func:`~stochastic_simulate.generate_trajectories`.
    scenario          : One of "A", "B", "C", "D", "E".
    forecast_col      : Name of the column containing forecast values
                        (default ``"forecast"``).
    date_col          : Name of the date column (default ``"date"``).
    trajectory_id_col : Name of the trajectory ID column
                        (default ``"trajectory_id"``).
    ve_initial        : Override VE_0 (default 0.55).
    waning_half_life_weeks: Override waning half-life (default 26 weeks).
    waned_floor       : Override waned plateau fraction (default 0.50).
    immune_escape_per_year: Override annual immune escape (default 0.35).

    Returns
    -------
    pd.DataFrame
        Same schema as ``trajectories``.  The ``forecast`` column is the
        adjusted (multiplied) value.  A new column ``multiplier`` is added
        so that callers can inspect the adjustment applied each week.
        The ``scenario_id`` column is added with the official hub scenario ID.

    Raises
    ------
    ValueError
        If ``scenario`` is not one of "A"–"E", or if required columns are
        missing from ``trajectories``.

    Notes
    -----
    - The same multiplier vector is applied uniformly across all 300
      trajectories for a given week (the multiplier is deterministic given
      the scenario spec and epidemiological parameters).  Stochastic
      variation between trajectories arises solely from the baseline random
      walk (Assumption A5 in ``stochastic_simulate.py``).
    - To produce properly paired scenario submissions, call
      ``generate_trajectories`` with the same random seed for all scenarios,
      then pass each resulting DataFrame through ``apply_scenario``.

    Examples
    --------
    >>> from stochastic_simulate import generate_trajectories
    >>> from load_data import build_epiweek_dates, ORIGIN_DATE
    >>> import numpy as np
    >>>
    >>> dates = build_epiweek_dates(ORIGIN_DATE)
    >>> obs = np.array([7000, 6000, 5000, 4000, 3000, 2000, 1500, 955])
    >>> base = generate_trajectories(obs, dates, seed=42)
    >>>
    >>> scen_b = apply_scenario(base, "B")
    >>> scen_b[["date", "trajectory_id", "forecast", "multiplier"]].head(3)
    """
    scenario = scenario.upper()
    if scenario not in SCENARIO_SPECS:
        raise ValueError(
            f"apply_scenario: unknown scenario '{scenario}'. "
            f"Valid values: {sorted(SCENARIO_SPECS.keys())}"
        )

    required_cols = {forecast_col, date_col, trajectory_id_col}
    missing = required_cols - set(trajectories.columns)
    if missing:
        raise ValueError(
            f"apply_scenario: trajectories DataFrame missing column(s): {missing}"
        )

    spec = SCENARIO_SPECS[scenario]
    logger.info(
        "apply_scenario: applying scenario %s (%s)  rows=%d",
        scenario, spec.name, len(trajectories),
    )

    # ── Build the multiplier schedule ─────────────────────────────────────
    # We need the unique sorted forecast dates to build the multiplier array.
    forecast_dates = pd.DatetimeIndex(
        sorted(trajectories[date_col].unique())
    )

    multipliers_arr = build_scenario_multipliers(
        forecast_dates=forecast_dates,
        spec=spec,
        ve_initial=ve_initial,
        waning_half_life_weeks=waning_half_life_weeks,
        waned_floor=waned_floor,
        immune_escape_per_year=immune_escape_per_year,
    )

    # Build a date → multiplier mapping for fast join
    mult_map = pd.Series(multipliers_arr, index=forecast_dates, name="multiplier")

    # ── Apply multipliers ─────────────────────────────────────────────────
    result = trajectories.copy()
    result["multiplier"] = result[date_col].map(mult_map)

    # Safety check: all dates must have a multiplier
    n_missing = result["multiplier"].isna().sum()
    if n_missing > 0:
        logger.warning(
            "apply_scenario: %d rows have no multiplier (dates outside forecast "
            "window?). Setting multiplier=1.0 for those rows.",
            n_missing,
        )
        result["multiplier"] = result["multiplier"].fillna(1.0)

    # Apply multiplier; do NOT impose a hard absolute floor here — the
    # trajectory generator already enforces DEFAULT_MIN_COUNT (200 for US
    # national) via its own clip.  Imposing clip(lower=1) here was masking
    # the per-scenario amplitude differences.
    result[forecast_col] = (result[forecast_col] * result["multiplier"]).clip(lower=0.0)

    # Attach official scenario metadata
    result["scenario_id"] = spec.scenario_id
    result["scenario_name"] = spec.name

    return result


# ===========================================================================
# Convenience: apply all five scenarios at once
# ===========================================================================

def apply_all_scenarios(
    baseline_trajectories: pd.DataFrame,
    **apply_scenario_kwargs,
) -> pd.DataFrame:
    """
    Apply all five Round 20 scenarios to a common set of baseline trajectories
    and return a single concatenated DataFrame.

    Parameters
    ----------
    baseline_trajectories : Base trajectories (Scenario A / no-vax baseline).
    **apply_scenario_kwargs: Passed through to :func:`apply_scenario`
                             (e.g. ``ve_initial``, ``waning_half_life_weeks``).

    Returns
    -------
    pd.DataFrame
        All five scenarios stacked vertically.  Column ``scenario_id``
        identifies the scenario for each row.

    Notes
    -----
    All scenarios are produced from the same baseline trajectories, ensuring
    the paired-trajectory requirement from round20.md (identical stochastic
    realisation in each scenario except for the multiplier).
    """
    frames: List[pd.DataFrame] = []
    for key in ("A", "B", "C", "D", "E"):
        adjusted = apply_scenario(
            baseline_trajectories, key, **apply_scenario_kwargs
        )
        frames.append(adjusted)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "apply_all_scenarios: combined %d rows across 5 scenarios.", len(combined)
    )
    return combined


# ===========================================================================
# Scenario summary table (for logging / reporting)
# ===========================================================================

def scenario_summary_table() -> pd.DataFrame:
    """
    Return a human-readable summary DataFrame of all five scenario specs.

    Returns
    -------
    pd.DataFrame
        One row per scenario with columns for all key parameters.
    """
    rows = []
    for spec in SCENARIO_SPECS.values():
        rows.append(
            {
                "scenario_key": spec.key,
                "scenario_id": spec.scenario_id,
                "name": spec.name,
                "has_fall_2026_campaign": spec.has_fall_2026,
                "fall_peak_coverage": f"{spec.fall_coverage:.1%}",
                "has_spring_2026_campaign": spec.has_spring_2026,
                "spring_peak_coverage_highrisk": f"{spec.spring_coverage:.1%}",
                "spring_ve_discount": f"{spec.spring_ve_discount:.0%}",
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# CLI smoke-test
# ===========================================================================
if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s – %(message)s",
    )

    import numpy as np
    from load_data import build_epiweek_dates, ORIGIN_DATE
    from stochastic_simulate import generate_trajectories

    print("=" * 65)
    print("Scenario summary")
    print("=" * 65)
    print(scenario_summary_table().to_string(index=False))

    # ── Build baseline trajectories ───────────────────────────────────────
    obs = np.array([7000, 6500, 6000, 5500, 4500, 3500, 2500, 955], dtype=float)
    dates = build_epiweek_dates(ORIGIN_DATE, n_weeks=104)

    print("\nGenerating 300 baseline trajectories …")
    base = generate_trajectories(obs, dates, seed=42, n_trajectories=300)

    # ── Apply all scenarios ───────────────────────────────────────────────
    print("\nApplying scenarios A–E …")
    all_scen = apply_all_scenarios(base)

    print(f"\nCombined shape: {all_scen.shape}")
    print(f"Columns: {all_scen.columns.tolist()}")

    # ── Compare median trajectories per scenario ──────────────────────────
    print("\n── Median weekly forecast by scenario (weeks 35–39, summer 2026) ──")
    summer = all_scen[
        (all_scen["date"] >= pd.Timestamp("2026-07-01")) &
        (all_scen["date"] <= pd.Timestamp("2026-08-15"))
    ]
    med = (
        summer.groupby(["scenario_id", "date"])["forecast"]
        .median()
        .reset_index()
        .pivot(index="date", columns="scenario_id", values="forecast")
    )
    print(med.to_string())

    print("\n── Median weekly forecast by scenario (weeks 70–74, winter 2026-27) ──")
    winter = all_scen[
        (all_scen["date"] >= pd.Timestamp("2026-12-01")) &
        (all_scen["date"] <= pd.Timestamp("2027-01-15"))
    ]
    med_w = (
        winter.groupby(["scenario_id", "date"])["forecast"]
        .median()
        .reset_index()
        .pivot(index="date", columns="scenario_id", values="forecast")
    )
    print(med_w.to_string())

    print("\n── Multiplier for scenario B (fall 2026 campaign window) ──")
    spec_b = SCENARIO_SPECS["B"]
    mults = build_scenario_multipliers(dates, spec_b)
    mult_df = pd.DataFrame({"date": dates, "multiplier_B": mults})
    active = mult_df[mult_df["multiplier_B"] < 0.999]
    print(active.to_string(index=False))

    print("\n✓ Smoke-test passed.")
