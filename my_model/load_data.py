"""
load_data.py
============
Production-quality data-loading layer for the Round 20 COVID-19 Scenario
Modeling Hub prototype.

Responsibilities
----------------
1. Load and validate ``target-data/time-series.csv`` (weekly inc hosp /
   inc death observations).
2. Load location metadata (FIPS codes, state populations).
3. Load Round 20 vaccination coverage curves.
4. Provide calibration-data and epi-week helpers consumed by the simulator.

All paths are resolved relative to the repository root so the module works
regardless of the current working directory.

Observed-data schema (``time-series.csv``)
------------------------------------------
Column       dtype     Notes
-----------  --------  -------------------------------------------------------
location     str       FIPS code ("US", "01" … "78")
date         str→date  Saturday end-of-epi-week (ISO 8601)
observation  int       Weekly incident count (non-negative integer)
age_group    str       "0-130" | "0-64" | "65-130"
target       str       "inc death" | "inc hosp"

Round 20 temporal scope (source: auxiliary-data/rounds/round20.md)
------------------------------------------------------------------
- Projection period : 2025-06-08 → 2027-06-05 (104 epi-weeks, Sun–Sat)
- Calibration cutoff: 2026-06-06 (last Saturday with complete data)
- Scenario IDs       : A-2026-05-11 … E-2026-05-11
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository root – two levels above my_model/load_data.py
# ---------------------------------------------------------------------------
REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
_TARGET_DATA_PATH = REPO_ROOT / "target-data" / "time-series.csv"
_LOCATIONS_PATH = REPO_ROOT / "auxiliary-data" / "data-locations" / "locations.csv"
_VAX_CURVES_PATH = (
    REPO_ROOT
    / "auxiliary-data"
    / "vaccination-coverage"
    / "COVID_RD20_Vaccination_curves.csv"
)

# Round 19 vaccination curves — "Historic coverage" scenario contains
# observed Aug–Dec 2024 uptake from the CDC National Immunization Survey.
_VAX_CURVES_RD19_PATH = (
    REPO_ROOT
    / "auxiliary-data"
    / "vaccination-coverage"
    / "COVID_RD19_Vaccination_curves.csv"
)
_VAX_CURVES_RD19_REQUIRED_COLS: List[str] = [
    "Geography", "Age", "Risk_group", "Pop", "Cum.Coverage.Percent",
    "Scenario", "Date",
]

# ---------------------------------------------------------------------------
# Required columns for each source file (validated on load)
# ---------------------------------------------------------------------------
_TARGET_DATA_REQUIRED_COLS: List[str] = [
    "location", "date", "observation", "age_group", "target",
]
_LOCATIONS_REQUIRED_COLS: List[str] = [
    "location", "location_name", "population",
]
_VAX_CURVES_REQUIRED_COLS: List[str] = [
    "geography", "age_group", "date", "coverage", "population", "scenario",
]

# ---------------------------------------------------------------------------
# Accepted domain values (used for validation warnings)
# ---------------------------------------------------------------------------
VALID_TARGETS: frozenset[str] = frozenset({"inc death", "inc hosp"})
VALID_AGE_GROUPS: frozenset[str] = frozenset({"0-130", "0-64", "65-130"})

# ---------------------------------------------------------------------------
# Round 20 temporal constants (source: round20.md)
# ---------------------------------------------------------------------------
# Projection period
ORIGIN_DATE: pd.Timestamp = pd.Timestamp("2025-06-08")   # first Sunday
SIM_END_DATE: pd.Timestamp = pd.Timestamp("2027-06-05")  # last Saturday
N_HORIZON_WEEKS: int = 104  # total epi-week horizons

# Maximum date allowed for model calibration / fitting
FIT_END_DATE: pd.Timestamp = pd.Timestamp("2026-06-06")

# Vaccination campaign windows (mid-week dates per round20.md)
VAX_FALL_2025_START: pd.Timestamp = pd.Timestamp("2025-08-13")
VAX_FALL_2025_END: pd.Timestamp   = pd.Timestamp("2026-02-14")
# Spring 2026 – Scenarios C & E only, high-risk groups
VAX_SPRING_2026_START: pd.Timestamp = pd.Timestamp("2026-02-15")
VAX_SPRING_2026_END: pd.Timestamp   = pd.Timestamp("2026-08-13")
# Fall 2026-27 – Scenarios B–E
VAX_FALL_2026_START: pd.Timestamp = pd.Timestamp("2026-08-14")
VAX_FALL_2026_END: pd.Timestamp   = pd.Timestamp("2027-02-13")

# Vaccine effectiveness against hospitalisation at campaign start (round20.md)
VE_HOSP: float = 0.55

# ---------------------------------------------------------------------------
# Round 20 scenario registry
# ---------------------------------------------------------------------------
SCENARIO_IDS: Dict[str, str] = {
    "A": "A-2026-05-11",  # counterfactual – no further boosters after Feb 2026
    "B": "B-2026-05-11",  # business-as-usual coverage, annual vaccination
    "C": "C-2026-05-11",  # business-as-usual coverage, semi-annual for high-risk
    "D": "D-2026-05-11",  # optimistic coverage, annual vaccination
    "E": "E-2026-05-11",  # optimistic coverage, semi-annual for high-risk
}

SCENARIO_NAMES: Dict[str, str] = {
    "A": "noVax",
    "B": "ModCov_AnnualVax",
    "C": "ModCov_SemiannualHRVax",
    "D": "OptCov_AnnualVax",
    "E": "OptCov_SemiannualHRVax",
}


# ===========================================================================
# Private helpers
# ===========================================================================

def _check_file_exists(path: pathlib.Path) -> None:
    """Raise ``FileNotFoundError`` with a clear message if *path* is absent."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "Ensure the repository is fully checked out and the working "
            "directory is correct."
        )


def _validate_columns(
    df: pd.DataFrame, required: List[str], source: str
) -> None:
    """
    Raise ``ValueError`` if *df* is missing any column in *required*.

    Parameters
    ----------
    df       : DataFrame to inspect.
    required : Column names that must be present.
    source   : Human-readable label for the data source (error message only).
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required column(s): {missing}. "
            f"Found: {df.columns.tolist()}"
        )


def _validate_target_domain(df: pd.DataFrame) -> None:
    """
    Emit warnings (not errors) for unknown ``target`` or ``age_group``
    values so the module keeps working when the hub adds new targets.
    """
    unknown_targets = set(df["target"].unique()) - VALID_TARGETS
    if unknown_targets:
        logger.warning(
            "load_target_data: unexpected target value(s) %s – "
            "included unless filtered out.",
            unknown_targets,
        )
    unknown_ages = set(df["age_group"].unique()) - VALID_AGE_GROUPS
    if unknown_ages:
        logger.warning(
            "load_target_data: unexpected age_group value(s) %s – "
            "included unless filtered out.",
            unknown_ages,
        )


def _coerce_observation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast ``observation`` to ``float64``.

    Non-numeric entries (e.g. suppressed counts stored as strings) become
    ``NaN``; the count of new nulls is logged as a warning.
    """
    df = df.copy()
    n_nulls_before = df["observation"].isna().sum()
    df["observation"] = pd.to_numeric(df["observation"], errors="coerce")
    new_nulls = int(df["observation"].isna().sum() - n_nulls_before)
    if new_nulls > 0:
        logger.warning(
            "load_target_data: %d observation value(s) could not be coerced "
            "to numeric and were set to NaN.",
            new_nulls,
        )
    return df



# ===========================================================================
# Public API – Section 1: Target data
# ===========================================================================

def load_target_data(
    *,
    location_filter: Optional[List[str]] = None,
    target_filter: Optional[List[str]] = None,
    age_group_filter: Optional[List[str]] = None,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
    drop_zero_obs: bool = False,
) -> pd.DataFrame:
    """
    Load and validate ``target-data/time-series.csv``.

    The function applies optional filters, converts the ``date`` column to
    ``datetime64[ns]``, coerces ``observation`` to ``float64``, and returns
    a consistently-sorted DataFrame.

    Parameters
    ----------
    location_filter  : Restrict to these FIPS codes (e.g. ``["US", "06"]``).
                       ``None`` keeps all locations.
    target_filter    : Restrict to these target names
                       (``"inc death"`` and/or ``"inc hosp"``).
                       ``None`` keeps all targets.
    age_group_filter : Restrict to these age groups
                       (``"0-130"``, ``"0-64"``, ``"65-130"``).
                       ``None`` keeps all age groups.
    min_date         : Inclusive lower bound on the ``date`` column.
    max_date         : Inclusive upper bound on the ``date`` column.
    drop_zero_obs    : If ``True``, drop rows where ``observation == 0``.

    Returns
    -------
    pd.DataFrame
        Columns: ``location`` (str), ``date`` (datetime64[ns]),
        ``observation`` (float64), ``age_group`` (str), ``target`` (str).
        Sorted by ``["location", "target", "age_group", "date"]``.

    Raises
    ------
    FileNotFoundError
        If ``target-data/time-series.csv`` does not exist.
    ValueError
        If required columns are absent or an invalid filter value is passed.

    Examples
    --------
    >>> df = load_target_data(
    ...     location_filter=["US"],
    ...     target_filter=["inc hosp"],
    ...     age_group_filter=["0-130"],
    ...     min_date=pd.Timestamp("2024-01-01"),
    ... )
    """
    _check_file_exists(_TARGET_DATA_PATH)

    df = pd.read_csv(
        _TARGET_DATA_PATH,
        dtype={"location": str, "age_group": str, "target": str},
    )
    logger.debug("Loaded %d rows from %s", len(df), _TARGET_DATA_PATH)

    _validate_columns(df, _TARGET_DATA_REQUIRED_COLS, source="time-series.csv")

    # ── Date parsing ───────────────────────────────────────────────────────
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    except Exception as exc:
        raise ValueError(
            f"time-series.csv: failed to parse 'date' column – {exc}"
        ) from exc

    n_bad_dates = int(df["date"].isna().sum())
    if n_bad_dates > 0:
        logger.warning(
            "load_target_data: %d date(s) could not be parsed and are NaT.",
            n_bad_dates,
        )

    # ── Numeric coercion ───────────────────────────────────────────────────
    df = _coerce_observation(df)

    # ── Domain-value warnings ──────────────────────────────────────────────
    _validate_target_domain(df)

    # ── Filters ────────────────────────────────────────────────────────────
    if location_filter is not None:
        unknown_locs = set(location_filter) - set(df["location"].unique())
        if unknown_locs:
            logger.warning(
                "load_target_data: location_filter contains unknown FIPS "
                "code(s): %s",
                sorted(unknown_locs),
            )
        df = df[df["location"].isin(location_filter)]

    if target_filter is not None:
        invalid = set(target_filter) - VALID_TARGETS
        if invalid:
            raise ValueError(
                f"target_filter contains invalid target(s): {invalid}. "
                f"Valid values: {sorted(VALID_TARGETS)}"
            )
        df = df[df["target"].isin(target_filter)]

    if age_group_filter is not None:
        invalid_ages = set(age_group_filter) - VALID_AGE_GROUPS
        if invalid_ages:
            raise ValueError(
                f"age_group_filter contains invalid age group(s): {invalid_ages}. "
                f"Valid values: {sorted(VALID_AGE_GROUPS)}"
            )
        df = df[df["age_group"].isin(age_group_filter)]

    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]

    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]

    if drop_zero_obs:
        df = df[df["observation"] != 0]

    # ── Sort & return ──────────────────────────────────────────────────────
    df = df.sort_values(
        ["location", "target", "age_group", "date"]
    ).reset_index(drop=True)

    if df.empty:
        logger.warning(
            "load_target_data: no rows remain after applying filters."
        )

    logger.debug("Returning %d rows after filtering.", len(df))
    return df


# ===========================================================================
# Public API – Section 2: Location metadata
# ===========================================================================

def load_locations() -> pd.DataFrame:
    """
    Load ``auxiliary-data/data-locations/locations.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``abbreviation`` (str), ``location`` (str, FIPS),
        ``location_name`` (str), ``population`` (Int64).
        Sorted by ``location``.

    Raises
    ------
    FileNotFoundError
        If ``locations.csv`` does not exist.
    ValueError
        If required columns are absent.
    """
    _check_file_exists(_LOCATIONS_PATH)

    df = pd.read_csv(
        _LOCATIONS_PATH,
        dtype={"location": str, "abbreviation": str, "location_name": str},
    )
    _validate_columns(df, _LOCATIONS_REQUIRED_COLS, source="locations.csv")

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    if df["population"].isna().any():
        logger.warning(
            "load_locations: %d population value(s) could not be parsed.",
            int(df["population"].isna().sum()),
        )
    df["population"] = df["population"].astype("Int64")

    return df.sort_values("location").reset_index(drop=True)


def get_population_map() -> Dict[str, int]:
    """
    Return a ``{fips_code: population}`` mapping for all locations.

    The national row ``"US"`` is included.

    Returns
    -------
    dict[str, int]

    Examples
    --------
    >>> pop = get_population_map()
    >>> pop["US"]
    328728466
    """
    locs = load_locations()
    return {
        row.location: int(row.population)
        for row in locs.itertuples(index=False)
        if pd.notna(row.population)
    }


# ===========================================================================
# Public API – Section 3: Vaccination coverage curves
# ===========================================================================

def load_vax_curves(
    *,
    scenario_filter: Optional[List[str]] = None,
    geography_filter: Optional[List[str]] = None,
    age_group_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load ``auxiliary-data/vaccination-coverage/COVID_RD20_Vaccination_curves.csv``.

    The ``coverage`` column is cumulative percentage vaccinated (0–100 scale).

    Parameters
    ----------
    scenario_filter  : Restrict to these scenario_id strings.
    geography_filter : Restrict to these US state names.
    age_group_filter : Restrict to these age-group labels.

    Returns
    -------
    pd.DataFrame
        Columns: ``geography``, ``age_group``, ``date`` (datetime64[ns]),
        ``coverage`` (float64), ``population`` (float64), ``scenario``.
        Sorted by ``["scenario", "geography", "age_group", "date"]``.

    Raises
    ------
    FileNotFoundError
        If the vaccination curves CSV does not exist.
    ValueError
        If required columns are absent.
    """
    _check_file_exists(_VAX_CURVES_PATH)

    df = pd.read_csv(_VAX_CURVES_PATH, dtype={"geography": str, "scenario": str})
    _validate_columns(
        df, _VAX_CURVES_REQUIRED_COLS, source="COVID_RD20_Vaccination_curves.csv"
    )

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    if scenario_filter is not None:
        df = df[df["scenario"].isin(scenario_filter)]
    if geography_filter is not None:
        df = df[df["geography"].isin(geography_filter)]
    if age_group_filter is not None:
        df = df[df["age_group"].isin(age_group_filter)]

    return df.sort_values(
        ["scenario", "geography", "age_group", "date"]
    ).reset_index(drop=True)


def get_national_vax_curve(scenario_id: str) -> pd.DataFrame:
    """
    Aggregate state-level vaccination curves to a US national weekly curve
    for *scenario_id*, weighted by sub-population size.

    Converts raw cumulative coverage percentage into a weekly incremental
    fraction of the total US population newly vaccinated.

    Parameters
    ----------
    scenario_id : One of the Round 20 scenario_id strings
                  (e.g. ``"B-2026-05-11"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (datetime64[ns]), ``cum_vax_frac`` (0–1),
        ``weekly_newly_vaccinated_frac`` (≥ 0).

    Raises
    ------
    ValueError
        If *scenario_id* is not found in the vaccination curves file.
    """
    df = load_vax_curves(scenario_filter=[scenario_id])

    if df.empty:
        raise ValueError(
            f"No vaccination coverage data found for scenario_id='{scenario_id}'."
        )

    df = df.copy()
    df["n_vax"] = df["coverage"] / 100.0 * df["population"]

    national = (
        df.groupby("date", as_index=False)
        .agg(total_vax=("n_vax", "sum"), total_pop=("population", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    national["cum_vax_frac"] = national["total_vax"] / national["total_pop"]

    # Weekly increment; clip to ≥ 0 to absorb floating-point jitter
    national["weekly_newly_vaccinated_frac"] = (
        national["cum_vax_frac"].diff().clip(lower=0.0)
    )
    # First row produces NaN from diff; fill from the cumulative value itself
    national.loc[0, "weekly_newly_vaccinated_frac"] = national.loc[0, "cum_vax_frac"]

    return national[["date", "cum_vax_frac", "weekly_newly_vaccinated_frac"]]


# ===========================================================================
# Private helper: coverage series → SEIRS multipliers  (O(N log N))
# ===========================================================================

def _coverage_to_multipliers(
    dates: pd.DatetimeIndex,
    cum_coverage_series: pd.Series,
) -> np.ndarray:
    """
    Convert a cumulative vaccination coverage series into weekly SEIRS
    susceptibility multipliers  M(t) = 1 - P(t).

    Uses the canonical VE function from scenario_adjustments.effective_ve so
    there is exactly ONE VE implementation in the repository.  There is no
    circular-import risk: scenario_adjustments does not import from load_data.

    Algorithm: O(N log N) via np.convolve.

        Δcov(s)      = weekly increment of cumulative vaccinated fraction
        ve_kernel[k] = effective_ve(k)         # VE k weeks after vaccination
        P(t)         = convolve(Δcov, ve_kernel)[:n]
        M(t)         = clip(1 - P(t), 0, 1)

    Parameters
    ----------
    dates               : Target epi-week Saturday DatetimeIndex.
    cum_coverage_series : pd.Series indexed by date; values = cumulative
                          fraction of total population vaccinated (0–1).
                          Reindexed internally — need not match dates exactly.

    Returns
    -------
    np.ndarray of shape (len(dates),), values in [0, 1].
    1.0 = no protection active.  Lower = vaccination-derived protection.
    """
    # Deferred import of the single canonical VE implementation.
    from scenario_adjustments import (  # noqa: PLC0415
        effective_ve,
        VE_HOSP_INITIAL,
        WANING_HALF_LIFE_WEEKS,
        WANED_FLOOR,
        IMMUNE_ESCAPE_PER_YEAR,
    )

    n = len(dates)
    if n == 0:
        return np.ones(0, dtype=float)

    # ── Align coverage to target dates ────────────────────────────────────
    # Reindex onto the union; forward-fill plateau between campaigns;
    # then select only requested dates; back-fill 0 for pre-campaign weeks.
    cov_aligned = (
        cum_coverage_series
        .reindex(cum_coverage_series.index.union(dates))
        .sort_index()
        .ffill()
        .reindex(dates)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # ── Weekly increments ─────────────────────────────────────────────────
    delta_cov = np.diff(cov_aligned, prepend=0.0)
    delta_cov = np.clip(delta_cov, 0.0, None)  # guard against float jitter

    if delta_cov.sum() == 0.0:
        return np.ones(n, dtype=float)  # fast path: no vaccination in window

    # ── VE kernel: VE(τ) for τ = 0 … n-1 weeks post-vaccination ─────────
    tau_vals = np.arange(n, dtype=float)
    ve_kernel = effective_ve(
        tau_vals,
        ve_initial=VE_HOSP_INITIAL,
        waning_half_life_weeks=WANING_HALF_LIFE_WEEKS,
        waned_floor=WANED_FLOOR,
        immune_escape_per_year=IMMUNE_ESCAPE_PER_YEAR,
    )

    # ── O(N log N) convolution ────────────────────────────────────────────
    # np.convolve(a, b) has length len(a)+len(b)-1; [:n] gives P(0..n-1).
    protection = np.convolve(delta_cov, ve_kernel)[:n]
    protection = np.clip(protection, 0.0, 1.0)

    return np.clip(1.0 - protection, 0.0, 1.0)  # M(t) = 1 - P(t)


# ===========================================================================
# Public API – Section 3b: Historical vaccination coverage
# ===========================================================================

def get_historical_vax_coverage() -> pd.DataFrame:
    """
    Construct a continuous cumulative national vaccination coverage series
    for the calibration/warm-start period, using only observed data.

    Responsibilities
    ----------------
    Returns raw coverage only (date → cum_coverage_frac).
    Does NOT perform VE calculations — see :func:`_coverage_to_multipliers`.

    Coverage timeline
    -----------------
    Period 1  Before 2024-08-04
        coverage = 0

    Period 2  2024-08-04 → 2024-12-29
        Source: COVID_RD19_Vaccination_curves.csv, Scenario == "Historic coverage".
        Observed NIS survey uptake (per vaccination-coverage/README.md).
        Population-weighted national aggregate using the same arithmetic as
        get_national_vax_curve():
            n_vax = (Cum.Coverage.Percent / 100) × Pop
            cum_coverage_frac = Σ n_vax / Σ Pop

    Period 3  2024-12-29 → 2025-08-17
        No campaign.  Coverage held at Dec 2024 plateau (forward-fill).
        Protection wanes; that is handled by the VE model.

    Period 4  2025-08-17 → 2026-02-14
        Source: COVID_RD20_Vaccination_curves.csv, Scenario "A-2026-05-11".
        Used ONLY because all Round 20 scenarios share identical 2025-26
        fall campaign uptake before scenario divergence (round20.md:
        "the 2025-26 vaccination campaign operates as observed in all
        scenarios A-E").  This is the shared historical campaign data;
        Scenario A itself is not described as "observed".

    Period 5  After 2026-02-14
        No additional campaign.  Coverage held at Feb 2026 plateau.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64[ns]), cum_coverage_frac (float64).
        One row per weekly Saturday, 2024-01-06 through FIT_END_DATE.
        Values are monotonically non-decreasing.
    """
    # ── Period 2: RD19 "Historic coverage" ───────────────────────────────
    _check_file_exists(_VAX_CURVES_RD19_PATH)
    rd19_raw = pd.read_csv(
        _VAX_CURVES_RD19_PATH,
        dtype={"Geography": str, "Scenario": str, "Age": str, "Risk_group": str},
    )
    _validate_columns(
        rd19_raw, _VAX_CURVES_RD19_REQUIRED_COLS,
        source="COVID_RD19_Vaccination_curves.csv",
    )
    rd19_raw["Date"] = pd.to_datetime(
        rd19_raw["Date"], format="%Y-%m-%d", errors="coerce"
    )
    rd19_raw["Cum.Coverage.Percent"] = pd.to_numeric(
        rd19_raw["Cum.Coverage.Percent"], errors="coerce"
    ).fillna(0.0)
    rd19_raw["Pop"] = pd.to_numeric(rd19_raw["Pop"], errors="coerce").fillna(0.0)

    rd19_hist = rd19_raw[rd19_raw["Scenario"] == "Historic coverage"].copy()
    if rd19_hist.empty:
        logger.warning(
            "get_historical_vax_coverage: no 'Historic coverage' rows in "
            "COVID_RD19_Vaccination_curves.csv; period 2 will be zero."
        )
        rd19_series: pd.Series = pd.Series(dtype=float)
    else:
        # Same population-weighting arithmetic as get_national_vax_curve()
        rd19_hist["n_vax"] = (
            rd19_hist["Cum.Coverage.Percent"] / 100.0 * rd19_hist["Pop"]
        )
        nat = (
            rd19_hist
            .groupby("Date", as_index=False)
            .agg(total_vax=("n_vax", "sum"), total_pop=("Pop", "sum"))
            .sort_values("Date")
            .reset_index(drop=True)
        )
        nat["cum_coverage_frac"] = (
            nat["total_vax"] / nat["total_pop"].replace(0.0, np.nan)
        ).fillna(0.0)
        rd19_series = nat.set_index("Date")["cum_coverage_frac"]

    # ── Period 4: RD20 Scenario A 2025-26 shared historical campaign ──────
    rd20_a = get_national_vax_curve("A-2026-05-11")
    rd20_hist = rd20_a.loc[
        (rd20_a["date"] >= VAX_FALL_2025_START)
        & (rd20_a["date"] <= VAX_FALL_2025_END)
    ].set_index("date")["cum_vax_frac"]

    # ── Build weekly Saturday spine and fill deterministically ───────────
    # Each period is applied by direct index assignment — no drop_duplicates,
    # no dataframe-order dependence.  Periods 2 and 4 are non-overlapping.
    full_spine = pd.Series(
        0.0,
        index=pd.date_range(
            start=pd.Timestamp("2024-01-06"),  # first Saturday >= 2024-01-01
            end=FIT_END_DATE,
            freq="7D",
        ),
        dtype=float,
    )
    # Period 2: overwrite with RD19 observed dates
    if not rd19_series.empty:
        overlap = full_spine.index.intersection(rd19_series.index)
        full_spine.loc[overlap] = rd19_series.loc[overlap]
    # Period 4: overwrite with RD20 observed dates (non-overlapping with P2)
    if not rd20_hist.empty:
        overlap = full_spine.index.intersection(rd20_hist.index)
        full_spine.loc[overlap] = rd20_hist.loc[overlap]

    # Periods 3 + 5: forward-fill plateau; enforce monotonicity
    full_spine = full_spine.ffill().fillna(0.0).cummax()

    result = full_spine.reset_index()
    result.columns = pd.Index(["date", "cum_coverage_frac"])
    logger.info(
        "get_historical_vax_coverage: n=%d  coverage [%.4f, %.4f]  %s → %s",
        len(result),
        float(result["cum_coverage_frac"].min()),
        float(result["cum_coverage_frac"].max()),
        result["date"].iloc[0].date() if len(result) else "n/a",
        result["date"].iloc[-1].date() if len(result) else "n/a",
    )
    return result


def build_historical_vax_multipliers(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert historical cumulative vaccination coverage into SEIRS
    susceptibility multipliers aligned to *dates*.

    Accepts dates explicitly so this function is reusable from:
    - the calibration path  (get_calibrated_params  in simulate.py)
    - the warm-start path   (generate_trajectories  in stochastic_simulate.py)

    VE mathematics are provided by _coverage_to_multipliers(), which imports
    effective_ve from scenario_adjustments.py — one canonical VE implementation.

    Parameters
    ----------
    dates : pd.DatetimeIndex of epi-week Saturdays.

    Returns
    -------
    np.ndarray of shape (len(dates),), values in [0, 1].
    1.0 = no protection.  Lower = active vaccination-derived protection.
    """
    cov_df = get_historical_vax_coverage()
    cov_series = cov_df.set_index("date")["cum_coverage_frac"]
    mult = _coverage_to_multipliers(dates, cov_series)
    logger.debug(
        "build_historical_vax_multipliers: n=%d  M∈[%.4f, %.4f]",
        len(mult),
        float(mult.min()) if len(mult) else float("nan"),
        float(mult.max()) if len(mult) else float("nan"),
    )
    return mult


def build_continuous_vax_multipliers(
    cal_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    scenario_id: str,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Build ONE continuous vaccination multiplier array spanning both the
    historical calibration period AND the forecast period for *scenario_id*.

    The VE convolution is performed once over the merged coverage timeline,
    so the immunity state is consistent across calibration → forecast.

    Timeline
    --------
    cal_dates[0] ─── historical observed ─── cal_dates[-1]
                                                   │
                                          forecast_dates[0] ─── scenario ─── forecast_dates[-1]

    Parameters
    ----------
    cal_dates      : Calibration/warm-start epi-week dates.
    forecast_dates : Forecast epi-week dates (104 weeks).
    scenario_id    : Hub scenario ID for the future campaign
                     (e.g. "B-2026-05-11"; use "A-2026-05-11" for no
                     additional future vaccination).

    Returns
    -------
    (cal_mult, forecast_mult) : tuple of two np.ndarrays.
        cal_mult         shape (len(cal_dates),)
        forecast_mult    shape (len(forecast_dates),)
        Both are slices of the same single VE convolution.
    """
    # ── Historical coverage (observed) ────────────────────────────────────
    hist_df = get_historical_vax_coverage()
    hist_series = hist_df.set_index("date")["cum_coverage_frac"]
    hist_plateau = float(hist_series.iloc[-1]) if len(hist_series) > 0 else 0.0

    # ── Future scenario coverage (incremental, on top of historical plateau) ─
    future_cov = pd.Series(0.0, index=forecast_dates, dtype=float)
    try:
        rd20 = get_national_vax_curve(scenario_id)
        # Only the campaign portion after the shared 2025-26 historical campaign
        rd20_future = rd20.loc[
            rd20["date"] > VAX_FALL_2025_END
        ].set_index("date")["cum_vax_frac"]
        if not rd20_future.empty:
            # Rebase to 0 so adding hist_plateau gives the correct running total
            rd20_future = (rd20_future - float(rd20_future.iloc[0])).clip(lower=0.0)
            overlap = forecast_dates.intersection(rd20_future.index)
            future_cov.loc[overlap] = rd20_future.loc[overlap]
    except ValueError:
        pass  # Scenario A or missing data → no additional future campaign

    # ── Merge onto a single date axis ─────────────────────────────────────
    all_dates = cal_dates.append(forecast_dates).sort_values().drop_duplicates()

    # Start from the forward-filled historical series
    combined = (
        hist_series
        .reindex(hist_series.index.union(all_dates))
        .sort_index()
        .ffill()
        .fillna(0.0)
        .reindex(all_dates)
    )
    # Add future incremental coverage for forecast dates
    future_on_full = future_cov.reindex(all_dates).fillna(0.0)
    combined = combined + future_on_full
    # Enforce monotonicity (cumulative coverage never decreases)
    combined = combined.cummax()

    # ── Single VE convolution over the full timeline ───────────────────
    all_mult = _coverage_to_multipliers(all_dates, combined)

    # ── Split back into calibration and forecast portions ─────────────
    cal_mask   = pd.Index(all_dates).isin(cal_dates)
    fcast_mask = pd.Index(all_dates).isin(forecast_dates)

    return all_mult[cal_mask], all_mult[fcast_mask]


# ===========================================================================
# Public API – Section 4: Calibration helpers
# ===========================================================================

def build_calibration_data(
    *,
    location: str = "US",
    min_date: pd.Timestamp = pd.Timestamp("2023-01-01"),
    max_date: pd.Timestamp = FIT_END_DATE,
) -> pd.DataFrame:
    """
    Build a wide weekly observation table for model calibration.

    Pivots the tidy ``time-series.csv`` into columns ``inc_hosp`` and
    ``inc_death``, restricted to all-ages (``age_group == "0-130"``).

    Parameters
    ----------
    location : FIPS code of the jurisdiction to load (default ``"US"``).
    min_date : Inclusive lower bound for calibration window.
    max_date : Inclusive upper bound (≤ ``FIT_END_DATE`` avoids look-ahead).

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (datetime64[ns]), ``inc_hosp`` (float64),
        ``inc_death`` (float64). One row per epi-week, sorted by ``date``.

    Raises
    ------
    ValueError
        If no observations are found for *location* in the date window.

    Examples
    --------
    >>> cal = build_calibration_data(location="US")
    >>> cal.dtypes
    date         datetime64[ns]
    inc_hosp            float64
    inc_death           float64
    dtype: object
    """
    df = load_target_data(
        location_filter=[location],
        target_filter=list(VALID_TARGETS),
        age_group_filter=["0-130"],
        min_date=min_date,
        max_date=max_date,
    )

    if df.empty:
        raise ValueError(
            f"build_calibration_data: no observations found for "
            f"location='{location}' between {min_date.date()} and {max_date.date()}."
        )

    cal = (
        df.pivot_table(
            index="date",
            columns="target",
            values="observation",
            aggfunc="sum",
        )
        .reset_index()
    )
    cal.columns.name = None

    rename_map = {"inc hosp": "inc_hosp", "inc death": "inc_death"}
    cal = cal.rename(columns=rename_map)

    for col in ("inc_hosp", "inc_death"):
        if col not in cal.columns:
            logger.warning(
                "build_calibration_data: column '%s' missing from target data; "
                "filling with NaN.",
                col,
            )
            cal[col] = np.nan

    return (
        cal.sort_values("date")
        .reset_index(drop=True)[["date", "inc_hosp", "inc_death"]]
    )


def build_epiweek_dates(
    origin: pd.Timestamp = ORIGIN_DATE,
    n_weeks: int = N_HORIZON_WEEKS,
) -> pd.DatetimeIndex:
    """
    Generate the Saturday end-dates for each epi-week in the projection horizon.

    Epi-weeks run Sunday–Saturday (US CDC convention).  Horizon *k*
    (1-indexed) corresponds to ``dates[k-1]``.

    Parameters
    ----------
    origin  : First Sunday of the projection period (default: 2025-06-08).
    n_weeks : Number of epi-week horizons (default: 104).

    Returns
    -------
    pd.DatetimeIndex
        Length *n_weeks*; each element is a Saturday.

    Examples
    --------
    >>> dates = build_epiweek_dates()
    >>> dates[0].date()    # horizon 1
    datetime.date(2025, 6, 14)
    >>> dates[-1].date()   # horizon 104
    datetime.date(2027, 6, 5)
    """
    saturdays = [origin + pd.Timedelta(days=6 + 7 * k) for k in range(n_weeks)]
    return pd.DatetimeIndex(saturdays)


# ===========================================================================
# Public API – Section 5: Convenience all-in-one loader
# ===========================================================================

def load_all(location: str = "US") -> Dict:
    """
    Load all data required by the stochastic simulator in a single call.

    Parameters
    ----------
    location : FIPS code for the jurisdiction to model (default ``"US"``).

    Returns
    -------
    dict with keys:

    ``calibration_data`` : pd.DataFrame
        ``(date, inc_hosp, inc_death)`` for the fitting window.
    ``population`` : int
        Total population of *location*.
    ``vax_curves`` : dict[str, pd.DataFrame]
        Keyed by scenario letter ``"A"`` … ``"E"``; each value is the
        DataFrame from :func:`get_national_vax_curve`.
    ``epiweek_dates`` : pd.DatetimeIndex
        The 104 Saturday end-dates for the projection horizon.
    ``scenario_ids`` : dict[str, str]
        ``{"A": "A-2026-05-11", …}`` – official submission scenario IDs.
    ``scenario_names`` : dict[str, str]
        ``{"A": "noVax", …}`` – human-readable names.
    """
    pop_map = get_population_map()
    if location not in pop_map:
        logger.warning(
            "load_all: location '%s' not found in locations.csv; "
            "falling back to US national population.",
            location,
        )
    population = pop_map.get(location, 328_728_466)

    cal = build_calibration_data(location=location)

    vax_curves: Dict[str, pd.DataFrame] = {}
    for letter, sid in SCENARIO_IDS.items():
        try:
            vax_curves[letter] = get_national_vax_curve(sid)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning(
                "load_all: could not load vax curve for scenario %s (%s): %s",
                letter, sid, exc,
            )
            vax_curves[letter] = pd.DataFrame(
                columns=["date", "cum_vax_frac", "weekly_newly_vaccinated_frac"]
            )

    return {
        "calibration_data": cal,
        "population": population,
        "vax_curves": vax_curves,
        "epiweek_dates": build_epiweek_dates(),
        "scenario_ids": SCENARIO_IDS,
        "scenario_names": SCENARIO_NAMES,
    }


# ===========================================================================
# CLI smoke-test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s – %(message)s",
    )

    print("─" * 60)
    print("load_target_data (US, inc hosp, 0-130, last 5 rows)")
    print("─" * 60)
    ts = load_target_data(
        location_filter=["US"],
        target_filter=["inc hosp"],
        age_group_filter=["0-130"],
        min_date=pd.Timestamp("2025-01-01"),
    )
    print(ts.tail())
    print(f"\n  dtypes:\n{ts.dtypes}\n")

    print("─" * 60)
    print("build_calibration_data (US)")
    print("─" * 60)
    cal = build_calibration_data(location="US")
    print(cal.tail())
    print(f"\n  shape: {cal.shape}  |  NaNs: {cal.isna().sum().to_dict()}\n")

    print("─" * 60)
    print("get_population_map (first 5 entries)")
    print("─" * 60)
    pop = get_population_map()
    for k, v in list(pop.items())[:5]:
        print(f"  {k}: {v:,}")

    print()
    print("─" * 60)
    print("build_epiweek_dates (first 3 and last 3 horizons)")
    print("─" * 60)
    dates = build_epiweek_dates()
    for i in [0, 1, 2, 101, 102, 103]:
        print(f"  horizon {i+1:3d}: {dates[i].date()}")

    print()
    print("─" * 60)
    print("get_national_vax_curve (scenario B, first 4 rows)")
    print("─" * 60)
    vc = get_national_vax_curve("B-2026-05-11")
    print(vc.head(4))
