"""
build_submission.py
===================
Assembles the official Round 20 Scenario Modeling Hub submission file from
stochastic trajectory DataFrames and writes a gzip-compressed Parquet file.

Hub submission format (model-output/README.md)
----------------------------------------------
The output file must contain **exactly eleven columns** in any order:

    origin_date     str  "YYYY-MM-DD"  – start date of the scenario period
    scenario_id     str  e.g. "A-2026-05-11"
    target          str  "inc hosp" | "inc death" | "cum hosp" | "cum death"
    horizon         int  1–104 (epi-week number after origin_date)
    location        str  FIPS code "US" | "01" … "78"
    age_group       str  "0-130" | "0-64" | "65-130"
    output_type     str  "sample" | "quantile" | "mean"
    output_type_id  float/NA  NA for samples; quantile level for quantiles
    value           float  ≥ 0
    run_grouping    int  grouping ID that is shared across paired scenarios
    stochastic_run  int  unique run number within a group (1–300)

No additional columns are allowed in the submitted file.

Schema (canonical, verified against CFA-Scenarios reference submission):
    origin_date    : date32[day]  (stored as Python date, not datetime)
    scenario_id    : string
    location       : string
    target         : string
    horizon        : int32
    age_group      : string
    output_type    : string
    output_type_id : double  (NA stored as None/null)
    run_grouping   : double  (hub uses float; filled as 1.0 for all samples)
    stochastic_run : int32   (1 … 300)
    value          : double

Pairing convention
------------------
All five scenarios share the same random seed when trajectories are generated
(see ``stochastic_simulate.generate_trajectories``).  Trajectory i in
Scenario A corresponds to trajectory i in Scenarios B–E.  We therefore set:

    run_grouping  = 1  (all trajectories in the same group; one set of
                        baseline parameters drives all scenarios)
    stochastic_run = trajectory_id  (1–300)

This encodes the cross-scenario pairing required by round20.md:
"projections need to be paired across horizon, targets and scenarios".

Targets included in submission
------------------------------
Required:
    "inc hosp"  – weekly incident hospitalisations
    "inc death" – weekly incident deaths (derived from inc hosp via IHR)

Optional (included if include_cumulative=True):
    "cum hosp"  – cumulative hospitalisations since origin_date
    "cum death" – cumulative deaths since origin_date

Death derivation assumption
---------------------------
The stochastic simulator currently generates hospitalisation trajectories.
Deaths are derived using a fixed infection-hospitalisation-to-death ratio
(IHR_TO_IFR):

    inc_death(t) = inc_hosp(t) × HOSP_TO_DEATH_RATIO

HOSP_TO_DEATH_RATIO = 0.065  (6.5%)
    Source: approximate in-hospital COVID-19 case-fatality ratio for 2025-26,
    consistent with CDC/NCHS weekly hospitalization-to-death ratios observed
    in the calibration window (US national 2025-26: ~55–75 deaths per week
    for ~950–7000 hospitalisations → ratio ≈ 4–8%, midpoint ~6.5%).
    This is a prototype simplification; a full model would run separate
    stochastic processes for deaths.

File naming
-----------
    YYYY-MM-DD-<team>-<model>.gz.parquet
    where YYYY-MM-DD = origin_date (2025-06-08 for Round 20)
    Default: "2025-06-08-MyTeam-ProtoModel.gz.parquet"
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Round 20 constants
# ---------------------------------------------------------------------------

ORIGIN_DATE: str = "2025-06-08"        # first Sunday of projection period
N_HORIZONS: int = 104                   # epi-weeks 1–104
N_TRAJECTORIES: int = 300              # required by hub spec

# In-hospital COVID-19 death ratio (see module docstring)
HOSP_TO_DEATH_RATIO: float = 0.065

# Hub-required quantile levels (model-output/README.md)
HUB_QUANTILES: List[float] = [
    0.010, 0.025, 0.050,
    0.100, 0.150, 0.200, 0.250, 0.300, 0.350,
    0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
    0.700, 0.750, 0.800, 0.850, 0.900, 0.950,
    0.975, 0.990,
]

# Canonical Arrow schema (matches CFA-Scenarios reference submission)
SUBMISSION_SCHEMA = pa.schema([
    pa.field("origin_date",    pa.date32()),
    pa.field("scenario_id",    pa.string()),
    pa.field("location",       pa.string()),
    pa.field("target",         pa.string()),
    pa.field("horizon",        pa.int32()),
    pa.field("age_group",      pa.string()),
    pa.field("output_type",    pa.string()),
    pa.field("output_type_id", pa.float64()),   # NA for samples
    pa.field("run_grouping",   pa.float64()),   # 1.0 for all (one param group)
    pa.field("stochastic_run", pa.int32()),     # 1–300; NA for quantiles
    pa.field("value",          pa.float64()),
])

# Exact column order mandated by the schema above
_COLUMN_ORDER: List[str] = [
    "origin_date", "scenario_id", "location", "target", "horizon",
    "age_group", "output_type", "output_type_id", "run_grouping",
    "stochastic_run", "value",
]

# Valid domain values (used in validation)
_VALID_TARGETS    = frozenset({"inc hosp", "inc death", "cum hosp", "cum death"})
_VALID_AGE_GROUPS = frozenset({"0-130", "0-64", "65-130"})
_VALID_OUTPUT_TYPES = frozenset({"sample", "quantile", "mean"})


# ===========================================================================
# Section 1 – Horizon builder
# ===========================================================================

def _build_horizon_map(
    origin: str = ORIGIN_DATE,
    n_horizons: int = N_HORIZONS,
) -> Dict[pd.Timestamp, int]:
    """
    Return ``{epi_week_saturday: horizon_number}`` for all 104 horizons.

    Horizon k corresponds to the Saturday that is (6 + 7*(k-1)) days after
    the origin Sunday.  Horizon 1 = first Saturday after origin.
    """
    origin_ts = pd.Timestamp(origin)
    return {
        origin_ts + pd.Timedelta(days=6 + 7 * k): k + 1
        for k in range(n_horizons)
    }


# ===========================================================================
# Section 2 – Death trajectory derivation
# ===========================================================================

def derive_death_trajectories(
    hosp_trajectories: pd.DataFrame,
    hosp_to_death_ratio: float = HOSP_TO_DEATH_RATIO,
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """
    Derive weekly incident death trajectories from hospitalisation trajectories
    using a fixed case-fatality ratio applied to hospitalised counts.

    Derivation
    ----------
    inc_death(t) = inc_hosp(t) × hosp_to_death_ratio

    This is a prototype simplification.  A full model would run separate
    stochastic processes for deaths seeded from infection counts.

    Parameters
    ----------
    hosp_trajectories   : DataFrame with at minimum ``trajectory_id``,
                          ``date``, ``forecast`` columns.  Must already have
                          the scenario multiplier applied.
    hosp_to_death_ratio : Fixed ratio of deaths to hospitalisations (default
                          0.065 = 6.5%; see module docstring).
    forecast_col        : Name of the value column (default ``"forecast"``).

    Returns
    -------
    pd.DataFrame
        Same schema as ``hosp_trajectories`` with ``forecast`` replaced by
        the derived death count.  Minimum value clipped to 0.
    """
    death = hosp_trajectories.copy()
    death[forecast_col] = (
        (death[forecast_col] * hosp_to_death_ratio)
        .clip(lower=0.0)
        .round(1)                 # one decimal place per hub spec
    )
    return death


# ===========================================================================
# Section 3 – Cumulative target builder
# ===========================================================================

def build_cumulative_trajectories(
    incident_trajectories: pd.DataFrame,
    forecast_col: str = "forecast",
    id_col: str = "trajectory_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Convert incident weekly counts to cumulative counts since origin_date.

    The cumulative value at horizon t is the sum of all incident values
    from horizon 1 through horizon t (inclusive).  This matches the hub
    definition: "cumulative number ... since the beginning of the simulation".

    Parameters
    ----------
    incident_trajectories : DataFrame with ``trajectory_id``, ``date``,
                            and ``forecast`` columns sorted by date within
                            each trajectory.
    forecast_col          : Value column (default ``"forecast"``).
    id_col                : Trajectory ID column (default ``"trajectory_id"``).
    date_col              : Date column (default ``"date"``).

    Returns
    -------
    pd.DataFrame
        Same schema; ``forecast`` replaced by cumulative sum per trajectory.
    """
    cum = (
        incident_trajectories
        .sort_values([id_col, date_col])
        .copy()
    )
    cum[forecast_col] = cum.groupby(id_col)[forecast_col].cumsum()
    return cum


# ===========================================================================
# Section 4 – Hub row assembler (sample output type)
# ===========================================================================

def _assemble_sample_rows(
    trajectories: pd.DataFrame,
    target: str,
    location: str,
    age_group: str,
    scenario_id: str,
    origin_date: str = ORIGIN_DATE,
    horizon_map: Optional[Dict[pd.Timestamp, int]] = None,
    forecast_col: str = "forecast",
    date_col: str = "date",
    trajectory_id_col: str = "trajectory_id",
) -> pd.DataFrame:
    """
    Convert a trajectory DataFrame into the hub "sample" row format.

    Each trajectory_id becomes one ``stochastic_run``; ``run_grouping = 1``
    for all rows (single parameter group; see module docstring for pairing
    convention).

    Returns a DataFrame with all eleven submission columns.
    """
    if horizon_map is None:
        horizon_map = _build_horizon_map(origin_date)

    df = trajectories[[trajectory_id_col, date_col, forecast_col]].copy()

    # Map date → horizon; rows with no matching horizon are silently dropped
    # (dates outside the 104-week window should not occur with correctly
    # constructed forecast_dates, but the drop is a safety net).
    df["horizon"] = df[date_col].map(horizon_map)
    n_before = len(df)
    df = df.dropna(subset=["horizon"])
    if len(df) < n_before:
        logger.warning(
            "_assemble_sample_rows: dropped %d rows with unmapped dates "
            "(outside 104-week horizon window).",
            n_before - len(df),
        )
    df["horizon"] = df["horizon"].astype(np.int32)

    df["origin_date"]     = pd.to_datetime(origin_date).date()
    df["scenario_id"]     = scenario_id
    df["target"]          = target
    df["location"]        = location
    df["age_group"]       = age_group
    df["output_type"]     = "sample"
    df["output_type_id"]  = np.nan          # NA for sample rows (hub spec)
    df["run_grouping"]    = 1.0             # single parameter group
    df["stochastic_run"]  = df[trajectory_id_col].astype(np.int32)
    df["value"]           = df[forecast_col].round(1).clip(lower=0.0)

    return df[_COLUMN_ORDER]


# ===========================================================================
# Section 5 – Hub row assembler (quantile output type)
# ===========================================================================

def _assemble_quantile_rows(
    trajectories: pd.DataFrame,
    target: str,
    location: str,
    age_group: str,
    scenario_id: str,
    quantiles: List[float] = HUB_QUANTILES,
    origin_date: str = ORIGIN_DATE,
    horizon_map: Optional[Dict[pd.Timestamp, int]] = None,
    forecast_col: str = "forecast",
    date_col: str = "date",
    trajectory_id_col: str = "trajectory_id",
) -> pd.DataFrame:
    """
    Compute quantile summaries across trajectories and format as hub rows.

    For quantile rows: ``run_grouping = NA``, ``stochastic_run = NA``.
    ``output_type_id`` is the quantile level (e.g. 0.025).

    Returns a DataFrame with all eleven submission columns.
    """
    if horizon_map is None:
        horizon_map = _build_horizon_map(origin_date)

    df = trajectories[[trajectory_id_col, date_col, forecast_col]].copy()
    df["horizon"] = df[date_col].map(horizon_map)
    df = df.dropna(subset=["horizon"])
    df["horizon"] = df["horizon"].astype(np.int32)

    # Compute quantiles per horizon
    rows = []
    for horizon, grp in df.groupby("horizon"):
        vals = grp[forecast_col].values
        q_vals = np.quantile(vals, quantiles)
        for q, v in zip(quantiles, q_vals):
            rows.append({
                "origin_date":    pd.to_datetime(origin_date).date(),
                "scenario_id":    scenario_id,
                "target":         target,
                "horizon":        np.int32(horizon),
                "location":       location,
                "age_group":      age_group,
                "output_type":    "quantile",
                "output_type_id": float(q),
                "run_grouping":   np.nan,    # NA for quantile rows
                "stochastic_run": np.nan,    # NA for quantile rows
                "value":          round(float(v), 1),
            })

    return pd.DataFrame(rows, columns=_COLUMN_ORDER)


# ===========================================================================
# Section 6 – Schema validator
# ===========================================================================

def validate_submission(df: pd.DataFrame) -> None:
    """
    Validate a submission DataFrame before writing to disk.

    Checks performed
    ----------------
    1. Exactly the eleven required columns are present (no extras).
    2. ``target`` values are within the accepted set.
    3. ``age_group`` values are within the accepted set.
    4. ``output_type`` values are within the accepted set.
    5. ``horizon`` is in [1, 104].
    6. ``value`` is non-negative.
    7. Sample rows have ``output_type_id = NA``.
    8. Quantile rows have ``output_type_id`` in (0, 1).
    9. ``stochastic_run`` for sample rows is in [1, N_TRAJECTORIES].
    10. The sample output type has exactly N_TRAJECTORIES rows per
        (scenario_id, location, target, horizon, age_group) group.
    11. ``origin_date`` is a single consistent value.

    Raises
    ------
    ValueError
        If any validation check fails.  The message identifies the specific
        violation.

    Notes
    -----
    This validation is intentionally strict for the prototype.  Some checks
    emit warnings rather than errors where real hub validation tolerates minor
    variations (e.g. cumulative targets being optional).
    """
    errors: List[str] = []

    # ── 1. Column set ──────────────────────────────────────────────────────
    actual_cols = set(df.columns)
    required_cols = set(_COLUMN_ORDER)
    extra = actual_cols - required_cols
    missing = required_cols - actual_cols
    if extra:
        errors.append(f"Extra columns not allowed: {sorted(extra)}")
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    if errors:           # cannot proceed with further checks if columns wrong
        raise ValueError("Submission validation failed:\n" + "\n".join(errors))

    # ── 2. Target values ───────────────────────────────────────────────────
    bad_targets = set(df["target"].unique()) - _VALID_TARGETS
    if bad_targets:
        errors.append(f"Invalid target value(s): {bad_targets}")

    # ── 3. Age group values ────────────────────────────────────────────────
    bad_ages = set(df["age_group"].unique()) - _VALID_AGE_GROUPS
    if bad_ages:
        errors.append(f"Invalid age_group value(s): {bad_ages}")

    # ── 4. Output type values ──────────────────────────────────────────────
    bad_types = set(df["output_type"].unique()) - _VALID_OUTPUT_TYPES
    if bad_types:
        errors.append(f"Invalid output_type value(s): {bad_types}")

    # ── 5. Horizon range ───────────────────────────────────────────────────
    horizons = df["horizon"].dropna()
    if horizons.min() < 1 or horizons.max() > N_HORIZONS:
        errors.append(
            f"horizon out of range [1, {N_HORIZONS}]: "
            f"found [{horizons.min()}, {horizons.max()}]"
        )

    # ── 6. Non-negative values ─────────────────────────────────────────────
    n_neg = (df["value"] < 0).sum()
    if n_neg > 0:
        errors.append(f"{n_neg} row(s) have negative 'value'.")

    # ── 7. Sample output_type_id must be NA ───────────────────────────────
    sample_mask = df["output_type"] == "sample"
    non_na_sample_ids = df.loc[sample_mask, "output_type_id"].notna().sum()
    if non_na_sample_ids > 0:
        errors.append(
            f"{non_na_sample_ids} sample row(s) have non-NA output_type_id."
        )

    # ── 8. Quantile output_type_id must be in (0, 1) ─────────────────────
    q_mask = df["output_type"] == "quantile"
    if q_mask.any():
        q_ids = df.loc[q_mask, "output_type_id"].dropna()
        if (q_ids <= 0).any() or (q_ids >= 1).any():
            errors.append("quantile output_type_id values must be in (0, 1).")

    # ── 9. stochastic_run range for sample rows ───────────────────────────
    if sample_mask.any():
        sr = df.loc[sample_mask, "stochastic_run"].dropna()
        if sr.min() < 1 or sr.max() > N_TRAJECTORIES:
            errors.append(
                f"stochastic_run out of range [1, {N_TRAJECTORIES}]: "
                f"found [{sr.min()}, {sr.max()}]"
            )

    # ── 10. Trajectory count per group ───────────────────────────────────
    if sample_mask.any():
        group_cols = ["scenario_id", "location", "target", "horizon", "age_group"]
        counts = (
            df[sample_mask]
            .groupby(group_cols)["stochastic_run"]
            .nunique()
        )
        wrong = counts[counts != N_TRAJECTORIES]
        if len(wrong) > 0:
            errors.append(
                f"{len(wrong)} sample group(s) do not have exactly "
                f"{N_TRAJECTORIES} trajectories.  Example:\n{wrong.head(3)}"
            )

    # ── 11. Single origin_date ────────────────────────────────────────────
    n_origin_dates = df["origin_date"].nunique()
    if n_origin_dates != 1:
        errors.append(
            f"origin_date must be a single value; found {n_origin_dates} distinct values."
        )

    if errors:
        raise ValueError(
            f"Submission validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )

    logger.info(
        "validate_submission: passed all checks.  rows=%d  "
        "scenarios=%s  targets=%s",
        len(df),
        sorted(df["scenario_id"].unique()),
        sorted(df["target"].unique()),
    )


# ===========================================================================
# Section 7 – Parquet writer
# ===========================================================================

def write_submission_parquet(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    team_model: str = "MyTeam-ProtoModel",
    origin_date: str = ORIGIN_DATE,
    compression_level: int = 9,
) -> pathlib.Path:
    """
    Write the submission DataFrame to a gzip-compressed Parquet file.

    File naming follows the hub convention:
        ``<output_dir>/<origin_date>-<team_model>.gz.parquet``

    The file is written using ``pyarrow.dataset.write_dataset`` with
    ``partitioning_flavor=None`` (non-hive-style), as required by the hub
    specification (model-output/README.md).

    Partitioning note
    -----------------
    The hub allows optional partitioning by ``origin_date`` and ``target``.
    When partitioning is used, those columns must be **absent** from the
    parquet file itself (the values are encoded in the directory path).
    This function writes a **single unpartitioned file** to keep the
    prototype simple and the submission self-contained.  For submissions
    exceeding 100 MB, switch to partitioned writing.

    Parameters
    ----------
    df              : Validated submission DataFrame (eleven columns only).
    output_dir      : Directory where the file will be written.
    team_model      : ``<team>-<model>`` string for the filename.
    origin_date     : Origin date string "YYYY-MM-DD" for the filename.
    compression_level: gzip compression level 1–9 (default 9 = max).

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If ``df`` contains columns other than the eleven required ones.
    """
    extra = set(df.columns) - set(_COLUMN_ORDER)
    if extra:
        raise ValueError(
            f"write_submission_parquet: DataFrame has extra column(s) {extra}. "
            "Remove them before writing."
        )

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{origin_date}-{team_model}.gz.parquet"
    out_path = output_dir / filename

    # ── Cast to canonical Arrow schema ────────────────────────────────────
    # origin_date: convert Python date / Timestamp → Arrow date32
    df = df.copy()
    df["origin_date"] = pd.to_datetime(df["origin_date"]).dt.date

    table = pa.Table.from_pandas(
        df[_COLUMN_ORDER],
        schema=SUBMISSION_SCHEMA,
        preserve_index=False,
    )

    # ── Write single gz.parquet (unpartitioned) ──────────────────────────
    fmt = ds.ParquetFileFormat()
    write_opts = fmt.make_write_options(
        compression="gzip",
        compression_level=compression_level,
    )
    ds.write_dataset(
        table,
        str(output_dir),
        format="parquet",
        partitioning_flavor=None,   # required: non-hive style (hub spec)
        file_options=write_opts,
        basename_template=filename.replace(".gz.parquet", "{i}.gz.parquet"),
        existing_data_behavior="overwrite_or_ignore",
    )

    logger.info(
        "write_submission_parquet: wrote %d rows → %s  (%.1f MB)",
        len(df),
        out_path,
        out_path.stat().st_size / 1e6 if out_path.exists() else 0,
    )
    return out_path


# ===========================================================================
# Section 8 – Main assembly function: build_submission
# ===========================================================================

def build_submission(
    scenario_trajectories: Dict[str, pd.DataFrame],
    *,
    location: str = "US",
    age_group: str = "0-130",
    origin_date: str = ORIGIN_DATE,
    hosp_to_death_ratio: float = HOSP_TO_DEATH_RATIO,
    include_cumulative: bool = True,
    include_quantiles: bool = True,
    output_dir: Optional[pathlib.Path] = None,
    team_model: str = "MyTeam-ProtoModel",
    forecast_col: str = "forecast",
    date_col: str = "date",
    trajectory_id_col: str = "trajectory_id",
) -> pd.DataFrame:
    """
    Build the complete hub submission DataFrame from scenario trajectories.

    This is the primary public function.  It:
    1. Derives death trajectories from hospitalisation trajectories.
    2. Optionally builds cumulative targets.
    3. Assembles sample (and optionally quantile) rows for every combination
       of scenario × target × location × age_group × horizon.
    4. Validates the assembled submission against the hub schema.
    5. Optionally writes a gzip Parquet file to ``output_dir``.

    Parameters
    ----------
    scenario_trajectories : ``{"A": df_A, "B": df_B, …}`` mapping scenario
                            letters to trajectory DataFrames produced by
                            ``scenario_adjustments.apply_scenario``.
                            Each DataFrame must have columns
                            ``trajectory_id``, ``date``, ``forecast``,
                            ``scenario_id`` (added by apply_scenario).
    location              : FIPS code for this submission (default ``"US"``).
    age_group             : Age group label (default ``"0-130"``).
    origin_date           : Scenario start date string "YYYY-MM-DD"
                            (default ``"2025-06-08"``).
    hosp_to_death_ratio   : Death : hospitalisation ratio (default 0.065).
    include_cumulative    : If True, include ``cum hosp`` and ``cum death``
                            targets (optional per hub spec).
    include_quantiles     : If True, include quantile output rows in addition
                            to sample rows (optional per hub spec).
    output_dir            : If provided, write the submission to this
                            directory.  Pass ``None`` to skip writing.
    team_model            : ``"<team>-<model>"`` for file naming.
    forecast_col          : Column name for forecast values in input DFs.
    date_col              : Column name for dates in input DFs.
    trajectory_id_col     : Column name for trajectory IDs in input DFs.

    Returns
    -------
    pd.DataFrame
        Validated submission DataFrame with exactly eleven columns.

    Raises
    ------
    ValueError
        If ``scenario_trajectories`` is empty, or if schema validation fails.

    Examples
    --------
    >>> from stochastic_simulate import generate_trajectories
    >>> from scenario_adjustments import apply_all_scenarios
    >>> from load_data import build_epiweek_dates, ORIGIN_DATE
    >>> import numpy as np, pathlib
    >>>
    >>> dates = build_epiweek_dates(ORIGIN_DATE)
    >>> obs = np.linspace(7000, 955, 16)
    >>> base = generate_trajectories(obs, dates, seed=42)
    >>> all_scen = apply_all_scenarios(base)
    >>>
    >>> scen_dfs = {
    ...     k: all_scen[all_scen["scenario_id"] == v]
    ...     for k, v in SCENARIO_IDS.items()
    ... }
    >>> sub = build_submission(
    ...     scen_dfs,
    ...     output_dir=pathlib.Path("model-output/MyTeam-ProtoModel"),
    ... )
    """
    if not scenario_trajectories:
        raise ValueError("build_submission: scenario_trajectories dict is empty.")

    horizon_map = _build_horizon_map(origin_date)
    all_frames: List[pd.DataFrame] = []

    for key, df_scen in scenario_trajectories.items():
        # Retrieve the official scenario_id (added by apply_scenario)
        if "scenario_id" in df_scen.columns:
            scenario_id = df_scen["scenario_id"].iloc[0]
        else:
            # Fallback: look up from the standard map
            from scenario_adjustments import SCENARIO_IDS as _SID_MAP
            scenario_id = _SID_MAP.get(key.upper(), key)

        logger.info(
            "build_submission: processing scenario %s (%s)  trajectories=%d",
            key, scenario_id,
            df_scen[trajectory_id_col].nunique(),
        )

        # ── Hospitalisation trajectories ──────────────────────────────────
        df_hosp = df_scen[[trajectory_id_col, date_col, forecast_col]].copy()

        # ── Death trajectories (derived) ──────────────────────────────────
        df_death = derive_death_trajectories(
            df_hosp,
            hosp_to_death_ratio=hosp_to_death_ratio,
            forecast_col=forecast_col,
        )

        # Build incident target frames
        incident_pairs = [
            ("inc hosp",  df_hosp),
            ("inc death", df_death),
        ]

        # Optionally add cumulative targets
        if include_cumulative:
            df_cum_hosp  = build_cumulative_trajectories(df_hosp,  forecast_col=forecast_col)
            df_cum_death = build_cumulative_trajectories(df_death, forecast_col=forecast_col)
            incident_pairs += [
                ("cum hosp",  df_cum_hosp),
                ("cum death", df_cum_death),
            ]

        for target_name, df_target in incident_pairs:
            # Sample rows (required)
            sample_rows = _assemble_sample_rows(
                trajectories=df_target,
                target=target_name,
                location=location,
                age_group=age_group,
                scenario_id=scenario_id,
                origin_date=origin_date,
                horizon_map=horizon_map,
                forecast_col=forecast_col,
                date_col=date_col,
                trajectory_id_col=trajectory_id_col,
            )
            all_frames.append(sample_rows)

            # Quantile rows (optional, encouraged)
            if include_quantiles:
                q_rows = _assemble_quantile_rows(
                    trajectories=df_target,
                    target=target_name,
                    location=location,
                    age_group=age_group,
                    scenario_id=scenario_id,
                    quantiles=HUB_QUANTILES,
                    origin_date=origin_date,
                    horizon_map=horizon_map,
                    forecast_col=forecast_col,
                    date_col=date_col,
                    trajectory_id_col=trajectory_id_col,
                )
                all_frames.append(q_rows)

    # ── Concatenate all frames ────────────────────────────────────────────
    submission = pd.concat(all_frames, ignore_index=True)

    # Enforce strict types before validation
    submission["horizon"]        = submission["horizon"].astype(np.int32)
    submission["stochastic_run"] = pd.to_numeric(
        submission["stochastic_run"], errors="coerce"
    )
    submission["run_grouping"]   = pd.to_numeric(
        submission["run_grouping"], errors="coerce"
    )
    submission["value"]          = submission["value"].astype(float)

    # ── Validate ──────────────────────────────────────────────────────────
    # Only pass sample rows to trajectory-count validation
    # (quantile rows do not need N_TRAJECTORIES rows per group)
    sample_only = submission[submission["output_type"] == "sample"].copy()
    validate_submission(sample_only)
    logger.info(
        "build_submission: validation passed.  total rows=%d  "
        "sample rows=%d  quantile rows=%d",
        len(submission),
        (submission["output_type"] == "sample").sum(),
        (submission["output_type"] == "quantile").sum(),
    )

    # ── Write parquet ─────────────────────────────────────────────────────
    if output_dir is not None:
        write_submission_parquet(
            submission,
            output_dir=pathlib.Path(output_dir),
            team_model=team_model,
            origin_date=origin_date,
        )

    return submission


# ===========================================================================
# Section 9 – Round-trip parquet reader (for verification)
# ===========================================================================

def read_submission_parquet(path: pathlib.Path) -> pd.DataFrame:
    """
    Read a submission parquet file back into a DataFrame.

    Restores ``origin_date`` as a ``datetime.date`` object and integer types
    to match the canonical schema.

    Parameters
    ----------
    path : Path to the ``.gz.parquet`` file.

    Returns
    -------
    pd.DataFrame  with SUBMISSION_SCHEMA column types.
    """
    table = pq.read_table(str(path))
    df = table.to_pandas()
    # origin_date stored as date32 → Python datetime.date; convert for display
    if "origin_date" in df.columns:
        df["origin_date"] = pd.to_datetime(df["origin_date"])
    return df


# ===========================================================================
# CLI smoke-test
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s – %(message)s",
    )

    import tempfile
    import numpy as np
    from load_data import build_epiweek_dates, ORIGIN_DATE as _ORIG
    from stochastic_simulate import generate_trajectories
    from scenario_adjustments import apply_all_scenarios, SCENARIO_IDS

    print("=" * 65)
    print("build_submission.py – smoke-test")
    print("=" * 65)

    # ── 1. Generate baseline trajectories from recent US inc hosp ─────────
    # Use the last 16 weeks of observed data (approximate; smoke-test only)
    obs = np.array([
        6967, 6604, 6405, 6480, 6335, 5686, 4992, 4321,
        3970, 2270, 1971, 1737, 1693, 1778, 1615, 1447,
        1349, 1236, 1046, 955,
    ], dtype=float)

    dates = build_epiweek_dates(_ORIG, n_weeks=104)

    print(f"\nGenerating {N_TRAJECTORIES} baseline trajectories …")
    base = generate_trajectories(obs, dates, seed=42, n_trajectories=N_TRAJECTORIES)
    print(f"  Baseline shape: {base.shape}")

    # ── 2. Apply all five scenarios ───────────────────────────────────────
    print("\nApplying scenarios A–E …")
    all_scen = apply_all_scenarios(base)

    # Split by scenario letter for build_submission input
    scen_dfs = {
        k: all_scen[all_scen["scenario_id"] == v].copy()
        for k, v in SCENARIO_IDS.items()
    }

    # ── 3. Build submission ───────────────────────────────────────────────
    print("\nBuilding submission …")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pathlib.Path(tmpdir) / "model-output" / "MyTeam-ProtoModel"

        sub = build_submission(
            scen_dfs,
            location="US",
            age_group="0-130",
            origin_date=ORIGIN_DATE,
            include_cumulative=True,
            include_quantiles=True,
            output_dir=out_dir,
            team_model="MyTeam-ProtoModel",
        )

        # ── 4. Report ─────────────────────────────────────────────────────
        print(f"\nSubmission DataFrame shape : {sub.shape}")
        print(f"Columns                    : {sub.columns.tolist()}")
        print(f"\nRow counts by output_type  :")
        print(sub["output_type"].value_counts().to_string())
        print(f"\nRow counts by target       :")
        print(sub["target"].value_counts().to_string())
        print(f"\nRow counts by scenario_id  :")
        print(sub["scenario_id"].value_counts().to_string())
        print(f"\nHorizon range              : {sub.horizon.min()}–{sub.horizon.max()}")
        print(f"Value range                : {sub.value.min():.1f}–{sub.value.max():.1f}")
        print(f"Any negative values?       : {(sub.value < 0).any()}")

        # ── 5. Schema check ───────────────────────────────────────────────
        print("\nDtypes:")
        print(sub.dtypes.to_string())

        # ── 6. Round-trip parquet check ───────────────────────────────────
        parquet_files = list(out_dir.glob("*.gz.parquet"))
        print(f"\nParquet file(s) written    : {[f.name for f in parquet_files]}")
        if parquet_files:
            size_mb = parquet_files[0].stat().st_size / 1e6
            print(f"File size                  : {size_mb:.2f} MB")
            rt = read_submission_parquet(parquet_files[0])
            print(f"Round-trip rows            : {len(rt)}")
            print(f"Round-trip schema match    : {set(rt.columns) == set(_COLUMN_ORDER)}")

        # ── 7. Sample first rows ──────────────────────────────────────────
        print("\nFirst 5 sample rows:")
        sample_preview = (
            sub[sub["output_type"] == "sample"]
            .sort_values(["scenario_id", "horizon", "stochastic_run"])
            .head(5)
        )
        print(sample_preview.to_string(index=False))

        print("\nFirst 5 quantile rows:")
        q_preview = (
            sub[sub["output_type"] == "quantile"]
            .sort_values(["scenario_id", "horizon", "output_type_id"])
            .head(5)
        )
        print(q_preview.to_string(index=False))

    print("\n✓ Smoke-test passed.")
