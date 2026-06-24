"""
tests/test_pipeline.py
======================
End-to-end integration test for the Round 20 COVID-19 Scenario Modeling Hub
prototype pipeline.

Pipeline under test
-------------------

  load_data.load_target_data()            ← target-data/time-series.csv
          │
          ▼
  stochastic_simulate.generate_trajectories()   ← 300 trajectories × 104 weeks
          │
          ▼
  scenario_adjustments.apply_all_scenarios()    ← Scenarios A–E
          │
          ▼
  build_submission.build_submission()           ← 11-column hub DataFrame
          │
          ▼
  write_submission_parquet()                    ← .gz.parquet
          │
          ▼
  plot_submission.generate_all_plots()          ← scenario_comparison.png
                                                   submission_plot.png

The test prints a structured report after each stage, then raises
``AssertionError`` (or propagates any exception from the module) if any
expectation is not met.  A final summary line confirms success.

Usage
-----
Run from the repository root:

    python3 tests/test_pipeline.py

Or with pytest (no pytest plugins required):

    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import logging
import pathlib
import sys
import tempfile
import traceback
from typing import Any, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make my_model/ importable when the script is run from the repo root or
# from within the tests/ directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "my_model"))

# ---------------------------------------------------------------------------
# Logging – INFO level so progress is visible but DEBUG is suppressed
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s – %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_pipeline")

# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on non-TTY terminals)
# ---------------------------------------------------------------------------
_USE_COLOUR = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _USE_COLOUR else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _USE_COLOUR else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOUR else s


def _section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(_bold(f"  {title}"))
    print("─" * 65)


def _ok(label: str, value: Any = "") -> None:
    tick = _green("✓")
    print(f"  {tick}  {label}" + (f":  {value}" if value != "" else ""))


def _fail(label: str, detail: str = "") -> None:
    cross = _red("✗")
    print(f"  {cross}  {label}" + (f":  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Assertion helper: raises AssertionError with a clear message on failure
# ---------------------------------------------------------------------------
def _assert(condition: bool, message: str, detail: str = "") -> None:
    if not condition:
        full = f"ASSERTION FAILED – {message}" + (f"\n  Detail: {detail}" if detail else "")
        _fail(message, detail)
        raise AssertionError(full)
    _ok(message)


# ===========================================================================
# Stage 1 – Load data
# ===========================================================================

def stage_load_data() -> Dict[str, Any]:
    """
    Load target observations, location map, and vaccination curves.

    Returns
    -------
    dict with keys:
        obs_series   – np.ndarray of US inc hosp weekly counts (recent 20 weeks)
        cal_df       – calibration DataFrame (date, inc_hosp, inc_death)
        pop_map      – dict {fips: population}
        epiweek_dates– pd.DatetimeIndex of 104 horizon Saturdays
    """
    _section("Stage 1 – Load data")

    from load_data import (
        load_target_data,
        build_calibration_data,
        get_population_map,
        build_epiweek_dates,
        ORIGIN_DATE,
        N_HORIZON_WEEKS,
        FIT_END_DATE,
        SCENARIO_IDS,
    )

    # ── 1a. load_target_data ───────────────────────────────────────────────
    ts = load_target_data(
        location_filter=["US"],
        target_filter=["inc hosp", "inc death"],
        age_group_filter=["0-130"],
        min_date=pd.Timestamp("2024-01-01"),
        max_date=FIT_END_DATE,
    )
    _assert(not ts.empty,
            "load_target_data returned non-empty DataFrame")
    _assert(set(ts.columns) == {"location", "date", "observation", "age_group", "target"},
            "load_target_data has correct columns",
            str(ts.columns.tolist()))
    _assert(ts["date"].dtype == "datetime64[us]" or np.issubdtype(ts["date"].dtype, np.datetime64),
            "date column is datetime dtype",
            str(ts["date"].dtype))
    _assert(ts["observation"].dtype in (np.float64, np.int64, "int64", "float64"),
            "observation column is numeric",
            str(ts["observation"].dtype))
    _assert(ts["location"].iloc[0] == "US",
            "location values are strings (FIPS preserved)")
    _assert((ts["target"].isin(["inc hosp", "inc death"])).all(),
            "target values in valid set")
    print(f"       rows loaded: {len(ts):,}  "
          f"date range: {ts.date.min().date()} → {ts.date.max().date()}")

    # ── 1b. build_calibration_data ────────────────────────────────────────
    cal = build_calibration_data(location="US")
    _assert(not cal.empty,
            "build_calibration_data returned non-empty DataFrame")
    _assert(list(cal.columns) == ["date", "inc_hosp", "inc_death"],
            "calibration_data has exactly [date, inc_hosp, inc_death]",
            str(cal.columns.tolist()))
    _assert(cal["inc_hosp"].notna().sum() > 0,
            "calibration_data has non-null inc_hosp values")
    print(f"       calibration rows: {len(cal)}  "
          f"NaN inc_hosp: {cal.inc_hosp.isna().sum()}  "
          f"NaN inc_death: {cal.inc_death.isna().sum()}")

    # ── 1c. get_population_map ────────────────────────────────────────────
    pop_map = get_population_map()
    _assert("US" in pop_map,
            "population map contains 'US' key")
    _assert(pop_map["US"] > 300_000_000,
            "US population > 300 million",
            f"{pop_map['US']:,}")
    _assert(len(pop_map) >= 50,
            "population map has at least 50 entries (50 states + territories)",
            str(len(pop_map)))
    print(f"       locations in population map: {len(pop_map)}")
    print(f"       US population: {pop_map['US']:,}")

    # ── 1d. build_epiweek_dates ───────────────────────────────────────────
    dates = build_epiweek_dates(ORIGIN_DATE, n_weeks=N_HORIZON_WEEKS)
    _assert(len(dates) == N_HORIZON_WEEKS,
            f"epiweek_dates has exactly {N_HORIZON_WEEKS} horizons",
            str(len(dates)))
    # Horizon 1 should be the Saturday of the first epi-week
    expected_h1 = ORIGIN_DATE + pd.Timedelta(days=6)
    _assert(dates[0] == expected_h1,
            f"horizon 1 date is {expected_h1.date()} (Saturday after origin)",
            str(dates[0].date()))
    expected_h104 = pd.Timestamp("2027-06-05")
    _assert(dates[-1] == expected_h104,
            f"horizon 104 date is {expected_h104.date()} (simulation end date)",
            str(dates[-1].date()))
    print(f"       epiweek horizon 1: {dates[0].date()}  "
          f"horizon 104: {dates[-1].date()}")

    # Extract the most recent 20 weeks of US inc hosp for trajectory seeding
    hosp = (
        ts[ts["target"] == "inc hosp"]
        .sort_values("date")
        .tail(20)["observation"]
        .values.astype(float)
    )
    _assert(len(hosp) >= 8,
            f"at least 8 recent inc hosp observations available for seeding",
            str(len(hosp)))

    return {
        "obs_series":    hosp,
        "cal_df":        cal,
        "pop_map":       pop_map,
        "epiweek_dates": dates,
    }


# ===========================================================================
# Stage 2 – Generate 300 stochastic trajectories
# ===========================================================================

def stage_generate_trajectories(
    obs_series: np.ndarray,
    epiweek_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Generate 300 stochastic trajectories and validate shape / content.

    Returns the raw baseline trajectory DataFrame.
    """
    _section("Stage 2 – Generate 300 stochastic trajectories")

    from stochastic_simulate import (
        estimate_log_growth_params,
        generate_trajectories,
        DEFAULT_N_TRAJECTORIES,
        DEFAULT_N_WEEKS,
        DEFAULT_MIN_COUNT,
    )

    # ── 2a. Parameter estimation ──────────────────────────────────────────
    mu, sigma = estimate_log_growth_params(obs_series, lookback_weeks=8)
    _assert(np.isfinite(mu),
            f"estimated log-growth mean μ is finite",
            f"μ = {mu:.4f}")
    _assert(sigma > 0,
            f"estimated log-growth std σ > 0",
            f"σ = {sigma:.4f}")
    print(f"       estimated μ = {mu:.4f}  σ = {sigma:.4f}")

    # ── 2b. Trajectory generation ─────────────────────────────────────────
    trajs = generate_trajectories(
        observations=obs_series,
        forecast_dates=epiweek_dates,
        n_trajectories=DEFAULT_N_TRAJECTORIES,
        seed=42,           # fixed seed ensures reproducibility
    )

    expected_rows = DEFAULT_N_TRAJECTORIES * DEFAULT_N_WEEKS
    _assert(len(trajs) == expected_rows,
            f"trajectory DataFrame has {expected_rows:,} rows "
            f"({DEFAULT_N_TRAJECTORIES} × {DEFAULT_N_WEEKS})",
            str(len(trajs)))
    _assert(set(trajs.columns) == {"trajectory_id", "date", "forecast"},
            "trajectory DataFrame has exactly [trajectory_id, date, forecast]",
            str(trajs.columns.tolist()))
    _assert(trajs["trajectory_id"].nunique() == DEFAULT_N_TRAJECTORIES,
            f"exactly {DEFAULT_N_TRAJECTORIES} unique trajectory_ids",
            str(trajs["trajectory_id"].nunique()))
    _assert(trajs["trajectory_id"].min() == 1,
            "trajectory_id starts at 1",
            str(trajs["trajectory_id"].min()))
    _assert(trajs["trajectory_id"].max() == DEFAULT_N_TRAJECTORIES,
            f"trajectory_id max = {DEFAULT_N_TRAJECTORIES}",
            str(trajs["trajectory_id"].max()))
    _assert(trajs["date"].nunique() == DEFAULT_N_WEEKS,
            f"exactly {DEFAULT_N_WEEKS} unique dates (one per horizon)",
            str(trajs["date"].nunique()))
    _assert(not trajs["forecast"].isna().any(),
            "no NaN in forecast column")
    _assert((trajs["forecast"] >= DEFAULT_MIN_COUNT).all(),
            f"all forecast values ≥ min_count ({DEFAULT_MIN_COUNT})",
            f"min forecast = {trajs.forecast.min():.2f}")

    print(f"       rows: {len(trajs):,}  "
          f"trajectories: {trajs.trajectory_id.nunique()}  "
          f"horizons: {trajs.date.nunique()}")
    print(f"       forecast range: {trajs.forecast.min():.1f} – "
          f"{trajs.forecast.max():.1f}")

    return trajs


# ===========================================================================
# Stage 3 – Apply all five Round 20 scenarios
# ===========================================================================

def stage_apply_scenarios(baseline: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Apply Scenarios A–E and validate multiplier semantics.

    Returns
    -------
    dict mapping scenario letter → adjusted trajectory DataFrame
    """
    _section("Stage 3 – Apply Round 20 scenarios A–E")

    from scenario_adjustments import (
        apply_scenario,
        apply_all_scenarios,
        SCENARIO_IDS,
        SCENARIO_SPECS,
    )

    # ── 3a. apply_all_scenarios ───────────────────────────────────────────
    combined = apply_all_scenarios(baseline)
    n_scen = combined["scenario_id"].nunique()
    _assert(n_scen == 5,
            "apply_all_scenarios produces exactly 5 scenario groups",
            str(n_scen))
    _assert(set(combined["scenario_id"].unique()) == set(SCENARIO_IDS.values()),
            "all five official scenario IDs are present",
            str(sorted(combined["scenario_id"].unique())))

    # Split into per-scenario dicts
    scen_dfs: Dict[str, pd.DataFrame] = {}
    for key, sid in SCENARIO_IDS.items():
        sub = combined[combined["scenario_id"] == sid].copy()
        _assert(not sub.empty,
                f"scenario {key} ({sid}) has rows",
                str(len(sub)))
        _assert(sub["trajectory_id"].nunique() == 300,
                f"scenario {key} has 300 trajectories",
                str(sub["trajectory_id"].nunique()))
        scen_dfs[key] = sub
        print(f"       scenario {key}: {len(sub):,} rows  "
              f"multiplier range [{sub.multiplier.min():.3f}, "
              f"{sub.multiplier.max():.3f}]")

    # ── 3b. Scenario A multiplier must equal 1.0 everywhere ───────────────
    a_mult = scen_dfs["A"]["multiplier"]
    _assert((a_mult == 1.0).all(),
            "Scenario A multiplier is exactly 1.0 for all rows (reference case)",
            f"unique multipliers: {a_mult.unique()[:3]}")

    # ── 3c. Vaccination scenarios reduce expected counts vs A ─────────────
    for key in ("B", "C", "D", "E"):
        sid = SCENARIO_IDS[key]
        # Compare median forecast of this scenario vs A during fall 2026 window
        fall_mask = (
            (combined["date"] >= pd.Timestamp("2026-10-01")) &
            (combined["date"] <= pd.Timestamp("2027-01-31"))
        )
        med_a = combined.loc[
            fall_mask & (combined["scenario_id"] == SCENARIO_IDS["A"]),
            "forecast"
        ].median()
        med_k = combined.loc[
            fall_mask & (combined["scenario_id"] == sid),
            "forecast"
        ].median()
        _assert(
            med_k <= med_a,
            f"Scenario {key} median forecast ≤ Scenario A during fall 2026 "
            f"(vaccination reduces burden)",
            f"A={med_a:.1f}  {key}={med_k:.1f}",
        )

    # ── 3d. Multipliers respect coverage ordering: D < B (higher coverage → more reduction)
    # Scenarios D (optimistic) should have a lower multiplier nadir than B (BaU)
    min_mult_b = scen_dfs["B"]["multiplier"].min()
    min_mult_d = scen_dfs["D"]["multiplier"].min()
    _assert(
        min_mult_d <= min_mult_b,
        "Scenario D (optimistic) minimum multiplier ≤ Scenario B (BaU) "
        "(higher coverage → greater hospitalisation reduction)",
        f"min_mult_B={min_mult_b:.3f}  min_mult_D={min_mult_d:.3f}",
    )

    # ── 3e. Invalid scenario raises ValueError ────────────────────────────
    raised = False
    try:
        apply_scenario(baseline, "Z")
    except ValueError:
        raised = True
    _assert(raised,
            "apply_scenario('Z') raises ValueError for unknown scenario key")

    return scen_dfs


# ===========================================================================
# Stage 4 – Build submission DataFrame
# ===========================================================================

def stage_build_submission(
    scen_dfs: Dict[str, pd.DataFrame],
    output_dir: pathlib.Path,
) -> tuple[pd.DataFrame, pathlib.Path]:
    """
    Assemble the hub submission, validate schema, and write parquet.

    Returns
    -------
    (submission_df, parquet_path)
    """
    _section("Stage 4 – Build and validate submission parquet")

    from build_submission import (
        build_submission,
        validate_submission,
        read_submission_parquet,
        ORIGIN_DATE,
        N_TRAJECTORIES,
        N_HORIZONS,
        HUB_QUANTILES,
        _COLUMN_ORDER,
    )

    sub = build_submission(
        scen_dfs,
        location="US",
        age_group="0-130",
        origin_date=ORIGIN_DATE,
        include_cumulative=True,
        include_quantiles=True,
        output_dir=output_dir,
        team_model="MyTeam-ProtoModel",
    )

    # ── 4a. Column set ────────────────────────────────────────────────────
    _assert(set(sub.columns) == set(_COLUMN_ORDER),
            "submission has exactly the 11 required hub columns",
            str(sorted(sub.columns.tolist())))

    # ── 4b. Row counts ────────────────────────────────────────────────────
    n_scenarios   = sub["scenario_id"].nunique()
    n_sample_rows = (sub["output_type"] == "sample").sum()
    n_q_rows      = (sub["output_type"] == "quantile").sum()

    # Expected sample rows:
    # 5 scenarios × 4 targets (inc+cum hosp+death) × 104 horizons × 300 trajectories
    expected_sample = 5 * 4 * N_HORIZONS * N_TRAJECTORIES
    _assert(n_sample_rows == expected_sample,
            f"exactly {expected_sample:,} sample rows "
            f"(5 scenarios × 4 targets × {N_HORIZONS} horizons × {N_TRAJECTORIES} trajectories)",
            f"actual: {n_sample_rows:,}")

    # Expected quantile rows: 5 × 4 × 104 × 23
    expected_q = 5 * 4 * N_HORIZONS * len(HUB_QUANTILES)
    _assert(n_q_rows == expected_q,
            f"exactly {expected_q:,} quantile rows "
            f"(5 scenarios × 4 targets × {N_HORIZONS} horizons × {len(HUB_QUANTILES)} quantiles)",
            f"actual: {n_q_rows:,}")

    print(f"       total rows:     {len(sub):,}")
    print(f"       sample rows:    {n_sample_rows:,}")
    print(f"       quantile rows:  {n_q_rows:,}")
    print(f"       scenarios:      {n_scenarios}")

    # ── 4c. Schema validation (runs validate_submission internally, but we
    #        call it again explicitly to test that it doesn't raise) ────────
    sample_only = sub[sub["output_type"] == "sample"].copy()
    try:
        validate_submission(sample_only)
        _ok("validate_submission() passes without raising")
    except ValueError as exc:
        _fail("validate_submission() raised ValueError", str(exc))
        raise

    # ── 4d. Value bounds ──────────────────────────────────────────────────
    _assert((sub["value"] >= 0).all(),
            "all values are non-negative",
            f"min value = {sub.value.min()}")
    _assert(sub["horizon"].min() == 1,
            "horizon minimum is 1",
            str(sub.horizon.min()))
    _assert(sub["horizon"].max() == N_HORIZONS,
            f"horizon maximum is {N_HORIZONS}",
            str(sub.horizon.max()))

    # ── 4e. Parquet file ──────────────────────────────────────────────────
    parquet_files = sorted(output_dir.glob("*.gz.parquet"))
    _assert(len(parquet_files) >= 1,
            "at least one .gz.parquet file was written",
            str(parquet_files))

    parquet_path = parquet_files[0]
    size_mb = parquet_path.stat().st_size / 1e6
    _assert(size_mb > 0,
            f"parquet file is non-empty",
            f"{size_mb:.2f} MB")
    print(f"       parquet path:   {parquet_path}")
    print(f"       file size:      {size_mb:.2f} MB")

    # ── 4f. Round-trip read ───────────────────────────────────────────────
    rt = read_submission_parquet(parquet_path)
    _assert(len(rt) == len(sub),
            f"round-trip row count matches ({len(sub):,} rows)",
            f"read back: {len(rt):,}")
    _assert(set(rt.columns) == set(_COLUMN_ORDER),
            "round-trip column set matches submission schema")
    _ok("parquet round-trip read successful")

    return sub, parquet_path


# ===========================================================================
# Stage 5 – Generate plots
# ===========================================================================

def stage_plot(
    parquet_path: pathlib.Path,
    plot_dir: pathlib.Path,
) -> Dict[str, pathlib.Path]:
    """
    Generate both standard figures and validate they were written to disk.

    Returns
    -------
    dict ``{"scenario_comparison": Path, "submission_plot": Path}``
    """
    _section("Stage 5 – Generate plots")

    from plot_submission import generate_all_plots

    plot_paths = generate_all_plots(
        parquet_path=parquet_path,
        output_dir=plot_dir,
        location="US",
        age_group="0-130",
    )

    expected_figures = {"scenario_comparison", "submission_plot"}
    _assert(set(plot_paths.keys()) == expected_figures,
            f"generate_all_plots returns both expected figures: {expected_figures}",
            str(set(plot_paths.keys())))

    for name, path in plot_paths.items():
        _assert(path.exists(),
                f"plot file exists: {name}",
                str(path))
        size_kb = path.stat().st_size / 1024
        _assert(size_kb > 50,
                f"{name}.png is at least 50 KB (non-trivial image)",
                f"{size_kb:.0f} KB")
        print(f"       {name}: {path.name}  ({size_kb:.0f} KB)")

    return plot_paths


# ===========================================================================
# Full pipeline runner
# ===========================================================================

def run_pipeline(tmp_dir: pathlib.Path) -> None:
    """
    Run all five pipeline stages in sequence.

    Parameters
    ----------
    tmp_dir : Temporary working directory for parquet and plot output.

    Raises
    ------
    Any exception propagated from a failing stage or assertion.
    """
    output_dir = tmp_dir / "model-output" / "MyTeam-ProtoModel"
    plot_dir   = tmp_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────
    stage1 = stage_load_data()

    # ── Stage 2 ───────────────────────────────────────────────────────────
    baseline = stage_generate_trajectories(
        obs_series=stage1["obs_series"],
        epiweek_dates=stage1["epiweek_dates"],
    )

    # ── Stage 3 ───────────────────────────────────────────────────────────
    scen_dfs = stage_apply_scenarios(baseline)

    # ── Stage 4 ───────────────────────────────────────────────────────────
    sub_df, parquet_path = stage_build_submission(scen_dfs, output_dir)

    # ── Stage 5 ───────────────────────────────────────────────────────────
    plot_paths = stage_plot(parquet_path, plot_dir)

    # ── Final report ──────────────────────────────────────────────────────
    _section("Pipeline summary")
    print(f"       Total submission rows:    {len(sub_df):,}")
    print(f"       Sample rows:              {(sub_df['output_type']=='sample').sum():,}")
    print(f"       Quantile rows:            {(sub_df['output_type']=='quantile').sum():,}")
    print(f"       Trajectories per group:   {sub_df[sub_df['output_type']=='sample']['stochastic_run'].max():.0f}")
    print(f"       Scenarios:                {sorted(sub_df['scenario_id'].unique())}")
    print(f"       Targets:                  {sorted(sub_df['target'].unique())}")
    print(f"       Horizon range:            {sub_df['horizon'].min()} – {sub_df['horizon'].max()}")
    print(f"       Output parquet:           {parquet_path}")
    for name, path in plot_paths.items():
        print(f"       Plot [{name}]: {path}")


# ===========================================================================
# pytest-compatible test function
# ===========================================================================

def test_end_to_end_pipeline(tmp_path: pathlib.Path) -> None:
    """
    pytest entry point.  ``tmp_path`` is provided by pytest's built-in fixture.

    Run with:
        pytest tests/test_pipeline.py -v
    """
    run_pipeline(tmp_path)
    print(_green("\n✓ All pipeline stages passed."))


# ===========================================================================
# Standalone runner (python3 tests/test_pipeline.py)
# ===========================================================================

if __name__ == "__main__":
    print(_bold("=" * 65))
    print(_bold("  Round 20 Pipeline – End-to-End Integration Test"))
    print(_bold("=" * 65))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        try:
            run_pipeline(tmp_path)
            print("\n" + "=" * 65)
            print(_green(_bold("  ✓  ALL STAGES PASSED")))
            print("=" * 65)
            sys.exit(0)
        except Exception:
            print("\n" + "=" * 65)
            print(_red(_bold("  ✗  PIPELINE FAILED")))
            print("=" * 65)
            traceback.print_exc()
            sys.exit(1)
