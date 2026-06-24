"""
plot_submission.py
==================
Publication-quality matplotlib figures for Round 20 Scenario Modeling Hub
submissions.

Produces two output files
--------------------------
``scenario_comparison.png``
    One panel per target (inc hosp / inc death), five coloured scenario
    curves on the same axes, observed data overlay, and a shared legend.
    Designed for direct inclusion in reports and model abstracts.

``submission_plot.png``
    2 × 2 panel grid (inc hosp, inc death, cum hosp, cum death).
    Each panel shows all five scenarios with median + 5th–95th shaded band.
    Includes a calibration data overlay and vertical lines marking key
    vaccination campaign dates.

Design choices
--------------
- Quantile bands are computed from ``output_type == "quantile"`` rows
  already present in the parquet (p05 / p50 / p95).  If quantile rows
  are absent, they are computed on-the-fly from sample trajectories.
- Observed data are read directly from ``target-data/time-series.csv``
  and shown as black scatter points behind the scenario curves.
- Vaccination campaign dates are marked with vertical dashed lines to
  help readers associate wave timing with intervention windows.
- All colour, style, and layout constants are defined at module level so
  they can be overridden without modifying function bodies.
- The ``Agg`` backend is forced so the module works in headless
  environments (CI, servers) without a display.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless-safe; must be set before pyplot import
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths (relative to the repository root my_model/ folder)
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent

# Default output directory is my_model/ itself; callers may override
DEFAULT_OUTPUT_DIR: pathlib.Path = _HERE

# ---------------------------------------------------------------------------
# Scenario display configuration
# ---------------------------------------------------------------------------
# Each entry: (display label, hex colour, linestyle)
SCENARIO_STYLE: Dict[str, Tuple[str, str, str]] = {
    "A-2026-05-11": ("Scenario A – No booster",             "#555555", "solid"),
    "B-2026-05-11": ("Scenario B – BaU annual",             "#1f77b4", "solid"),
    "C-2026-05-11": ("Scenario C – BaU semi-annual HR",     "#ff7f0e", "solid"),
    "D-2026-05-11": ("Scenario D – Optimistic annual",      "#2ca02c", "dashed"),
    "E-2026-05-11": ("Scenario E – Optimistic semi-annual HR", "#d62728", "dashed"),
}

# Shade alpha for the 5th–95th percentile band
BAND_ALPHA: float = 0.15

# ---------------------------------------------------------------------------
# Target display names and y-axis labels
# ---------------------------------------------------------------------------
TARGET_META: Dict[str, Dict[str, str]] = {
    "inc hosp":  {"title": "Weekly Incident Hospitalisations",
                  "ylabel": "Weekly new hospitalisations"},
    "inc death": {"title": "Weekly Incident Deaths",
                  "ylabel": "Weekly new deaths"},
    "cum hosp":  {"title": "Cumulative Hospitalisations (since Jun 2025)",
                  "ylabel": "Cumulative hospitalisations"},
    "cum death": {"title": "Cumulative Deaths (since Jun 2025)",
                  "ylabel": "Cumulative deaths"},
}

# ---------------------------------------------------------------------------
# Vaccination campaign reference dates (from round20.md)
# ---------------------------------------------------------------------------
VAX_LINES: List[Tuple[str, str]] = [
    ("2025-08-13", "Fall 2025-26\ncampaign start"),
    ("2026-02-15", "Spring 2026\ncampaign start\n(C & E only)"),
    ("2026-08-14", "Fall 2026-27\ncampaign start"),
]

# Calibration data cutoff (no observations after this date)
FIT_END_DATE: pd.Timestamp = pd.Timestamp("2026-06-06")

# Round 20 origin date
ORIGIN_DATE: pd.Timestamp = pd.Timestamp("2025-06-08")

# ---------------------------------------------------------------------------
# Global matplotlib style settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.fontsize":    8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
})


# ===========================================================================
# Section 1 – Data loading helpers
# ===========================================================================

def load_submission(path: pathlib.Path) -> pd.DataFrame:
    """
    Load a ``.gz.parquet`` submission file and add a ``date`` column.

    The ``date`` column is computed from ``origin_date + horizon weeks``
    (Saturday of epi-week).

    Parameters
    ----------
    path : Path to the submission parquet file.

    Returns
    -------
    pd.DataFrame with all eleven hub columns plus ``date`` (datetime64).
    """
    table = pq.read_table(str(path))
    df = table.to_pandas()

    df["origin_date"] = pd.to_datetime(df["origin_date"])
    origin = df["origin_date"].iloc[0]

    # horizon k → Saturday date = origin + 6 days + 7*(k-1) days
    # (origin is a Sunday; horizon 1 ends on the first Saturday)
    df["date"] = df["horizon"].apply(
        lambda h: origin + pd.Timedelta(days=6 + 7 * (int(h) - 1))
    )

    logger.info(
        "load_submission: loaded %d rows from %s  "
        "horizons %d–%d  scenarios=%s",
        len(df), path.name,
        df["horizon"].min(), df["horizon"].max(),
        sorted(df["scenario_id"].unique()),
    )
    return df


def load_observed(
    location: str = "US",
    age_group: str = "0-130",
) -> pd.DataFrame:
    """
    Load observed inc hosp and inc death from ``target-data/time-series.csv``.

    Returns a tidy DataFrame with columns ``date``, ``target``, ``value``.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    path = repo_root / "target-data" / "time-series.csv"
    if not path.exists():
        logger.warning("load_observed: %s not found; skipping observed overlay.", path)
        return pd.DataFrame(columns=["date", "target", "value"])

    raw = pd.read_csv(path, dtype={"location": str})
    raw["date"] = pd.to_datetime(raw["date"])
    obs = raw[
        (raw["location"] == location)
        & (raw["age_group"] == age_group)
        & (raw["target"].isin(["inc hosp", "inc death"]))
        & (raw["date"] >= ORIGIN_DATE - pd.Timedelta(weeks=8))
        & (raw["date"] <= FIT_END_DATE)
    ].rename(columns={"observation": "value"})[["date", "target", "value"]]
    return obs


def _get_quantile_series(
    df: pd.DataFrame,
    scenario_id: str,
    target: str,
    quantile: float,
    age_group: str = "0-130",
) -> pd.Series:
    """
    Extract a quantile time-series from the submission DataFrame.

    If quantile rows are present, use them directly.  Otherwise fall back
    to computing the quantile from sample rows (slower but always works).

    Returns a pd.Series indexed by ``date``, sorted chronologically.
    """
    mask = (
        (df["scenario_id"] == scenario_id)
        & (df["target"] == target)
        & (df["age_group"] == age_group)
    )
    q_mask = mask & (df["output_type"] == "quantile") & (df["output_type_id"] == quantile)
    q_rows = df[q_mask].sort_values("date")

    if len(q_rows) > 0:
        return q_rows.set_index("date")["value"]

    # Fallback: compute from sample rows
    s_mask = mask & (df["output_type"] == "sample")
    s_rows = df[s_mask]
    if s_rows.empty:
        return pd.Series(dtype=float)

    return (
        s_rows.groupby("date")["value"]
        .quantile(quantile)
        .sort_index()
    )


# ===========================================================================
# Section 2 – Axes-level drawing primitives
# ===========================================================================

def _draw_scenario_band(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    p05: np.ndarray,
    p50: np.ndarray,
    p95: np.ndarray,
    color: str,
    linestyle: str,
    label: str,
    band_alpha: float = BAND_ALPHA,
) -> None:
    """
    Draw median line + shaded 5th–95th band for one scenario on *ax*.

    Parameters
    ----------
    ax        : Matplotlib axes to draw on.
    dates     : x-axis date values.
    p05       : 5th-percentile values.
    p50       : Median (50th-percentile) values.
    p95       : 95th-percentile values.
    color     : Hex colour string.
    linestyle : Matplotlib linestyle ("solid", "dashed", etc.).
    label     : Legend label for the median line.
    band_alpha: Transparency of the shaded band (default 0.15).
    """
    ax.fill_between(dates, p05, p95, color=color, alpha=band_alpha, linewidth=0)
    ax.plot(dates, p50, color=color, linestyle=linestyle, linewidth=1.8,
            label=label, zorder=3)


def _draw_observed(
    ax: plt.Axes,
    obs: pd.DataFrame,
    target: str,
) -> None:
    """Overlay observed data points on *ax* as black scatter with connecting line."""
    sub = obs[obs["target"] == target].sort_values("date")
    if sub.empty:
        return
    ax.plot(sub["date"], sub["value"], color="black", linewidth=1.2,
            marker="o", markersize=3, zorder=5, label="Observed (NHSN/NCHS)",
            alpha=0.85)


def _draw_vax_lines(
    ax: plt.Axes,
    scenario_id: Optional[str] = None,
) -> None:
    """
    Add vertical dashed reference lines for vaccination campaign start dates.

    The spring 2026 line is shown in orange and only annotated with "(C & E)"
    to signal it is scenario-specific.
    """
    for i, (date_str, label) in enumerate(VAX_LINES):
        ts = pd.Timestamp(date_str)
        # Spring line uses a different colour to signal it is C/E-only
        is_spring = "Spring" in label
        color  = "#ff7f0e" if is_spring else "#888888"
        lw     = 1.0
        zorder = 2
        ax.axvline(ts, color=color, linewidth=lw, linestyle=":",
                   alpha=0.7, zorder=zorder)
        # Rotate annotation to avoid overlap; position at top of axes
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.97
        ax.text(
            ts, y_pos, f" {label}",
            rotation=90, fontsize=6, color=color, va="top",
            ha="left", zorder=zorder + 1,
        )


def _format_date_axis(ax: plt.Axes, n_weeks: int = 104) -> None:
    """Apply consistent date formatting and rotation to the x-axis."""
    # Major ticks every 3 months, minor every month
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=2)


def _format_yaxis(ax: plt.Axes) -> None:
    """Use thousands-separator formatting on y-axis."""
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x == int(x) else f"{x:,.0f}"
    ))


# ===========================================================================
# Section 3 – Figure 1: scenario_comparison.png
# ===========================================================================

def plot_scenario_comparison(
    df: pd.DataFrame,
    observed: pd.DataFrame,
    output_path: pathlib.Path,
    targets: List[str] = ("inc hosp", "inc death"),
    age_group: str = "0-130",
    location: str = "US",
) -> pathlib.Path:
    """
    Two-panel figure comparing all five scenarios side-by-side for
    ``inc hosp`` and ``inc death``.

    Layout
    ------
    Left panel : Weekly incident hospitalisations
    Right panel: Weekly incident deaths

    Each panel shows:
    - Median forecast line per scenario (coloured, styled per SCENARIO_STYLE)
    - Shaded 5th–95th percentile band (same colour, semi-transparent)
    - Observed data overlay (black points)
    - Vertical reference lines for vaccination campaign dates
    - Shared legend on the right

    Parameters
    ----------
    df          : Submission DataFrame (from :func:`load_submission`).
    observed    : Observed data DataFrame (from :func:`load_observed`).
    output_path : Destination file path (e.g. ``…/scenario_comparison.png``).
    targets     : Targets to plot (default inc hosp + inc death).
    age_group   : Age group filter (default "0-130").
    location    : FIPS location filter (default "US").

    Returns
    -------
    pathlib.Path  The saved file path.
    """
    df = df[df["location"] == location].copy() if "location" in df.columns else df

    n_panels = len(targets)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7.5 * n_panels, 5.5),
        sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    scenario_ids = [s for s in SCENARIO_STYLE if s in df["scenario_id"].unique()]

    for ax, target in zip(axes, targets):
        meta = TARGET_META.get(target, {"title": target, "ylabel": "Count"})

        # Draw observed data first (background layer)
        _draw_observed(ax, observed, target)

        # Draw each scenario band
        for sid in scenario_ids:
            label, color, linestyle = SCENARIO_STYLE[sid]
            p05 = _get_quantile_series(df, sid, target, 0.05,  age_group)
            p50 = _get_quantile_series(df, sid, target, 0.50,  age_group)
            p95 = _get_quantile_series(df, sid, target, 0.95,  age_group)

            if p50.empty:
                logger.warning(
                    "plot_scenario_comparison: no data for scenario=%s target=%s",
                    sid, target,
                )
                continue

            # Align all three series to the same index
            common_idx = p50.index
            p05 = p05.reindex(common_idx).ffill().bfill()
            p95 = p95.reindex(common_idx).ffill().bfill()

            _draw_scenario_band(
                ax, common_idx,
                p05.values, p50.values, p95.values,
                color, linestyle, label,
            )

        # Axis decoration
        ax.set_title(meta["title"])
        ax.set_xlabel("Date")
        ax.set_ylabel(meta["ylabel"])
        _format_date_axis(ax)
        _format_yaxis(ax)

        # Shade the retrospective period (Jun 2025 → Jun 2026)
        ax.axvspan(ORIGIN_DATE, FIT_END_DATE,
                   color="#eeeeee", alpha=0.4, zorder=0,
                   label="Retrospective period")

        ax.set_xlim(
            ORIGIN_DATE - pd.Timedelta(weeks=8),
            ORIGIN_DATE + pd.Timedelta(weeks=105),
        )
        ax.set_ylim(bottom=0)

        # Draw vaccination lines after ylim is set
        _draw_vax_lines(ax)

    # ── Legend ─────────────────────────────────────────────────────────────
    # Collect handles from the last axis and add a custom band patch
    handles, labels_leg = axes[-1].get_legend_handles_labels()
    band_patch = mpatches.Patch(
        facecolor="#888888", alpha=0.35, label="5th–95th percentile"
    )
    handles.append(band_patch)
    labels_leg.append("5th–95th percentile")

    fig.legend(
        handles, labels_leg,
        loc="lower center",
        ncol=min(len(handles), 4),
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        fontsize=8,
    )

    fig.suptitle(
        "COVID-19 Round 20 — Scenario Comparison\n"
        "United States · Jun 2025 – Jun 2027",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_scenario_comparison: saved → %s", output_path)
    return output_path


# ===========================================================================
# Section 4 – Figure 2: submission_plot.png
# ===========================================================================

def plot_submission(
    df: pd.DataFrame,
    observed: pd.DataFrame,
    output_path: pathlib.Path,
    targets: List[str] = ("inc hosp", "inc death", "cum hosp", "cum death"),
    age_group: str = "0-130",
    location: str = "US",
) -> pathlib.Path:
    """
    Four-panel 2 × 2 publication figure covering all four submission targets.

    Layout (2 rows × 2 columns)
    ---------------------------
    Row 0: inc hosp  |  inc death
    Row 1: cum hosp  |  cum death

    Each panel shows:
    - Median forecast line per scenario
    - Shaded 5th–95th band (semi-transparent fill)
    - Observed data overlay (black circles + connecting line)
    - Vertical dashed lines for vaccination campaign dates
    - Grey shading for the retrospective period (Jun 2025 – Jun 2026)

    Parameters
    ----------
    df          : Submission DataFrame (from :func:`load_submission`).
    observed    : Observed data DataFrame (from :func:`load_observed`).
    output_path : Destination path.
    targets     : Four targets to fill the 2×2 grid.
    age_group   : Age group (default "0-130").
    location    : FIPS code (default "US").

    Returns
    -------
    pathlib.Path  The saved file path.
    """
    df = df[df["location"] == location].copy() if "location" in df.columns else df

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 10),
        sharey=False,
    )
    axes_flat = axes.flatten()

    scenario_ids = [s for s in SCENARIO_STYLE if s in df["scenario_id"].unique()]

    for ax, target in zip(axes_flat, targets):
        meta = TARGET_META.get(target, {"title": target, "ylabel": "Count"})

        # Retrospective shading (behind everything)
        ax.axvspan(ORIGIN_DATE, FIT_END_DATE,
                   color="#eeeeee", alpha=0.45, zorder=0,
                   label="Retrospective period (Jun 2025 – Jun 2026)")

        # Observed data
        _draw_observed(ax, observed, target)

        # Scenario bands
        for sid in scenario_ids:
            label, color, linestyle = SCENARIO_STYLE[sid]
            p05 = _get_quantile_series(df, sid, target, 0.05, age_group)
            p50 = _get_quantile_series(df, sid, target, 0.50, age_group)
            p95 = _get_quantile_series(df, sid, target, 0.95, age_group)

            if p50.empty:
                continue

            common_idx = p50.index
            p05 = p05.reindex(common_idx).ffill().bfill()
            p95 = p95.reindex(common_idx).ffill().bfill()

            _draw_scenario_band(
                ax, common_idx,
                p05.values, p50.values, p95.values,
                color, linestyle, label,
            )

        # Axis decoration
        ax.set_title(meta["title"], pad=6)
        ax.set_ylabel(meta["ylabel"])
        ax.set_xlabel("")
        _format_date_axis(ax)
        _format_yaxis(ax)
        ax.set_xlim(
            ORIGIN_DATE - pd.Timedelta(weeks=8),
            ORIGIN_DATE + pd.Timedelta(weeks=105),
        )
        ax.set_ylim(bottom=0)

        # Vaccination lines (after ylim is set so annotation y is correct)
        _draw_vax_lines(ax)

    # Add x-label only to bottom row
    for ax in axes[1]:
        ax.set_xlabel("Date")

    # ── Legend ─────────────────────────────────────────────────────────────
    # Build legend from last panel; add a manual band patch
    handles, labels_leg = axes_flat[0].get_legend_handles_labels()
    band_patch = mpatches.Patch(
        facecolor="#888888", alpha=0.35, label="5th–95th percentile band"
    )
    retro_patch = mpatches.Patch(
        facecolor="#cccccc", alpha=0.6, label="Retrospective period"
    )
    # Deduplicate: keep only scenario lines + band patch + retro patch
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen and "Retrospective" not in l:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    unique_handles += [band_patch, retro_patch]
    unique_labels  += [band_patch.get_label(), retro_patch.get_label()]

    fig.legend(
        unique_handles, unique_labels,
        loc="lower center",
        ncol=min(len(unique_handles), 4),
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        fontsize=8.5,
    )

    fig.suptitle(
        "COVID-19 Round 20 — Full Submission Overview\n"
        "United States · All Targets · Jun 2025 – Jun 2027\n"
        "Shaded band = 5th–95th percentile across 300 trajectories",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_submission: saved → %s", output_path)
    return output_path


# ===========================================================================
# Section 5 – Convenience wrapper: generate_all_plots
# ===========================================================================

def generate_all_plots(
    parquet_path: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
    location: str = "US",
    age_group: str = "0-130",
) -> Dict[str, pathlib.Path]:
    """
    Load a submission parquet and generate both standard figures.

    Parameters
    ----------
    parquet_path : Path to the ``.gz.parquet`` submission file.
    output_dir   : Directory where PNGs are saved.  Defaults to the same
                   directory as the parquet file.
    location     : FIPS code to plot (default ``"US"``).
    age_group    : Age group to plot (default ``"0-130"``).

    Returns
    -------
    dict with keys ``"scenario_comparison"`` and ``"submission_plot"``,
    values are the absolute paths of the written PNG files.
    """
    if output_dir is None:
        output_dir = parquet_path.parent

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("generate_all_plots: loading %s", parquet_path)
    df = load_submission(parquet_path)
    obs = load_observed(location=location, age_group=age_group)

    paths: Dict[str, pathlib.Path] = {}

    # Figure 1 – scenario comparison (inc hosp + inc death, all scenarios)
    p1 = plot_scenario_comparison(
        df, obs,
        output_path=output_dir / "scenario_comparison.png",
        targets=["inc hosp", "inc death"],
        age_group=age_group,
        location=location,
    )
    paths["scenario_comparison"] = p1

    # Figure 2 – full submission overview (all four targets)
    available_targets = [
        t for t in ["inc hosp", "inc death", "cum hosp", "cum death"]
        if t in df["target"].unique()
    ]
    p2 = plot_submission(
        df, obs,
        output_path=output_dir / "submission_plot.png",
        targets=available_targets[:4],
        age_group=age_group,
        location=location,
    )
    paths["submission_plot"] = p2

    return paths


# ===========================================================================
# CLI smoke-test / standalone runner
# ===========================================================================
if __name__ == "__main__":
    sys.path.insert(0, str(_HERE))

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s – %(message)s",
    )

    import tempfile
    from load_data import build_epiweek_dates, ORIGIN_DATE as _ORIG
    from stochastic_simulate import generate_trajectories, DEFAULT_N_TRAJECTORIES
    from scenario_adjustments import apply_all_scenarios, SCENARIO_IDS
    from build_submission import build_submission, ORIGIN_DATE as _ORIG_STR

    print("=" * 65)
    print("plot_submission.py – smoke-test")
    print("=" * 65)

    # ── 1. Build a full submission in a temp directory ────────────────────
    obs_array = np.array([
        6967, 6604, 6405, 6480, 6335, 5686, 4992, 4321,
        3970, 2270, 1971, 1737, 1693, 1778, 1615, 1447,
        1349, 1236, 1046,  955,
    ], dtype=float)

    import numpy as np

    dates = build_epiweek_dates(_ORIG, n_weeks=104)
    print(f"\nGenerating {DEFAULT_N_TRAJECTORIES} trajectories …")
    base = generate_trajectories(obs_array, dates, seed=42,
                                 n_trajectories=DEFAULT_N_TRAJECTORIES)

    print("Applying scenarios A–E …")
    all_scen = apply_all_scenarios(base)
    scen_dfs = {
        k: all_scen[all_scen["scenario_id"] == v].copy()
        for k, v in SCENARIO_IDS.items()
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        sub_dir = pathlib.Path(tmpdir) / "model-output" / "MyTeam-ProtoModel"
        plot_dir = pathlib.Path(tmpdir) / "figures"

        print("\nBuilding submission parquet …")
        sub_df = build_submission(
            scen_dfs,
            location="US",
            age_group="0-130",
            origin_date=_ORIG_STR,
            include_cumulative=True,
            include_quantiles=True,
            output_dir=sub_dir,
        )
        parquet_files = list(sub_dir.glob("*.gz.parquet"))
        assert parquet_files, "No parquet file written!"
        parquet_path = parquet_files[0]
        print(f"  Parquet: {parquet_path.name}  ({parquet_path.stat().st_size/1e6:.1f} MB)")

        # ── 2. Generate plots ─────────────────────────────────────────────
        print("\nGenerating plots …")
        plot_paths = generate_all_plots(
            parquet_path=parquet_path,
            output_dir=plot_dir,
            location="US",
            age_group="0-130",
        )

        for name, path in plot_paths.items():
            size_kb = path.stat().st_size / 1024
            print(f"  {name}: {path.name}  ({size_kb:.0f} KB)")

        # ── 3. Copy to my_model/ for inspection ──────────────────────────
        import shutil
        for name, src in plot_paths.items():
            dst = _HERE / src.name
            shutil.copy(src, dst)
            print(f"  Copied → {dst}")

    print("\n✓ Smoke-test passed. Check my_model/ for PNG files.")
