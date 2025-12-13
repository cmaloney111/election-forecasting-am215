#!/usr/bin/env python3
"""
Shared data loading and preprocessing utilities.

This version supports multiple election cycles (e.g. 2012, 2016, 2020) and
both the original 2016 timeseries file and FiveThirtyEight-style long polls
files (like 2020_president_polls.csv).
"""
# mypy: ignore-errors

from typing import Dict, List, Optional
import pandas as pd  # type: ignore[import-untyped]
import numpy as np

# ---------------------------------------------------------------------
# Global election configuration
# ---------------------------------------------------------------------

CURRENT_ELECTION_YEAR: int = 2016
CURRENT_POLLS_FILE: Optional[str] = None  # if None, use sensible default per year


def set_election_config(year: int = 2016, polls_file: Optional[str] = None) -> None:
    """
    Configure which election cycle the rest of the module should use.

    Args:
        year: Election year (e.g. 2012, 2016, 2020).
        polls_file: Optional path to a FiveThirtyEight-style polls CSV.
            If None, we:
              - use the original 2016 timeseries file for year=2016
              - otherwise fall back to f"data/polls/{year}_president_polls.csv"
    """
    global CURRENT_ELECTION_YEAR, CURRENT_POLLS_FILE
    CURRENT_ELECTION_YEAR = int(year)
    CURRENT_POLLS_FILE = polls_file


def get_election_date(year: int) -> str:
    """
    Return the election day (YYYY-MM-DD) for a given year.
    """
    election_dates = {
        2012: "2012-11-06",
        2016: "2016-11-08",
        2020: "2020-11-03",
    }
    return election_dates.get(year, f"{year}-11-01")


def get_current_election_date() -> str:
    """
    Convenience wrapper that uses the currently configured election year.
    """
    return get_election_date(CURRENT_ELECTION_YEAR)


# ---------------------------------------------------------------------
# Common state name → postal abbreviation mapping
# ---------------------------------------------------------------------

_STATE_NAME_TO_ABBREV: Dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


# ---------------------------------------------------------------------
# 2016-specific polling loader (original timeseries)
# ---------------------------------------------------------------------


def _load_polling_data_2016() -> pd.DataFrame:
    """
    Load and preprocess 2016 polling data from FiveThirtyEight timeseries.

    Returns:
        DataFrame with columns:
            middate, dem, rep, margin, dem_proportion,
            samplesize, pollster, state_code
    """
    polls = pd.read_csv("data/polls/fivethirtyeight_2016_polls_timeseries.csv")
    polls["startdate"] = pd.to_datetime(polls["startdate"])
    polls["enddate"] = pd.to_datetime(polls["enddate"])
    polls["middate"] = polls["startdate"] + (polls["enddate"] - polls["startdate"]) / 2

    polls["dem"] = polls["rawpoll_clinton"]
    polls["rep"] = polls["rawpoll_trump"]
    polls["total"] = polls["dem"] + polls["rep"]

    mask = polls["total"] > 0
    polls = polls.loc[mask].copy()

    polls["margin"] = (polls["dem"] - polls["rep"]) / polls["total"]
    polls["dem_proportion"] = polls["dem"] / polls["total"]

    polls["state_code"] = polls["state"].map(_STATE_NAME_TO_ABBREV)

    return polls


# ---------------------------------------------------------------------
# Generic FiveThirtyEight-style polling loader (e.g. 2020)
# ---------------------------------------------------------------------


def _load_polling_data_fte_long(polls_file: str, cycle: int) -> pd.DataFrame:
    """
    Load and preprocess raw FiveThirtyEight-style presidential polling data.

    Designed for files like `2020_president_polls.csv` where each row is a
    (poll, candidate) combo and support is in a `pct` column.

    Args:
        polls_file: Path to CSV.
        cycle: Election cycle year (used to filter the `cycle` column).

    Returns:
        DataFrame with the same core columns as `_load_polling_data_2016`.
    """
    polls_raw = pd.read_csv(polls_file)
    polls = polls_raw.copy()

    # Filter by cycle if present
    if "cycle" in polls.columns:
        polls = polls[polls["cycle"] == cycle]

    # Keep only presidential general election polls if columns exist
    if "office_type" in polls.columns:
        polls = polls[polls["office_type"].str.contains("President", na=False)]

    if "stage" in polls.columns:
        polls = polls[polls["stage"] == "general"]

    # Require a state
    polls = polls[polls["state"].notna()].copy()

    # We expect candidate_party & pct
    if "candidate_party" not in polls.columns or "pct" not in polls.columns:
        raise ValueError(
            "Expected columns `candidate_party` and `pct` in polls file "
            f"{polls_file}, but they were not found."
        )

    polls = polls[polls["candidate_party"].isin(["DEM", "REP"])].copy()

    # Parse dates
    polls["start_date"] = pd.to_datetime(polls["start_date"])
    polls["end_date"] = pd.to_datetime(polls["end_date"])

    index_cols = [
        "poll_id",
        "state",
        "pollster",
        "sample_size",
        "start_date",
        "end_date",
    ]

    table = polls.pivot_table(
        index=index_cols,
        columns="candidate_party",
        values="pct",
        aggfunc="mean",
    )

    wide = table.reset_index()

    # Rename columns to align with 2016 loader
    wide.rename(
        columns={
            "DEM": "dem",
            "REP": "rep",
            "sample_size": "samplesize",
            "start_date": "startdate",
            "end_date": "enddate",
        },
        inplace=True,
    )

    wide["total"] = wide["dem"] + wide["rep"]
    wide = wide[wide["total"] > 0].copy()

    wide["margin"] = (wide["dem"] - wide["rep"]) / wide["total"]
    wide["dem_proportion"] = wide["dem"] / wide["total"]
    wide["middate"] = wide["startdate"] + (wide["enddate"] - wide["startdate"]) / 2

    wide["state_code"] = wide["state"].map(_STATE_NAME_TO_ABBREV)

    cols = [
        "middate",
        "dem",
        "rep",
        "margin",
        "dem_proportion",
        "samplesize",
        "pollster",
        "state_code",
        "startdate",
        "enddate",
        "total",
    ]
    wide = wide[cols]

    return wide


# ---------------------------------------------------------------------
# Public polling loaders used by models
# ---------------------------------------------------------------------


def load_polling_data() -> pd.DataFrame:
    """
    Load polling data for the currently configured election.

    Behaviour:
      - If CURRENT_ELECTION_YEAR == 2016 and CURRENT_POLLS_FILE is None,
        this uses the original `_load_polling_data_2016()` to preserve
        backwards compatibility.
      - Otherwise, it expects a FiveThirtyEight-style CSV (either provided via
        CURRENT_POLLS_FILE or inferred as `data/polls/{year}_president_polls.csv`)
        and parses it with `_load_polling_data_fte_long`.
    """
    year = CURRENT_ELECTION_YEAR
    polls_file = CURRENT_POLLS_FILE

    if year == 2016 and polls_file is None:
        return _load_polling_data_2016()

    if polls_file is None:
        polls_file = f"data/polls/{year}_president_polls.csv"

    return _load_polling_data_fte_long(polls_file=polls_file, cycle=year)


# ---------------------------------------------------------------------
# Election results loaders
# ---------------------------------------------------------------------


def _load_election_results_year(year: int) -> Dict[str, float]:
    """
    Load actual election results for a given year from MIT Election Lab.

    Returns:
        dict mapping state code to actual Democratic margin for that year.
    """
    results = pd.read_csv(
        "data/election_results/mit_president_state_1976_2020.csv", sep="\t"
    )
    results_year = results[results["year"] == year].copy()

    state_results = (
        results_year.groupby(["state_po", "party_simplified"])
        .agg({"candidatevotes": "sum"})
        .reset_index()
    )
    dem = state_results[state_results["party_simplified"] == "DEMOCRAT"].set_index(
        "state_po"
    )["candidatevotes"]
    rep = state_results[state_results["party_simplified"] == "REPUBLICAN"].set_index(
        "state_po"
    )["candidatevotes"]

    actual_margin = ((dem - rep) / (dem + rep)).to_dict()

    return actual_margin


def _load_election_results_2016() -> Dict[str, float]:
    """
    Backwards-compatible wrapper for 2016 results.
    """
    return _load_election_results_year(2016)


def load_election_results() -> Dict[str, float]:
    """
    Public wrapper used by models.

    Uses the currently configured election year.
    """
    return _load_election_results_year(CURRENT_ELECTION_YEAR)


# ---------------------------------------------------------------------
# Fundamentals prior (unchanged)
# ---------------------------------------------------------------------


def load_fundamentals() -> Dict[str, Dict[str, float]]:
    """
    Load historical election results for fundamentals prior.

    Computes weighted average of 2012 (70%) and 2008 (30%) results.

    NOTE: This is still the same 2016-oriented prior as in the original
    project. If you want a 2012 or 2020-specific fundamentals prior, you
    can generalise this function further (e.g. use (2008, 2004) for 2012,
    or (2016, 2012) for 2020).

    Returns:
        dict mapping state code to fundamentals dict with keys:
            margin, margin_2012, margin_2008
    """
    results = pd.read_csv(
        "data/election_results/mit_president_state_1976_2020.csv", sep="\t"
    )

    fundamentals: Dict[str, Dict[str, float]] = {}

    for state in results["state_po"].unique():
        state_results = results[results["state_po"] == state]

        # Get 2012 and 2008 results
        margins_2012: Dict[str, float] = {}
        margins_2008: Dict[str, float] = {}

        for year, margins_dict in [(2012, margins_2012), (2008, margins_2008)]:
            year_results = state_results[state_results["year"] == year]
            year_grouped = year_results.groupby("party_simplified")[
                "candidatevotes"
            ].sum()

            if "DEMOCRAT" in year_grouped.index and "REPUBLICAN" in year_grouped.index:
                dem = year_grouped["DEMOCRAT"]
                rep = year_grouped["REPUBLICAN"]
                margins_dict[state] = (dem - rep) / (dem + rep)

        # Compute weighted average (70% weight on 2012)
        if state in margins_2012 and state in margins_2008:
            fundamentals[state] = {
                "margin": 0.7 * margins_2012[state] + 0.3 * margins_2008[state],
                "margin_2012": margins_2012[state],
                "margin_2008": margins_2008[state],
            }
        elif state in margins_2012:
            fundamentals[state] = {
                "margin": margins_2012[state],
                "margin_2012": margins_2012[state],
                "margin_2008": 0.0,
            }

    return fundamentals


# ---------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------


def get_state_list(polls: pd.DataFrame, actual_results: Dict[str, float]) -> List[str]:
    """
    Get list of states with sufficient polling data.

    Args:
        polls: DataFrame of polling data
        actual_results: dict of actual election results

    Returns:
        list of state codes
    """
    states = [
        s for s in polls["state_code"].unique() if pd.notna(s) and s in actual_results
    ]
    return states


def compute_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute evaluation metrics from predictions.

    Args:
        predictions_df: DataFrame with columns:
            forecast_date, win_probability, predicted_margin, actual_margin

    Returns:
        DataFrame with columns:
            forecast_date, n_states, brier_score, log_loss, mae_margin
    """
    metrics = []
    forecast_dates = predictions_df["forecast_date"].unique()

    for fdate in forecast_dates:
        subset = predictions_df[predictions_df["forecast_date"] == fdate].copy()
        subset["actual_win"] = (subset["actual_margin"] > 0).astype(int)
        subset = subset[subset["actual_margin"].notna()]

        if len(subset) == 0:
            continue

        brier = np.mean((subset["win_probability"] - subset["actual_win"]) ** 2)
        eps = 1e-10
        log_loss = -np.mean(
            subset["actual_win"] * np.log(subset["win_probability"] + eps)
            + (1 - subset["actual_win"]) * np.log(1 - subset["win_probability"] + eps)
        )
        mae = np.mean(np.abs(subset["predicted_margin"] - subset["actual_margin"]))

        metrics.append(
            {
                "forecast_date": pd.to_datetime(fdate).date(),
                "n_states": int(len(subset)),
                "brier_score": float(brier),
                "log_loss": float(log_loss),
                "mae_margin": float(mae),
            }
        )

    return pd.DataFrame(metrics)
