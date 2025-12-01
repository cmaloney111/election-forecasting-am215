#!/usr/bin/env python3
"""
Shared data loading and preprocessing utilities
"""
from typing import Dict, List

import pandas as pd
import numpy as np


def load_polling_data() -> pd.DataFrame:
    """
    Load and preprocess 2016 polling data from FiveThirtyEight

    Returns:
        DataFrame with columns: middate, dem, rep, margin, dem_proportion, samplesize, pollster, state_code
    """
    polls = pd.read_csv("data/polls/fivethirtyeight_2016_polls_timeseries.csv")
    polls["startdate"] = pd.to_datetime(polls["startdate"])
    polls["enddate"] = pd.to_datetime(polls["enddate"])
    polls["middate"] = polls["startdate"] + (polls["enddate"] - polls["startdate"]) / 2
    polls["dem"] = polls["rawpoll_clinton"]
    polls["rep"] = polls["rawpoll_trump"]
    polls["total"] = polls["dem"] + polls["rep"]
    polls = polls[polls["total"] > 0].copy()

    polls["margin"] = (polls["dem"] - polls["rep"]) / polls["total"]
    polls["dem_proportion"] = polls["dem"] / polls["total"]

    state_map = {
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
    polls["state_code"] = polls["state"].map(state_map)

    return polls


def load_election_results() -> Dict[str, float]:
    """
    Load actual 2016 election results from MIT Election Lab

    Returns:
        dict mapping state code to actual Democratic margin
    """
    results = pd.read_csv(
        "data/election_results/mit_president_state_1976_2020.csv", sep="\t"
    )
    results_2016 = results[results["year"] == 2016].copy()

    state_results = (
        results_2016.groupby(["state_po", "party_simplified"])
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


def load_fundamentals() -> Dict[str, Dict[str, float]]:
    """
    Load historical election results for fundamentals prior

    Computes weighted average of 2012 (70%) and 2008 (30%) results

    Returns:
        dict mapping state code to fundamentals dict with keys: margin, margin_2012, margin_2008
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
                "margin_2008": None,
            }

    return fundamentals


def get_state_list(polls: pd.DataFrame, actual_results: Dict[str, float]) -> List[str]:
    """
    Get list of states with sufficient polling data

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
    Compute evaluation metrics from predictions

    Args:
        predictions_df: DataFrame with columns: forecast_date, win_probability, predicted_margin, actual_margin

    Returns:
        DataFrame with columns: forecast_date, n_states, brier_score, log_loss, mae_margin
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
                "n_states": len(subset),
                "brier_score": brier,
                "log_loss": log_loss,
                "mae_margin": mae,
            }
        )

    return pd.DataFrame(metrics)
