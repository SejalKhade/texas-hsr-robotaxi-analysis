"""
Data validation schemas using Pandera.
Call these functions before running the model to catch bad data early.
"""

import pandas as pd
from pandera.pandas import Column, DataFrameSchema, Check


ERCOT_SCHEMA = DataFrameSchema(
    columns={
        "HourEnding": Column(str,   nullable=False),
        "ERCOT_MWh":  Column(float, checks=[
            Check.greater_than(0,       error="ERCOT load must be positive"),
            Check.less_than(100_000,    error="ERCOT load unrealistically high — check units"),
        ]),
    },
    name="ERCOT Load Data",
)

POPULATION_SCHEMA = DataFrameSchema(
    columns={
        "city":       Column(str, nullable=False),
        "population": Column(int, checks=[
            Check.greater_than(10_000,    error="Population too small to be a Texas city"),
            Check.less_than(10_000_000,   error="Population exceeds plausible Texas city size"),
        ]),
    },
    name="City Population",
)

BASELINE_SCHEMA = DataFrameSchema(
    columns={
        "route":                          Column(str,   nullable=False),
        "distance_miles":                 Column(float, checks=Check.in_range(50, 2_000)),
        "annual_flight_passengers_est":   Column(int,   checks=Check.greater_than_or_equal_to(0)),
        "annual_road_person_trips_proxy": Column(int,   checks=Check.greater_than(0)),
        "population_gravity_score":       Column(float, checks=Check.greater_than(0)),
        "car_cost_usd":                   Column(float, checks=Check.greater_than(0)),
        "hsr_robotaxi_cost_usd":          Column(float, checks=Check.greater_than(0)),
    },
    name="Route Baseline",
)

RESULTS_SCHEMA = DataFrameSchema(
    columns={
        "route":                    Column(str),
        "scenario":                 Column(str,   checks=Check.isin(["Low", "Medium", "High"])),
        "hsr_annual_riders":        Column(int,   checks=Check.greater_than_or_equal_to(0)),
        "avoided_metric_tons_co2":  Column(float),
        "annual_revenue_usd":       Column(float, checks=Check.greater_than_or_equal_to(0)),
    },
    name="Scenario Results",
)


def validate_ercot(df: pd.DataFrame) -> pd.DataFrame:
    return ERCOT_SCHEMA.validate(df)

def validate_population(df: pd.DataFrame) -> pd.DataFrame:
    return POPULATION_SCHEMA.validate(df)

def validate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    return BASELINE_SCHEMA.validate(df)

def validate_results(df: pd.DataFrame) -> pd.DataFrame:
    return RESULTS_SCHEMA.validate(df)