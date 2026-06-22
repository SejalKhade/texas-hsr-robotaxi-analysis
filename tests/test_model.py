"""
Unit tests for src/model.py — gravity model, emissions, scenarios, Monte Carlo.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
from src.model import allocate_riders, add_energy_emissions, run_scenario, run_all_scenarios, monte_carlo
from src.transform import normalize_series


# ── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def baseline():
    return pd.DataFrame({
        "route":                          ["Dallas-Houston",     "Dallas-Austin"],
        "city_origin":                    ["Dallas",             "Dallas"],
        "city_dest":                      ["Houston",            "Austin"],
        "distance_miles":                 [239.0,                195.0],
        "drive_time_hr":                  [3.75,                 3.25],
        "flight_time_hr":                 [1.10,                 0.90],
        "population_origin":              [1_326_093,            1_326_093],
        "population_dest":                [2_387_910,            1_000_000],
        "population_gravity_score":       [44_068_432_123.0,     32_000_000_000.0],
        "annual_flight_passengers_est":   [3_200_000,            1_800_000],
        "annual_road_person_trips_proxy": [28_000_000,           18_000_000],
        "car_cost_usd":                   [65.19,                55.95],
        "flight_cost_usd":                [180.0,                180.0],
        "hsr_robotaxi_cost_usd":          [77.58,                67.90],
        "drive_door_to_door_hr":          [4.25,                 3.75],
        "flight_door_to_door_hr":         [3.60,                 3.40],
        "hsr_robotaxi_door_to_door_hr":   [2.195,                1.975],
    })

@pytest.fixture
def ef():
    return {
        "car_kg_per_vehicle_mile":      0.404,
        "flight_kg_per_passenger_mile": 0.255,
        "ercot_kg_per_kwh":             0.386,
    }


# ── normalize_series ───────────────────────────────────────────────────────

def test_normalize_min_is_zero():
    s = pd.Series([10.0, 20.0, 30.0])
    assert normalize_series(s).min() == pytest.approx(0.0)

def test_normalize_max_is_one():
    s = pd.Series([10.0, 20.0, 30.0])
    assert normalize_series(s).max() == pytest.approx(1.0)

def test_normalize_constant_returns_zeros():
    s = pd.Series([5.0, 5.0, 5.0])
    assert (normalize_series(s) == 0.0).all()


# ── allocate_riders ────────────────────────────────────────────────────────

def test_route_weights_sum_to_one(baseline):
    result = allocate_riders(baseline, weekly_total=35_000)
    assert abs(result["route_weight"].sum() - 1.0) < 0.001

def test_annual_riders_is_52x_weekly(baseline):
    result = allocate_riders(baseline, weekly_total=35_000)
    assert (result["hsr_annual_riders"] == result["hsr_weekly_riders"] * 52).all()

def test_total_weekly_riders_close_to_input(baseline):
    weekly = 35_000
    result = allocate_riders(baseline, weekly_total=weekly)
    assert abs(result["hsr_weekly_riders"].sum() - weekly) <= 5

def test_no_negative_mode_shift(baseline):
    result = allocate_riders(baseline, weekly_total=35_000)
    assert (result["shift_from_flights"] >= 0).all()
    assert (result["shift_from_cars"]    >= 0).all()

def test_flight_shift_capped_by_available_passengers(baseline):
    result = allocate_riders(baseline, weekly_total=35_000)
    assert (result["shift_from_flights"] <= result["annual_flight_passengers_est"]).all()

def test_zero_weekly_riders_gives_zero_annual(baseline):
    result = allocate_riders(baseline, weekly_total=0)
    assert (result["hsr_annual_riders"] == 0).all()


# ── add_energy_emissions ───────────────────────────────────────────────────

def test_electric_kwh_is_positive(baseline, ef):
    df = allocate_riders(baseline, 35_000)
    result = add_energy_emissions(df, ef)
    assert (result["annual_total_electric_kwh"] > 0).all()

def test_avoided_co2_positive_for_medium_adoption(baseline, ef):
    df = allocate_riders(baseline, 35_000)
    result = add_energy_emissions(df, ef)
    assert result["avoided_metric_tons_co2"].sum() > 0

def test_hsr_cheaper_than_car_on_long_routes(baseline, ef):
    df = allocate_riders(baseline, 35_000)
    result = add_energy_emissions(df, ef)
    long = result[result["distance_miles"] > 200]
    assert (long["cost_savings_vs_car_usd"] > 0).all()

def test_time_savings_positive_vs_driving(baseline, ef):
    df = allocate_riders(baseline, 35_000)
    result = add_energy_emissions(df, ef)
    assert (result["time_savings_vs_drive_hr"] > 0).all()

def test_mwh_equals_kwh_divided_by_1000(baseline, ef):
    df = allocate_riders(baseline, 35_000)
    result = add_energy_emissions(df, ef)
    ratio = result["annual_total_electric_mwh"] / result["annual_total_electric_kwh"]
    assert (ratio == pytest.approx(0.001)).all()


# ── run_scenario ───────────────────────────────────────────────────────────

def test_high_more_riders_than_low(baseline, ef):
    low  = run_scenario(baseline, ef, "Low")
    high = run_scenario(baseline, ef, "High")
    assert high["hsr_annual_riders"].sum() > low["hsr_annual_riders"].sum()

def test_scenario_column_correct(baseline, ef):
    result = run_scenario(baseline, ef, "Medium")
    assert (result["scenario"] == "Medium").all()

def test_invalid_scenario_raises_value_error(baseline, ef):
    with pytest.raises(ValueError, match="Unknown scenario"):
        run_scenario(baseline, ef, "Ultra")

def test_all_scenarios_returns_three_times_routes(baseline, ef):
    result = run_all_scenarios(baseline, ef)
    assert len(result) == len(baseline) * 3


# ── monte_carlo ────────────────────────────────────────────────────────────

def test_monte_carlo_correct_row_count(baseline, ef):
    result = monte_carlo(baseline, ef, n=50)
    assert len(result) == 50

def test_monte_carlo_no_negative_riders(baseline, ef):
    result = monte_carlo(baseline, ef, n=100)
    assert (result["total_annual_riders"] >= 0).all()

def test_monte_carlo_p10_less_than_p90(baseline, ef):
    result = monte_carlo(baseline, ef, n=200)
    p10 = result["total_avoided_tons_co2"].quantile(0.10)
    p90 = result["total_avoided_tons_co2"].quantile(0.90)
    assert p10 < p90

def test_monte_carlo_reproducible_with_same_seed(baseline, ef):
    r1 = monte_carlo(baseline, ef, n=50, seed=0)
    r2 = monte_carlo(baseline, ef, n=50, seed=0)
    pd.testing.assert_frame_equal(r1, r2)
