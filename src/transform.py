"""
Data transformation — cleaning, feature engineering, route baseline builder.
All functions: DataFrame in → DataFrame out.
"""

import pandas as pd
from src.config import ROUTES, CITY_POP_FALLBACK


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]. Returns zeros if all values identical."""
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.0, index=s.index)


def build_route_baseline(
    pop_df:  pd.DataFrame,
    aadt_df: pd.DataFrame,
    bts_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per route with baseline traffic, flight, and gravity metrics.

    Steps:
      1. Look up population for origin and destination cities
      2. Compute population gravity score (product / distance²)
      3. Estimate annual flight passengers from BTS DB1B
      4. Estimate annual road person-trips from TxDOT AADT
      5. Add cost and time columns for mode comparison
    """
    pop_lookup = dict(zip(pop_df["city"], pop_df["population"]))

    records = []
    for route_name, cfg in ROUTES.items():
        city_a, city_b = cfg["cities"]
        pop_a = pop_lookup.get(city_a, CITY_POP_FALLBACK.get(city_a, 1_000_000))
        pop_b = pop_lookup.get(city_b, CITY_POP_FALLBACK.get(city_b, 1_000_000))

        gravity = (pop_a * pop_b) / (cfg["distance_miles"] ** 2)

        records.append({
            "route":                         route_name,
            "city_origin":                   city_a,
            "city_dest":                     city_b,
            "distance_miles":                float(cfg["distance_miles"]),
            "drive_time_hr":                 cfg["drive_time_hr"],
            "flight_time_hr":                cfg["flight_time_hr"],
            "population_origin":             pop_a,
            "population_dest":               pop_b,
            "population_gravity_score":      float(gravity),
            "annual_flight_passengers_est":  _estimate_flights(bts_df, city_a, city_b),
            "annual_road_person_trips_proxy": _estimate_road_trips(aadt_df, cfg.get("corridor_roads", [])),
        })

    df = pd.DataFrame(records)

    # ── Cost per trip (door-to-door) ──────────────────────────────────────
    df["car_cost_usd"]            = df["distance_miles"] * 0.21 + 15.0   # IRS mileage + parking
    df["flight_cost_usd"]         = 180.0                                 # avg Southwest TX fare
    df["hsr_robotaxi_cost_usd"]   = df["distance_miles"] * 0.22 + 25.0   # HSR ticket + robotaxi

    # ── Time per trip (door-to-door hours) ────────────────────────────────
    df["drive_door_to_door_hr"]         = df["drive_time_hr"] + 0.50     # + parking/walk
    df["flight_door_to_door_hr"]        = df["flight_time_hr"] + 2.50    # + airport overhead
    df["hsr_robotaxi_door_to_door_hr"]  = df["distance_miles"] / 200.0 + 1.0  # HSR + robotaxi each end

    return df


def _estimate_flights(bts_df: pd.DataFrame, city_a: str, city_b: str) -> int:
    """Estimate annual flight passengers between two cities from BTS DB1B data."""
    AIRPORT_MAP = {
        "Dallas":      ["DFW", "DAL"],
        "Houston":     ["IAH", "HOU"],
        "Austin":      ["AUS"],
        "San Antonio": ["SAT"],
        "Fort Worth":  ["DFW"],
    }
    FALLBACK = {
        "Dallas-Houston":     3_200_000,
        "Dallas-Austin":      1_800_000,
        "Houston-San Antonio":  900_000,
    }

    airports_a = AIRPORT_MAP.get(city_a, [])
    airports_b = AIRPORT_MAP.get(city_b, [])
    key = f"{city_a}-{city_b}"

    if bts_df.empty or not airports_a or not airports_b:
        return FALLBACK.get(key, 1_000_000)

    pax_col  = next((c for c in bts_df.columns if "pass" in c.lower()), None)
    orig_col = next((c for c in bts_df.columns if "orig" in c.lower()), None)
    dest_col = next((c for c in bts_df.columns if "dest" in c.lower()), None)

    if not all([pax_col, orig_col, dest_col]):
        return FALLBACK.get(key, 1_000_000)

    mask = (
        (bts_df[orig_col].isin(airports_a) & bts_df[dest_col].isin(airports_b)) |
        (bts_df[orig_col].isin(airports_b) & bts_df[dest_col].isin(airports_a))
    )
    quarterly = pd.to_numeric(bts_df.loc[mask, pax_col], errors="coerce").sum()
    return int(quarterly * 4) if quarterly > 0 else FALLBACK.get(key, 1_000_000)


def _estimate_road_trips(aadt_df: pd.DataFrame, corridor_roads: list) -> int:
    """Estimate annual person-trips on corridor from TxDOT AADT."""
    FALLBACK = 25_000_000

    if aadt_df.empty or not corridor_roads:
        return FALLBACK

    route_col = aadt_df.columns[0]
    aadt_col  = aadt_df.columns[-1]

    mask = aadt_df[route_col].astype(str).str.upper().apply(
        lambda r: any(road.upper() in r for road in corridor_roads)
    )
    avg_aadt = pd.to_numeric(aadt_df.loc[mask, aadt_col], errors="coerce").mean()

    if pd.isna(avg_aadt) or avg_aadt <= 0:
        return FALLBACK

    return int(avg_aadt * 365 * 1.6)   # AADT × days × avg occupancy
