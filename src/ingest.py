"""
Data ingestion — one function per data source.
Each function loads a raw file and returns a clean DataFrame.
"""

import warnings
import pandas as pd
import duckdb

from src.config import FILES, DB_PATH, CITY_POP_FALLBACK

warnings.filterwarnings("ignore")


# ── Per-source loaders ─────────────────────────────────────────────────────

def load_population() -> pd.DataFrame:
    """Load Census population for key Texas cities. Falls back to known values."""
    records = []

    for city, csv_key in [("Dallas", "census_dallas"), ("Houston", "census_houston")]:
        try:
            df = pd.read_csv(FILES[csv_key], header=1)
            pop_col = next(
                (c for c in df.columns if "Total population" in c or "Total" in c),
                df.columns[1]
            )
            raw = str(df.iloc[0][pop_col]).replace(",", "").replace(" ", "")
            val = pd.to_numeric(raw, errors="coerce")
            pop = int(val) if pd.notna(val) and val > 0 else CITY_POP_FALLBACK[city]
        except Exception:
            pop = CITY_POP_FALLBACK[city]
        records.append({"city": city, "population": pop})

    # Add remaining cities from fallback
    for city in ["Austin", "San Antonio", "Fort Worth"]:
        records.append({"city": city, "population": CITY_POP_FALLBACK[city]})

    return pd.DataFrame(records)


def load_ercot() -> pd.DataFrame:
    """Load ERCOT Native Load 2025 (hourly MWh). Returns cleaned hourly DataFrame."""
    df = pd.read_excel(FILES["ercot"], sheet_name=0, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]

    hour_col  = next((c for c in df.columns if "Hour"  in c), df.columns[0])
    ercot_col = next((c for c in df.columns if "ERCOT" in c.upper()), df.columns[1])

    df = df[[hour_col, ercot_col]].rename(
        columns={hour_col: "HourEnding", ercot_col: "ERCOT_MWh"}
    )
    df["ERCOT_MWh"] = pd.to_numeric(df["ERCOT_MWh"], errors="coerce")
    return df.dropna(subset=["ERCOT_MWh"]).reset_index(drop=True)


def load_epa_factors() -> dict:
    """
    Load EPA GHG emission factors.
    Returns dict: {car_kg_per_vehicle_mile, flight_kg_per_passenger_mile, ercot_kg_per_kwh}
    """
    DEFAULTS = {
        "car_kg_per_vehicle_mile":      0.404,
        "flight_kg_per_passenger_mile": 0.255,
        "ercot_kg_per_kwh":             0.386,
    }
    try:
        df = pd.read_csv(FILES["epa"])
        df.columns = [c.strip().lower() for c in df.columns]

        def _find(keyword: str, default: float) -> float:
            mask = df.apply(
                lambda r: r.astype(str).str.contains(keyword, case=False).any(), axis=1
            )
            if mask.any():
                num_cols = df.select_dtypes(include="number").columns
                if len(num_cols):
                    return float(df[mask].iloc[0][num_cols[0]])
            return default

        return {
            "car_kg_per_vehicle_mile":      _find("passenger car",  DEFAULTS["car_kg_per_vehicle_mile"]),
            "flight_kg_per_passenger_mile": _find("air travel",     DEFAULTS["flight_kg_per_passenger_mile"]),
            "ercot_kg_per_kwh":             DEFAULTS["ercot_kg_per_kwh"],
        }
    except Exception:
        return DEFAULTS


def load_aadt() -> pd.DataFrame:
    """Load TxDOT Annual Average Daily Traffic. Returns route/county/AADT DataFrame."""
    df = pd.read_csv(FILES["aadt"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    aadt_col   = next((c for c in df.columns if "aadt" in c.lower()), None)
    route_col  = next((c for c in df.columns if "route" in c.lower()), None)
    county_col = next((c for c in df.columns if "county" in c.lower()), None)

    if aadt_col:
        df[aadt_col] = pd.to_numeric(df[aadt_col], errors="coerce")

    keep = [c for c in [route_col, county_col, aadt_col] if c]
    return df[keep].dropna().reset_index(drop=True) if keep else df


def load_bts(sample_rows: int = 500_000) -> pd.DataFrame:
    """
    Load BTS DB1B Origin-Destination air passenger data.
    Samples to 500K rows by default to manage memory.
    """
    try:
        return pd.read_csv(
            FILES["bts"],
            sep=",",
            nrows=sample_rows,
            low_memory=False,
            on_bad_lines="skip",
        )
    except Exception:
        return pd.DataFrame(columns=["origin", "dest", "passengers", "fare"])


def check_all_files() -> dict:
    """Return {filename: True/False} showing which source files exist."""
    return {name: path.exists() for name, path in FILES.items()}


# ── Database writer ────────────────────────────────────────────────────────

def write_to_db(
    baseline:   pd.DataFrame,
    results:    pd.DataFrame,
    mc_results: pd.DataFrame,
    ef:         dict,
) -> None:
    """Write all model outputs to DuckDB for SQL querying and dashboard."""
    conn = duckdb.connect(str(DB_PATH))

    for table in ["route_baseline", "scenario_results", "monte_carlo_results", "emission_factors"]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    conn.execute("CREATE TABLE route_baseline      AS SELECT * FROM baseline")
    conn.execute("CREATE TABLE scenario_results    AS SELECT * FROM results")
    conn.execute("CREATE TABLE monte_carlo_results AS SELECT * FROM mc_results")

    ef_df = pd.DataFrame([ef])  # noqa: F841
    conn.register("ef_df", ef_df)
    conn.execute("CREATE TABLE emission_factors AS SELECT * FROM ef_df")

    conn.close()
    print(f"  Database written to: {DB_PATH}")


def query_db(sql: str) -> pd.DataFrame:
    """Run a SQL query against the project DuckDB and return a DataFrame."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    result = conn.execute(sql).df()
    conn.close()
    return result
