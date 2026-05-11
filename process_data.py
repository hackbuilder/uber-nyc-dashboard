"""
process_data.py  –  Run once to build trips_combined_2025.parquet

Reads all fhvhv_tripdata_*.parquet files in the current directory,
filters to Uber (HV0003) and Lyft (HV0005), joins borough names,
and aggregates to (date, Borough, hour, day_of_week, provider).

Usage:
    cd "Desktop/2026 class/product marketing/hw5"
    python process_data.py
"""

import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

NEEDED_COLS = [
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "PULocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "shared_match_flag",
]

PROVIDER_MAP = {"HV0003": "Uber", "HV0005": "Lyft"}


def process_file(path: str, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    print(f"  Processing {os.path.basename(path)} ...", end=" ", flush=True)

    schema = pq.read_schema(path)
    cols = [c for c in NEEDED_COLS if c in schema.names]
    df = pq.read_table(path, columns=cols).to_pandas()

    # Keep only Uber + Lyft
    df = df[df["hvfhs_license_num"].isin(PROVIDER_MAP)].copy()
    if df.empty:
        print("0 rows after filter")
        return pd.DataFrame()

    df["provider"] = df["hvfhs_license_num"].map(PROVIDER_MAP)

    # Parse datetimes
    df["request_datetime"] = pd.to_datetime(df["request_datetime"], utc=False, errors="coerce")
    df["pickup_datetime"]  = pd.to_datetime(df["pickup_datetime"],  utc=False, errors="coerce")

    df["request_date"] = df["request_datetime"].dt.date
    df["request_hour"] = df["request_datetime"].dt.hour
    df["day_of_week"]  = df["request_datetime"].dt.dayofweek  # 0 = Monday

    # Derived metrics
    df["wait_time_sec"] = (
        (df["pickup_datetime"] - df["request_datetime"]).dt.total_seconds().clip(lower=0)
    )
    df["is_airport"] = (df["airport_fee"].fillna(0) > 0).astype(int)
    df["is_shared"]  = (df["shared_match_flag"] == "Y").astype(int)

    # Clip extreme fare-per-mile before aggregating
    raw_fpm = np.where(
        df["trip_miles"] > 0.1,
        df["base_passenger_fare"] / df["trip_miles"],
        np.nan,
    )
    p99 = np.nanpercentile(raw_fpm, 99)
    df["fare_per_mile"] = np.clip(raw_fpm, a_min=None, a_max=p99)

    # Borough join (drop EWR / Unknown for cleanliness – keep if you want)
    df = df.merge(
        zone_lookup[["LocationID", "Borough"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    )
    df["Borough"] = df["Borough"].fillna("Unknown")

    # Aggregate: store SUMS so app can compute correct weighted averages
    agg = (
        df.groupby(["request_date", "Borough", "request_hour", "day_of_week", "provider"])
        .agg(
            trip_count       = ("trip_miles",           "count"),
            trip_miles_sum   = ("trip_miles",           "sum"),
            trip_time_sum    = ("trip_time",            "sum"),   # seconds
            wait_time_sum    = ("wait_time_sec",        "sum"),
            base_fare_sum    = ("base_passenger_fare",  "sum"),
            driver_pay_sum   = ("driver_pay",           "sum"),
            tips_sum         = ("tips",                 "sum"),
            congestion_sum   = ("congestion_surcharge", "sum"),
            airport_trips    = ("is_airport",           "sum"),
            shared_trips     = ("is_shared",            "sum"),
        )
        .reset_index()
    )

    print(f"{len(df):,} trips → {len(agg):,} rows")
    return agg


def main():
    zone_lookup = pd.read_csv(os.path.join(DATA_DIR, "taxi_zone_lookup.csv"))
    # Normalize borough names to match GeoJSON
    zone_lookup["Borough"] = zone_lookup["Borough"].str.strip()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "fhvhv_tripdata_*.parquet")))
    if not files:
        print("No fhvhv_tripdata_*.parquet files found in", DATA_DIR)
        return

    print(f"Found {len(files)} file(s):")
    results = []
    for path in files:
        chunk = process_file(path, zone_lookup)
        if not chunk.empty:
            results.append(chunk)

    if not results:
        print("No data processed.")
        return

    combined = pd.concat(results, ignore_index=True)
    combined["request_date"] = pd.to_datetime(combined["request_date"])

    uber = combined[combined["provider"] == "Uber"]["trip_count"].sum()
    lyft = combined[combined["provider"] == "Lyft"]["trip_count"].sum()
    print(f"\nTotal  Uber trips : {uber:>15,.0f}")
    print(f"Total  Lyft trips : {lyft:>15,.0f}")
    print(f"Date range        : {combined['request_date'].min().date()} → {combined['request_date'].max().date()}")

    out = os.path.join(DATA_DIR, "trips_combined_2025.parquet")
    combined.to_parquet(out, index=False)
    print(f"\nSaved → {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
