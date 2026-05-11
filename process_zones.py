"""
process_zones.py  –  Run once to build trips_zone_summary.parquet

Much faster than process_data.py: reads only 2 columns per file,
no datetime parsing. Generates zone-level Uber/Lyft trip counts
for the choropleth maps.

Usage:
    python process_zones.py
"""

import os
import glob
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    zone_lookup = pd.read_csv(os.path.join(DATA_DIR, "taxi_zone_lookup.csv"))
    zone_lookup["Borough"] = zone_lookup["Borough"].str.strip()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "fhvhv_tripdata_*.parquet")))
    print(f"Found {len(files)} file(s)")

    results = []
    for path in files:
        print(f"  {os.path.basename(path)} ...", end=" ", flush=True)
        df = pq.read_table(path, columns=["hvfhs_license_num", "PULocationID"]).to_pandas()
        df = df[df["hvfhs_license_num"].isin(["HV0003", "HV0005"])]
        df["provider"] = df["hvfhs_license_num"].map({"HV0003": "Uber", "HV0005": "Lyft"})
        agg = df.groupby(["PULocationID", "provider"]).size().reset_index(name="trip_count")
        results.append(agg)
        print(f"{len(df):,} trips → {len(agg)} zone-provider rows")

    combined = pd.concat(results, ignore_index=True)
    combined = combined.groupby(["PULocationID", "provider"])["trip_count"].sum().reset_index()

    # Join zone metadata
    combined = combined.merge(
        zone_lookup[["LocationID", "Borough", "Zone"]],
        left_on="PULocationID", right_on="LocationID", how="left"
    )
    combined["Borough"] = combined["Borough"].fillna("Unknown")
    combined["Zone"]    = combined["Zone"].fillna("Unknown")

    out = os.path.join(DATA_DIR, "trips_zone_summary.parquet")
    combined.to_parquet(out, index=False)

    uber = combined[combined["provider"] == "Uber"]["trip_count"].sum()
    lyft = combined[combined["provider"] == "Lyft"]["trip_count"].sum()
    print(f"\nUber: {uber:,.0f}  Lyft: {lyft:,.0f}  zones: {combined['PULocationID'].nunique()}")
    print(f"Saved → {out}  ({os.path.getsize(out)/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
