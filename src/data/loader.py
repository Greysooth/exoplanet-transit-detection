"""
NASA Exoplanet Archive TOI Catalog Ingestion
Fetches ground-truth confirmed planets (KP, CP) and false positives (FP)
with required physical metadata and transit midpoints.
"""

import os
import sys
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import RAW_DATA_DIR, SEED

def fetch_toi_catalog(save_path=None, random_state=SEED):
    """
    Queries NASA Exoplanet Archive for the TOI table, filtering for ground-truth
    confirmed planets (KP, CP) and false positives (FP) with non-null ephemerides.
    """
    print("[1/2] Querying NASA Exoplanet Archive (TOI Table)...")
    table = NasaExoplanetArchive.query_criteria(
        table="toi",
        select="tid, toi, tfopwg_disp, pl_orbper, pl_tranmid, st_teff, st_rad"
    )
    df = table.to_pandas()

    print(f"      Retrieved {len(df)} total TOI records.")
    df = df.dropna(subset=["tfopwg_disp", "pl_orbper", "pl_tranmid", "st_teff", "st_rad"])

    # Strict ground-truth classification
    positives = df[df["tfopwg_disp"].isin(["KP", "CP"])].copy()
    negatives = df[df["tfopwg_disp"] == "FP"].copy()

    print(f"      Filtered valid records: {len(positives)} KP+CP (Positives) | {len(negatives)} FP (Negatives)")

    # Sample without replacement for balanced representation
    pos_sample = positives.sample(n=len(positives), random_state=random_state)
    neg_sample = negatives.sample(n=len(negatives), random_state=random_state)

    dataset = pd.concat([pos_sample, neg_sample], ignore_index=True)
    dataset["label"] = dataset["tfopwg_disp"].map({"KP": 1, "CP": 1, "FP": 0})
    dataset = dataset[["tid", "pl_orbper", "pl_tranmid", "label", "st_teff", "st_rad"]]

    if save_path is None:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        save_path = os.path.join(RAW_DATA_DIR, "toi_clean_labels.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.to_csv(save_path, index=False)
    print(f"[2/2] Clean ground-truth catalog successfully saved to: {save_path}")
    print(f"      Total targets: {len(dataset)} | Class counts:\n{dataset['label'].value_counts().to_string()}")
    return dataset

if __name__ == "__main__":
    fetch_toi_catalog()
