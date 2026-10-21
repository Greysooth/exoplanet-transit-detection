"""
TESS Light Curve Ingestion & Phase-Folding Downloader
Downloads PDCSAP flux via Lightkurve from MAST, applies Savitzky-Golay detrending,
phase-folds using exact BTJD epoch centering, and caches compressed .npz archives.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DATA_DIR, RAW_DATA_DIR
from src.data.preprocessing import detrend_flux

FOLDED_CACHE_DIR = os.path.join(DATA_DIR, "folded_curves")

def download_and_cache_lightcurve(tic_id, row, cache_dir=FOLDED_CACHE_DIR):
    """
    Downloads TESS light curve for a single TIC target, detrends, phase-folds,
    and caches compressed arrays.
    """
    import lightkurve as lk
    import astropy.units as u
    from lightkurve import LightCurve

    try:
        period = float(row["pl_orbper"])
        t0 = float(row["pl_tranmid"])
        t0_btjd = t0 - 2457000.0  # Crucial JD -> BTJD alignment

        if pd.isna(period) or period <= 0 or pd.isna(t0):
            return False

        tic_id = int(tic_id)
        cache_path = os.path.join(cache_dir, f"{tic_id}.npz")

        # Skip download if already cached locally
        if os.path.exists(cache_path):
            return True

        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            return False

        # Download primary observation sector and remove NaNs
        lc = search[0].download().remove_nans()
        raw_flux = lc["pdcsap_flux"].value

        # Baseline cleaning via Savitzky-Golay filtering
        detrended = detrend_flux(raw_flux)
        if detrended is None or len(detrended) == 0:
            return False

        lc_detrended = LightCurve(time=lc.time, flux=detrended * u.dimensionless_unscaled)
        folded = lc_detrended.fold(period=period, epoch_time=t0_btjd)

        # Save compressed raw phase-folded representation
        np.savez_compressed(
            cache_path,
            phase=folded.phase.value,
            flux=folded.flux.value,
            label=int(row["label"]),
            metadata=np.array([float(row["st_teff"]), float(row["pl_orbper"]), float(row["st_rad"])])
        )
        return True

    except Exception:
        return False


def build_raw_cache(catalog_path=None, cache_dir=FOLDED_CACHE_DIR, max_targets=None):
    """
    Iterates over target catalog and caches phase-folded light curves.
    """
    if catalog_path is None:
        catalog_path = os.path.join(RAW_DATA_DIR, "toi_clean_labels.csv")

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Target catalog not found at {catalog_path}. Run loader.py first!")

    df = pd.read_csv(catalog_path)
    if max_targets is not None:
        df = df.iloc[:max_targets]

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading & caching phase-folded TESS light curves for {len(df)} targets...")
    print(f"Destination: {cache_dir}")

    success_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching TESS Curves"):
        if download_and_cache_lightcurve(row["tid"], row, cache_dir=cache_dir):
            success_count += 1

    print(f"\nCaching complete! Locally stored {success_count}/{len(df)} valid light curves.")
    return success_count

if __name__ == "__main__":
    build_raw_cache()
