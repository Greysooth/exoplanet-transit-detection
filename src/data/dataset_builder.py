"""
Phase-Grid Dataset Builder
Transforms cached raw phase-folded curves (.npz) into the canonical 80-bin
center-cropped, standardized dataset ready for neural network ingestion.
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DATA_DIR, PROCESSED_DATA_DIR
from src.data.preprocessing import phase_grid_representation, normalize_processed_curve

FOLDED_CACHE_DIR = os.path.join(DATA_DIR, "folded_curves")

def build_phasegrid_dataset(cache_dir=FOLDED_CACHE_DIR, output_dir=PROCESSED_DATA_DIR):
    """
    Processes all cached .npz light curves into equidistant 200-bin grids,
    crops to central 80 bins (transit window), standardizes, and saves .npy arrays.
    """
    npz_files = glob.glob(os.path.join(cache_dir, "*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"No cached .npz files found in {cache_dir}. Run downloader.py first!")

    processed_curves = []
    labels = []
    metadata = []

    print(f"Synthesizing 80-bin phase-grid dataset from {len(npz_files)} cached light curves...")

    for file_path in tqdm(npz_files, desc="Processing Phase-Grids"):
        try:
            with np.load(file_path) as data:
                phase = data["phase"]
                flux = data["flux"]
                lbl = data["label"]
                meta = data["metadata"]

            # 200-bin phase grid representation with central 80-bin crop
            grid_flux = phase_grid_representation(phase, flux, n_bins=200, crop_start=60, crop_end=140)
            if grid_flux is None or len(grid_flux) != 80:
                continue

            # Standardization to zero-mean and unit-variance
            normalized_flux = normalize_processed_curve(grid_flux)

            processed_curves.append(normalized_flux)
            labels.append(lbl)
            metadata.append(meta)

        except Exception as e:
            continue

    X_processed = np.array(processed_curves, dtype=np.float64)
    y_labels = np.array(labels, dtype=np.int64)
    z_metadata = np.array(metadata, dtype=np.float64)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "lightcurves.npy"), X_processed)
    np.save(os.path.join(output_dir, "labels.npy"), y_labels)
    np.save(os.path.join(output_dir, "metadata.npy"), z_metadata)

    unique, counts = np.unique(y_labels, return_counts=True)
    print("\n========================= DATASET TELEMETRY =========================")
    print(f"Phase-Grid Dataset Built Successfully -> {output_dir}")
    print(f"Lightcurves Array Matrix Shape : {X_processed.shape} (N={len(X_processed)}, Bins=80)")
    print(f"Class Distribution Balance     : {dict(zip(unique.astype(int), counts))}")
    print(f"Metadata Matrix Shape          : {z_metadata.shape} (Features=3)")
    print("=====================================================================")
    return X_processed, y_labels, z_metadata

if __name__ == "__main__":
    build_phasegrid_dataset()
