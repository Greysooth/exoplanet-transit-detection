import numpy as np
from scipy.signal import savgol_filter

# Canonical feature dimension for cropped transit representation
TRANSIT_BIN_COUNT = 80

def detrend_flux(flux):
    """
    Applies Savitzky-Golay filtering to remove stellar low-frequency trends.
    """
    flux = flux[np.isfinite(flux)]
    if len(flux) == 0:
        return None

    window = int(len(flux) * 0.1)
    if window % 2 == 0:
        window += 1
    window = max(window, 301)

    if len(flux) > window:
        trend = savgol_filter(flux, window_length=window, polyorder=2)
        flux = flux - trend
    return flux


def phase_grid_representation(phase, flux, n_bins=200, crop_start=60, crop_end=140):
    """
    Bins phase-folded data onto an equidistant grid, interpolates gaps,
    and crops to the central transit dip (indices crop_start:crop_end).
    """
    idx = np.argsort(phase)
    phase, flux = phase[idx], flux[idx]

    bin_edges = np.linspace(phase.min(), phase.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_assignments = np.digitize(phase, bin_edges) - 1

    bin_means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_assignments == i
        if np.any(mask):
            bin_means[i] = flux[mask].mean()
        else:
            bin_means[i] = np.nan

    valid = ~np.isnan(bin_means)
    if valid.sum() < 10:
        return None

    # Linear interpolation across full phase
    bin_means = np.interp(bin_centers, bin_centers[valid], bin_means[valid])

    # Crop to the central transit window (80 bins)
    cropped_transit = bin_means[crop_start:crop_end]
    return cropped_transit


def normalize_processed_curve(flux):
    """
    Standardizes flux to zero mean and unit variance.
    """
    std = np.std(flux)
    if std == 0:
        return flux
    return (flux - np.mean(flux)) / std
