# Astro-Informed Lightweight Multimodal Deep Learning for Exoplanet Transit Vetting in TESS M-Dwarf Systems

## 1. Project Overview
This repository contains a self-contained, turnkey, publication-ready research framework for automated exoplanet transit vetting in photometric time-series from NASA's Transiting Exoplanet Survey Satellite (TESS).

Focusing specifically on active **M-dwarf systems** (where magnetic flares and rotational starspot modulation frequently mimic planetary dips), this framework couples **80-bin phase-grid light-curve representations** with **astrophysical stellar host metadata** (`st_teff`, `pl_orbper`, `st_rad`).

### Key Results Summary (Test Set N=437):
- **Heterogeneous Ensemble ROC-AUC:** `0.8839` (outperforming all standalone models)
- **F1-Score:** `0.8114` (95% Bootstrap Confidence Interval: `[0.7709, 0.8463]`)
- **Accuracy:** `78.72%`
- **False Negative Rate (Missed Planets):** `13.42%` (lowest in project)
- **CPU Inference Latency:** `1.05 ms per star` (<1 MB memory footprint)

---

## 2. System Requirements & Hardware

- **Operating System:** Windows 10/11, Ubuntu/Debian Linux (>= 20.04), or macOS (Intel/Apple Silicon).
- **Python Version:** Python 3.10 or 3.11 recommended.
- **Hardware:** Standard commodity CPU (Intel Core i5/i7, AMD Ryzen, or Apple M1/M2/M3). No GPU is required.
- **RAM:** Minimum 4 GB RAM (8 GB recommended).
- **Disk Space:**
  - Fast Evaluation & Training Mode: ~150 MB (bundled preprocessed data + checkpoints).
  - Full Raw Download Mode: ~1.5 GB (stores cached `.npz` curves for ~2,200 stars).

---

## 3. Installation & Environment Setup

### Option A: Using Conda (Recommended)
```bash
# 1. Clone or extract the repository
git clone <repository-url>
cd exoplanet-transit-unified

# 2. Create the virtual environment from environment.yml
conda env create -f environment.yml

# 3. Activate the environment
conda activate exoplanet-dl
```

### Option B: Using Standard Python (venv / pip)
```bash
# 1. Clone or extract the repository
git clone <repository-url>
cd exoplanet-transit-unified

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution Modes (Master `run.py` Script)

All operations are unified under the single script `run.py`.

### Mode 1: Fast Evaluation & Reproduction (Default)
Evaluates all pre-trained model checkpoints on the test set (N=437), generates 1,000-iteration bootstrap confidence intervals, runs 5-fold cross-validation, benchmarks CPU inference latency, and exports publication figures and tables:
```bash
python run.py
# or explicitly:
python run.py --mode eval
```
*Execution Time: ~20 seconds on standard CPU.*

### Mode 2: Retrain All Models From Scratch
Retrains all baseline and multimodal architectures (Two-Stage Dual CNN, ResExoNet, Hybrid Transformer, Fusion MLP, Logistic Regression) on the phase-grid dataset, updates checkpoints, and saves evaluation metrics:
```bash
python run.py --mode train
```

### Mode 3: Download & Build Dataset From Scratch
Queries the NASA Exoplanet Archive for the TOI table, downloads light curves via Lightkurve from MAST, applies Savitzky-Golay detrending and phase-folding, and synthesizes the 80-bin phase-grid dataset:
```bash
python run.py --mode download
```
*Tip: To test data ingestion on a small sample of stars first, use:*
```bash
python run.py --mode download --max_download 10
```

### Mode 4: Full End-to-End Pipeline
Executes all four phases sequentially: (1) NASA/MAST download -> (2) Phase-grid synthesis -> (3) Model training -> (4) Rigorous evaluation and plot export:
```bash
python run.py --mode full
```

### Additional Command Line Flags:
```bash
python run.py --help          # Show full parameter documentation
python run.py --skip_plots    # Skip image rendering for headless servers
```

---

## 5. Repository Structure

```text
exoplanet-transit-unified/
├── run.py                             # Master entry point (eval / train / download / full)
├── requirements.txt                   # Pip package dependencies
├── environment.yml                    # Conda environment definition
├── README.md                          # Comprehensive user guide
├── Research_Manuscript_Foundation.docx # Academic manuscript monograph (Word document)
│
├── checkpoints/                       # Pre-trained PyTorch weights (.pth) & scaler
│   ├── two_stage_dual_model.pth       # Flagship Two-Stage Dual CNN weights
│   ├── res_exonet_model.pth           # Tri-Branch Residual network weights
│   ├── hybrid_dual.pth                # CNN-Transformer hybrid weights
│   ├── cnn_baseline.pth               # Unimodal 1D-CNN baseline weights
│   └── metadata_scaler.pkl            # StandardScaler fitted on training partition
│
├── data/
│   ├── processed/                     # Canonical 80-bin phase-grid arrays
│   │   ├── lightcurves.npy            # (2181, 80) normalized flux matrix
│   │   ├── metadata.npy               # (2181, 3) host-star parameters
│   │   └── labels.npy                 # (2181,) binary labels (0=FP, 1=Confirmed)
│   ├── raw/                           # Curated NASA catalog
│   │   └── toi_clean_labels.csv       # Verified TOI ground-truth table
│   └── folded_curves/                 # Cached raw folded light curves (.npz archives)
│
├── results/
│   ├── plots/                         # Publication-quality diagnostic figures
│   │   ├── discrimination_curves.png  # Side-by-side ROC and Precision-Recall curves
│   │   ├── confusion_matrices.png     # Multi-panel normalized confusion matrices
│   │   ├── f1_bootstrap_ci.png        # 1000-iteration 95% Bootstrap CIs error bars
│   │   ├── ablation_study.png         # Architectural progression bar chart
│   │   └── latency_frontier.png       # Inference Latency vs F1 Pareto frontier
│   └── tables/                        # Exported metric tables
│       ├── final_metrics_summary.csv  # Machine-readable performance benchmarks
│       └── final_metrics_summary.md   # Markdown summary table
│
└── src/
    ├── data/
    │   ├── loader.py                  # Queries NASA Exoplanet Archive for TOI table
    │   ├── downloader.py              # Downloads TESS light curves from MAST via Lightkurve
    │   ├── dataset_builder.py         # Bins and center-crops curves into 80-bin grids
    │   ├── preprocessing.py           # Savitzky-Golay detrending and phase-grid logic
    │   └── dataset.py                 # PyTorch Dataset classes
    ├── models/
    │   ├── cnn.py                     # TransitCNN (3-block 1D-CNN)
    │   ├── mlp.py                     # TransitMLP & FusionMLP
    │   ├── rnn.py                     # TransitRNN (2-layer GRU)
    │   ├── metadata_mlp.py            # Metadata projection sub-network
    │   ├── dual_input.py              # Late-fusion dual-input network
    │   ├── two_stage_dual.py          # Disentangled two-stage dual CNN
    │   ├── hybrid.py                  # CNN-Transformer hybrid with attention pooling
    │   └── res_exonet.py              # Tri-branch residual skip connection network
    ├── training/
    │   └── trainer.py                 # Single-stage and two-stage training routines
    └── utils/
        ├── config.py                  # Hyperparameters and dynamic path resolution
        ├── metrics.py                 # Classification metrics, bootstrap CI, latency
        └── plotting.py                # Academic plotting functions (Matplotlib/Seaborn)
```

---

## 6. Experimental Benchmark Results (Test Set N=437)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Alarm Rate (FPR) | Missed Planet Rate (FNR) | 95% Bootstrap CI | CPU Latency | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (83 feats) | 75.06% | 74.40% | 80.52% | 0.7734 | 0.8252 | 31.07% | 19.48% | `[0.7302, 0.8113]` | 0.057 ms | 84 |
| **Fusion MLP** (83 feats) | 74.14% | 72.87% | 81.39% | 0.7689 | 0.8501 | 33.98% | 18.61% | `[0.7291, 0.8085]` | 0.156 ms | 19,522 |
| **Two-Stage Dual CNN** | 76.20% | 74.33% | 83.98% | 0.7886 | 0.8253 | 32.52% | 16.02% | `[0.7469, 0.8257]` | 0.840 ms | 61,742 |
| **ResExoNet** (Residual 3-Branch) | 73.46% | 73.66% | 77.49% | 0.7553 | 0.7925 | 31.07% | 22.51% | `[0.7120, 0.7966]` | 0.924 ms | 92,807 |
| **CNN-Transformer Hybrid** | 69.11% | 70.17% | 72.29% | 0.7122 | 0.7485 | 34.47% | 27.71% | `[0.6652, 0.7546]` | 1.434 ms | 188,525 |
| **Heterogeneous Ensemble (Uniform)** | **78.72%** | **76.34%** | **86.58%** | **0.8114** | **0.8839** | **30.10%** | **13.42%** | **`[0.7709, 0.8463]`** | **1.052 ms** | **81,348** |
| **Performance-Weighted Ensemble** | 78.26% | 76.15% | 85.71% | 0.8065 | 0.8839 | 30.10% | 14.29% | `[0.7651, 0.8428]` | 1.052 ms | 81,348 |

---

## 7. Troubleshooting & FAQ

### Q1: Lightkurve warning: `tpfmodel submodule is not available without oktopus`
- **Cause:** Lightkurve prints an informational warning if the optional `oktopus` package is not installed.
- **Impact:** Harmless. Target Pixel File (TPF) modeling is not used in this project; our pipeline uses PDCSAP time-series flux directly.

### Q2: Network Timeout during `python run.py --mode download`
- **Cause:** MAST or NASA Exoplanet Archive rate-limits simultaneous downloads or encounters temporary network lag.
- **Fix:** The downloader checks if a star is already cached in `data/folded_curves/{TIC_ID}.npz`. If interrupted, simply re-run `python run.py --mode download`; it resumes from where it stopped and skips already downloaded curves.

### Q3: How do I inspect the full research paper?
- Open [`Research_Manuscript_Foundation.docx`](./Research_Manuscript_Foundation.docx) in Microsoft Word, LibreOffice Writer, or Google Docs. It contains the complete academic text, structured sections, mathematical formulas, and citation list.

---

## 8. Data Sources & Acknowledgments
- **Photometry:** NASA Transiting Exoplanet Survey Satellite (TESS).
- **Target Archive:** NASA Exoplanet Archive (TESS Objects of Interest - TOI catalog).
- **Software Libraries:** Lightkurve, Astropy, Astroquery, PyTorch, Scikit-learn, Matplotlib, Seaborn.
