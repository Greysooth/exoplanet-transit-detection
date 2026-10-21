import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Reproducibility seed
SEED = 42

# Preprocessing hyper-parameters
TRANSIT_POINTS = 80
METADATA_DIM = 3
TOTAL_FEATURES = TRANSIT_POINTS + METADATA_DIM  # 83

# Training hyper-parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS_SINGLE_STAGE = 50
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2A = 10
EPOCHS_STAGE2B = 30

# Evaluation settings
TEST_SPLIT_RATIO = 0.20
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
CV_FOLDS = 5
LATENCY_BENCHMARK_ITERATIONS = 200

# Color palette for publication plots
PALETTE = {
    "LR": "#2E75B6",
    "MLP": "#ED7D31",
    "CNN": "#70AD47",
    "DualInput": "#41719C",
    "TwoStageDual": "#5B9BD5",
    "HybridDual": "#FFC000",
    "ResExoNet": "#00B050",
    "Ensemble_Uniform": "#7030A0",
    "Ensemble_Weighted": "#C00000"
}
