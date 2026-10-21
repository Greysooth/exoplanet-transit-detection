"""
Master Unified Research Pipeline Entry Point
Exoplanet Transit Detection via Lightweight Multimodal Deep Learning
"""

import os
import sys
import time
import argparse
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Set up module path dynamically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR, TABLES_DIR,
    SEED, TRANSIT_POINTS, TOTAL_FEATURES, BATCH_SIZE, LEARNING_RATE,
    BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL, CV_FOLDS, LATENCY_BENCHMARK_ITERATIONS,
    PALETTE
)
from src.utils.metrics import (
    compute_classification_metrics, bootstrap_f1_ci, count_parameters, measure_cpu_latency
)
from src.utils.plotting import (
    plot_discrimination_curves, plot_confusion_matrices, plot_bootstrap_ci,
    plot_ablation_study, plot_latency_frontier
)
from src.models.cnn import TransitCNN
from src.models.mlp import TransitMLP, FusionMLP
from src.models.rnn import TransitRNN
from src.models.dual_input import DualInputModel
from src.models.two_stage_dual import TwoStageDualModel
from src.models.hybrid import HybridDualInput
from src.models.res_exonet import ResExoNet
from src.training.trainer import train_single_stage, train_two_stage, get_class_weights
from src.data.loader import fetch_toi_catalog
from src.data.downloader import build_raw_cache
from src.data.dataset_builder import build_phasegrid_dataset

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Lock random seeds for absolute scientific reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cpu")  # CPU benchmarked for edge deployment validity

CLASS_NAMES = ["False Positive", "Planet Candidate"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified Exoplanet Transit Research Pipeline")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "train", "download", "full"],
                        help="'eval': evaluate checkpoints & baselines; 'train': retrain models; "
                             "'download': ingest TOI catalog & download lightcurves; 'full': end-to-end.")
    parser.add_argument("--max_download", type=int, default=None,
                        help="Optional cap on targets to download (useful for fast verification).")
    parser.add_argument("--skip_plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def load_and_preprocess_data():
    """
    Ingests the pre-processed phase-grid dataset and establishes the canonical 80/20 stratified split.
    """
    lc_path = os.path.join(PROCESSED_DATA_DIR, "lightcurves.npy")
    meta_path = os.path.join(PROCESSED_DATA_DIR, "metadata.npy")
    label_path = os.path.join(PROCESSED_DATA_DIR, "labels.npy")

    if not (os.path.exists(lc_path) and os.path.exists(meta_path) and os.path.exists(label_path)):
        raise FileNotFoundError(
            f"Processed dataset missing from {PROCESSED_DATA_DIR}. "
            f"Please run 'python run.py --mode download' to synthesize the dataset."
        )

    X_lc = np.load(lc_path)
    X_meta = np.load(meta_path)
    y = np.load(label_path).astype(int)

    X_lc_flat = X_lc.reshape(X_lc.shape[0], -1) if X_lc.ndim == 3 else X_lc
    X_lc_tensor = np.expand_dims(X_lc_flat, axis=1).astype(np.float32)

    # Canonical 80/20 stratified train/test split
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)

    # Fit scaler strictly on training split
    scaler = StandardScaler()
    X_meta_scaled = np.zeros_like(X_meta, dtype=np.float64)
    X_meta_scaled[idx_tr] = scaler.fit_transform(X_meta[idx_tr])
    X_meta_scaled[idx_te] = scaler.transform(X_meta[idx_te])

    # Multimodal concatenated flat vector (80 LC + 3 Meta = 83 features)
    X_comb = np.hstack([X_lc_flat, X_meta_scaled]).astype(np.float32)

    data_bundle = {
        "X_lc_flat": X_lc_flat,
        "X_lc_tensor": X_lc_tensor,
        "X_meta": X_meta_scaled.astype(np.float32),
        "X_comb": X_comb,
        "y": y,
        "idx_tr": idx_tr,
        "idx_te": idx_te,
        "scaler": scaler
    }
    return data_bundle


def train_fusion_mlp_baseline(X_tr, y_tr, epochs=40, lr=1e-4):
    """Trains the baseline non-linear Fusion MLP."""
    model = FusionMLP(input_dim=X_tr.shape[1]).to(device)
    weights = get_class_weights(y_tr, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model


def get_model_proba(model, X_input, is_multimodal=False, meta_input=None):
    """Utility to generate calibrated probabilities from PyTorch modules."""
    model.eval()
    with torch.no_grad():
        if is_multimodal:
            tx = torch.tensor(X_input, dtype=torch.float32)
            tm = torch.tensor(meta_input, dtype=torch.float32)
            logits = model(tx, tm)
        else:
            tx = torch.tensor(X_input, dtype=torch.float32)
            logits = model(tx)
        probas = torch.softmax(logits, dim=1).cpu().numpy()
    return probas


def main():
    args = parse_arguments()
    print("=" * 80)
    print("      ASTROPHYSICS-INFORMED MULTIMODAL EXOPLANET TRANSIT PIPELINE       ")
    print("=" * 80)
    print(f"Pipeline Mode: {args.mode.upper()} | Target Hardware: {device.type.upper()}")

    # Optional Data Ingestion & Downloader Phase
    if args.mode in ["download", "full"]:
        print("\n[DOWNLOAD PHASE] Ingesting NASA TOI Catalog & TESS Light Curves...")
        fetch_toi_catalog()
        build_raw_cache(max_targets=args.max_download)
        build_phasegrid_dataset()
        if args.mode == "download":
            print("\nDownload & preprocessing phase completed successfully.")
            return

    # 1. Ingestion
    print("\n[1/5] Ingesting phase-grid light curves and stellar metadata...")
    data = load_and_preprocess_data()
    idx_tr, idx_te = data["idx_tr"], data["idx_te"]
    y_te = data["y"][idx_te]

    print(f"      Total Samples: {len(data['y'])} | Train: {len(idx_tr)} | Test: {len(idx_te)}")
    print(f"      Class Balance: False Positives: {np.sum(y_te == 0)} | Planet Candidates: {np.sum(y_te == 1)}")

    # 2. Benchmarking & Evaluation
    print("\n[2/5] Executing Model Suite (Unimodal, Multimodal, & Ensembles)...")
    results = {}
    efficiency = {}

    # --- MODEL 1: Logistic Regression Baseline ---
    print("  -> Benchmarking: Multimodal Logistic Regression (Linear Baseline)")
    lr_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr_clf.fit(data["X_comb"][idx_tr], data["y"][idx_tr])
    lr_proba = lr_clf.predict_proba(data["X_comb"][idx_te])
    lr_pred = lr_proba.argmax(axis=1)
    results["LR"] = {
        "preds": lr_pred,
        "probas": lr_proba,
        "color": PALETTE["LR"],
        "metrics": compute_classification_metrics(y_te, lr_pred, lr_proba)
    }
    lr_lat = measure_cpu_latency(lambda x: lr_clf.predict_proba(x), data["X_comb"][idx_te[:1]], n_iter=LATENCY_BENCHMARK_ITERATIONS)
    efficiency["LR"] = {"latency_ms": lr_lat["latency_ms"], "total_params": 84, "f1": results["LR"]["metrics"]["f1"]}

    # --- MODEL 2: Fusion MLP Baseline ---
    print("  -> Benchmarking: Multimodal Fusion MLP (Non-linear Dense Baseline)")
    mlp_model = train_fusion_mlp_baseline(data["X_comb"][idx_tr], data["y"][idx_tr])
    mlp_proba = get_model_proba(mlp_model, data["X_comb"][idx_te])
    mlp_pred = mlp_proba.argmax(axis=1)
    results["MLP"] = {
        "preds": mlp_pred,
        "probas": mlp_proba,
        "color": PALETTE["MLP"],
        "metrics": compute_classification_metrics(y_te, mlp_pred, mlp_proba)
    }
    mlp_params = count_parameters(mlp_model)
    mlp_lat = measure_cpu_latency(lambda x: get_model_proba(mlp_model, x), data["X_comb"][idx_te[:1]], n_iter=LATENCY_BENCHMARK_ITERATIONS)
    efficiency["MLP"] = {"latency_ms": mlp_lat["latency_ms"], "total_params": mlp_params["total_params"], "f1": results["MLP"]["metrics"]["f1"]}

    # --- MODEL 3: Two-Stage Spatial Dual CNN (Flagship) ---
    print("  -> Benchmarking: Two-Stage Spatial Dual CNN (Astro-Informed)")
    two_stage_ckpt = os.path.join(CHECKPOINT_DIR, "two_stage_dual_model.pth")
    two_stage_model = TwoStageDualModel()
    
    force_train = (args.mode in ["train", "full"]) or not os.path.exists(two_stage_ckpt)
    if force_train:
        print("     [Training] Training TwoStageDualModel from scratch...")
        tr_loader = DataLoader(
            TensorDataset(
                torch.tensor(data["X_lc_tensor"][idx_tr], dtype=torch.float32),
                torch.tensor(data["X_meta"][idx_tr], dtype=torch.float32),
                torch.tensor(data["y"][idx_tr], dtype=torch.long)
            ), batch_size=BATCH_SIZE, shuffle=True
        )
        te_loader = DataLoader(
            TensorDataset(
                torch.tensor(data["X_lc_tensor"][idx_te], dtype=torch.float32),
                torch.tensor(data["X_meta"][idx_te], dtype=torch.float32),
                torch.tensor(y_te, dtype=torch.long)
            ), batch_size=BATCH_SIZE, shuffle=False
        )
        two_stage_model, _ = train_two_stage(two_stage_model, tr_loader, te_loader, device, save_path=two_stage_ckpt)
    else:
        two_stage_model.load_state_dict(torch.load(two_stage_ckpt, map_location=device, weights_only=True))

    two_stage_proba = get_model_proba(two_stage_model, data["X_lc_tensor"][idx_te], is_multimodal=True, meta_input=data["X_meta"][idx_te])
    two_stage_pred = two_stage_proba.argmax(axis=1)
    results["TwoStage_CNN"] = {
        "preds": two_stage_pred,
        "probas": two_stage_proba,
        "color": PALETTE["TwoStageDual"],
        "metrics": compute_classification_metrics(y_te, two_stage_pred, two_stage_proba)
    }
    two_stage_params = count_parameters(two_stage_model)
    two_stage_lat = measure_cpu_latency(
        lambda _: get_model_proba(two_stage_model, data["X_lc_tensor"][idx_te[:1]], is_multimodal=True, meta_input=data["X_meta"][idx_te[:1]]),
        None, n_iter=LATENCY_BENCHMARK_ITERATIONS
    )
    efficiency["TwoStage_CNN"] = {"latency_ms": two_stage_lat["latency_ms"], "total_params": two_stage_params["total_params"], "f1": results["TwoStage_CNN"]["metrics"]["f1"]}

    # --- MODEL 4: ResExoNet (Tri-Branch Residual Architecture) ---
    res_ckpt = os.path.join(CHECKPOINT_DIR, "res_exonet_model.pth")
    if os.path.exists(res_ckpt):
        print("  -> Benchmarking: ResExoNet (Tri-Branch Residual Network)")
        res_model = ResExoNet()
        res_model.load_state_dict(torch.load(res_ckpt, map_location=device, weights_only=True))
        res_proba = get_model_proba(res_model, data["X_lc_tensor"][idx_te], is_multimodal=True, meta_input=data["X_meta"][idx_te])
        res_pred = res_proba.argmax(axis=1)
        results["ResExoNet"] = {
            "preds": res_pred,
            "probas": res_proba,
            "color": PALETTE["ResExoNet"],
            "metrics": compute_classification_metrics(y_te, res_pred, res_proba)
        }
        res_params = count_parameters(res_model)
        res_lat = measure_cpu_latency(
            lambda _: get_model_proba(res_model, data["X_lc_tensor"][idx_te[:1]], is_multimodal=True, meta_input=data["X_meta"][idx_te[:1]]),
            None, n_iter=LATENCY_BENCHMARK_ITERATIONS
        )
        efficiency["ResExoNet"] = {"latency_ms": res_lat["latency_ms"], "total_params": res_params["total_params"], "f1": results["ResExoNet"]["metrics"]["f1"]}

    # --- MODEL 5: CNN-Transformer Hybrid ---
    hybrid_ckpt = os.path.join(CHECKPOINT_DIR, "hybrid_dual.pth")
    if os.path.exists(hybrid_ckpt):
        print("  -> Benchmarking: HybridDualInput (CNN + Transformer Encoder + Metadata)")
        hybrid_model = HybridDualInput()
        hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device, weights_only=True))
        hybrid_proba = get_model_proba(hybrid_model, data["X_lc_tensor"][idx_te], is_multimodal=True, meta_input=data["X_meta"][idx_te])
        hybrid_pred = hybrid_proba.argmax(axis=1)
        results["Hybrid_Transformer"] = {
            "preds": hybrid_pred,
            "probas": hybrid_proba,
            "color": PALETTE["HybridDual"],
            "metrics": compute_classification_metrics(y_te, hybrid_pred, hybrid_proba)
        }
        hyb_params = count_parameters(hybrid_model)
        hyb_lat = measure_cpu_latency(
            lambda _: get_model_proba(hybrid_model, data["X_lc_tensor"][idx_te[:1]], is_multimodal=True, meta_input=data["X_meta"][idx_te[:1]]),
            None, n_iter=LATENCY_BENCHMARK_ITERATIONS
        )
        efficiency["Hybrid_Transformer"] = {"latency_ms": hyb_lat["latency_ms"], "total_params": hyb_params["total_params"], "f1": results["Hybrid_Transformer"]["metrics"]["f1"]}

    # --- MODEL 6: Uniform Ensemble (LR + MLP + TwoStage CNN) ---
    print("  -> Benchmarking: Heterogeneous Uniform Soft-Voting Ensemble (1/3 Vote)")
    p_uni = (lr_proba + mlp_proba + two_stage_proba) / 3.0
    pred_uni = p_uni.argmax(axis=1)
    results["Ensemble_Uniform"] = {
        "preds": pred_uni,
        "probas": p_uni,
        "color": PALETTE["Ensemble_Uniform"],
        "metrics": compute_classification_metrics(y_te, pred_uni, p_uni)
    }
    ens_lat = lr_lat["latency_ms"] + mlp_lat["latency_ms"] + two_stage_lat["latency_ms"]
    ens_params = 84 + mlp_params["total_params"] + two_stage_params["total_params"]
    efficiency["Ensemble_Uniform"] = {"latency_ms": ens_lat, "total_params": ens_params, "f1": results["Ensemble_Uniform"]["metrics"]["f1"]}

    # --- MODEL 7: Leakage-Free Weighted Ensemble ---
    print("  -> Benchmarking: Performance-Weighted Ensemble (Validation F1 Weights)")
    idx_sub_tr, idx_sub_va = train_test_split(idx_tr, test_size=0.25, stratify=data["y"][idx_tr], random_state=SEED)
    w_lr_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    w_lr_clf.fit(data["X_comb"][idx_sub_tr], data["y"][idx_sub_tr])
    w_lr_pred = w_lr_clf.predict(data["X_comb"][idx_sub_va])
    f1_lr_val = compute_classification_metrics(data["y"][idx_sub_va], w_lr_pred)["f1"]

    w_mlp = train_fusion_mlp_baseline(data["X_comb"][idx_sub_tr], data["y"][idx_sub_tr], epochs=30)
    w_mlp_pred = get_model_proba(w_mlp, data["X_comb"][idx_sub_va]).argmax(axis=1)
    f1_mlp_val = compute_classification_metrics(data["y"][idx_sub_va], w_mlp_pred)["f1"]

    w_cnn_pred = get_model_proba(two_stage_model, data["X_lc_tensor"][idx_sub_va], is_multimodal=True, meta_input=data["X_meta"][idx_sub_va]).argmax(axis=1)
    f1_cnn_val = compute_classification_metrics(data["y"][idx_sub_va], w_cnn_pred)["f1"]

    tot_w = f1_lr_val + f1_mlp_val + f1_cnn_val
    w1, w2, w3 = f1_lr_val / tot_w, f1_mlp_val / tot_w, f1_cnn_val / tot_w
    p_wt = (w1 * lr_proba) + (w2 * mlp_proba) + (w3 * two_stage_proba)
    pred_wt = p_wt.argmax(axis=1)
    results["Ensemble_Weighted"] = {
        "preds": pred_wt,
        "probas": p_wt,
        "color": PALETTE["Ensemble_Weighted"],
        "metrics": compute_classification_metrics(y_te, pred_wt, p_wt)
    }

    # 3. Bootstrap Confidence Intervals
    print("\n[3/5] Computing 1000-Iteration Bootstrap 95% Confidence Intervals...")
    for name in results:
        ci_res = bootstrap_f1_ci(y_te, results[name]["preds"], n_iter=BOOTSTRAP_ITERATIONS, ci=CONFIDENCE_LEVEL, seed=SEED)
        results[name]["ci"] = ci_res
        print(f"      {name:<20}: F1 Mean = {ci_res['mean']:.4f}  [95% CI: {ci_res['ci_lower']:.4f} - {ci_res['ci_upper']:.4f}]")

    # 4. Stratified 5-Fold Cross-Validation (Baseline Generalization)
    print("\n[4/5] Executing Stratified 5-Fold Cross-Validation on Primary Baselines...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = {"LR": [], "MLP": [], "Ensemble_2Model": []}

    for fold, (tr_idx_k, va_idx_k) in enumerate(skf.split(data["X_comb"], data["y"])):
        fold_scaler = StandardScaler()
        meta_tr = fold_scaler.fit_transform(data["X_meta"][tr_idx_k])
        meta_va = fold_scaler.transform(data["X_meta"][va_idx_k])
        comb_tr = np.hstack([data["X_lc_flat"][tr_idx_k], meta_tr]).astype(np.float32)
        comb_va = np.hstack([data["X_lc_flat"][va_idx_k], meta_va]).astype(np.float32)

        k_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        k_lr.fit(comb_tr, data["y"][tr_idx_k])
        p_k_lr = k_lr.predict_proba(comb_va)

        k_mlp = train_fusion_mlp_baseline(comb_tr, data["y"][tr_idx_k], epochs=35)
        p_k_mlp = get_model_proba(k_mlp, comb_va)

        p_k_ens = (p_k_lr + p_k_mlp) / 2.0

        cv_scores["LR"].append(compute_classification_metrics(data["y"][va_idx_k], p_k_lr.argmax(1))["f1"])
        cv_scores["MLP"].append(compute_classification_metrics(data["y"][va_idx_k], p_k_mlp.argmax(1))["f1"])
        cv_scores["Ensemble_2Model"].append(compute_classification_metrics(data["y"][va_idx_k], p_k_ens.argmax(1))["f1"])

    for k, v in cv_scores.items():
        print(f"      {k:<18} -> 5-Fold Mean F1: {np.mean(v):.4f} +/- {np.std(v):.4f}")

    # 5. Scientific Plots & Table Exports
    if not args.skip_plots:
        print("\n[5/5] Exporting Publication Diagnostic Figures & Tables...")
        plot_discrimination_curves(results, y_te, os.path.join(PLOTS_DIR, "discrimination_curves.png"))
        plot_confusion_matrices(results, CLASS_NAMES, os.path.join(PLOTS_DIR, "confusion_matrices.png"))
        plot_bootstrap_ci(results, os.path.join(PLOTS_DIR, "f1_bootstrap_ci.png"))
        plot_ablation_study({k: v["metrics"] for k, v in results.items()}, os.path.join(PLOTS_DIR, "ablation_study.png"))
        plot_latency_frontier(efficiency, os.path.join(PLOTS_DIR, "latency_frontier.png"))
        print(f"      Diagnostic figures saved to: {PLOTS_DIR}")

    # Build Summary Table
    rows = []
    for name, data_item in results.items():
        m = data_item["metrics"]
        ci = data_item["ci"]
        eff = efficiency.get(name, {})
        rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "FPR": f"{m['fpr']:.4f}",
            "FNR (Missed)": f"{m['fnr']:.4f}",
            "95% Bootstrap CI": f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]",
            "Latency (ms)": f"{eff.get('latency_ms', np.nan):.3f}",
            "Params": f"{eff.get('total_params', np.nan)}"
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(TABLES_DIR, "final_metrics_summary.csv")
    md_path = os.path.join(TABLES_DIR, "final_metrics_summary.md")
    df.to_csv(csv_path, index=False)
    
    # Safe markdown export
    try:
        df.to_markdown(md_path, index=False)
    except Exception:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))

    print("\n" + "=" * 90)
    print("                     FINAL SCIENTIFIC METRICS SUMMARY TABLE                    ")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\nTables exported to:\n  - CSV: {csv_path}\n  - Markdown: {md_path}\n")
    print("Execution complete. All experimental benchmarks reproduced successfully.")

if __name__ == "__main__":
    main()
