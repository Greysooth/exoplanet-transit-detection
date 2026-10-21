import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes a full suite of astronomical classification metrics.
    Includes True Positives, False Positives, False Negatives, False Alarm Rate,
    and Missed Transit Rate.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Missed planet candidates
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr)
    }
    
    if y_prob is not None:
        if y_prob.ndim == 2:
            prob_pos = y_prob[:, 1]
        else:
            prob_pos = y_prob
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, prob_pos)
            metrics["pr_auc"] = average_precision_score(y_true, prob_pos)
        except Exception:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        
    return metrics


def bootstrap_f1_ci(y_true, y_pred, n_iter=1000, ci=0.95, seed=42):
    """
    Computes non-parametric bootstrap confidence intervals on F1 score.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    f1_boot = []
    
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        # Verify both classes present
        if len(np.unique(y_true[idx])) < 2:
            continue
        f1_boot.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        
    f1_boot = np.array(f1_boot)
    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(f1_boot, alpha * 100)
    upper = np.percentile(f1_boot, (1.0 - alpha) * 100)
    mean_val = np.mean(f1_boot)
    std_val = np.std(f1_boot)
    
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "distribution": f1_boot
    }


def count_parameters(model):
    """
    Counts total and trainable parameters in a PyTorch module.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 * 1024)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "size_mb": size_mb
    }


def measure_cpu_latency(predict_fn, sample_input, n_iter=200, warmup=20):
    """
    Measures mean single-sample CPU inference latency in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        _ = predict_fn(sample_input)
        
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = predict_fn(sample_input)
    elapsed = time.perf_counter() - t0
    
    latency_ms = (elapsed / n_iter) * 1000.0
    throughput_hz = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return {
        "latency_ms": latency_ms,
        "throughput_hz": throughput_hz
    }
