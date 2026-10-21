import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# Configure academic plot styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif",
    "figure.dpi": 200
})

def plot_discrimination_curves(results_dict, y_true, save_path):
    """
    Plots side-by-side ROC Curves and Precision-Recall Curves for all evaluated models.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. ROC Curves
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")
    for name, data in results_dict.items():
        if "probas" in data:
            fpr, tpr, _ = roc_curve(y_true, data["probas"][:, 1])
            auc_val = data["metrics"]["roc_auc"]
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2, color=data.get("color"))
            
    ax_roc.set_title("(a) Receiver Operating Characteristic (ROC)")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR / Recall)")
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    
    # 2. Precision-Recall Curves
    baseline_pr = np.mean(y_true)
    ax_pr.axhline(y=baseline_pr, color="k", linestyle="--", alpha=0.6, label=f"Prevalence ({baseline_pr:.3f})")
    for name, data in results_dict.items():
        if "probas" in data:
            precision, recall, _ = precision_recall_curve(y_true, data["probas"][:, 1])
            ap_val = data["metrics"]["pr_auc"]
            ax_pr.plot(recall, precision, label=f"{name} (AP = {ap_val:.3f})", lw=2, color=data.get("color"))
            
    ax_pr.set_title("(b) Precision-Recall Curve")
    ax_pr.set_xlabel("Recall (True Positive Rate)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.legend(loc="lower left", frameon=True)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results_dict, class_names, save_path):
    """
    Plots a multi-panel grid of normalized confusion matrices.
    """
    models_to_plot = [k for k, v in results_dict.items() if "preds" in v]
    n_models = len(models_to_plot)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, name in enumerate(models_to_plot):
        ax = axes[i]
        m = results_dict[name]["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        
        annot = np.empty_like(cm, dtype=object)
        for r in range(2):
            for c in range(2):
                annot[r, c] = f"{cm[r, c]}\n({cm_norm[r, c]:.1%})"
                
        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"{name}\nAcc: {m['accuracy']:.1%} | F1: {m['f1']:.3f}", fontsize=11)
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_bootstrap_ci(results_dict, save_path):
    """
    Plots horizontal error-bar comparisons of 95% Bootstrap Confidence Intervals for F1 scores.
    """
    models = []
    means = []
    lowers = []
    uppers = []
    colors = []
    
    for name, data in results_dict.items():
        if "ci" in data:
            models.append(name)
            means.append(data["ci"]["mean"])
            lowers.append(data["ci"]["mean"] - data["ci"]["ci_lower"])
            uppers.append(data["ci"]["ci_upper"] - data["ci"]["mean"])
            colors.append(data.get("color", "#2E75B6"))
            
    fig, ax = plt.subplots(figsize=(9, max(4, len(models) * 0.6)))
    y_pos = np.arange(len(models))
    
    ax.errorbar(means, y_pos, xerr=[lowers, uppers], fmt="o", color="navy",
                ecolor="dimgray", elinewidth=2, capsize=5, capthick=2, markersize=7)
    
    for i, (m, y_p, c) in enumerate(zip(means, y_pos, colors)):
        ax.scatter(m, y_p, color=c, s=90, zorder=3)
        ax.text(m + 0.005, y_p + 0.1, f"{m:.3f} [{data['ci']['ci_lower']:.3f}, {data['ci']['ci_upper']:.3f}]",
                va="center", fontsize=9, fontweight="bold")
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Validation F1 Score (1000-Iteration 95% Bootstrap CI)")
    ax.set_title("Statistical Robustness Bounds Across Architectural Candidates")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim([0.65, 0.88])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_ablation_study(metrics_summary, save_path):
    """
    Generates a grouped bar chart visualizing the progression from
    Unimodal -> Late Fusion -> Disentangled Two-Stage -> Multimodal Ensemble.
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    models = list(metrics_summary.keys())
    
    x = np.arange(len(categories))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(models):
        m = metrics_summary[name]
        vals = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]]
        ax.bar(x + i * width, vals, width, label=name)
        
    ax.set_ylabel("Score")
    ax.set_title("Ablation Progress: Unimodal Baselines vs Multimodal Architectures")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim([0.45, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_latency_frontier(efficiency_data, save_path):
    """
    Plots the Accuracy/F1 vs CPU Inference Latency Pareto frontier.
    Highlights that lightweight models achieve competitive performance within <1ms.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in efficiency_data.items():
        latency = d["latency_ms"]
        f1 = d["f1"]
        params = d["total_params"]
        ax.scatter(latency, f1, s=max(40, params / 1000), alpha=0.8, edgecolors="black", label=f"{name} ({params:,} p)")
        ax.annotate(name, (latency * 1.03, f1), fontsize=9, fontweight="medium")
        
    ax.set_xscale("log")
    ax.set_xlabel("CPU Inference Latency per Sample (ms) [Log Scale]")
    ax.set_ylabel("F1 Score")
    ax.set_title("Operational Efficiency Envelope: Inference Speed vs Detection Performance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
