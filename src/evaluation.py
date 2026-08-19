import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from pathlib import Path
from src.utils import get_logger, CLASS_ORDER, REPORTS_DIR

logger = get_logger("evaluation")

def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Computes classification metrics for a multi-class problem.
    """
    accuracy = accuracy_score(y_true, y_pred)
    # Use macro averaging to give equal weight to each class
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Multiclass ROC-AUC (One-vs-Rest, macro-averaged)
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        roc_auc = np.nan
        
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_macro": float(f1),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None
    }


def save_confusion_matrix_plot(y_true, y_pred, model_name: str, save_path: Path):
    """
    Plots and saves a confusion matrix in the correct class order.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=CLASS_ORDER, 
        yticklabels=CLASS_ORDER
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual Performance")
    plt.xlabel("Predicted Performance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")


def save_roc_curves_plot(y_true, y_prob, model_name: str, save_path: Path):
    """
    Plots and saves multi-class ROC curves using One-vs-Rest strategy.
    """
    # Binarize labels for multi-class ROC calculation
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    n_classes = 4
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(8, 6))
    colors = ["red", "orange", "blue", "green"]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i], tpr[i], 
            color=colors[i], 
            lw=2, 
            label=f"ROC curve of class {CLASS_ORDER[i]} (AUC = {roc_auc[i]:0.2f})"
        )
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curves plot to {save_path}")


def save_feature_importance_plot(model_pipeline, feature_names: list, model_name: str, save_path: Path):
    """
    Identifies and plots feature importances or model coefficients.
    """
    # Extract the classifier from the pipeline
    classifier = model_pipeline.named_steps["classifier"]
    
    importances = None
    title = ""
    
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        title = f"Feature Importance - {model_name}"
    elif hasattr(classifier, "coef_"):
        # For multi-class linear models, we average absolute coefficients across classes
        importances = np.mean(np.abs(classifier.coef_), axis=0)
        title = f"Average Coefficient Magnitude - {model_name}"
        
    if importances is not None:
        # Match features
        if len(importances) == len(feature_names):
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
            
            plt.figure(figsize=(10, 6))
            feat_imp.plot(kind="barh", color="teal")
            plt.title(title)
            plt.xlabel("Importance / Coefficient Magnitude")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            logger.info(f"Saved feature importance plot to {save_path}")
        else:
            logger.warning(
                f"Feature importance size ({len(importances)}) mismatch with feature names size ({len(feature_names)}). "
                "Skipping plot."
            )
    else:
        logger.info(f"Model '{model_name}' does not support feature importances or coefficients. Skipping plot.")
