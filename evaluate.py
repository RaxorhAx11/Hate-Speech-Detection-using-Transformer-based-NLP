import os
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from config import load_config, AppConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Map indices to label names
LABEL_MAP = {0: "Safe", 1: "Offensive", 2: "Hate Speech"}

def evaluate_model(config: AppConfig) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Loads the best model and evaluates it against the test set."""
    best_model_path = os.path.join(config.training.output_dir, "best_model")
    test_path = os.path.join(config.dataset.data_dir, "test.csv")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model path {best_model_path} does not exist. Run train.py first.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data {test_path} does not exist. Run dataset_builder.py first.")
        
    logger.info("Loading best model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info("Loading test dataset...")
    test_df = pd.read_csv(test_path)
    
    # Prepare batch inference
    text_col = "clean_text" if "clean_text" in test_df.columns else "text"
    texts = test_df[text_col].tolist()
    labels = test_df["label"].tolist()
    
    all_probabilities = []
    all_predictions = []
    
    # Process in batches to avoid OOM
    batch_size = config.training.batch_size
    logger.info("Running evaluation predictions...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.model.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Calculate probabilities via softmax
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            
            all_probabilities.extend(probs)
            all_predictions.extend(preds)
            
    all_probabilities = np.array(all_probabilities)
    all_predictions = np.array(all_predictions)
    labels = np.array(labels)
    
    # Compute Overall Metrics
    accuracy = accuracy_score(labels, all_predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, all_predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, all_predictions, average="weighted", zero_division=0
    )
    
    # Compute Per-Class Metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, all_predictions, average=None, zero_division=0
    )
    
    # ROC AUC Score (one-vs-rest)
    try:
        roc_auc = roc_auc_score(labels, all_probabilities, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"ROC-AUC calculation failed: {e}. Setting to 0.0")
        roc_auc = 0.0
        
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc
    }
    for i in range(3):
        label_name = LABEL_MAP[i]
        metrics[f"{label_name.lower().replace(' ', '_')}_precision"] = precision_per_class[i]
        metrics[f"{label_name.lower().replace(' ', '_')}_recall"] = recall_per_class[i]
        metrics[f"{label_name.lower().replace(' ', '_')}_f1"] = f1_per_class[i]
        metrics[f"{label_name.lower().replace(' ', '_')}_support"] = int(support_per_class[i])
    
    # Create evaluation outputs folders
    eval_dir = "evaluation_plots"
    assets_dir = "assets"
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, all_predictions)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{LABEL_MAP[i].replace(' ', '_')}" for i in range(3)],
        columns=[f"Pred_{LABEL_MAP[i].replace(' ', '_')}" for i in range(3)]
    )
    cm_df.to_csv(os.path.join(eval_dir, "confusion_matrix.csv"))
    cm_df.to_csv(os.path.join(assets_dir, "confusion_matrix.csv"))

    # Write Classification Report
    class_report_str = classification_report(
        labels, all_predictions, target_names=[LABEL_MAP[i] for i in range(3)], digits=4, zero_division=0
    )
    report_file = os.path.join(eval_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(class_report_str)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Macro): {precision_macro:.4f}\n")
        f.write(f"Recall (Macro): {recall_macro:.4f}\n")
        f.write(f"F1 (Macro): {f1_macro:.4f}\n")
        f.write(f"ROC-AUC (Macro): {roc_auc:.4f}\n\n")
        f.write("=== CONFUSION MATRIX ===\n")
        f.write(cm_df.to_string())
        f.write("\n")
    logger.info(f"Classification report saved to {report_file}")
    
    # Generate confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_MAP[i] for i in range(3)],
        yticklabels=[LABEL_MAP[i] for i in range(3)]
    )
    plt.title("Confusion Matrix - Hate Speech Detection (Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    cm_assets_path = os.path.join(assets_dir, "confusion_matrix.png")
    plt.savefig(cm_assets_path, dpi=300)
    plt.close()
    logger.info(f"Confusion Matrix saved to {cm_path} and {cm_assets_path}")
    
    # Generate ROC Curve Plot
    plt.figure(figsize=(8, 6))
    for i in range(3):
        # binarize labels for ovr
        bin_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(bin_labels, all_probabilities[:, i])
        roc_auc_class = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {LABEL_MAP[i]} (AUC = {roc_auc_class:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(eval_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"ROC curve saved to {roc_path}")
    
    # Generate Precision-Recall Curve Plot
    plt.figure(figsize=(8, 6))
    for i in range(3):
        bin_labels = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(bin_labels, all_probabilities[:, i])
        ap = average_precision_score(bin_labels, all_probabilities[:, i])
        plt.plot(recall, precision, label=f"Class {LABEL_MAP[i]} (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(eval_dir, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    logger.info(f"PR curve saved to {pr_path}")
    
    # Add predictions back to df for misclassification analysis
    test_df["predicted_label"] = all_predictions
    test_df["predicted_name"] = test_df["predicted_label"].map(LABEL_MAP)
    test_df["true_name"] = test_df["label"].map(LABEL_MAP)
    for i in range(3):
        test_df[f"prob_{LABEL_MAP[i]}"] = all_probabilities[:, i]
    test_df["confidence"] = all_probabilities.max(axis=-1)
    
    # Isolate misclassifications
    misclassified_df = test_df[test_df["label"] != test_df["predicted_label"]].copy()
    
    # Sort misclassifications by confidence (highest confidence errors first)
    misclassified_df = misclassified_df.sort_values(by="confidence", ascending=False)
    
    misclass_file = os.path.join(eval_dir, "misclassified_analysis.csv")
    misclassified_df.to_csv(misclass_file, index=False)
    logger.info(f"Misclassified samples analysis saved to {misclass_file} (Found {len(misclassified_df)} errors out of {len(test_df)} samples)")
    
    # Print summary metrics
    print("\n" + "="*40)
    print("=== MODEL EVALUATION METRICS ===")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("="*40)
    print(class_report_str)
    print("="*40)
    
    return metrics, test_df

if __name__ == "__main__":
    config = load_config()
    evaluate_model(config)
