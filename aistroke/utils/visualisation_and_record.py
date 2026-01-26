import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.close()

def plot_pr_curve(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.savefig(save_path)
    plt.close()

def save_error_records(FP_records, FN_records, save_path):
    """
    将 FP/FN 详细记录保存为 JSON 文件
    """
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "FP": FP_records,
        "FN": FN_records
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)