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

import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# ===== Your upper-body joint mask =====
UPPER_BODY_MASK = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
IDX_MAP = {j: i for i, j in enumerate(UPPER_BODY_MASK)}

# ===== H36M upper-body edges =====
H36M_UPPER_EDGES_RAW = [
    (0, 7),
    (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

H36M_UPPER_EDGES = [
    (IDX_MAP[i], IDX_MAP[j])
    for i, j in H36M_UPPER_EDGES_RAW
    if i in IDX_MAP and j in IDX_MAP
]

def save_upper_body_skeleton_3d_cam(
    joints_3d,
    cam,
    save_path,
    edges=H36M_UPPER_EDGES,
    title=None,
    elev=15,
    azim=70,
    dpi=150,
):
    """
    Args:
        joints_3d: (V, 3)
        cam: (V,)
        save_path: str, e.g. xxx/vis_cam_left.png
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # ---- bones ----
    for i, j in edges:
        ax.plot(
            [joints_3d[i, 0], joints_3d[j, 0]],
            [joints_3d[i, 1], joints_3d[j, 1]],
            [joints_3d[i, 2], joints_3d[j, 2]],
            color="gray",
            linewidth=2,
            alpha=0.8,
        )

    # ---- joints ----
    sc = ax.scatter(
        joints_3d[:, 0],
        joints_3d[:, 1],
        joints_3d[:, 2],
        c=cam,
        cmap="jet",
        s=70,
        vmin=0.0,
        vmax=1.0,
    )

    plt.colorbar(sc, ax=ax, shrink=0.6)

    ax.set_title(title or "Upper-body Skeleton Grad-CAM")
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)