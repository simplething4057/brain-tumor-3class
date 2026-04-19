"""
Brain Tumor Classification - Result Visualization
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from loguru import logger

FEAT_PATH = _ROOT / "outputs" / "features" / "features.csv"
OUT_DIR   = _ROOT / "outputs" / "plots"

CLASSES   = ["GLI", "MEN", "MET"]
COLORS    = {"GLI": "#E74C3C", "MEN": "#3498DB", "MET": "#2ECC71"}
LABEL_MAP = {"GLI": 0, "MEN": 1, "MET": 2}
INV_MAP   = {0: "GLI", 1: "MEN", 2: "MET"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def prepare():
    df = pd.read_csv(FEAT_PATH)
    df = df.dropna(subset=["true_label"])
    df = df[df["true_label"].isin(CLASSES)].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ["subject_id", "true_label"]]
    X = df[feat_cols].fillna(0)
    y = df["true_label"].tolist()
    logger.info(f"samples={len(y)}  GLI={y.count('GLI')} MEN={y.count('MEN')} MET={y.count('MET')}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr.values, [LABEL_MAP[l] for l in y_tr])
    logger.info(f"RF fitted  train={len(y_tr)}  test={len(y_te)}")

    y_pred  = [INV_MAP[p] for p in rf.predict(X_te.values)]
    y_proba = pd.DataFrame(rf.predict_proba(X_te.values), columns=CLASSES)
    imp_df  = (
        pd.DataFrame({"feature": feat_cols, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )
    return y_te, y_pred, y_proba, imp_df


def feat_color(name):
    if name.startswith("gli_"): return COLORS["GLI"]
    if name.startswith("men_"): return COLORS["MEN"]
    if name.startswith("met_"): return COLORS["MET"]
    return "#95A5A6"


def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p)
    plt.close(fig)
    logger.info(f"saved: {p.name}")


# 1. Confusion Matrix
def plot_cm(y_te, y_pred):
    cm      = confusion_matrix(y_te, y_pred, labels=CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmap    = LinearSegmentedColormap.from_list("c", ["#FDFEFE", "#2471A3"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold", y=1.02)

    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm], ["Count", "Normalized (Recall)"], ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    linewidths=0.5, linecolor="#BDC3C7",
                    annot_kws={"size": 13, "weight": "bold"},
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    plt.tight_layout()
    save(fig, "01_confusion_matrix.png")


# 2. Per-Class Metrics
def plot_metrics(y_te, y_pred):
    rep = classification_report(y_te, y_pred, labels=CLASSES, output_dict=True)
    x   = np.arange(len(CLASSES))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Per-Class Performance Metrics", fontsize=15, fontweight="bold")

    for ax, m, lbl in zip(axes,
                           ["precision", "recall", "f1-score"],
                           ["Precision", "Recall", "F1-Score"]):
        vals = [rep[c][m] for c in CLASSES]
        bars = ax.bar(x, vals, width=0.55,
                      color=[COLORS[c] for c in CLASSES],
                      edgecolor="white", linewidth=1.2, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=11)
        ax.set_title(lbl, fontsize=12, pad=8)
        ax.set_ylabel("Score", fontsize=10)
        ax.axhline(y=0.8, color="#AAB7B8", linestyle="--", linewidth=0.8, zorder=2)
        ax.grid(axis="y", alpha=0.3, zorder=1)

    fig.text(0.5, -0.04,
             f"Accuracy: {rep['accuracy']:.3f}   Macro F1: {rep['macro avg']['f1-score']:.3f}   "
             f"Test: GLI {rep['GLI']['support']:.0f} / "
             f"MEN {rep['MEN']['support']:.0f} / MET {rep['MET']['support']:.0f}",
             ha="center", fontsize=11, color="#566573")
    plt.tight_layout()
    save(fig, "02_class_metrics.png")


# 3. Feature Importance Top15
def plot_feat_top15(imp_df):
    top    = imp_df.head(15).iloc[::-1].copy()
    colors = [feat_color(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["feature"], top["importance"],
                   color=colors, edgecolor="white", linewidth=0.8, height=0.7)
    for bar, val in zip(bars, top["importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#2C3E50")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importance (Top 15)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    leg = [Patch(facecolor=COLORS[c], label=f"{c} features") for c in CLASSES]
    ax.legend(handles=leg, loc="lower right", fontsize=10)
    plt.tight_layout()
    save(fig, "03_feature_importance.png")


# 4. Feature Importance Grouped
def plot_feat_grouped(imp_df):
    df = imp_df.copy()

    def grp(name):
        if name.startswith("gli_"): return "GLI"
        if name.startswith("men_"): return "MEN"
        if name.startswith("met_"): return "MET"
        return "Other"

    def ftype(name):
        for sfx in ["total_voxels", "total_volume_mm3", "et_ratio",
                    "edema_ratio", "core_ratio", "lesion_count", "has_tumor"]:
            if name.endswith(sfx): return sfx
        return name

    df["group"]     = df["feature"].apply(grp)
    df["feat_type"] = df["feature"].apply(ftype)
    pivot = df.pivot_table(index="feat_type", columns="group",
                           values="importance", fill_value=0)
    pivot = pivot[CLASSES].sort_values("GLI", ascending=False)

    x = np.arange(len(pivot)); w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cls in enumerate(CLASSES):
        ax.bar(x + (i - 1) * w, pivot[cls], width=w,
               label=cls, color=COLORS[cls],
               edgecolor="white", linewidth=0.8, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_title("Feature Importance by Type & Model",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(title="Model", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "04_feature_grouped.png")


# 5. Confidence Distribution
def plot_confidence(y_te, y_proba):
    df = y_proba.copy(); df["true_label"] = y_te
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Predicted Confidence Distribution (per true class)",
                 fontsize=14, fontweight="bold")
    for ax, true_cls in zip(axes, CLASSES):
        sub = df[df["true_label"] == true_cls]
        for pred_cls, color in COLORS.items():
            ax.hist(sub[pred_cls].values, bins=20, range=(0, 1),
                    color=color, alpha=0.65, label=pred_cls,
                    edgecolor="white", linewidth=0.5)
        ax.axvline(x=0.5, color="#AAB7B8", linestyle="--", linewidth=1)
        ax.set_title(f"True: {true_cls}  (n={len(sub)})",
                     fontsize=11, fontweight="bold", color=COLORS[true_cls])
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_xlim(0, 1); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[-1].legend(title="Predicted as", fontsize=9, loc="upper left")
    plt.tight_layout()
    save(fig, "05_confidence_dist.png")


# 6. Precision-Recall Curve
def plot_pr_curve(y_te, y_proba):
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color in COLORS.items():
        y_bin  = [1 if y == cls else 0 for y in y_te]
        scores = y_proba[cls].values
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)
        ax.plot(rec, prec, color=color, linewidth=2.2, label=f"{cls}  (AP={ap:.3f})")
        ax.fill_between(rec, prec, alpha=0.08, color=color)
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve (One-vs-Rest)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save(fig, "06_pr_curve.png")


# 0. Dashboard
def plot_dashboard(y_te, y_pred, y_proba, imp_df):
    rep     = classification_report(y_te, y_pred, labels=CLASSES, output_dict=True)
    cm_norm = confusion_matrix(y_te, y_pred, labels=CLASSES).astype(float)
    cm_norm /= cm_norm.sum(axis=1, keepdims=True)
    top10   = imp_df.head(10).iloc[::-1].copy()
    cmap    = LinearSegmentedColormap.from_list("c", ["#FDFEFE", "#2471A3"])

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Brain Tumor Classification - Result Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # (0,0) confusion matrix
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, linecolor="#BDC3C7",
                annot_kws={"size": 12, "weight": "bold"},
                ax=ax0, cbar=False)
    ax0.set_title("Confusion Matrix\n(Normalized)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Predicted", fontsize=9); ax0.set_ylabel("True", fontsize=9)

    # (0,1) per-class metrics
    ax1 = fig.add_subplot(gs[0, 1])
    x   = np.arange(len(CLASSES)); w = 0.22
    for i, (m, ml) in enumerate(zip(
        ["precision", "recall", "f1-score"], ["Precision", "Recall", "F1"]
    )):
        ax1.bar(x + (i - 1) * w, [rep[c][m] for c in CLASSES],
                width=w, label=ml, edgecolor="white", linewidth=0.8, alpha=0.88)
    ax1.set_xticks(x); ax1.set_xticklabels(CLASSES, fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_title(f"Per-Class Metrics\nAcc={rep['accuracy']:.3f}  "
                  f"Macro-F1={rep['macro avg']['f1-score']:.3f}",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right"); ax1.grid(axis="y", alpha=0.3)
    for tick in ax1.get_xticklabels():
        tick.set_color(COLORS.get(tick.get_text(), "black")); tick.set_fontweight("bold")

    # (0,2) PR curve
    ax2 = fig.add_subplot(gs[0, 2])
    for cls, color in COLORS.items():
        y_bin  = [1 if y == cls else 0 for y in y_te]
        scores = y_proba[cls].values
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)
        ax2.plot(rec, prec, color=color, linewidth=2, label=f"{cls} AP={ap:.2f}")
        ax2.fill_between(rec, prec, alpha=0.07, color=color)
    ax2.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=9); ax2.set_ylabel("Precision", fontsize=9)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1.01); ax2.set_ylim(0, 1.05)

    # (1,0-1) feature importance top10
    ax3 = fig.add_subplot(gs[1, :2])
    colors3 = [feat_color(f) for f in top10["feature"]]
    bars3 = ax3.barh(top10["feature"], top10["importance"],
                     color=colors3, edgecolor="white", linewidth=0.8, height=0.65)
    for bar, val in zip(bars3, top10["importance"]):
        ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)
    ax3.set_title("Feature Importance (Top 10)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Importance", fontsize=9); ax3.grid(axis="x", alpha=0.3)
    leg = [Patch(facecolor=COLORS[c], label=f"{c} features") for c in CLASSES]
    ax3.legend(handles=leg, fontsize=9, loc="lower right")

    # (1,2) confidence distribution
    ax4 = fig.add_subplot(gs[1, 2])
    df_p = y_proba.copy(); df_p["true_label"] = y_te
    for cls, color in COLORS.items():
        sub = df_p[df_p["true_label"] == cls]
        ax4.hist(sub[cls].values, bins=15, range=(0, 1),
                 color=color, alpha=0.65, label=cls, edgecolor="white", linewidth=0.5)
    ax4.axvline(x=0.5, color="#AAB7B8", linestyle="--", linewidth=1)
    ax4.set_title("Confidence (True Class Prob.)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("P(correct class)", fontsize=9); ax4.set_ylabel("Count", fontsize=9)
    ax4.legend(fontsize=9); ax4.grid(alpha=0.25)

    save(fig, "00_dashboard.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"output: {OUT_DIR}")

    y_te, y_pred, y_proba, imp_df = prepare()

    plot_dashboard(y_te, y_pred, y_proba, imp_df)
    plot_cm(y_te, y_pred)
    plot_metrics(y_te, y_pred)
    plot_feat_top15(imp_df)
    plot_feat_grouped(imp_df)
    plot_confidence(y_te, y_proba)
    plot_pr_curve(y_te, y_proba)

    files = sorted(OUT_DIR.glob("*.png"))
    logger.info(f"=== done: {len(files)} files ===")
    for f in files:
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
