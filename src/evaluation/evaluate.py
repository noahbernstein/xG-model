"""
Model evaluation module for xG model.

Computes metrics that matter for probability estimation:
- Log loss (primary metric)
- Brier score (calibration)
- AUC-ROC (discrimination)
- Calibration curves

Key insight: accuracy is the wrong metric for xG. A model predicting
"no goal" every time gets 90% accuracy. Calibration is what matters.
"""

import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from src.features.build_features import FEATURE_COLUMNS, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_NAMES = ["distance_only", "logistic_regression", "random_forest", "xgboost"]
MODEL_DISPLAY = {
    "distance_only": "Distance Only",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "distance_only": "#95a5a6",
    "logistic_regression": "#3498db",
    "random_forest": "#2ecc71",
    "xgboost": "#e74c3c",
}


def load_models() -> dict:
    """Load all trained models."""
    models = {}
    for name in MODEL_NAMES:
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


def load_test_data() -> pd.DataFrame:
    """Load test set."""
    return pd.read_parquet(PROCESSED_DIR / "test.parquet")


def predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Get predicted probabilities from a model."""
    X = df[features].values
    return model.predict_proba(X)[:, 1]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    return {
        "log_loss": float(log_loss(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_pred)),
    }


def evaluate_all(models: dict, test_df: pd.DataFrame) -> dict:
    """Evaluate all models on the test set."""
    y_true = test_df[TARGET].values
    results = {}

    # Load training summary for CV scores
    with open(MODEL_DIR / "training_summary.json") as f:
        train_summary = json.load(f)

    for name in MODEL_NAMES:
        features = FEATURE_COLUMNS if name != "distance_only" else ["distance_to_goal"]
        y_pred = predict(models[name], test_df, features)

        metrics = compute_metrics(y_true, y_pred)
        metrics["cv_log_loss"] = train_summary[name]["cv_log_loss"]

        results[name] = {
            "metrics": metrics,
            "predictions": y_pred,
        }

        logger.info(
            f"{MODEL_DISPLAY[name]:25s} | "
            f"Log loss: {metrics['log_loss']:.4f} | "
            f"Brier: {metrics['brier_score']:.4f} | "
            f"AUC: {metrics['auc_roc']:.4f}"
        )

    return results


def plot_calibration_curves(results: dict, y_true: np.ndarray, save_path: Path | None = None):
    """Plot calibration curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)

    for name in MODEL_NAMES:
        y_pred = results[name]["predictions"]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", label=MODEL_DISPLAY[name], color=MODEL_COLORS[name])

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)

    # ROC curves
    ax = axes[1]
    for name in MODEL_NAMES:
        y_pred = results[name]["predictions"]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = results[name]["metrics"]["auc_roc"]
        ax.plot(fpr, tpr, label=f"{MODEL_DISPLAY[name]} (AUC={auc:.3f})", color=MODEL_COLORS[name])

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved calibration plot to {save_path}")
    plt.show()


def plot_metrics_comparison(results: dict, save_path: Path | None = None):
    """Bar chart comparing all models across metrics."""
    metrics_df = pd.DataFrame({
        MODEL_DISPLAY[name]: results[name]["metrics"]
        for name in MODEL_NAMES
    }).T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title, lower_better in zip(
        axes,
        ["log_loss", "brier_score", "auc_roc"],
        ["Log Loss (lower = better)", "Brier Score (lower = better)", "AUC-ROC (higher = better)"],
        [True, True, False],
    ):
        colors = [MODEL_COLORS[name] for name in MODEL_NAMES]
        bars = ax.bar(metrics_df.index, metrics_df[metric], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

        # Add value labels
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_results(results: dict) -> None:
    """Save evaluation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    serialisable = {
        name: {
            "metrics": data["metrics"],
        }
        for name, data in results.items()
    }
    output_path = RESULTS_DIR / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    models = load_models()
    test_df = load_test_data()
    y_true = test_df[TARGET].values

    results = evaluate_all(models, test_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration_curves(results, y_true, save_path=RESULTS_DIR / "calibration_curves.png")
    plot_metrics_comparison(results, save_path=RESULTS_DIR / "metrics_comparison.png")
    save_results(results)
