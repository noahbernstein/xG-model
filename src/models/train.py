"""
Model training pipeline for xG model.

Trains four models and saves them for evaluation:
1. Distance-only logistic regression (naive baseline)
2. Full logistic regression
3. Random forest
4. XGBoost

Train/test split by competition: trained on club football,
tested on international tournaments (World Cup 2022, Euro 2024,
Women's World Cup 2023) to demonstrate genuine generalisation.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from src.features.build_features import FEATURE_COLUMNS, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# Test set: international tournaments for generalisation testing
TEST_COMPETITIONS = {
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2024"),
    ("Women's World Cup", "2023"),
}


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature matrix and split into train/test by competition."""
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    test_mask = df.apply(
        lambda row: (row["competition"], row["season"]) in TEST_COMPETITIONS, axis=1
    )
    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    logger.info(f"Train: {len(train_df):,} shots ({train_df[TARGET].mean():.1%} goal rate)")
    logger.info(f"Test:  {len(test_df):,} shots ({test_df[TARGET].mean():.1%} goal rate)")
    logger.info(f"Test competitions: {test_df['competition'].unique().tolist()}")

    return train_df, test_df


def get_Xy(df: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target from DataFrame."""
    if features is None:
        features = FEATURE_COLUMNS
    X = df[features].values
    y = df[TARGET].values
    return X, y


def build_preprocessing():
    """Standard preprocessing: impute NaN then scale."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def train_models(train_df: pd.DataFrame) -> dict:
    """Train all four models and return them with CV scores."""
    X_train, y_train = get_Xy(train_df)
    X_train_dist, _ = get_Xy(train_df, features=["distance_to_goal"])

    models = {}

    # 1. Distance-only baseline
    logger.info("Training distance-only baseline...")
    dist_pipe = Pipeline([
        ("preprocess", build_preprocessing()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    dist_cv = cross_val_score(dist_pipe, X_train_dist, y_train, cv=5, scoring="neg_log_loss")
    dist_pipe.fit(X_train_dist, y_train)
    models["distance_only"] = {
        "pipeline": dist_pipe,
        "features": ["distance_to_goal"],
        "cv_log_loss": -dist_cv.mean(),
        "cv_std": dist_cv.std(),
    }
    logger.info(f"  CV log loss: {-dist_cv.mean():.4f} (±{dist_cv.std():.4f})")

    # 2. Logistic regression (all features)
    logger.info("Training logistic regression...")
    lr_pipe = Pipeline([
        ("preprocess", build_preprocessing()),
        ("model", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    lr_cv = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring="neg_log_loss")
    lr_pipe.fit(X_train, y_train)
    models["logistic_regression"] = {
        "pipeline": lr_pipe,
        "features": FEATURE_COLUMNS,
        "cv_log_loss": -lr_cv.mean(),
        "cv_std": lr_cv.std(),
    }
    logger.info(f"  CV log loss: {-lr_cv.mean():.4f} (±{lr_cv.std():.4f})")

    # 3. Random forest
    logger.info("Training random forest...")
    rf_pipe = Pipeline([
        ("preprocess", build_preprocessing()),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_cv = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring="neg_log_loss")
    rf_pipe.fit(X_train, y_train)
    models["random_forest"] = {
        "pipeline": rf_pipe,
        "features": FEATURE_COLUMNS,
        "cv_log_loss": -rf_cv.mean(),
        "cv_std": rf_cv.std(),
    }
    logger.info(f"  CV log loss: {-rf_cv.mean():.4f} (±{rf_cv.std():.4f})")

    # 4. XGBoost
    logger.info("Training XGBoost...")
    xgb_pipe = Pipeline([
        ("preprocess", build_preprocessing()),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ])
    xgb_cv = cross_val_score(xgb_pipe, X_train, y_train, cv=5, scoring="neg_log_loss")
    xgb_pipe.fit(X_train, y_train)
    models["xgboost"] = {
        "pipeline": xgb_pipe,
        "features": FEATURE_COLUMNS,
        "cv_log_loss": -xgb_cv.mean(),
        "cv_std": xgb_cv.std(),
    }
    logger.info(f"  CV log loss: {-xgb_cv.mean():.4f} (±{xgb_cv.std():.4f})")

    return models


def save_models(models: dict) -> None:
    """Save trained models and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model_data in models.items():
        model_path = MODEL_DIR / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data["pipeline"], f)
        logger.info(f"Saved {name} to {model_path}")

    # Save summary
    summary = {
        name: {
            "cv_log_loss": data["cv_log_loss"],
            "cv_std": data["cv_std"],
            "features": data["features"],
        }
        for name, data in models.items()
    }
    summary_path = MODEL_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to {summary_path}")


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train/test splits for reproducibility."""
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    logger.info("Saved train/test splits")


if __name__ == "__main__":
    train_df, test_df = load_and_split()
    save_splits(train_df, test_df)
    models = train_models(train_df)
    save_models(models)
