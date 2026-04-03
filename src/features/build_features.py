"""
Feature engineering pipeline for xG model.

Transforms raw StatsBomb shot data into a model-ready feature matrix.
Deliberately excludes player/team identity — the model learns what makes
a good chance, not who takes it.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# StatsBomb pitch dimensions: 120 x 80 yards
# Goal sits at x=120, centred at y=40
GOAL_X = 120.0
GOAL_CENTRE_Y = 40.0
GOAL_WIDTH = 8.0  # yards (7.32m ≈ 8 yards)
GOAL_POST_LEFT_Y = GOAL_CENTRE_Y - GOAL_WIDTH / 2  # 36
GOAL_POST_RIGHT_Y = GOAL_CENTRE_Y + GOAL_WIDTH / 2  # 44


# ---------------------------------------------------------------------------
# Location features
# ---------------------------------------------------------------------------

def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract x, y coordinates and compute distance and angle to goal."""
    df["x"] = df["location"].apply(lambda loc: float(loc[0]) if hasattr(loc, "__len__") else np.nan)
    df["y"] = df["location"].apply(lambda loc: float(loc[1]) if hasattr(loc, "__len__") else np.nan)

    # Distance to goal centre
    df["distance_to_goal"] = np.sqrt((GOAL_X - df["x"]) ** 2 + (GOAL_CENTRE_Y - df["y"]) ** 2)

    # Visible angle of the goal from the shot position (radians)
    angle_left = np.arctan2(GOAL_POST_LEFT_Y - df["y"], GOAL_X - df["x"])
    angle_right = np.arctan2(GOAL_POST_RIGHT_Y - df["y"], GOAL_X - df["x"])
    df["angle_to_goal"] = np.abs(angle_right - angle_left)

    # Distance to nearest post (useful for tight-angle shots)
    dist_left = np.sqrt((GOAL_X - df["x"]) ** 2 + (GOAL_POST_LEFT_Y - df["y"]) ** 2)
    dist_right = np.sqrt((GOAL_X - df["x"]) ** 2 + (GOAL_POST_RIGHT_Y - df["y"]) ** 2)
    df["distance_to_nearest_post"] = np.minimum(dist_left, dist_right)

    # How central the shot is (0 = in line with centre of goal, 1 = at the post line)
    df["off_centre"] = np.abs(df["y"] - GOAL_CENTRE_Y) / GOAL_CENTRE_Y

    return df


# ---------------------------------------------------------------------------
# Shot-level features
# ---------------------------------------------------------------------------

def add_shot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode shot-level categorical and boolean features."""
    # Body part — one-hot
    df["is_header"] = (df["shot_body_part"] == "Head").astype(int)
    df["is_right_foot"] = (df["shot_body_part"] == "Right Foot").astype(int)
    df["is_left_foot"] = (df["shot_body_part"] == "Left Foot").astype(int)

    # Technique
    df["is_volley"] = df["shot_technique"].isin(["Volley", "Half Volley"]).astype(int)
    df["is_half_volley"] = (df["shot_technique"] == "Half Volley").astype(int)
    df["is_lob"] = (df["shot_technique"] == "Lob").astype(int)
    df["is_overhead_kick"] = (df["shot_technique"] == "Overhead Kick").astype(int)
    df["is_backheel"] = (df["shot_technique"] == "Backheel").astype(int)
    df["is_diving_header"] = (df["shot_technique"] == "Diving Header").astype(int)

    # Boolean flags (NaN = False for these)
    df["is_first_time"] = df["shot_first_time"].fillna(False).astype(int)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["is_one_on_one"] = df["shot_one_on_one"].fillna(False).astype(int)
    df["is_open_goal"] = df["shot_open_goal"].fillna(False).astype(int)
    df["is_deflected"] = df["shot_deflected"].fillna(False).astype(int)
    df["follows_dribble"] = df["shot_follows_dribble"].fillna(False).astype(int)
    df["is_redirect"] = df["shot_redirect"].fillna(False).astype(int)

    return df


# ---------------------------------------------------------------------------
# Situation / play pattern features
# ---------------------------------------------------------------------------

def add_situation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode play pattern and shot type context."""
    # Shot type
    df["is_penalty"] = (df["shot_type"] == "Penalty").astype(int)
    df["is_free_kick"] = (df["shot_type"] == "Free Kick").astype(int)

    # Play pattern — one-hot the common ones
    df["is_open_play"] = (df["play_pattern"] == "Regular Play").astype(int)
    df["is_from_corner"] = (df["play_pattern"] == "From Corner").astype(int)
    df["is_counter_attack"] = (df["play_pattern"] == "From Counter").astype(int)
    df["is_from_free_kick"] = (df["play_pattern"] == "From Free Kick").astype(int)
    df["is_from_throw_in"] = (df["play_pattern"] == "From Throw In").astype(int)
    df["is_from_goal_kick"] = (df["play_pattern"] == "From Goal Kick").astype(int)
    df["is_from_keeper"] = (df["play_pattern"] == "From Keeper").astype(int)

    return df


# ---------------------------------------------------------------------------
# Game state features
# ---------------------------------------------------------------------------

def add_game_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add minute and period features."""
    df["minute"] = df["minute"].astype(float)
    df["is_second_half"] = (df["period"] == 2).astype(int)
    # Late-game pressure indicator
    df["is_last_15"] = (df["minute"] >= 75).astype(int)

    return df


# ---------------------------------------------------------------------------
# Freeze frame features (defender/goalkeeper positioning)
# ---------------------------------------------------------------------------

def _count_defenders_in_cone(row: pd.Series) -> dict:
    """
    From the freeze frame, count:
    - Opponents between the shot and the goal line
    - Opponents inside the shot-to-goal triangle (the "cone")
    - Whether the goalkeeper is visible and their angle coverage
    """
    result = {
        "n_defenders_in_cone": 0,
        "n_opponents_in_path": 0,
        "gk_distance": np.nan,
        "gk_angle_coverage": np.nan,
    }

    freeze = row.get("shot_freeze_frame")
    if freeze is None or not hasattr(freeze, "__len__") or len(freeze) == 0:
        return result

    shot_x = row.get("x", np.nan)
    shot_y = row.get("y", np.nan)
    if np.isnan(shot_x) or np.isnan(shot_y):
        return result

    # Angle from shot to each goal post
    angle_to_left_post = np.arctan2(GOAL_POST_LEFT_Y - shot_y, GOAL_X - shot_x)
    angle_to_right_post = np.arctan2(GOAL_POST_RIGHT_Y - shot_y, GOAL_X - shot_x)
    min_angle = min(angle_to_left_post, angle_to_right_post)
    max_angle = max(angle_to_left_post, angle_to_right_post)

    for player_data in freeze:
        is_teammate = player_data.get("teammate", True)
        if is_teammate:
            continue

        ploc = player_data.get("location")
        if ploc is None or not hasattr(ploc, "__len__"):
            continue

        px, py = float(ploc[0]), float(ploc[1])

        # Is this player between the shot and the goal line?
        if px > shot_x:
            result["n_opponents_in_path"] += 1

            # Is this player inside the cone (triangle from shot to goal)?
            angle_to_player = np.arctan2(py - shot_y, px - shot_x)
            if min_angle <= angle_to_player <= max_angle:
                result["n_defenders_in_cone"] += 1

        # Check if this is the goalkeeper (position id 1)
        pos = player_data.get("position", {})
        if pos.get("id") == 1:
            gk_dist = np.sqrt((px - shot_x) ** 2 + (py - shot_y) ** 2)
            result["gk_distance"] = gk_dist

            # How much of the goal does the GK "cover"?
            # Simplified: angle from shot to GK relative to goal angle
            if result.get("gk_angle_coverage") is np.nan:
                goal_angle = max_angle - min_angle
                if goal_angle > 0:
                    gk_angle = np.arctan2(py - shot_y, px - shot_x)
                    # How centred is the GK in the cone? 0 = perfectly centred
                    cone_centre = (min_angle + max_angle) / 2
                    result["gk_angle_coverage"] = 1.0 - abs(gk_angle - cone_centre) / (goal_angle / 2 + 1e-8)

    return result


def add_freeze_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract defender and goalkeeper features from freeze frame data."""
    logger.info("Extracting freeze frame features (this may take a moment)...")

    ff_features = df.apply(_count_defenders_in_cone, axis=1, result_type="expand")
    for col in ff_features.columns:
        df[col] = ff_features[col]

    # Cap GK coverage at [0, 1]
    df["gk_angle_coverage"] = df["gk_angle_coverage"].clip(0, 1)

    logger.info(f"Freeze frame features extracted. GK distance available for {df['gk_distance'].notna().sum():,} shots")
    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target: 1 = goal, 0 = no goal."""
    df["is_goal"] = (df["shot_outcome"] == "Goal").astype(int)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Features used for modelling (in order)
FEATURE_COLUMNS = [
    # Location
    "distance_to_goal",
    "angle_to_goal",
    "distance_to_nearest_post",
    "off_centre",
    # Shot
    "is_header",
    "is_right_foot",
    "is_left_foot",
    "is_volley",
    "is_half_volley",
    "is_lob",
    "is_overhead_kick",
    "is_backheel",
    "is_diving_header",
    "is_first_time",
    "under_pressure",
    "is_one_on_one",
    "is_open_goal",
    "is_deflected",
    "follows_dribble",
    "is_redirect",
    # Situation
    "is_penalty",
    "is_free_kick",
    "is_open_play",
    "is_from_corner",
    "is_counter_attack",
    "is_from_free_kick",
    "is_from_throw_in",
    "is_from_goal_kick",
    "is_from_keeper",
    # Game state
    "minute",
    "is_second_half",
    "is_last_15",
    # Freeze frame
    "n_defenders_in_cone",
    "n_opponents_in_path",
    "gk_distance",
    "gk_angle_coverage",
]

TARGET = "is_goal"

# Columns to keep for analysis (not used in training)
METADATA_COLUMNS = [
    "competition",
    "season",
    "competition_id",
    "season_id",
    "match_id",
    "player",
    "team",
    "shot_statsbomb_xg",
    "shot_outcome",
    "shot_type",
    "play_pattern",
    "x",
    "y",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    logger.info(f"Starting feature engineering on {len(df):,} shots")

    df = add_location_features(df)
    df = add_shot_features(df)
    df = add_situation_features(df)
    df = add_game_state_features(df)
    df = add_freeze_frame_features(df)
    df = add_target(df)

    # Drop rows without valid location (can't compute core features)
    before = len(df)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows without valid location")

    logger.info(f"Feature matrix: {len(df):,} rows, {len(FEATURE_COLUMNS)} features")
    return df


def save_feature_matrix(df: pd.DataFrame) -> Path:
    """Save the processed feature matrix to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save full dataset (features + metadata + target)
    keep_cols = FEATURE_COLUMNS + [TARGET] + [c for c in METADATA_COLUMNS if c in df.columns]
    output = df[keep_cols].copy()

    output_path = PROCESSED_DIR / "features.parquet"
    output.to_parquet(output_path, index=False)
    logger.info(f"Saved feature matrix to {output_path}")

    # Print summary
    logger.info(f"  Rows: {len(output):,}")
    logger.info(f"  Features: {len(FEATURE_COLUMNS)}")
    logger.info(f"  Goal rate: {output[TARGET].mean():.1%}")
    logger.info(f"  Penalties: {output['is_penalty'].sum():,}")
    logger.info(f"  Missing GK distance: {output['gk_distance'].isna().sum():,}")

    return output_path


if __name__ == "__main__":
    raw_df = pd.read_parquet(RAW_DIR / "shots_raw.parquet")
    processed_df = build_features(raw_df)
    save_feature_matrix(processed_df)
