"""
Collect all shot events from StatsBomb open data.

Pulls events from all available competitions/seasons, filters to shots,
and saves the raw shot-level dataset as parquet.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from statsbombpy import sb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def get_all_competitions() -> pd.DataFrame:
    """Fetch all available competitions from StatsBomb open data."""
    comps = sb.competitions()
    logger.info(f"Found {len(comps)} competition-seasons")
    return comps


def get_shots_for_match(match_id: int) -> pd.DataFrame:
    """Extract shot events from a single match."""
    events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"].copy()
    shots["match_id"] = match_id
    return shots


def collect_all_shots() -> pd.DataFrame:
    """Pull all shot events across all available competitions."""
    comps = get_all_competitions()
    all_shots = []

    for _, comp in comps.iterrows():
        comp_id = comp["competition_id"]
        season_id = comp["season_id"]
        comp_name = comp["competition_name"]
        season_name = comp["season_name"]

        logger.info(f"Processing {comp_name} {season_name} (comp={comp_id}, season={season_id})")

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as e:
            logger.warning(f"Failed to get matches for {comp_name} {season_name}: {e}")
            continue

        match_count = 0
        for _, match in matches.iterrows():
            match_id = match["match_id"]
            try:
                shots = get_shots_for_match(match_id)
                if len(shots) > 0:
                    # Add competition context
                    shots["competition"] = comp_name
                    shots["season"] = season_name
                    shots["competition_id"] = comp_id
                    shots["season_id"] = season_id
                    all_shots.append(shots)
                match_count += 1
            except Exception as e:
                logger.warning(f"Failed to get events for match {match_id}: {e}")
                continue

        logger.info(f"  Processed {match_count} matches")

    if not all_shots:
        raise RuntimeError("No shots collected — check StatsBomb API access")

    df = pd.concat(all_shots, ignore_index=True)
    logger.info(f"Total shots collected: {len(df)}")
    logger.info(f"Total goals: {df['shot_outcome'].eq('Goal').sum()}")
    return df


def save_raw_shots(df: pd.DataFrame) -> Path:
    """Save raw shots dataset to parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / "shots_raw.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} shots to {output_path}")

    # Also save a summary
    summary = {
        "total_shots": len(df),
        "total_goals": int(df["shot_outcome"].eq("Goal").sum()),
        "goal_rate": float(df["shot_outcome"].eq("Goal").mean()),
        "competitions": df["competition"].nunique(),
        "columns": list(df.columns),
    }
    summary_path = RAW_DIR / "shots_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return output_path


if __name__ == "__main__":
    df = collect_all_shots()
    save_raw_shots(df)
