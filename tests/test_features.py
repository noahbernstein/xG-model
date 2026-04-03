"""Unit tests for feature engineering calculations."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    GOAL_X,
    GOAL_CENTRE_Y,
    GOAL_POST_LEFT_Y,
    GOAL_POST_RIGHT_Y,
    GOAL_WIDTH,
    add_location_features,
    add_shot_features,
    add_situation_features,
    add_game_state_features,
    add_target,
    FEATURE_COLUMNS,
)


def _make_shot(x=100.0, y=40.0, **kwargs):
    """Helper to create a single-row shot DataFrame."""
    data = {
        "location": [np.array([x, y])],
        "shot_body_part": "Right Foot",
        "shot_technique": "Normal",
        "shot_first_time": None,
        "under_pressure": None,
        "shot_one_on_one": None,
        "shot_open_goal": None,
        "shot_deflected": None,
        "shot_follows_dribble": None,
        "shot_redirect": None,
        "shot_type": "Open Play",
        "play_pattern": "Regular Play",
        "shot_outcome": "Off T",
        "minute": 45,
        "period": 1,
    }
    data.update(kwargs)
    return pd.DataFrame(data)


class TestLocationFeatures:
    def test_distance_from_centre(self):
        """Shot from the centre of the pitch, 20 yards out."""
        df = _make_shot(x=100.0, y=40.0)
        df = add_location_features(df)
        assert df["distance_to_goal"].iloc[0] == pytest.approx(20.0, abs=0.01)

    def test_distance_from_penalty_spot(self):
        """Penalty spot is 12 yards from goal (x=108)."""
        df = _make_shot(x=108.0, y=40.0)
        df = add_location_features(df)
        assert df["distance_to_goal"].iloc[0] == pytest.approx(12.0, abs=0.01)

    def test_angle_from_centre(self):
        """Shot dead centre should have the widest angle for that distance."""
        df_centre = _make_shot(x=100.0, y=40.0)
        df_centre = add_location_features(df_centre)

        df_wide = _make_shot(x=100.0, y=20.0)
        df_wide = add_location_features(df_wide)

        assert df_centre["angle_to_goal"].iloc[0] > df_wide["angle_to_goal"].iloc[0]

    def test_angle_symmetry(self):
        """Shots equidistant from centre on opposite sides should have equal angles."""
        df_left = _make_shot(x=100.0, y=30.0)
        df_left = add_location_features(df_left)

        df_right = _make_shot(x=100.0, y=50.0)
        df_right = add_location_features(df_right)

        assert df_left["angle_to_goal"].iloc[0] == pytest.approx(
            df_right["angle_to_goal"].iloc[0], abs=0.001
        )

    def test_off_centre_zero_at_middle(self):
        """Shot from y=40 (centre) should have off_centre=0."""
        df = _make_shot(x=100.0, y=40.0)
        df = add_location_features(df)
        assert df["off_centre"].iloc[0] == pytest.approx(0.0, abs=0.001)

    def test_off_centre_increases_to_side(self):
        df_centre = _make_shot(x=100.0, y=40.0)
        df_centre = add_location_features(df_centre)

        df_side = _make_shot(x=100.0, y=10.0)
        df_side = add_location_features(df_side)

        assert df_side["off_centre"].iloc[0] > df_centre["off_centre"].iloc[0]

    def test_closer_shot_has_wider_angle(self):
        """Closer to goal = wider visible angle."""
        df_close = _make_shot(x=115.0, y=40.0)
        df_close = add_location_features(df_close)

        df_far = _make_shot(x=80.0, y=40.0)
        df_far = add_location_features(df_far)

        assert df_close["angle_to_goal"].iloc[0] > df_far["angle_to_goal"].iloc[0]


class TestShotFeatures:
    def test_header_flag(self):
        df = _make_shot(shot_body_part="Head")
        df = add_location_features(df)
        df = add_shot_features(df)
        assert df["is_header"].iloc[0] == 1
        assert df["is_right_foot"].iloc[0] == 0

    def test_volley_includes_half_volley(self):
        df = _make_shot(shot_technique="Half Volley")
        df = add_location_features(df)
        df = add_shot_features(df)
        assert df["is_volley"].iloc[0] == 1
        assert df["is_half_volley"].iloc[0] == 1

    def test_first_time_false_when_missing(self):
        df = _make_shot(shot_first_time=None)
        df = add_location_features(df)
        df = add_shot_features(df)
        assert df["is_first_time"].iloc[0] == 0

    def test_first_time_true(self):
        df = _make_shot(shot_first_time=True)
        df = add_location_features(df)
        df = add_shot_features(df)
        assert df["is_first_time"].iloc[0] == 1


class TestSituationFeatures:
    def test_penalty(self):
        df = _make_shot(shot_type="Penalty")
        df = add_situation_features(df)
        assert df["is_penalty"].iloc[0] == 1
        assert df["is_free_kick"].iloc[0] == 0

    def test_counter_attack(self):
        df = _make_shot(play_pattern="From Counter")
        df = add_situation_features(df)
        assert df["is_counter_attack"].iloc[0] == 1
        assert df["is_open_play"].iloc[0] == 0

    def test_open_play(self):
        df = _make_shot(play_pattern="Regular Play", shot_type="Open Play")
        df = add_situation_features(df)
        assert df["is_open_play"].iloc[0] == 1
        assert df["is_penalty"].iloc[0] == 0


class TestGameStateFeatures:
    def test_second_half(self):
        df = _make_shot(period=2, minute=60)
        df = add_game_state_features(df)
        assert df["is_second_half"].iloc[0] == 1

    def test_last_15(self):
        df = _make_shot(minute=80)
        df = add_game_state_features(df)
        assert df["is_last_15"].iloc[0] == 1

    def test_not_last_15(self):
        df = _make_shot(minute=74)
        df = add_game_state_features(df)
        assert df["is_last_15"].iloc[0] == 0


class TestTarget:
    def test_goal(self):
        df = _make_shot(shot_outcome="Goal")
        df = add_target(df)
        assert df["is_goal"].iloc[0] == 1

    def test_not_goal(self):
        df = _make_shot(shot_outcome="Saved")
        df = add_target(df)
        assert df["is_goal"].iloc[0] == 0
