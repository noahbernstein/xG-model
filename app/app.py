"""
Interactive xG Match Explorer — Dash app.

Select a competition, season, and match to see:
- Cumulative xG timeline over 90 minutes
- Shot map on the pitch
- Match stats summary
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from src.features.build_features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/xgboost.pkl")


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df["xg"] = model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]

    # Build match labels
    match_info = (
        df.groupby("match_id")
        .agg(
            team_a=("team", "first"),
            team_b=("team", "last"),
            competition=("competition", "first"),
            season=("season", "first"),
            total_shots=("is_goal", "count"),
            total_goals=("is_goal", "sum"),
        )
        .reset_index()
    )
    # Deduplicate team names properly
    for _, row in match_info.iterrows():
        teams = df.loc[df["match_id"] == row["match_id"], "team"].unique()
        match_info.loc[match_info["match_id"] == row["match_id"], "team_a"] = teams[0]
        if len(teams) > 1:
            match_info.loc[match_info["match_id"] == row["match_id"], "team_b"] = teams[1]

    match_info["label"] = match_info.apply(
        lambda r: f"{r['team_a']} vs {r['team_b']} ({r['total_goals']}G, {r['total_shots']}S)", axis=1
    )
    df = df.merge(match_info[["match_id", "label"]], on="match_id", how="left")
    return df, match_info


df, match_info = load_data()

# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

PITCH_GREEN = "#2d572c"
LINE_WHITE = "rgba(255,255,255,0.6)"


def build_xg_timeline(match_df: pd.DataFrame) -> go.Figure:
    teams = match_df["team"].unique()
    colors = ["#3498db", "#e74c3c"]

    fig = go.Figure()

    for i, team in enumerate(teams):
        team_shots = match_df[match_df["team"] == team].sort_values("minute").copy()
        team_shots["cum_xg"] = team_shots["xg"].cumsum()

        minutes = [0] + team_shots["minute"].tolist()
        xg_vals = [0] + team_shots["cum_xg"].tolist()
        # Extend to 90+
        minutes.append(max(95, match_df["minute"].max() + 2))
        xg_vals.append(xg_vals[-1])

        fig.add_trace(go.Scatter(
            x=minutes, y=xg_vals,
            mode="lines",
            name=f"{team} (xG: {team_shots['xg'].sum():.2f})",
            line=dict(color=colors[i], width=3, shape="hv"),
        ))

        # Mark goals
        goals = team_shots[team_shots["is_goal"] == 1]
        if len(goals) > 0:
            fig.add_trace(go.Scatter(
                x=goals["minute"].tolist(),
                y=goals["cum_xg"].tolist(),
                mode="markers",
                name=f"{team} goals",
                marker=dict(color=colors[i], size=14, symbol="circle",
                            line=dict(color="white", width=2)),
                hovertext=[f"{row['player']} {row['minute']}'" for _, row in goals.iterrows()],
                hoverinfo="text",
            ))

    # Half time line
    fig.add_vline(x=45, line_dash="dot", line_color="rgba(255,255,255,0.3)")

    teams_list = list(teams)
    goals_a = match_df[match_df["team"] == teams_list[0]]["is_goal"].sum()
    goals_b = match_df[match_df["team"] == teams_list[1]]["is_goal"].sum() if len(teams_list) > 1 else 0

    fig.update_layout(
        title=f"{teams_list[0]} {goals_a} - {goals_b} {teams_list[1] if len(teams_list) > 1 else ''}",
        xaxis_title="Minute",
        yaxis_title="Cumulative xG",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        legend=dict(x=0.02, y=0.98),
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def build_shot_map(match_df: pd.DataFrame) -> go.Figure:
    """Shot map on a half-pitch using Plotly."""
    fig = go.Figure()

    # Draw pitch outline (StatsBomb coordinates: 120x80, attacking half x=60-120)
    # Goal
    fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=120.5,
                  fillcolor="white", line=dict(color="white", width=2))
    # 18-yard box
    fig.add_shape(type="rect", x0=18, y0=102, x1=62, y1=120,
                  line=dict(color=LINE_WHITE, width=1.5), fillcolor="rgba(0,0,0,0)")
    # 6-yard box
    fig.add_shape(type="rect", x0=30, y0=114, x1=50, y1=120,
                  line=dict(color=LINE_WHITE, width=1.5), fillcolor="rgba(0,0,0,0)")
    # Penalty spot
    fig.add_shape(type="circle", x0=39.5, y0=107.5, x1=40.5, y1=108.5,
                  fillcolor=LINE_WHITE, line=dict(color=LINE_WHITE))
    # Half-way line
    fig.add_shape(type="line", x0=0, y0=60, x1=80, y1=60,
                  line=dict(color=LINE_WHITE, width=1))
    # Pitch border
    fig.add_shape(type="rect", x0=0, y0=60, x1=80, y1=120,
                  line=dict(color=LINE_WHITE, width=2), fillcolor="rgba(0,0,0,0)")

    teams = match_df["team"].unique()
    colors = ["#3498db", "#e74c3c"]

    for i, team in enumerate(teams):
        team_shots = match_df[match_df["team"] == team]
        goals = team_shots[team_shots["is_goal"] == 1]
        misses = team_shots[team_shots["is_goal"] == 0]

        # Misses
        if len(misses) > 0:
            fig.add_trace(go.Scatter(
                x=misses["y"], y=misses["x"],
                mode="markers",
                name=f"{team} (miss)",
                marker=dict(
                    color=colors[i], size=misses["xg"] * 60 + 5,
                    opacity=0.5, symbol="circle",
                    line=dict(color="white", width=0.5),
                ),
                hovertext=[f"{row['player']}<br>{row['minute']}' xG:{row['xg']:.2f}" for _, row in misses.iterrows()],
                hoverinfo="text",
            ))

        # Goals
        if len(goals) > 0:
            fig.add_trace(go.Scatter(
                x=goals["y"], y=goals["x"],
                mode="markers",
                name=f"{team} (goal)",
                marker=dict(
                    color=colors[i], size=goals["xg"] * 60 + 10,
                    opacity=0.9, symbol="star",
                    line=dict(color="white", width=1.5),
                ),
                hovertext=[f"GOAL! {row['player']}<br>{row['minute']}' xG:{row['xg']:.2f}" for _, row in goals.iterrows()],
                hoverinfo="text",
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor=PITCH_GREEN,
        xaxis=dict(range=[-2, 82], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[58, 122], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        showlegend=True,
        legend=dict(x=0.02, y=0.02),
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def build_stats_table(match_df: pd.DataFrame) -> list:
    """Build match stats as a list of dbc components."""
    teams = match_df["team"].unique()
    rows = []

    for team in teams:
        t = match_df[match_df["team"] == team]
        rows.append({
            "Team": team,
            "Shots": len(t),
            "Goals": int(t["is_goal"].sum()),
            "xG": f"{t['xg'].sum():.2f}",
            "SB xG": f"{t['shot_statsbomb_xg'].sum():.2f}",
            "On Target": int(t["shot_outcome"].isin(["Goal", "Saved", "Saved to Post"]).sum()),
            "Headers": int(t["is_header"].sum()),
            "Penalties": int(t["is_penalty"].sum()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "xG Match Explorer"

competitions = sorted(match_info["competition"].unique())

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("xG Match Explorer", className="text-center mt-3 mb-1"),
            html.P("Select a match to explore shot-by-shot expected goals",
                   className="text-center text-muted mb-3"),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Competition"),
            dcc.Dropdown(
                id="comp-dropdown",
                options=[{"label": c, "value": c} for c in competitions],
                value="La Liga",
                clearable=False,
                style={"color": "#000"},
            ),
        ], md=3),
        dbc.Col([
            html.Label("Season"),
            dcc.Dropdown(id="season-dropdown", clearable=False, style={"color": "#000"}),
        ], md=2),
        dbc.Col([
            html.Label("Match"),
            dcc.Dropdown(id="match-dropdown", clearable=False, style={"color": "#000"}),
        ], md=7),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="xg-timeline"),
        ], md=12),
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Shot Map", className="text-center mt-2"),
            dcc.Graph(id="shot-map"),
        ], md=7),
        dbc.Col([
            html.H5("Match Stats", className="text-center mt-2"),
            html.Div(id="stats-table"),
        ], md=5),
    ]),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                "Built from scratch using ",
                html.A("StatsBomb open data", href="https://github.com/statsbomb/open-data",
                       target="_blank"),
                ". Model: XGBoost with 36 engineered features. ",
                "Bubble size = xG. Stars = goals.",
            ], className="text-muted text-center small"),
        ])
    ]),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(Output("season-dropdown", "options"), Output("season-dropdown", "value"),
          Input("comp-dropdown", "value"))
def update_seasons(comp):
    seasons = sorted(match_info[match_info["competition"] == comp]["season"].unique())
    options = [{"label": s, "value": s} for s in seasons]
    return options, seasons[0] if seasons else None


@callback(Output("match-dropdown", "options"), Output("match-dropdown", "value"),
          Input("comp-dropdown", "value"), Input("season-dropdown", "value"))
def update_matches(comp, season):
    filtered = match_info[(match_info["competition"] == comp) & (match_info["season"] == season)]
    filtered = filtered.sort_values("total_goals", ascending=False)
    options = [{"label": row["label"], "value": row["match_id"]} for _, row in filtered.iterrows()]
    return options, options[0]["value"] if options else None


@callback(
    Output("xg-timeline", "figure"),
    Output("shot-map", "figure"),
    Output("stats-table", "children"),
    Input("match-dropdown", "value"),
)
def update_match(match_id):
    if match_id is None:
        empty = go.Figure()
        return empty, empty, ""

    match_df = df[df["match_id"] == match_id].copy()

    timeline = build_xg_timeline(match_df)
    shot_map = build_shot_map(match_df)
    stats = build_stats_table(match_df)

    table = dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True,
                                      dark=True, size="sm", className="mt-2")

    return timeline, shot_map, table


if __name__ == "__main__":
    app.run(debug=True, port=8050)
