"""Elo ratings and Poisson goal model."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from worldcup2026 import NAME_MAP, ALL_TEAMS

TODAY = datetime.now()
TWO_YEARS_AGO = TODAY - timedelta(days=730)
FIVE_YEARS_AGO = TODAY - timedelta(days=1825)
EIGHT_YEARS_AGO = TODAY - timedelta(days=8 * 365)


def get_tournament_mult(tourn):
    t = tourn.lower()
    if "world cup" in t and "qualification" not in t and "fifa" in t:
        return 2.0
    if "qualification" in t:
        return 1.25
    for name in ["uefa euro", "copa america", "africa cup", "african cup", "afcon",
                  "afc asian cup", "asian cup", "gold cup"]:
        if name in t:
            return 1.5
    if "friendly" in t:
        return 0.5
    return 1.0


def get_recency_mult(match_date):
    if match_date >= TWO_YEARS_AGO:
        return 1.4
    if match_date >= FIVE_YEARS_AGO:
        return 1.1
    return 0.8


def compute_elo_and_model():
    print("Loading match data...")
    df = pd.read_csv("results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)
    print(f"Loaded {len(df)} matches")

    print("Computing Elo ratings...")
    elo = {}
    poisson_rows = []
    for row in df.itertuples():
        ht, at = row.home_team, row.away_team
        elo_h = elo.get(ht, 1500)
        elo_a = elo.get(at, 1500)

        if row.date >= EIGHT_YEARS_AGO and pd.notna(row.home_score) and pd.notna(row.away_score):
            tm = get_tournament_mult(row.tournament)
            is_neutral = bool(row.neutral)
            poisson_rows.append({"attack_elo": elo_h, "defense_elo": elo_a, "elo_diff": elo_h - elo_a,
                                 "is_home": 0 if is_neutral else 1, "tourn_weight": tm, "goals": row.home_score})
            poisson_rows.append({"attack_elo": elo_a, "defense_elo": elo_h, "elo_diff": elo_a - elo_h,
                                 "is_home": 0, "tourn_weight": tm, "goals": row.away_score})

        tm = get_tournament_mult(row.tournament)
        rm = get_recency_mult(row.date)
        K = 30 * tm * rm

        home_bonus = 100 if not row.neutral else 0
        exp_h = 1 / (1 + 10 ** ((elo_a - (elo_h + home_bonus)) / 400))
        exp_a = 1 - exp_h

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue
        if row.home_score > row.away_score:
            act_h, act_a = 1, 0
        elif row.home_score < row.away_score:
            act_h, act_a = 0, 1
        else:
            act_h, act_a = 0.5, 0.5

        goal_diff = abs(row.home_score - row.away_score)
        margin_mult = min(1 + goal_diff / 3, 2.0)

        elo[ht] = elo_h + K * margin_mult * (act_h - exp_h)
        elo[at] = elo_a + K * margin_mult * (act_a - exp_a)

    team_elo = {t: elo.get(t, 1500) for t in ALL_TEAMS}
    print("Elo ratings computed.")

    # Train Poisson model
    train = pd.DataFrame(poisson_rows)
    features = ["attack_elo", "defense_elo", "elo_diff", "is_home", "tourn_weight"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=1.0, max_iter=300)),
    ])
    model.fit(train[features], train["goals"])
    print("Poisson model trained.")

    # Precompute pairwise lambdas
    print("Precomputing pairwise lambdas...")
    lambda_cache = {}
    all_pairs = [(a, b) for a in ALL_TEAMS for b in ALL_TEAMS if a != b]
    rows_a, rows_b = [], []
    for a, b in all_pairs:
        ea, eb = elo.get(a, 1500), elo.get(b, 1500)
        rows_a.append([ea, eb, ea - eb, 0, 2.0])
        rows_b.append([eb, ea, eb - ea, 0, 2.0])
    la = np.maximum(model.predict(np.array(rows_a)), 0.1)
    lb = np.maximum(model.predict(np.array(rows_b)), 0.1)
    for i, (a, b) in enumerate(all_pairs):
        lambda_cache[(a, b)] = (la[i], lb[i])

    return elo, team_elo, model, lambda_cache


def get_lambdas(model, elo, team_a, team_b, neutral=True):
    ea = elo.get(team_a, 1500)
    eb = elo.get(team_b, 1500)
    xa = pd.DataFrame([{"attack_elo": ea, "defense_elo": eb, "elo_diff": ea - eb,
                         "is_home": 0 if neutral else 1, "tourn_weight": 2.0}])
    xb = pd.DataFrame([{"attack_elo": eb, "defense_elo": ea, "elo_diff": eb - ea,
                         "is_home": 0, "tourn_weight": 2.0}])
    la = max(model.predict(xa)[0], 0.1)
    lb = max(model.predict(xb)[0], 0.1)
    return la, lb
