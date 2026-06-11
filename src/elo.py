"""Leakage-free Elo ratings for international football (World Football Elo style).

Walks every match in chronological order and maintains a rating per national
team. For each match we record the PRE-match ratings (the only thing a
leakage-free model may use as a feature) and the POST-match ratings (so we can
ask "what was each team's rating as of date T?" for any T).

Elo update:  R' = R + K * G * (W - E)
  E (expected home score) = 1 / (1 + 10**(-(Rh + H - Ra)/400))
  W in {1, 0.5, 0}             home win / draw / loss
  G = goal-difference multiplier (bigger wins move ratings more)
  K = base step, scaled by how important the match is (WC >> friendly)
  H = home advantage in Elo points, set to 0 at neutral venues
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ----- default hyper-parameters (Phase 2b's tuner will sweep these) -----
DEFAULTS = dict(
    base_rating=1500.0,
    home_adv=100.0,      # Elo points added to the home side (0 if neutral)
    k_world_cup=60.0,
    k_continental=50.0,  # Euro / Copa / AFCON / Asian Cup / Gold Cup / Confeds
    k_qualifier=40.0,    # WC + continental qualifiers, Nations League
    k_other=30.0,
    k_friendly=20.0,
)


def _k_for_tournament(t: str, p: dict) -> float:
    t = (t or "").lower()
    if "friendly" in t:
        return p["k_friendly"]
    if "world cup" in t and "qualif" not in t:
        return p["k_world_cup"]
    if "qualif" in t or "nations league" in t:
        return p["k_qualifier"]
    for kw in ("copa", "uefa euro", "european championship", "african cup",
               "afc asian cup", "gold cup", "confederations"):
        if kw in t:
            return p["k_continental"]
    return p["k_other"]


def _goal_diff_multiplier(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(results: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Return `results` (rows with valid scores, date-sorted) augmented with
    elo_home_pre/away_pre, elo_home_post/away_post, and the K/expected used.
    """
    p = {**DEFAULTS, **(params or {})}
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    ratings: dict[str, float] = {}
    base = p["base_rating"]
    n = len(df)
    hp = np.empty(n); ap = np.empty(n)
    hpost = np.empty(n); apost = np.empty(n)

    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    neutral = df["neutral"].to_numpy() if "neutral" in df else np.zeros(n, bool)
    tours = df["tournament"].to_numpy() if "tournament" in df else np.array([""] * n)

    for i in range(n):
        h, a = homes[i], aways[i]
        Rh = ratings.get(h, base)
        Ra = ratings.get(a, base)
        H = 0.0 if bool(neutral[i]) else p["home_adv"]
        E_home = 1.0 / (1.0 + 10.0 ** (-(Rh + H - Ra) / 400.0))
        if hs[i] > as_[i]:
            W = 1.0
        elif hs[i] == as_[i]:
            W = 0.5
        else:
            W = 0.0
        K = _k_for_tournament(tours[i], p)
        G = _goal_diff_multiplier(hs[i] - as_[i])
        delta = K * G * (W - E_home)
        hp[i], ap[i] = Rh, Ra
        ratings[h] = Rh + delta
        ratings[a] = Ra - delta
        hpost[i], apost[i] = ratings[h], ratings[a]

    df["elo_home_pre"] = hp
    df["elo_away_pre"] = ap
    df["elo_home_post"] = hpost
    df["elo_away_post"] = apost
    df["home_adv_used"] = np.where(neutral, 0.0, p["home_adv"])
    return df


def ratings_asof(elo_df: pd.DataFrame, date) -> dict[str, float]:
    """Each team's rating as of `date` = its post-rating in its last match
    strictly before `date`. Teams unseen before `date` are absent (caller
    falls back to base_rating)."""
    date = pd.to_datetime(date)
    sub = elo_df[elo_df["date"] < date]
    out: dict[str, float] = {}
    # build a long frame (team, date, post-rating) and take the latest per team
    long = pd.concat([
        sub[["home_team", "date", "elo_home_post"]].rename(
            columns={"home_team": "team", "elo_home_post": "rating"}),
        sub[["away_team", "date", "elo_away_post"]].rename(
            columns={"away_team": "team", "elo_away_post": "rating"}),
    ], ignore_index=True)
    if len(long):
        long = long.sort_values("date", kind="mergesort")
        out = long.groupby("team")["rating"].last().to_dict()
    return out


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = pd.read_csv(os.path.join(here, "data", "results.csv"))
    elo = compute_elo(res)
    final = ratings_asof(elo, "2026-06-11")
    top = sorted(final.items(), key=lambda kv: -kv[1])[:20]
    print("Top 20 teams by Elo as of 2026-06-11:")
    for t, r in top:
        print(f"  {r:7.1f}  {t}")
