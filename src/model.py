"""Dixon-Coles goals model on top of Elo (Phase 2a base layer).

Idea: a team's Elo difference predicts how many goals each side scores. We fit
two Poisson regressions — home goals and away goals — on the pre-match Elo
difference D = (Elo_home + home_adv) - Elo_away, then apply the Dixon-Coles
low-score correction (which fixes plain Poisson's under-counting of 0-0/1-0/
0-1/1-1). From the resulting scoreline matrix we read off P(win/draw/loss),
the expected goals, and the most likely scoreline — and crucially the full
goal-difference distribution the Monte-Carlo simulator needs for tiebreakers.

Everything here is leakage-free as long as it is fit on matches strictly before
the date being predicted (the backtest harness enforces this).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor

from elo import DEFAULTS, ratings_asof


def _dc_tau(h, a, lh, la, rho):
    """Dixon-Coles correction factor for low scores (vectorised)."""
    tau = np.ones_like(lh, dtype=float)
    m00 = (h == 0) & (a == 0); tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    m01 = (h == 0) & (a == 1); tau[m01] = 1.0 + lh[m01] * rho
    m10 = (h == 1) & (a == 0); tau[m10] = 1.0 + la[m10] * rho
    m11 = (h == 1) & (a == 1); tau[m11] = 1.0 - rho
    return tau


class DixonColesModel:
    def __init__(self, home_adv: float = DEFAULTS["home_adv"],
                 max_goals: int = 10, alpha: float = 1e-4):
        self.home_adv = home_adv
        self.max_goals = max_goals
        self.alpha = alpha            # tiny L2 on the Poisson GLMs
        self.fitted = False

    # ---- fit ----
    def fit(self, elo_df: pd.DataFrame, asof=None, train_from=None, xi: float = 0.0):
        df = elo_df
        if asof is not None:
            df = df[df["date"] < pd.to_datetime(asof)]
        if train_from is not None:
            df = df[df["date"] >= pd.to_datetime(train_from)]
        df = df.dropna(subset=["home_score", "away_score"])
        if len(df) < 500:
            raise ValueError(f"not enough training matches ({len(df)})")

        D = (df["elo_home_pre"] + df["home_adv_used"] - df["elo_away_pre"]).to_numpy() / 100.0
        hg = df["home_score"].to_numpy().astype(int)
        ag = df["away_score"].to_numpy().astype(int)

        # recency weights (xi = decay per year); xi=0 -> uniform
        if xi and xi > 0:
            age_yrs = (df["date"].max() - df["date"]).dt.days.to_numpy() / 365.25
            w = np.exp(-xi * age_yrs)
        else:
            w = None

        X = D.reshape(-1, 1)
        self.m_home = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(X, hg, sample_weight=w)
        self.m_away = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(X, ag, sample_weight=w)

        lh = self.m_home.predict(X)
        la = self.m_away.predict(X)
        self.rho = self._fit_rho(hg, ag, lh, la, w)
        self.fitted = True
        return self

    def _fit_rho(self, hg, ag, lh, la, w):
        low = (hg <= 1) & (ag <= 1)            # only low scores carry rho info
        h, a, lhh, laa = hg[low], ag[low], lh[low], la[low]
        ww = None if w is None else w[low]

        def neg_ll(rho):
            tau = _dc_tau(h, a, lhh, laa, rho)
            tau = np.clip(tau, 1e-9, None)
            ll = np.log(tau)
            return -(ll if ww is None else ll * ww).sum()

        res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
        return float(res.x)

    # ---- predict ----
    def score_matrix(self, elo_home: float, elo_away: float, neutral: bool = True):
        H = 0.0 if neutral else self.home_adv
        D = np.array([[(elo_home + H - elo_away) / 100.0]])
        lh = float(self.m_home.predict(D)[0])
        la = float(self.m_away.predict(D)[0])
        g = np.arange(self.max_goals + 1)
        ph = poisson.pmf(g, lh)
        pa = poisson.pmf(g, la)
        M = np.outer(ph, pa)                   # rows = home goals, cols = away
        # apply DC correction to the 2x2 low-score block
        for i in (0, 1):
            for j in (0, 1):
                M[i, j] *= _dc_tau(np.array([i]), np.array([j]),
                                   np.array([lh]), np.array([la]), self.rho)[0]
        M = M / M.sum()
        return M, lh, la

    def predict(self, elo_home: float, elo_away: float, neutral: bool = True) -> dict:
        M, lh, la = self.score_matrix(elo_home, elo_away, neutral)
        p_home = np.tril(M, -1).sum()          # home goals > away goals
        p_draw = np.trace(M)
        p_away = np.triu(M, 1).sum()
        i, j = np.unravel_index(np.argmax(M), M.shape)
        return dict(p_home=float(p_home), p_draw=float(p_draw), p_away=float(p_away),
                    exp_home=float(lh), exp_away=float(la),
                    top_score=(int(i), int(j)), matrix=M)

    def predict_teams(self, team_a, team_b, ratings: dict, neutral: bool = True,
                      base: float = DEFAULTS["base_rating"]) -> dict:
        return self.predict(ratings.get(team_a, base), ratings.get(team_b, base), neutral)


if __name__ == "__main__":
    import os
    from elo import compute_elo
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = pd.read_csv(os.path.join(here, "data", "results.csv"))
    elo = compute_elo(res)
    model = DixonColesModel().fit(elo, train_from="1990-01-01")
    ratings = ratings_asof(elo, "2026-06-11")
    print(f"fitted rho = {model.rho:.4f}\n")
    for a, b in [("Spain", "Germany"), ("Argentina", "Mexico"),
                 ("United States", "England"), ("Brazil", "Japan")]:
        p = model.predict_teams(a, b, ratings, neutral=True)
        print(f"{a} vs {b} (neutral):  "
              f"W {p['p_home']:.0%} / D {p['p_draw']:.0%} / L {p['p_away']:.0%}  "
              f"| xG {p['exp_home']:.2f}-{p['exp_away']:.2f}  "
              f"| likely {p['top_score'][0]}-{p['top_score'][1]}")
