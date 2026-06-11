"""Walk-forward backtest harness (Phase 2a).

Reusable scoring rig: for a test period it refits the model once per year on a
trailing window of matches strictly BEFORE that year (leakage-free), predicts
every test match, and scores the probabilities. Phase 2b's tuner will call
run_backtest() with different hyper-parameters and optimise on RPS.

Metrics:
  accuracy  - argmax outcome correct (blunt; ~0.55 ceiling for internationals)
  log_loss  - rewards calibrated probabilities
  RPS       - Ranked Probability Score for ordered [home, draw, away];
              the standard football-forecasting metric (lower = better)

Compared models:
  dc           - our Elo + Dixon-Coles goals model
  elo_logistic - baseline: multinomial logistic on the Elo difference alone
  base_rate    - reference: always predict the training-set W/D/L frequencies
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss

from elo import compute_elo
from model import DixonColesModel

OUTCOMES = ["home", "draw", "away"]            # 0, 1, 2


def _outcome(hs, as_):
    return np.where(hs > as_, 0, np.where(hs == as_, 1, 2))


def rps(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean Ranked Probability Score for ordered 3-outcome forecasts."""
    onehot = np.eye(3)[y]
    cp = np.cumsum(probs, axis=1)[:, :2]
    co = np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2.0))


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    acc = float(np.mean(np.argmax(probs, axis=1) == y))
    ll = float(sk_log_loss(y, probs, labels=[0, 1, 2]))
    return dict(accuracy=acc, log_loss=ll, rps=rps(probs, y), n=int(len(y)))


def _feature_D(df):
    return ((df["elo_home_pre"] + df["home_adv_used"] - df["elo_away_pre"]) / 100.0).to_numpy()


def run_backtest(elo_df: pd.DataFrame, test_start=2002, test_end=2022,
                 subset="wc", train_window_years=20, model_params=None,
                 verbose=False, return_raw=False) -> dict:
    """Return {dc, elo_logistic, base_rate} -> metrics dict, over `subset`
    matches in [test_start, test_end]. subset: 'wc' | 'competitive' | 'all'.
    If return_raw, also include 'raw' = (dc_probs, y) for calibration analysis."""
    elo_df = elo_df.copy()
    elo_df["year"] = elo_df["date"].dt.year
    elo_df["outcome"] = _outcome(elo_df["home_score"].to_numpy(),
                                 elo_df["away_score"].to_numpy())

    tour = elo_df["tournament"].str.lower()
    if subset == "wc":
        test_pool = elo_df[tour.eq("fifa world cup")]
    elif subset == "competitive":
        test_pool = elo_df[~tour.str.contains("friendly")]
    else:
        test_pool = elo_df

    dc_p, elo_p, base_p, ys = [], [], [], []
    for yr in range(test_start, test_end + 1):
        test = test_pool[test_pool["year"] == yr]
        if test.empty:
            continue
        train = elo_df[(elo_df["year"] < yr) & (elo_df["year"] >= yr - train_window_years)]
        train = train.dropna(subset=["home_score", "away_score"])
        if len(train) < 500:
            continue

        # our model
        dc = DixonColesModel(**(model_params or {})).fit(train)
        # baseline: logistic on Elo diff
        base_lr = LogisticRegression(max_iter=1000)
        base_lr.fit(_feature_D(train).reshape(-1, 1), train["outcome"].to_numpy())
        # reference: training base rates
        rates = np.bincount(train["outcome"].to_numpy(), minlength=3) / len(train)

        for _, r in test.iterrows():
            neutral = r["home_adv_used"] == 0
            p = dc.predict(r["elo_home_pre"], r["elo_away_pre"], neutral=neutral)
            dc_p.append([p["p_home"], p["p_draw"], p["p_away"]])
            elo_p.append(base_lr.predict_proba(
                np.array([[(r["elo_home_pre"] + r["home_adv_used"] - r["elo_away_pre"]) / 100.0]]))[0])
            base_p.append(rates)
            ys.append(int(r["outcome"]))
        if verbose:
            print(f"  {yr}: {len(test)} {subset} matches  (train {len(train)})")

    y = np.array(ys)
    out = {
        "dc": evaluate(np.array(dc_p), y),
        "elo_logistic": evaluate(np.array(elo_p), y),
        "base_rate": evaluate(np.array(base_p), y),
    }
    if return_raw:
        out["raw"] = (np.array(dc_p), y)
    return out


def _print_table(title, res):
    print(f"\n=== {title} (n={res['dc']['n']}) ===")
    print(f"{'model':<14}{'accuracy':>10}{'log_loss':>10}{'RPS':>10}")
    for name in ("dc", "elo_logistic", "base_rate"):
        m = res[name]
        print(f"{name:<14}{m['accuracy']:>10.3f}{m['log_loss']:>10.3f}{m['rps']:>10.4f}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = pd.read_csv(os.path.join(here, "data", "results.csv"))
    print("computing Elo over full history ...")
    elo = compute_elo(res)

    wc = run_backtest(elo, 2002, 2022, subset="wc", verbose=True)
    _print_table("World Cup matches 2002-2022", wc)

    comp = run_backtest(elo, 2010, 2022, subset="competitive")
    _print_table("All competitive matches 2010-2022", comp)

    print("\nLower RPS / log_loss = better. Target: dc beats elo_logistic, "
          "both beat base_rate.")
