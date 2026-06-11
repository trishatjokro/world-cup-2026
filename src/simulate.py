"""Monte-Carlo tournament simulator (Phase 3).

Simulates the full 48-team 2026 World Cup thousands of times from the Phase-2a
match model, locking any matches that have already been played to their real
results, and aggregates each team's odds of: winning its group, qualifying,
reaching each knockout round, and winning the cup.

Group stage  : 6 matches/group, scorelines sampled from the Dixon-Coles matrix
               (so goal difference / goals-for tiebreakers are real).
Tiebreakers  : points -> GD -> GF -> drawing of lots (random).  [H2H: TODO 2b]
Best thirds  : the 12 third-placed teams are ranked; the top 8 qualify and are
               slotted into the official R32 "3rd-<groups>" placeholders via a
               constraint-respecting assignment (eligibility from the fixtures).
Knockouts    : single elimination; a regulation draw is resolved to a winner
               with prob proportional to the two sides' relative strength
               (extra-time / penalties approximation).

Output: predictions.json consumed by the dashboard.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from elo import compute_elo, ratings_asof, DEFAULTS
from model import DixonColesModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
WC_TOURNAMENT = "FIFA World Cup"
HOSTS = {"United States", "Canada", "Mexico"}
HOST_BONUS = 50.0          # small Elo bump for host nations at "home" (EDA: ~+0.3 goal)


# ----------------------------- helpers ---------------------------------------
def _sample_scores(matrix: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (home_goals, away_goals) pairs from a scoreline matrix."""
    flat = matrix.ravel()
    idx = rng.choice(flat.size, size=n, p=flat / flat.sum())
    ncols = matrix.shape[1]
    return idx // ncols, idx % ncols


class Predictor:
    """Wraps the model + current ratings and caches per-matchup quantities."""

    def __init__(self, model: DixonColesModel, ratings: dict, ko_luck: float = 0.5):
        self.model = model
        self.ratings = ratings
        # ko_luck in [0,1]: how a regulation draw (-> extra time / penalties) is
        # resolved.  1 = purely by team strength; 0 = pure coin-flip (shootouts
        # are ~50/50).  0.5 blends the two -> realistic knockout "luck".
        self.ko_luck = ko_luck
        self._mat: dict = {}
        self._adv: dict = {}

    def _rating(self, team, host=False):
        r = self.ratings.get(team, DEFAULTS["base_rating"])
        return r + (HOST_BONUS if host else 0.0)

    def matrix(self, a, b):
        key = (a, b)
        if key not in self._mat:
            ra = self._rating(a, host=a in HOSTS)
            rb = self._rating(b, host=b in HOSTS)
            M, _, _ = self.model.score_matrix(ra, rb, neutral=True)
            self._mat[key] = M
        return self._mat[key]

    def adv_prob(self, a, b):
        """P(a advances past b) in a knockout (regulation + ET/pens)."""
        key = (a, b)
        if key not in self._adv:
            M = self.matrix(a, b)
            ph = np.tril(M, -1).sum()
            pd_ = np.trace(M)
            pa = np.triu(M, 1).sum()
            # resolve the regulation-draw mass: blend strength vs coin-flip
            strength = ph / (ph + pa) if (ph + pa) > 0 else 0.5
            draw_share = 0.5 + self.ko_luck * (strength - 0.5)
            self._adv[key] = float(ph + pd_ * draw_share)
        return self._adv[key]


# ----------------------------- group stage -----------------------------------
def _standings(group_teams, matches, scores):
    """scores: list of (hg, ag) aligned with matches rows. Return ranked teams."""
    pts = {t: 0 for t in group_teams}
    gf = {t: 0 for t in group_teams}
    ga = {t: 0 for t in group_teams}
    for (h, a), (hg, ag) in zip(matches, scores):
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    # points -> GD -> GF -> random jitter (drawing of lots)
    rng_key = {t: np.random.random() for t in group_teams}
    ranked = sorted(group_teams,
                    key=lambda t: (pts[t], gf[t] - ga[t], gf[t], rng_key[t]),
                    reverse=True)
    stats = {t: dict(pts=pts[t], gd=gf[t] - ga[t], gf=gf[t]) for t in group_teams}
    return ranked, stats


# ----------------------------- main sim ---------------------------------------
def run(n_sims=10000, asof="2026-06-11", seed=0, ko_luck=0.5, n_sims_progress=False):
    rng = np.random.default_rng(seed)
    groups = pd.read_csv(os.path.join(DATA, "groups.csv"))
    fixtures = pd.read_csv(os.path.join(DATA, "fixtures.csv"))
    results = pd.read_csv(os.path.join(DATA, "results.csv"))

    elo = compute_elo(results)
    model = DixonColesModel().fit(elo, train_from="1994-01-01")
    ratings = ratings_asof(elo, asof)
    pred = Predictor(model, ratings, ko_luck=ko_luck)

    team_group = dict(zip(groups.team, groups.group))
    group_teams = {g: list(groups[groups.group == g].team) for g in sorted(groups.group.unique())}

    gf = fixtures[fixtures.stage == "group"]
    group_matches = {g: [(r.home_team, r.away_team) for _, r in gf[gf.group == g].iterrows()]
                     for g in group_teams}

    # already-played 2026 WC results (lock them); none expected before Jun 11
    played = results.copy()
    played["date"] = pd.to_datetime(played["date"])
    played = played[(played.tournament == WC_TOURNAMENT) & (played.date < pd.to_datetime(asof))
                    & played.home_score.notna()]
    locked = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
              for _, r in played.iterrows()}

    # precompute group-match scoreline matrices once
    gm_matrix = {g: [pred.matrix(h, a) for (h, a) in group_matches[g]] for g in group_teams}

    # knockout fixture structure
    ko = fixtures[fixtures.stage != "group"].sort_values("match_id")
    third_slots = []  # (match_id, side, eligible_groups)
    for _, r in ko.iterrows():
        for side in ("home_team", "away_team"):
            slot = r[side]
            if isinstance(slot, str) and slot.startswith("3rd-"):
                third_slots.append((r.match_id, side, list(slot.split("-")[1])))

    # accumulators
    teams = list(groups.team)
    acc = {t: dict(win_group=0, second=0, third_qualify=0, reach_r16=0, reach_qf=0,
                   reach_sf=0, reach_final=0, win_cup=0) for t in teams}

    for s in range(n_sims):
        # ----- group stage -----
        winners, runners, thirds, third_stats = {}, {}, [], {}
        for g in group_teams:
            scores = []
            for k, (h, a) in enumerate(group_matches[g]):
                if (h, a) in locked:
                    scores.append(locked[(h, a)])
                else:
                    hg, ag = _sample_scores(gm_matrix[g][k], 1, rng)
                    scores.append((int(hg[0]), int(ag[0])))
            ranked, stats = _standings(group_teams[g], group_matches[g], scores)
            winners[g], runners[g] = ranked[0], ranked[1]
            acc[ranked[0]]["win_group"] += 1
            acc[ranked[1]]["second"] += 1
            thirds.append(ranked[2])
            third_stats[ranked[2]] = (stats[ranked[2]], g)

        # ----- rank thirds, top 8 qualify -----
        thirds_sorted = sorted(thirds,
                               key=lambda t: (third_stats[t][0]["pts"], third_stats[t][0]["gd"],
                                              third_stats[t][0]["gf"], np.random.random()),
                               reverse=True)
        qual_thirds = thirds_sorted[:8]
        for t in qual_thirds:
            acc[t]["third_qualify"] += 1

        # assign qualifying thirds to the 8 R32 third-slots (eligibility-respecting)
        cost = np.full((8, len(third_slots)), 1e6)
        for i, t in enumerate(qual_thirds):
            tg = third_stats[t][1]
            for j, (_, _, elig) in enumerate(third_slots):
                if tg in elig:
                    cost[i, j] = 0.0
        ri, cj = linear_sum_assignment(cost)
        slot_team = {}
        for i, j in zip(ri, cj):
            mid, side, _ = third_slots[j]
            slot_team[(mid, side)] = qual_thirds[i]

        # ----- knockouts -----
        wins, loses = {}, {}

        def resolve(mid, side, slot):
            if slot.startswith("1"):
                return winners[slot[1:]]
            if slot.startswith("2"):
                return runners[slot[1:]]
            if slot.startswith("3rd-"):
                return slot_team[(mid, side)]
            if slot.startswith("WM"):
                return wins[int(slot[2:])]
            if slot.startswith("LM"):
                return loses[int(slot[2:])]
            raise ValueError(slot)

        round_reached = {"R16": "reach_r16", "QF": "reach_qf", "SF": "reach_sf"}
        for _, r in ko.iterrows():
            a = resolve(r.match_id, "home_team", r.home_team)
            b = resolve(r.match_id, "away_team", r.away_team)
            pa = pred.adv_prob(a, b)
            if rng.random() < pa:
                w, l = a, b
            else:
                w, l = b, a
            wins[r.match_id] = w
            loses[r.match_id] = l
            # credit reaching the NEXT round (i.e. winning this match), per stage
            if r.stage == "R32":
                acc[w]["reach_r16"] += 1
            elif r.stage == "R16":
                acc[w]["reach_qf"] += 1
            elif r.stage == "QF":
                acc[w]["reach_sf"] += 1
            elif r.stage == "SF":
                acc[w]["reach_final"] += 1
            elif r.stage == "final":
                acc[w]["win_cup"] += 1

        if n_sims_progress and (s + 1) % 2000 == 0:
            print(f"  ...{s+1}/{n_sims} sims")

    # ----- aggregate -----
    out_teams = {}
    for t in teams:
        a = acc[t]
        qualify = (a["win_group"] + a["second"] + a["third_qualify"]) / n_sims
        out_teams[t] = dict(
            group=team_group[t], elo=round(ratings.get(t, DEFAULTS["base_rating"]), 1),
            win_group=a["win_group"] / n_sims, runner_up=a["second"] / n_sims,
            qualify=qualify, reach_r16=a["reach_r16"] / n_sims, reach_qf=a["reach_qf"] / n_sims,
            reach_sf=a["reach_sf"] / n_sims, reach_final=a["reach_final"] / n_sims,
            win_cup=a["win_cup"] / n_sims)

    # per-match probabilities for known-team (group-stage) fixtures
    matches = []
    for _, r in gf.iterrows():
        p = pred_match(pred, r.home_team, r.away_team)
        played_res = locked.get((r.home_team, r.away_team))
        matches.append(dict(match_id=int(r.match_id), date=r.date, stage="group", group=r.group,
                            home=r.home_team, away=r.away_team, venue=r.venue,
                            played=played_res is not None, result=played_res, **p))

    payload = dict(asof=asof, n_sims=n_sims, generated_ts=None,
                   model=dict(rho=round(model.rho, 4), home_bonus_hosts=HOST_BONUS),
                   teams=out_teams, matches=matches)
    return payload


def pred_match(pred: Predictor, a, b):
    M = pred.matrix(a, b)
    ph = float(np.tril(M, -1).sum()); pdr = float(np.trace(M)); pa = float(np.triu(M, 1).sum())
    i, j = np.unravel_index(np.argmax(M), M.shape)
    exp_h = float((np.arange(M.shape[0])[:, None] * M).sum())
    exp_a = float((np.arange(M.shape[1])[None, :] * M).sum())
    return dict(p_home=round(ph, 4), p_draw=round(pdr, 4), p_away=round(pa, 4),
                exp_home=round(exp_h, 2), exp_away=round(exp_a, 2),
                top_score=[int(i), int(j)])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-sims", type=int, default=10000)
    ap.add_argument("--asof", default="2026-06-11")
    args = ap.parse_args()

    print(f"simulating {args.n_sims} tournaments (asof {args.asof}) ...")
    payload = run(n_sims=args.n_sims, asof=args.asof, n_sims_progress=True)
    payload["generated_ts"] = datetime.now(timezone.utc).isoformat()
    out = os.path.join(HERE, "predictions.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {out}\n")

    teams = payload["teams"]
    top = sorted(teams.items(), key=lambda kv: -kv[1]["win_cup"])[:16]
    print(f"{'team':<22}{'grp':>4}{'winGrp':>8}{'qualify':>9}{'reachSF':>9}{'winCup':>8}")
    for t, d in top:
        print(f"{t:<22}{d['group']:>4}{d['win_group']:>8.0%}{d['qualify']:>9.0%}"
              f"{d['reach_sf']:>9.0%}{d['win_cup']:>8.1%}")
