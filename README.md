# 🏆 World Cup 2026 — Live ML Predictor & Squad Explorer

A self-updating, Telegraph-style "supercomputer" for the 2026 FIFA World Cup. It uses machine learning to forecast every match, runs a Monte Carlo simulation of the whole tournament thousands of times, and lets you drill from groups → countries → players. It **retrains itself after every match** and compares its own probabilities against live prediction-market odds.

> **Status:** 🚧 In active development (the tournament kicks off **11 June 2026**). See the [roadmap](#-roadmap--status) for what's live vs. in progress.

<!-- Add once deployed: [![Live demo](https://img.shields.io/badge/demo-streamlit-FF4B4B?logo=streamlit)](https://your-app.streamlit.app) -->

---

## What it does

- **🔮 Match predictions** — win / draw / loss probabilities and a most-likely scoreline for every fixture, from an Elo + Dixon–Coles goals model.
- **📊 Tournament simulation** — a 10,000-run Monte Carlo of the full 48-team bracket (real group-stage tiebreakers → Round of 32 → Final) producing each team's odds of **winning their group, advancing, and lifting the cup**.
- **🌍 Squad explorer** — click a country to see its 26-man squad; click a player to see their stats, attribute radar, and a KMeans-derived **playing-style tag** ("poacher", "deep-lying playmaker", …).
- **💹 Model vs. Market** — the model's probabilities side-by-side with prediction-market / bookmaker implied odds, highlighting where the model **disagrees with the crowd**.
- **♻️ Always up to date** — a scheduled job re-pulls results, retrains, and re-simulates **after every completed match**, so the public site is never stale.
- **📈 Learns from the tournament** — group-stage **shots/xG** feed back into the model, so knockout-round predictions reflect how teams have *actually* played in 2026, not just their history.

---

## How it works

```
┌─ DATA ───────────────────────────────────────────────┐
│ Historical SCORES: 49k matches (1872→2026)             │  → base layer
│ SHOTS/xG (recent + live 2026): FBref / API-Football    │  → enhancement layer
│ Live 2026:  results + match stats re-pulled per round  │  → "always updated"
│ Static:     48 teams · 12 groups · 104-match schedule  │
│ Squads:     2026 rosters + EA FC player attributes     │  → drill-down
└──────────────────────┬────────────────────────────────┘
┌─ MODEL (two layers) ─▼────────────────────────────────┐
│ BASE:    Elo + Dixon–Coles on scores (all 49k matches) │  → P(W/D/L) + scoreline grid
│ ENHANCE: shots/xG form rating blended in; its weight   │  → sharpens knockout odds from
│          grows as the 2026 tournament is played        │     real group-stage performance
└──────────────────────┬────────────────────────────────┘
┌─ SIMULATION ─────────▼────────────────────────────────┐
│ Monte Carlo 10k× the bracket (real WC tiebreakers).    │  → P(win group / advance /
│ Played matches locked to real results.                 │     reach SF·F / win cup)
└──────────────────────┬────────────────────────────────┘
┌─ DASHBOARD (Streamlit + Plotly) ──────────────────────┐
│ Groups · Match · Knockout/title · Country → Squad →    │
│ Player · Model-vs-Market · Model credibility           │
└───────────────────────────────────────────────────────┘
```

The heavy compute (retrain + simulation) runs in a **GitHub Action** that commits a fresh `predictions.json`; the Streamlit app just reads that file, so the public site stays fast and updates automatically on every push.

### The model — a two-layer design

Scores exist for all 49k matches back to 1872, but **shots/xG only exist for recent years and live**. So the model has two layers:

- **Base layer (deep history, scores):** **Elo** computed from the match history itself (leakage-free, updates after every result) + **Dixon–Coles** (time-weighted Poisson), which models each team's expected goals → a full scoreline probability matrix → honest **W/D/L *and draw*** probabilities (plain Poisson under-predicts draws). It also yields the scorelines the simulator needs for goal-difference tiebreakers — something a plain classifier can't.
- **Enhancement layer (recent + live 2026, shots/xG):** a performance rating from **shots and xG** that captures whether results were *deserved* — a team winning while being out-shot is overrated by goals alone. It's **blended** with the Elo prior (not replacing it — 3 group games is a tiny sample), and its weight **grows as the tournament unfolds**, so by the knockouts the model reflects how teams have actually played in 2026.

**Honesty rails:** the app reports calibrated probabilities (not just labels), evaluated with **time-based walk-forward** validation on **RPS + log-loss** against an Elo baseline *and* bookmaker odds. The realistic accuracy ceiling for international football is **~55–58%** (draws are the hardest class) — so the real target is **calibration (RPS) near the bookmakers'**, not high accuracy. The in-tournament idea is validated by replaying the 2022 World Cup (predicting its knockouts from its group-stage xG).

---

## Key findings from the EDA (49,373 real matches, 1872–2026)

See [`eda/EDA_FINDINGS.md`](eda/EDA_FINDINGS.md) for the full write-up and plots. Highlights baked into the model:

- **Use neutral-venue base rates ≈ 44% / 22% / 33%** (home/draw/away) — *not* the overall 49/23/28, because ~87% of World Cup matches are at neutral venues.
- **Home advantage collapses on neutral ground** (+0.58 → +0.30 goals) → no home edge except a small host bonus (USA / Canada / Mexico).
- **World Cup draw rate is a stable 22.2%** → draws are modelled explicitly (goals are over-dispersed, var/mean = 1.49).
- Scoring stabilised post-1980 → the model **calibrates on 1980+** data.

---

## Tech stack

`Python` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` (KMeans) · `Plotly` · `Streamlit` · `matplotlib`/`seaborn` (EDA) · `GitHub Actions` (live retrain)

## Data sources & credits

| Data | Source | License |
|---|---|---|
| Historical international results (scores) | [martj42/international_results](https://github.com/martj42/international_results) | CC0 (public domain) |
| Shots / xG (recent + live 2026) | [FBref](https://fbref.com) via `soccerdata` · [API-Football](https://www.api-football.com/) · [StatsBomb open-data](https://github.com/statsbomb/open-data) (2018/2022 WC, validation) | per provider terms |
| 2026 fixtures, groups, squads | Wikipedia "2026 FIFA World Cup" / squads | CC BY-SA |
| Player attributes | EA Sports FC dataset (Kaggle) | per dataset terms |
| Market odds | The Odds API / Polymarket / Kalshi | per provider terms |

---

## Project structure

```
.
├── data/            results.csv (49k matches), shootouts, goalscorers,
│                    groups.csv (48 teams), fixtures.csv (104 matches), squads.csv,
│                    match_stats.csv (shots/xG — WIP)
├── eda/             eda.py, EDA_FINDINGS.md, plots (*.png)
├── src/             build_tournament.py  (groups + fixtures)
│                    elo.py, model.py, backtest.py   (base model — WIP)
│                    match_stats.py   (shots/xG enhancement — WIP)
│                    simulate.py, build_players.py, markets.py, run_pipeline.py   (WIP)
├── .github/workflows/   live-retrain.yml   (retrain-on-new-match — WIP)
├── app.py           Streamlit dashboard (WIP)
├── notes/           research notes (predictor, players, sentiment)
├── OUTLINE.md       full design doc
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/trishatjokro/world-cup-2026.git
cd world-cup-2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# reproduce the EDA
python eda/eda.py

# (once the predictor lands) run the pipeline + app
# python src/run_pipeline.py
# streamlit run app.py
```

## Deploy as a public website

Push to GitHub and deploy free on **[Streamlit Community Cloud](https://streamlit.io/cloud)** — connect the repo, pick `app.py`, and you get a public `*.streamlit.app` URL. The retrain GitHub Action commits new predictions, and Streamlit Cloud auto-redeploys on each push, so the live site updates itself. (For a custom domain, host on Render/Railway + Cloudflare instead.)

---

## 🗺️ Roadmap / status

- [x] **Phase 0** — Workspace, data download, EDA on 49k matches
- [x] **Phase 1** — 48 teams, 12 groups, 104-match schedule (`groups.csv`, `fixtures.csv`) + 1,245-player squads (`squads.csv`)
- [ ] **Phase 2a** — Base Elo + Dixon–Coles model + walk-forward backtest harness
- [ ] **Phase 2b** — Shots/xG enhancement layer + hyperparameter tuning loop (validated by replaying 2022)
- [ ] **Phase 3** — Monte Carlo tournament simulator (re-runs each matchday)
- [ ] **Phase 4** — Player data + style clustering
- [ ] **Phase 5** — Prediction-market adapter (Model vs. Market)
- [ ] **Phase 6** — Retrain-on-new-match pipeline
- [ ] **Phase 7** — Streamlit dashboard
- [ ] **Phase 8** — Automate (GitHub Actions) + deploy

---

## ⚠️ Disclaimer

This is a portfolio / educational data-science project. Predictions come from a statistical model (Elo + Poisson), not insider knowledge — it's a credible "supercomputer," not Opta. "Model vs. Market" edges indicate where the model disagrees with bookmakers; **they are not betting advice.**
