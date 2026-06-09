# EDA Findings — International Football Results

**Dataset:** martj42/international_results
**Generated:** 2026-06-09
**Source files:** `data/results.csv`, `data/shootouts.csv`, `data/goalscorers.csv`
**Script:** `eda/eda.py`

---

## 1. Dataset overview

- **results.csv shape:** 49,445 rows × 9 columns
- **shootouts.csv shape:** 678 × 5
- **goalscorers.csv shape:** 47,601 × 8
- **Date range:** 1872-11-30 → 2026-06-27 (a handful of future/scheduled fixtures included)
- **Total matches (with scores):** 49,373 after dropping 72 rows with missing `home_score`/`away_score`
- **Columns:** date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
- **Missing values:** only 72 each in home_score / away_score; everything else complete.

## 2. Result distribution & home advantage

| Split | n | Home win | Draw | Away win |
|---|---|---|---|---|
| **Overall** | 49,373 | **49.01%** | **22.74%** | **28.26%** |
| Non-neutral venue | 36,339 | 50.74% | 22.85% | 26.41% |
| **Neutral venue** | 13,034 | **44.17%** | **22.43%** | **33.40%** |

- Avg home goals 1.757 vs avg away goals 1.182 → **home advantage = +0.575 goals/match** overall.
- On **neutral** ground the goal gap shrinks to **+0.300** (1.672 vs 1.371); on non-neutral it is +0.674.
- The draw rate is remarkably stable (~22.4–22.9%) regardless of venue.

**Key insight for a neutral-venue World Cup:** the residual +0.30 "home" edge on neutral ground is just an artifact of which team is *listed* as home (often the better-seeded / organising side). The true no-advantage baseline for two evenly-matched teams on neutral ground is close to a coin flip with a fixed ~22% draw mass.

## 3. Goals per match & Poisson fit

- Avg total goals/match: **2.939**; median 3; variance 4.392.
- **var/mean ratio = 1.494** → total goals are **over-dispersed** relative to a pure Poisson (1.0).
- Per-team goals: mean 1.470, var 2.640 (also over-dispersed).
- Poisson(λ=2.94) tracks the body of the distribution but **under-predicts 0-goal games** (observed 8.0% vs Poisson 5.3%) and **fat right tail** (observed 0.97% vs 0.09% for 10+).
- Implication: a **bivariate / negative-binomial or Dixon–Coles-adjusted Poisson** will fit better than naive independent Poisson, especially the low-score and draw cells.

## 4. Tournament breakdown

- Most common: Friendly (18,364), FIFA World Cup qualification (8,771), UEFA Euro qualification (2,824).
- **FIFA World Cup (finals):** 964 matches, 1930-07-13 → 2022-12-18.
  - Home win **45.54%** / **draw 22.20%** / away win **32.26%**.
  - **World Cup draw rate: 22.20%.**
  - Avg total goals 2.822 (slightly below the all-match 2.939).
  - **87.4% of WC finals matches are at neutral venues** — confirming the neutral-venue base rates are the right reference, not the overall ones.
- WC qualification (8,772 matches): 50.84% / 21.15% / 28.01% — more home-skewed (home-and-away legs).

## 5. Trend over time (by decade)

- Match volume exploded: 13 matches in the 1870s → ~9,500–9,800/decade in the 2000s–2010s.
- **Scoring has fallen sharply:** ~4.0–5.6 goals/match pre-1950s, settling to **~2.7–2.8** since the 1980s.
- Draw rate rose from ~15% (early eras) to a stable ~23% from the 1980s onward.
- Modern era (1980+) is the relevant regime for calibration; older data is a different scoring environment.

## 6. Team performance

- Top win rates (min 100 matches): Jersey 65.1% and Guernsey 60.4% (weak-opposition islands), then **Brazil 63.5%, Spain 58.8%, Germany 58.1%, England 57.3%, Iran 57.0%, Argentina 55.3%**.
- Most active: Sweden, England, Argentina, Brazil, Germany — all >1,000 matches.

## 7. Recent form (since 2021-06-09, ~last 5 years)

- Top contenders by win rate: **Argentina 77.9%, Morocco 73.2%, Japan 68.8%, Iran 67.8%, Algeria 67.1%, Portugal 66.2%, Spain 65.2%, Senegal 64.5%, England 62.7%**.
- Hosts: USA 52.3%, Mexico 51.0%, Canada 48.0%.
- Recent form diverges meaningfully from all-time win rate — a recency-weighted strength feature is warranted.

---

## Modeling implications for the 2026 predictor

1. **Use neutral-venue base rates as the baseline prior, not overall.** Because ~87% of World Cup matches are neutral, the reference outcome split is ≈ **44% / 22% / 33%** (listed-home / draw / listed-away), not the 49/23/28 all-match figure.
2. **Don't bake in home-field advantage for the tournament.** The +0.575 goal home edge collapses to ~+0.30 on neutral ground, and even that reflects listing/seeding bias rather than a real venue effect. For the host nations (USA/Canada/Mexico) a *small* explicit host bonus may be justified, but treat all other matches as venue-neutral.
3. **Model the draw explicitly (~22%).** The draw mass is stable across venue and across the World Cup specifically (22.2%). Independent Poisson under-predicts draws and 0-0s — apply a **Dixon–Coles low-score correction** or fit a negative-binomial/bivariate-Poisson to capture the over-dispersion (var/mean ≈ 1.49).
4. **Calibrate on the modern regime (1980+).** Scoring and draw rates shifted; pre-1980 data comes from a higher-scoring environment and will bias goal-rate estimates upward. Optionally down-weight by recency.
5. **Feature ideas:** recency-weighted team attack/defence strength (Elo or rolling goals-for/against), recent-form win rate (last ~5 yrs / last N matches), opponent-adjusted goal rates, confederation/region, rest days, and a host-nation flag. Predict **goal rates per team** (λ_home, λ_away) and derive the win/draw/loss probabilities from the score matrix rather than classifying outcomes directly.

## Plots saved

- `eda/plot1_result_distribution.png` — result split, all matches vs neutral venue
- `eda/plot2_goals_poisson.png` — total-goals histogram with Poisson(λ=2.94) overlay
- `eda/plot3_over_time.png` — matches and avg goals per decade
- `eda/plot4_home_advantage.png` — avg home vs away goals by venue type
