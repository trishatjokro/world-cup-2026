#!/usr/bin/env python3
"""EDA on martj42/international_results to support a 2026 World Cup match outcome predictor."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
DATA = "/Users/trishatjokrosapoetro/Documents/Projects/World Cup 2026/data"
OUT = "/Users/trishatjokrosapoetro/Documents/Projects/World Cup 2026/eda"

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------- load
df = pd.read_csv(f"{DATA}/results.csv", parse_dates=["date"])
shootouts = pd.read_csv(f"{DATA}/shootouts.csv", parse_dates=["date"])
goalscorers = pd.read_csv(f"{DATA}/goalscorers.csv", parse_dates=["date"])

hr("1. DATASET OVERVIEW")
print("results.csv shape :", df.shape)
print("shootouts.csv shape:", shootouts.shape)
print("goalscorers.csv shape:", goalscorers.shape)
print("\nDate range:", df["date"].min().date(), "->", df["date"].max().date())
print("Total matches:", len(df))
print("\nColumns / dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())

# drop rows with missing scores for outcome analysis
df = df.dropna(subset=["home_score", "away_score"]).copy()
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df["total_goals"] = df["home_score"] + df["away_score"]
df["goal_diff"] = df["home_score"] - df["away_score"]
df["year"] = df["date"].dt.year
df["decade"] = (df["year"] // 10) * 10


def outcome(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    if row["home_score"] < row["away_score"]:
        return "away_win"
    return "draw"


df["outcome"] = df.apply(outcome, axis=1)


def dist(frame, label):
    n = len(frame)
    vc = frame["outcome"].value_counts(normalize=True) * 100
    hw = vc.get("home_win", 0.0)
    dr = vc.get("draw", 0.0)
    aw = vc.get("away_win", 0.0)
    print(f"{label:28s} n={n:6d} | home_win={hw:5.2f}%  draw={dr:5.2f}%  away_win={aw:5.2f}%")
    return n, hw, dr, aw


# ---------------------------------------------------------------- result distribution
hr("2. RESULT DISTRIBUTION & HOME ADVANTAGE")
all_n, all_hw, all_dr, all_aw = dist(df, "Overall")
print(f"\nAvg home goals: {df['home_score'].mean():.3f}")
print(f"Avg away goals: {df['away_score'].mean():.3f}")
print(f"Home advantage (home - away avg goals): {df['home_score'].mean() - df['away_score'].mean():.3f}")
print(f"Avg goal diff (home - away): {df['goal_diff'].mean():.3f}")

# ---------------------------------------------------------------- neutral vs non-neutral
hr("3. NEUTRAL vs NON-NEUTRAL VENUE (key for a neutral-venue World Cup)")
neutral = df[df["neutral"] == True]
non_neutral = df[df["neutral"] == False]
print(f"Neutral matches: {len(neutral)} ({len(neutral)/len(df)*100:.1f}%)")
print(f"Non-neutral matches: {len(non_neutral)} ({len(non_neutral)/len(df)*100:.1f}%)\n")
neu_stats = dist(neutral, "Neutral venue")
non_stats = dist(non_neutral, "Non-neutral venue")
print(f"\nNeutral  : avg home goals={neutral['home_score'].mean():.3f}  avg away goals={neutral['away_score'].mean():.3f}  diff={neutral['home_score'].mean()-neutral['away_score'].mean():.3f}")
print(f"NonNeutral: avg home goals={non_neutral['home_score'].mean():.3f}  avg away goals={non_neutral['away_score'].mean():.3f}  diff={non_neutral['home_score'].mean()-non_neutral['away_score'].mean():.3f}")
print("\n=> On neutral ground 'home_team' is just the nominally-listed team; the gap shrinks toward a true no-advantage baseline.")

# ---------------------------------------------------------------- goals / Poisson
hr("4. GOALS PER MATCH & POISSON FIT")
print(f"Avg total goals per match: {df['total_goals'].mean():.3f}")
print(f"Median total goals: {df['total_goals'].median()}")
print(f"Variance of total goals: {df['total_goals'].var():.3f}")
print("\nDistribution of total goals (0..10+):")
tg = df["total_goals"].clip(upper=10)
obs = tg.value_counts(normalize=True).sort_index() * 100
lam = df["total_goals"].mean()
for g in range(0, 11):
    label = f"{g}+" if g == 10 else str(g)
    o = obs.get(g, 0.0)
    if g < 10:
        p = stats.poisson.pmf(g, lam) * 100
    else:
        p = (1 - stats.poisson.cdf(9, lam)) * 100
    print(f"  {label:>3s} goals: observed={o:5.2f}%   poisson(lam={lam:.2f})={p:5.2f}%")

# chi-square style goodness comparison (informal)
print(f"\nMean total goals (lambda) = {lam:.3f}; var/mean ratio = {df['total_goals'].var()/lam:.3f} (1.0 => pure Poisson)")
# per-team goals Poisson check
print(f"Per-team goals: mean={pd.concat([df['home_score'], df['away_score']]).mean():.3f}, "
      f"var={pd.concat([df['home_score'], df['away_score']]).var():.3f}")

# ---------------------------------------------------------------- tournaments
hr("5. TOURNAMENT BREAKDOWN")
print("Top 15 tournament types by match count:")
print(df["tournament"].value_counts().head(15).to_string())

wc = df[df["tournament"] == "FIFA World Cup"]
print(f"\n--- FIFA World Cup matches: {len(wc)} ---")
print(f"Date range: {wc['date'].min().date()} -> {wc['date'].max().date()}")
wc_n, wc_hw, wc_dr, wc_aw = dist(wc, "FIFA World Cup")
print(f"FIFA World Cup DRAW RATE: {wc_dr:.2f}%")
print(f"FIFA World Cup avg total goals: {wc['total_goals'].mean():.3f}")
print(f"FIFA World Cup neutral share: {(wc['neutral']==True).mean()*100:.1f}%")

# also qualifiers for reference
wcq = df[df["tournament"].str.contains("World Cup qualification", case=False, na=False)]
print(f"\nWorld Cup qualification matches: {len(wcq)}")
dist(wcq, "WC qualification")

# ---------------------------------------------------------------- trend over time
hr("6. TREND OVER TIME (by decade)")
dec = df.groupby("decade").agg(
    matches=("outcome", "size"),
    avg_total_goals=("total_goals", "mean"),
    avg_home_goals=("home_score", "mean"),
    avg_away_goals=("away_score", "mean"),
    draw_rate=("outcome", lambda s: (s == "draw").mean() * 100),
    home_win_rate=("outcome", lambda s: (s == "home_win").mean() * 100),
)
print(dec.round(3).to_string())

# ---------------------------------------------------------------- team win rates
hr("7. TEAM PERFORMANCE")
home = df[["home_team", "outcome"]].rename(columns={"home_team": "team"})
home["win"] = home["outcome"] == "home_win"
home["draw"] = home["outcome"] == "draw"
away = df[["away_team", "outcome"]].rename(columns={"away_team": "team"})
away["win"] = away["outcome"] == "away_win"
away["draw"] = away["outcome"] == "draw"
long = pd.concat([home[["team", "win", "draw"]], away[["team", "win", "draw"]]])
team = long.groupby("team").agg(matches=("win", "size"), wins=("win", "sum"), draws=("draw", "sum"))
team["win_rate"] = team["wins"] / team["matches"] * 100
team["draw_rate"] = team["draws"] / team["matches"] * 100

print("Top 20 teams by win rate (min 100 matches):")
top_win = team[team["matches"] >= 100].sort_values("win_rate", ascending=False).head(20)
print(top_win.round(2).to_string())

print("\nMost active teams (by matches played):")
print(team.sort_values("matches", ascending=False).head(15).round(2).to_string())

# ---------------------------------------------------------------- recent form for 2026 WC participants
hr("8. RECENT FORM (last 5 years, since 2021-06-09) — 2026 World Cup context")
recent_cut = pd.Timestamp("2021-06-09")
recent = df[df["date"] >= recent_cut]
rh = recent[["home_team", "outcome"]].rename(columns={"home_team": "team"})
rh["win"] = rh["outcome"] == "home_win"
ra = recent[["away_team", "outcome"]].rename(columns={"away_team": "team"})
ra["win"] = ra["outcome"] == "away_win"
rlong = pd.concat([rh[["team", "win"]], ra[["team", "win"]]])
rteam = rlong.groupby("team").agg(matches=("win", "size"), wins=("win", "sum"))
rteam["win_rate"] = rteam["wins"] / rteam["matches"] * 100

# Known/likely 2026 qualified or strong contenders (hosts auto + major nations)
participants_2026 = [
    "United States", "Canada", "Mexico",  # hosts
    "Argentina", "Brazil", "France", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Italy", "Belgium", "Croatia", "Uruguay", "Colombia",
    "Morocco", "Japan", "South Korea", "Senegal", "Switzerland", "Denmark",
    "Mexico", "Ecuador", "Australia", "Iran", "Saudi Arabia", "Poland",
    "Serbia", "Ghana", "Cameroon", "Tunisia", "Nigeria", "Egypt",
    "Austria", "Ukraine", "Wales", "Sweden", "Norway", "Turkey",
    "Peru", "Chile", "Paraguay", "Qatar", "Ivory Coast", "Algeria",
]
participants_2026 = sorted(set(participants_2026))
present = [t for t in participants_2026 if t in rteam.index]
sub = rteam.loc[present].sort_values("win_rate", ascending=False)
print(f"Recent (>= 2021-06-09) win rate for {len(present)} likely contenders (min coverage):")
print(sub[sub["matches"] >= 10].round(2).to_string())

print("\nTop 20 teams by recent matches played (activity proxy):")
print(rteam.sort_values("matches", ascending=False).head(20).round(2).to_string())

# ================================================================ PLOTS
hr("9. SAVING PLOTS")

# Plot 1: result distribution overall vs neutral
fig, ax = plt.subplots(figsize=(8, 5))
cats = ["home_win", "draw", "away_win"]
overall_pct = [df["outcome"].value_counts(normalize=True).get(c, 0) * 100 for c in cats]
neutral_pct = [neutral["outcome"].value_counts(normalize=True).get(c, 0) * 100 for c in cats]
x = np.arange(len(cats))
w = 0.38
ax.bar(x - w/2, overall_pct, w, label="All matches", color="#4C72B0")
ax.bar(x + w/2, neutral_pct, w, label="Neutral venue", color="#DD8452")
for i, (o, n) in enumerate(zip(overall_pct, neutral_pct)):
    ax.text(i - w/2, o + 0.5, f"{o:.1f}", ha="center", fontsize=9)
    ax.text(i + w/2, n + 0.5, f"{n:.1f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["Home win", "Draw", "Away win"])
ax.set_ylabel("% of matches")
ax.set_title("Result distribution: all matches vs neutral venue")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/plot1_result_distribution.png", dpi=120)
plt.close(fig)

# Plot 2: goals histogram with Poisson overlay
fig, ax = plt.subplots(figsize=(8, 5))
maxg = 12
counts = df["total_goals"].clip(upper=maxg)
ax.hist(counts, bins=np.arange(-0.5, maxg + 1.5, 1), density=True,
        color="#55A868", alpha=0.75, label="Observed total goals")
xs = np.arange(0, maxg + 1)
ax.plot(xs, stats.poisson.pmf(xs, lam), "o-", color="#C44E52",
        label=f"Poisson(λ={lam:.2f})")
ax.set_xlabel("Total goals in match")
ax.set_ylabel("Probability")
ax.set_title("Total goals per match vs Poisson fit")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/plot2_goals_poisson.png", dpi=120)
plt.close(fig)

# Plot 3: matches and avg goals over time (by decade)
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(dec.index, dec["matches"], width=8, color="#4C72B0", alpha=0.7, label="Matches")
ax1.set_xlabel("Decade")
ax1.set_ylabel("Matches per decade", color="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(dec.index, dec["avg_total_goals"], "o-", color="#C44E52", label="Avg total goals")
ax2.set_ylabel("Avg total goals/match", color="#C44E52")
ax1.set_title("Matches and average goals per decade")
fig.tight_layout()
fig.savefig(f"{OUT}/plot3_over_time.png", dpi=120)
plt.close(fig)

# Plot 4: home advantage — avg home vs away goals, neutral vs non-neutral
fig, ax = plt.subplots(figsize=(8, 5))
groups = ["All", "Non-neutral", "Neutral"]
home_g = [df["home_score"].mean(), non_neutral["home_score"].mean(), neutral["home_score"].mean()]
away_g = [df["away_score"].mean(), non_neutral["away_score"].mean(), neutral["away_score"].mean()]
x = np.arange(len(groups))
ax.bar(x - w/2, home_g, w, label="Home/listed team goals", color="#4C72B0")
ax.bar(x + w/2, away_g, w, label="Away/opponent goals", color="#DD8452")
for i, (h, a) in enumerate(zip(home_g, away_g)):
    ax.text(i - w/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("Avg goals")
ax.set_title("Home advantage: home vs away goals by venue type")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/plot4_home_advantage.png", dpi=120)
plt.close(fig)

print("Saved: plot1_result_distribution.png, plot2_goals_poisson.png, plot3_over_time.png, plot4_home_advantage.png")
print("\nEDA complete.")
